"""Line segment detection, vanishing points, and collinear merging.

Run on the all-white capture, which is the one frame in the set that is simply a
photograph of the house with the projector acting as a floodlight. Flat, even,
shadowless illumination is close to ideal for edge detection -- far better than
whatever the streetlights were doing.

On the detector
---------------
``cv2.ximgproc.createFastLineDetector`` from opencv-contrib, not
``cv2.createLineSegmentDetector``: the latter was removed from mainline OpenCV
over a patent dispute and is absent or a stub depending on your build.

The pipeline
------------
1. **Detect** raw segments. A house yields hundreds, most of them siding
   courses, shingle edges and brick joints.
2. **Drop short ones.** This single threshold removes most of that texture,
   because architectural edges -- rooflines, window heads, corner boards -- are
   long and siding courses are not.
3. **Find vanishing points** by RANSAC over pairs of segments. A house
   photographed from the ground shows two or three dominant directions: one
   vertical family, and one or two horizontal families depending on how oblique
   the view is.
4. **Snap** each segment to point exactly at its vanishing point. Detector noise
   of a degree or two is enough to stop two pieces of the same roofline from
   merging, and snapping removes it.
5. **Merge** collinear segments back into whole architectural edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import DetectConfig

#: Segments are stored as an (N, 4) float array of x1, y1, x2, y2.
Segments = np.ndarray


# --------------------------------------------------------------------------- #
# Basic segment geometry
# --------------------------------------------------------------------------- #
def lengths(segments: Segments) -> np.ndarray:
    return np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])


def midpoints(segments: Segments) -> np.ndarray:
    return np.stack([(segments[:, 0] + segments[:, 2]) / 2.0,
                     (segments[:, 1] + segments[:, 3]) / 2.0], axis=-1)


def directions(segments: Segments) -> np.ndarray:
    """Unit direction of each segment."""
    d = np.stack([segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]], -1)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return d / np.maximum(n, 1e-12)


def homogeneous_lines(segments: Segments) -> np.ndarray:
    """The infinite line through each segment, as a homogeneous 3-vector."""
    p1 = np.stack([segments[:, 0], segments[:, 1], np.ones(len(segments))], -1)
    p2 = np.stack([segments[:, 2], segments[:, 3], np.ones(len(segments))], -1)
    lines = np.cross(p1, p2)
    norm = np.linalg.norm(lines[:, :2], axis=-1, keepdims=True)
    return lines / np.maximum(norm, 1e-12)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def detect_segments(image: np.ndarray, cfg: DetectConfig | None = None) -> Segments:
    """Raw line segments from a grayscale or colour image."""
    import cv2

    cfg = cfg or DetectConfig()
    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError(
            "cv2.ximgproc is missing. Install opencv-contrib-python (not "
            "opencv-python); FastLineDetector lives in the contrib modules, and "
            "cv2.createLineSegmentDetector is not a substitute -- it was removed "
            "from mainline OpenCV."
        )

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray * 255 if gray.max() <= 1.0 else gray, 0, 255).astype(np.uint8)

    detector = cv2.ximgproc.createFastLineDetector(
        length_threshold=int(cfg.fld_length_threshold),
        distance_threshold=float(cfg.fld_distance_threshold),
        canny_th1=float(cfg.fld_canny_th1),
        canny_th2=float(cfg.fld_canny_th2),
        canny_aperture_size=int(cfg.fld_canny_aperture_size),
        do_merge=bool(cfg.fld_do_merge),
    )
    raw = detector.detect(gray)
    if raw is None or len(raw) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray(raw, dtype=np.float64).reshape(-1, 4)


def detect_depth_segments(depth_edges: np.ndarray,
                          cfg: DetectConfig | None = None) -> Segments:
    """Line segments along boundaries in the decoded correspondence.

    These are *geometric* edges: places where the surface steps toward or away
    from the projector. They exist regardless of how the two surfaces compare in
    brightness, which is exactly where photometric edge detection gives up.

    The binary edge map is thickened slightly first. A one-pixel line gives
    Canny very little gradient to work with, and the detector reads a dashed
    trail of fragments instead of an edge.
    """
    import cv2

    cfg = cfg or DetectConfig()
    edges = np.where(np.asarray(depth_edges), 255, 0).astype(np.uint8)
    thick = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    softened = np.asarray(cv2.GaussianBlur(thick, (0, 0), sigmaX=1.0), dtype=np.uint8)
    return detect_segments(softened, cfg)


def filter_short(segments: Segments, min_length: float) -> Segments:
    """Discard segments below ``min_length``.

    This is the main texture rejector: siding courses, shingle edges and brick
    joints are short, architectural edges are long.
    """
    if len(segments) == 0:
        return segments
    return segments[lengths(segments) >= min_length]


# --------------------------------------------------------------------------- #
# Vanishing points
# --------------------------------------------------------------------------- #
@dataclass
class VanishingPoint:
    """A dominant direction in the image.

    Stored homogeneously so that a truly parallel family -- a vanishing point at
    infinity, which is what you get photographing a facade dead-on -- is
    representable rather than an overflow.
    """

    #: Homogeneous 3-vector, unit norm.
    point: np.ndarray
    #: Indices into the segment array that voted for it.
    inliers: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    #: Total length of those segments, in pixels. The strength of the evidence.
    support: float = 0.0

    @property
    def at_infinity(self) -> bool:
        return abs(float(self.point[2])) < 1e-9

    @property
    def image_point(self) -> np.ndarray | None:
        """Pixel coordinates, or None if the point is at infinity."""
        if self.at_infinity:
            return None
        return self.point[:2] / self.point[2]

    def direction_from(self, points: np.ndarray) -> np.ndarray:
        """Unit direction from each of ``points`` toward this vanishing point.

        ``v[:2] - v[2] * p`` is the projective form: it degrades gracefully to
        the plain direction ``v[:2]`` as ``v[2]`` goes to zero, so there is no
        special case for a point at infinity and no division by a tiny number.
        """
        points = np.atleast_2d(points)
        d = self.point[:2][None, :] - self.point[2] * points
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        return d / np.maximum(n, 1e-12)


def _consistency(segments: Segments, point: np.ndarray) -> np.ndarray:
    """Angle, in degrees, between each segment and the direction to ``point``."""
    vp = VanishingPoint(point=point)
    toward = vp.direction_from(midpoints(segments))
    cosine = np.abs(np.sum(toward * directions(segments), axis=-1))
    return np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))


def _refine(lines: np.ndarray) -> np.ndarray:
    """Least-squares vanishing point: the null vector of the inlier line matrix.

    A vanishing point lies on every line of its family, so ``l . v = 0`` for all
    of them, so ``v`` is the smallest right singular vector of the stacked lines.
    """
    _, _, vt = np.linalg.svd(lines)
    v = vt[-1]
    return v / max(float(np.linalg.norm(v)), 1e-12)


def estimate_vanishing_points(segments: Segments,
                              cfg: DetectConfig | None = None) -> list[VanishingPoint]:
    """Find the dominant directions by RANSAC over pairs of segments.

    Each pair of segments proposes a vanishing point (the intersection of their
    infinite lines); the proposal is scored by the total *length* of segments
    consistent with it, because one long roofline is better evidence than four
    short bits of siding. The best family is refined, removed, and the search
    repeats on what is left.
    """
    cfg = cfg or DetectConfig()
    if len(segments) < 2:
        return []

    lines = homogeneous_lines(segments)
    seg_lengths = lengths(segments)
    remaining = np.arange(len(segments))
    rng = np.random.default_rng(cfg.vp_random_seed)
    found: list[VanishingPoint] = []

    while len(found) < cfg.max_vanishing_points and len(remaining) >= cfg.vp_min_inliers:
        pool = segments[remaining]
        pool_lines = lines[remaining]
        pool_lengths = seg_lengths[remaining]

        best_point: np.ndarray | None = None
        best_support = 0.0
        best_inliers = np.zeros(0, dtype=int)

        for _ in range(cfg.vp_ransac_iterations):
            i, j = rng.choice(len(pool), size=2, replace=False)
            candidate = np.cross(pool_lines[i], pool_lines[j])
            norm = float(np.linalg.norm(candidate))
            if norm < 1e-9:      # the two lines coincide
                continue
            candidate = candidate / norm

            inliers = np.nonzero(_consistency(pool, candidate) <= cfg.vp_inlier_angle_deg)[0]
            support = float(pool_lengths[inliers].sum())
            if support > best_support:
                best_point, best_support, best_inliers = candidate, support, inliers

        if best_point is None or len(best_inliers) < cfg.vp_min_inliers:
            break

        refined = _refine(pool_lines[best_inliers])
        refined_inliers = np.nonzero(
            _consistency(pool, refined) <= cfg.vp_inlier_angle_deg
        )[0]
        if len(refined_inliers) >= len(best_inliers):
            best_point, best_inliers = refined, refined_inliers
            best_support = float(pool_lengths[best_inliers].sum())

        found.append(VanishingPoint(point=best_point,
                                    inliers=remaining[best_inliers].copy(),
                                    support=best_support))
        remaining = np.delete(remaining, best_inliers)

    found.sort(key=lambda v: -v.support)
    return found


def snap_to_vanishing_points(segments: Segments, vanishing_points: list[VanishingPoint],
                             cfg: DetectConfig | None = None) -> Segments:
    """Rotate each segment about its midpoint to point exactly at its VP.

    Detector noise of a degree or two is plenty to stop two halves of the same
    roofline, interrupted by a downpipe, from being recognised as collinear.
    Snapping removes that noise. Length and midpoint are preserved, so nothing
    moves anywhere it was not already.
    """
    cfg = cfg or DetectConfig()
    if not cfg.vp_snap or len(segments) == 0:
        return segments.copy()

    out = segments.copy()
    for vp in vanishing_points:
        if len(vp.inliers) == 0:
            continue
        idx = vp.inliers
        mids = midpoints(segments[idx])
        half = lengths(segments[idx])[:, None] / 2.0
        toward = vp.direction_from(mids)

        # Keep each segment pointing the way it already pointed, so that
        # endpoint order is stable for anything downstream.
        sign = np.sign(np.sum(toward * directions(segments[idx]), axis=-1))[:, None]
        sign[sign == 0] = 1.0
        toward = toward * sign

        out[idx, 0:2] = mids - toward * half
        out[idx, 2:4] = mids + toward * half
    return out


# --------------------------------------------------------------------------- #
# Collinear merging
# --------------------------------------------------------------------------- #
def _point_line_distance(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Distance from points to a normalised homogeneous line."""
    return np.abs(points @ line[:2] + line[2])


def merge_collinear(segments: Segments, cfg: DetectConfig | None = None) -> Segments:
    """Fuse segments that are pieces of the same architectural edge.

    Three tests, all of which must pass:

    ``merge_angle_deg``
        the two directions agree;
    ``merge_perp_px``
        each segment's midpoint lies on the other's infinite line, so two
        parallel window heads one above the other are not fused;
    ``merge_gap_px``
        the two pieces are actually near each other along that shared line.
        Without this last one, a window sill and a roofline that happen to be
        collinear get welded into a single edge running across the whole house.
    """
    cfg = cfg or DetectConfig()
    n = len(segments)
    if n < 2:
        return segments.copy()

    lines = homogeneous_lines(segments)
    dirs = directions(segments)
    mids = midpoints(segments)
    cos_limit = np.cos(np.radians(cfg.merge_angle_deg))

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j in range(i + 1, n):
            if abs(float(dirs[i] @ dirs[j])) < cos_limit:
                continue
            perp = max(float(_point_line_distance(mids[j:j + 1], lines[i])[0]),
                       float(_point_line_distance(mids[i:i + 1], lines[j])[0]))
            if perp > cfg.merge_perp_px:
                continue

            axis = dirs[i]
            a = np.sort(np.array([segments[i, 0:2] @ axis, segments[i, 2:4] @ axis]))
            b = np.sort(np.array([segments[j, 0:2] @ axis, segments[j, 2:4] @ axis]))
            gap = max(0.0, max(b[0] - a[1], a[0] - b[1]))
            if gap > cfg.merge_gap_px:
                continue
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(segments[members[0]])
            continue
        member_dirs = dirs[members]
        # Flip to a common orientation before averaging, or opposite-pointing
        # halves of one edge would cancel out.
        reference = member_dirs[int(np.argmax(lengths(segments[members])))]
        flipped = member_dirs * np.sign(member_dirs @ reference)[:, None]
        axis = flipped.mean(axis=0)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)

        endpoints = np.concatenate([segments[members][:, 0:2], segments[members][:, 2:4]])
        centre = endpoints.mean(axis=0)
        t = (endpoints - centre) @ axis
        merged.append(np.concatenate([centre + axis * t.min(), centre + axis * t.max()]))

    return np.array(merged, dtype=np.float64).reshape(-1, 4)


def _point_segment_distance(point: np.ndarray, segments: Segments) -> np.ndarray:
    """Distance from one point to each segment's nearest point."""
    a = segments[:, 0:2]
    b = segments[:, 2:4]
    ab = b - a
    denominator = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
    t = np.clip(np.sum((point - a) * ab, axis=1) / denominator, 0.0, 1.0)
    nearest = a + ab * t[:, None]
    return np.linalg.norm(point - nearest, axis=1)


def snap_endpoints(segments: Segments, cfg: DetectConfig | None = None) -> Segments:
    """Pull segment ends onto the corners they were reaching for.

    Line detectors stop short at corners, where two edges meet and local
    contrast momentarily vanishes. The blunt fix is to grow every segment at
    both ends until the near-misses overlap, but the amount of growth needed is
    set by the *worst* corner in the frame. On a real scan that was 220 pixels,
    and extending every segment by 220 pixels in both directions manufactured
    fifty spurious faces to recover one real one.

    Snapping is the precise version. For each loose end, the intersection of its
    own line with a nearby, sufficiently non-parallel neighbour is computed, and
    the end is moved exactly there -- if the intersection is close to the end
    (``snap_radius_px``) *and* close to the neighbour, so an edge cannot be
    dragged to a line on the far side of the house. Corners then close exactly
    where they belong, and nothing else moves at all.

    Near-parallel pairs are skipped: their intersection is wildly sensitive to a
    fraction of a degree of noise, so it is not evidence of a corner.
    """
    cfg = cfg or DetectConfig()
    n = len(segments)
    if n < 2 or cfg.snap_radius_px <= 0:
        return segments.copy()

    lines = homogeneous_lines(segments)
    dirs = directions(segments)
    cos_limit = np.cos(np.radians(cfg.snap_min_angle_deg))
    # How far each end may credibly reach: bounded by the segment's own length,
    # so a long roofline can close an obvious near-miss while a short fragment
    # cannot invent a corner two hundred pixels away.
    allowance = np.minimum(cfg.snap_radius_px,
                           cfg.snap_length_frac * lengths(segments))
    out = segments.copy()

    for i in range(n):
        crossings = np.cross(lines[i], lines)          # (n, 3) homogeneous
        w = crossings[:, 2]
        usable = np.abs(w) > 1e-9
        usable[i] = False
        # Ill-conditioned when the two lines are nearly parallel.
        usable &= np.abs(dirs @ dirs[i]) <= cos_limit
        if not usable.any():
            continue

        points = np.full((n, 2), np.inf)
        points[usable] = crossings[usable, :2] / w[usable, None]

        for columns in (slice(0, 2), slice(2, 4)):
            tip = segments[i, columns]
            to_tip = np.linalg.norm(points - tip, axis=1)
            # A corner has to be plausible for both segments.
            reach = np.minimum(allowance[i], allowance)
            near_tip = usable & (to_tip <= reach)
            if not near_tip.any():
                continue
            # The corner must belong to the other segment too, not merely to
            # its infinite line.
            to_other = np.full(n, np.inf)
            for j in np.nonzero(near_tip)[0]:
                to_other[j] = _point_segment_distance(points[j], segments[j:j + 1])[0]
            candidates = near_tip & (to_other <= reach)
            if not candidates.any():
                continue
            best = int(np.argmin(np.where(candidates, to_tip, np.inf)))
            out[i, columns] = points[best]

    return out


def detect_lines(image: np.ndarray, cfg: DetectConfig | None = None
                 ) -> tuple[Segments, list[VanishingPoint]]:
    """The whole line pipeline: detect, filter, find VPs, snap, merge."""
    cfg = (cfg or DetectConfig()).resolve(image.shape)
    raw = detect_segments(image, cfg)
    long_enough = filter_short(raw, cfg.min_segment_length_px)
    vanishing_points = estimate_vanishing_points(long_enough, cfg)
    snapped = snap_to_vanishing_points(long_enough, vanishing_points, cfg)
    return snap_endpoints(merge_collinear(snapped, cfg), cfg), vanishing_points
