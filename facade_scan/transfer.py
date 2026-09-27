"""Carrying camera-space geometry into projector space.

Once the decoder has a dense camera-pixel -> projector-pixel map, this stage is
a lookup and nothing more. There is no calibration, no homography, and no
assumption that the facade is planar -- which is the entire reason for going to
the trouble of a structured-light scan in the first place.

Two things still need care.

Straight lines do not stay straight
-----------------------------------
A line drawn across the camera's view of the house, from the wall onto the
garage bump-out, is straight in camera space and **bent** in projector space,
because the two surfaces are a metre apart in depth. Transferring only the
endpoints and joining them would put the middle of that line into thin air. So
every edge is resampled at a fixed spacing first and carried point by point.

Depth discontinuities are boundaries, not gradients
---------------------------------------------------
Adjacent camera pixels looking at the same wall differ by a fraction of a
projector pixel. Adjacent camera pixels straddling the edge of the bump-out
differ by many, and there is no surface in between -- the projector pixels in
that gap land on wall that is hidden behind the garage. Nothing is interpolated
or morphologically bridged across such a jump.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .config import DecodeConfig, TransferConfig
from .decode import DecodeResult, discontinuity_mask
from .detect.regions import Region


def _resolved(cfg: TransferConfig | None, decoded: DecodeResult) -> TransferConfig:
    """Turn step-relative thresholds into projector pixels for this scan."""
    return (cfg or TransferConfig()).resolve(
        decoded.valid.shape, (decoded.projector_width, decoded.projector_height)
    )


@dataclass
class TransferResult:
    """Camera-space geometry, expressed in projector pixels."""

    regions: list[Region] = field(default_factory=list)
    #: Projector-resolution boolean mask, if one was transferred.
    mask: np.ndarray | None = None
    projector_width: int = 0
    projector_height: int = 0
    #: Labels of regions that could not be transferred, and why.
    dropped: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.projector_height, self.projector_width


# --------------------------------------------------------------------------- #
# Point lookup
# --------------------------------------------------------------------------- #
def densify(ring: np.ndarray, step: float) -> np.ndarray:
    """Resample a closed ring so that no edge is longer than ``step`` pixels.

    Vertices are always kept; points are inserted between them. This is what
    lets a transferred edge bend where the surface underneath it does.
    """
    ring = np.asarray(ring, dtype=np.float64)
    if len(ring) < 2:
        return ring.copy()
    step = max(float(step), 1e-6)

    out = []
    closed = np.vstack([ring, ring[:1]])
    for a, b in itertools.pairwise(closed):
        out.append(a)
        distance = float(np.linalg.norm(b - a))
        n = int(distance // step)
        if n >= 1:
            fractions = np.arange(1, n + 1) / (n + 1)
            out.extend(a + (b - a) * fractions[:, None])
    return np.array(out, dtype=np.float64)


def _spiral_offsets(radius: int) -> list[tuple[int, int]]:
    """Integer offsets within ``radius``, nearest first."""
    offsets = [(dy, dx)
               for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)]
    offsets.sort(key=lambda o: (o[0] * o[0] + o[1] * o[1]))
    return offsets


def lookup_points(decoded: DecodeResult, points: np.ndarray,
                  radius: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Map camera points to projector coordinates.

    A point landing on an undecoded camera pixel -- dark brick, a window, the
    rim of a shadow -- is retried against its nearest valid neighbours out to
    ``radius``. If nothing valid is within reach the point is reported as not
    found, and callers drop it rather than guess.

    Returns ``(projector_xy, found)``.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    height, width = decoded.valid.shape
    out = np.zeros((len(points), 2), dtype=np.float64)
    found = np.zeros(len(points), dtype=bool)

    base_x = np.rint(points[:, 0]).astype(np.int64)
    base_y = np.rint(points[:, 1]).astype(np.int64)

    for dy, dx in _spiral_offsets(max(0, int(radius))):
        pending = ~found
        if not pending.any():
            break
        ys = base_y[pending] + dy
        xs = base_x[pending] + dx
        inside = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
        ys, xs = np.clip(ys, 0, height - 1), np.clip(xs, 0, width - 1)
        usable = inside & decoded.valid[ys, xs]

        idx = np.nonzero(pending)[0][usable]
        out[idx] = decoded.proj_map[ys[usable], xs[usable]]
        found[idx] = True

    return out, found


# --------------------------------------------------------------------------- #
# Vector transfer
# --------------------------------------------------------------------------- #
def drop_ring_spikes(points: np.ndarray, threshold: float) -> np.ndarray:
    """Remove isolated outliers from a transferred ring, keeping real bends.

    A point that is far from both neighbours *while they are close to each
    other* has jumped away and come straight back -- a bad lookup, not a
    surface. A point at a genuine depth step is far from one neighbour and close
    to the other, so it survives, which it must: that bend is the whole reason
    edges are transferred point by point.
    """
    if len(points) < 3 or threshold <= 0:
        return points

    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    to_previous = np.linalg.norm(points - previous, axis=1)
    to_following = np.linalg.norm(points - following, axis=1)
    across = np.linalg.norm(following - previous, axis=1)

    spike = (to_previous > threshold) & (to_following > threshold) & (across <= threshold)
    return points[~spike]


def ring_is_plausible(points: np.ndarray, max_ratio: float) -> bool:
    """Reject a transferred ring that zigzags instead of tracing an outline.

    Compares total path length against the ring's own bounding-box diagonal.
    A real outline is a few times its diagonal; a ring whose points alternate
    between two surfaces is tens of times, and projecting it throws a bundle of
    parallel beams across the target.
    """
    if max_ratio <= 0 or len(points) < 3:
        return True
    extent = np.ptp(points, axis=0)
    diagonal = float(np.hypot(*extent))
    if diagonal < 1e-6:
        return False
    closed = np.vstack([points, points[:1]])
    perimeter = float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())
    return perimeter <= max_ratio * diagonal


def transfer_ring(decoded: DecodeResult, ring: np.ndarray,
                  cfg: TransferConfig | None = None) -> np.ndarray | None:
    """Transfer one closed ring, densified, dropping points that do not decode.

    A ring that crosses a depth discontinuity comes back with a genuine kink in
    it, and that kink is correct: to lay light along a straight line on a house
    with a bump-out in it, the projector must bend the line.
    """
    cfg = _resolved(cfg, decoded)
    dense = densify(ring, cfg.densify_step_px)
    if len(dense) < 3:
        return None
    mapped, found = lookup_points(decoded, dense, cfg.lookup_radius_px)
    kept = drop_ring_spikes(mapped[found], cfg.ring_outlier_px)
    if len(kept) < 3:
        return None
    if not ring_is_plausible(kept, cfg.ring_max_perimeter_ratio):
        return None
    return kept


def transfer_region(decoded: DecodeResult, region: Region,
                    cfg: TransferConfig | None = None) -> Region | None:
    """Transfer a region, holes and all, into projector space."""
    cfg = _resolved(cfg, decoded)
    exterior = transfer_ring(decoded, region.polygon, cfg)
    if exterior is None:
        return None

    transferred = Region(polygon=exterior, label=region.label,
                         attributes=dict(region.attributes))
    for hole in region.holes:
        mapped = transfer_ring(decoded, hole, cfg)
        if mapped is not None and len(mapped) >= 3:
            transferred.holes.append(mapped)

    if transferred.area < cfg.min_polygon_area_px:
        return None
    transferred.attributes["projector_area_px"] = float(transferred.area)
    transferred.attributes["camera_area_px"] = float(region.area)
    return transferred


def transfer_regions(decoded: DecodeResult, regions: list[Region],
                     cfg: TransferConfig | None = None) -> TransferResult:
    """Transfer a whole detection, reporting anything that had to be dropped."""
    cfg = _resolved(cfg, decoded)
    result = TransferResult(projector_width=decoded.projector_width,
                            projector_height=decoded.projector_height)
    for region in regions:
        moved = transfer_region(decoded, region, cfg)
        if moved is None:
            result.dropped.append(
                f"{region.label}: did not transfer cleanly -- too little of "
                "it decoded, or its outline came back tangled"
            )
        else:
            result.regions.append(moved)
    return result


# --------------------------------------------------------------------------- #
# Raster transfer
# --------------------------------------------------------------------------- #
def scatter_mask(decoded: DecodeResult, camera_mask: np.ndarray,
                 cfg: TransferConfig | None = None) -> np.ndarray:
    """Scatter a camera-space mask into projector space, with no filling.

    Camera pixels sitting on a depth discontinuity are excluded. Their decode is
    the least trustworthy in the frame -- projector blur straddles two surfaces
    there -- and they are exactly the pixels that would seed a fill across the
    gap between two surfaces at different depths.
    """
    cfg = _resolved(cfg, decoded)
    usable = camera_mask & decoded.valid
    usable &= ~discontinuity_mask(decoded.proj_map, decoded.valid,
                                  cfg.discontinuity_jump_px)

    out = np.zeros((decoded.projector_height, decoded.projector_width), dtype=bool)
    xs = decoded.proj_map[..., 0][usable]
    ys = decoded.proj_map[..., 1][usable]
    inside = ((xs >= 0) & (xs < decoded.projector_width)
              & (ys >= 0) & (ys < decoded.projector_height))
    out[ys[inside], xs[inside]] = True
    return out


def fill_small_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
    """Fill enclosed holes up to ``max_area``, leaving anything open alone.

    An enclosed hole is a resolution artefact -- a projector pixel no camera
    pixel happened to land on. A gap that reaches the edge of the mask is a
    boundary, and boundaries are left where they are.
    """
    import cv2

    inverse = (~mask).astype(np.uint8)
    # The bundled OpenCV stub does not list uint8 here, though uint8 is exactly
    # what connectedComponentsWithStats requires at runtime.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverse, 8)  # type: ignore[call-overload]
    height, width = mask.shape
    out = mask.copy()
    for i in range(1, count):
        x, y, w, h, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                            stats[i, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if not touches_border and area <= max_area:
            out[labels == i] = True
    return out


def transfer_mask(decoded: DecodeResult, camera_mask: np.ndarray,
                  cfg: TransferConfig | None = None) -> np.ndarray:
    """Scatter a camera-space mask to projector space, then tidy it.

    Scattering leaves pinholes wherever the projector locally out-resolves the
    camera -- an oblique wall, or the far end of a wide facade. Closing and
    hole-filling remove them.

    Keep ``close_kernel_px`` below the disparity jump at your deepest step, or
    the close will weld two surfaces together across the gap between them. The
    jump in projector pixels is roughly ``fx * B * (1/z_near - 1/z_far)`` for a
    camera-projector baseline ``B``; with the camera mounted right beside the
    projector as it should be, that is small, and so should this be.
    """
    import cv2

    cfg = _resolved(cfg, decoded)
    mask = scatter_mask(decoded, camera_mask, cfg)

    if cfg.close_kernel_px > 1:
        k = int(cfg.close_kernel_px) | 1        # odd sizes only
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                kernel).astype(bool)

    if cfg.max_hole_area_px > 0:
        mask = fill_small_holes(mask, cfg.max_hole_area_px)
    return mask


# --------------------------------------------------------------------------- #
# Isolating what stands in front
# --------------------------------------------------------------------------- #
def dominant_plane_residual(decoded: DecodeResult, sample: int = 60_000,
                            seed: int = 0) -> np.ndarray:
    """Per-pixel distance from the scene's dominant plane, in projector pixels.

    The decoded map over any *planar* surface is exactly a homography. Fit one
    robustly and whatever most of the scene is becomes the reference; anything
    at a different depth has a large residual, however similar it looks.

    This is geometry, not photometry. A cardboard cutout the same colour as the
    door behind it is invisible to any amount of image processing and obvious
    here, because it is 200 mm nearer and the correspondence says so.
    """
    import cv2

    out = np.zeros(decoded.valid.shape, np.float32)
    ys, xs = np.nonzero(decoded.valid)
    if len(ys) < 500:
        return out

    rng = np.random.default_rng(seed)
    pick = (rng.choice(len(ys), sample, replace=False)
            if len(ys) > sample else np.arange(len(ys)))
    camera = np.stack([xs[pick], ys[pick]], -1).astype(np.float32)
    projector = decoded.proj_map[ys[pick], xs[pick]].astype(np.float32)

    homography, _ = cv2.findHomography(camera, projector, cv2.RANSAC, 2.0)
    if homography is None:
        return out

    everything = np.stack([xs, ys], -1).astype(np.float32)
    predicted = cv2.perspectiveTransform(everything.reshape(-1, 1, 2),
                                         homography).reshape(-1, 2)
    actual = decoded.proj_map[ys, xs].astype(np.float32)
    out[ys, xs] = np.linalg.norm(predicted - actual, axis=1)
    return out


def foreground_mask(decoded: DecodeResult,
                    cfg: TransferConfig | None = None) -> np.ndarray:
    """Camera-space mask of everything standing in front of the background.

    For a facade this separates the house from whatever is behind it. For a
    cutout on a stand it separates the cutout from the wall. It is the mask you
    want when the goal is "light *that*, and nothing else" -- which is usually
    the goal, since `house_mask` covers every surface the projector reaches,
    background included.

    How well it works depends on the camera-projector baseline, and this is the
    one place where the usual advice inverts. Everywhere else you want the
    camera as close to the projector as possible, to minimise the shadows it
    cannot see into. But depth resolution is stereo disparity, and disparity is
    proportional to that same baseline: with the two devices together, a step
    that is obvious to the eye may be a pixel or two in the correspondence.

    Measured on the synthetic house, separating a 1 m bump-out at 14 m:

    ========  ==================  =================
    baseline  covered of bump-out  leaked onto wall
    ========  ==================  =================
    0.6 m     59%                 13%
    2.2 m     100%                0%
    ========  ==================  =================

    So if isolating the foreground matters more than filling shadows, move the
    camera further from the projector. What also helps, for free, is relative
    depth: a cutout 200 mm off a wall at 1 m separates far more cleanly than a
    1 m bump-out at 14 m, because disparity scales with ``1/z_near - 1/z_far``.
    """
    import cv2

    cfg = _resolved(cfg, decoded)
    residual = dominant_plane_residual(decoded, cfg.foreground_sample)
    values = residual[decoded.valid]
    if values.size < 500:
        return np.zeros(decoded.valid.shape, bool)

    threshold = cfg.foreground_residual_px
    if threshold <= 0:
        # The two surfaces separate cleanly, so let Otsu find the gap rather
        # than making the operator guess a depth in projector pixels.
        scale = max(float(np.percentile(values, 99)), 1e-6)
        as_bytes = np.clip(values / scale * 255.0, 0, 255).astype(np.uint8)
        level, _ = cv2.threshold(as_bytes, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = float(level) / 255.0 * scale

    mask = decoded.valid & (residual > threshold)
    if cfg.foreground_close_px > 1:
        k = int(cfg.foreground_close_px) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                kernel).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                                kernel).astype(bool)

    # Keep only blobs comparable in size to the largest: a foreground object is
    # a thing, not a scattering of speckle.
    if cfg.foreground_min_blob_frac > 0 and mask.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8)  # type: ignore[call-overload]
        if count > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep = 1 + np.nonzero(areas >= cfg.foreground_min_blob_frac * areas.max())[0]
            mask = np.isin(labels, keep)
    return mask


def largest_blob(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected piece of a mask."""
    import cv2

    if not mask.any():
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)  # type: ignore[call-overload]
    if count <= 2:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def inset_mask(mask: np.ndarray, pixels: float) -> np.ndarray:
    """Pull a mask's boundary inward by ``pixels``."""
    import cv2

    if pixels <= 0 or not mask.any():
        return mask
    size = round(pixels) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


def foreground_projector_mask(decoded: DecodeResult,
                              cfg: TransferConfig | None = None) -> np.ndarray:
    """The foreground, transferred into projector space and tidied.

    Reduced to a single connected piece and pulled in slightly, so what is
    projected is one clean shape that lands on the subject rather than a spray
    of dots with its edge spilling onto the background.
    """
    cfg = _resolved(cfg, decoded)
    mask = transfer_mask(decoded, foreground_mask(decoded, cfg), cfg)
    if cfg.foreground_single_blob:
        mask = largest_blob(mask)
    return inset_mask(mask, cfg.foreground_inset_px)


# --------------------------------------------------------------------------- #
# The shortcut that is most of the practical value
# --------------------------------------------------------------------------- #
def house_mask(decoded: DecodeResult, cfg: TransferConfig | None = None,
               decode_cfg: DecodeConfig | None = None,
               exclude_glass: bool = False) -> np.ndarray:
    """Projector-space mask of the house: "project here, and nowhere else".

    This is a one-step operation and the single most useful thing the tool
    produces. Thresholding the all-white capture against the all-black one finds
    every surface the projector is actually landing on -- and nothing else, no
    sky, no neighbour's hedge, no driveway -- and the decoded map carries that
    straight into projector pixels.

    With nothing more than this you can light a house accurately and stop light
    spilling onto everything around it, which is most of what projection mapping
    a facade is for.

    ``exclude_glass`` additionally drops anything the decoder flagged as
    glazing, for when you would rather not throw light through the windows at
    whoever is inside.
    """
    decode_cfg = decode_cfg or DecodeConfig()
    lit = decoded.illumination >= decode_cfg.illumination_threshold
    if exclude_glass:
        lit &= ~decoded.likely_glass
    return transfer_mask(decoded, lit, cfg)
