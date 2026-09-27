"""Gray-code decoding: captured images -> dense camera-pixel -> projector-pixel map.

This is the step that makes the whole approach work. Once every camera pixel
knows which projector pixel lit it, *anything* found in camera space transfers
to projector space by lookup. There is no camera calibration, no projector
calibration, no homography, and -- crucially -- no assumption that the facade is
one plane. The garage bump-out, the gables, the window reveals and the eaves are
all at different depths, and none of that matters here, because the
correspondence is measured per-pixel rather than modelled.

Reading a bit
-------------
Each pattern is compared against its own inverse, per pixel::

    bit = (pattern > inverse)

Surface albedo, ambient light, lens vignetting and the projector's own falloff
multiply both exposures equally, so they cancel and no global threshold is ever
needed. Dark brick and white trim decode the same way.

Confidence
----------
``|pattern - inverse|`` is a free per-pixel, per-bit confidence. It collapses to
zero wherever the surface returns no pattern modulation, which happens for three
physically distinct reasons:

- **deep shadow** -- the projector cannot reach the point at all
- **very dark surfaces** -- so little light comes back that sensor noise wins
- **glass** -- windows are specular and mostly transparent, so the pattern goes
  through the pane or bounces off at an angle that misses the camera, while
  glare from the projector body and the street keeps the pixel *bright*

A pixel is decoded only if **every** bit clears the threshold, because Gray code
or not, one wrong bit is a wrong coordinate. Pixels that fail are marked invalid
rather than decoded into garbage -- garbage coordinates are far worse than
missing ones, since they scatter light onto random parts of the house.

The third case above is worth exposing on its own. A pixel that is bright in the
white frame but carries almost no pattern modulation is, on a house at night,
overwhelmingly likely to be glazing. That falls out of the decode for free and
is published as :attr:`DecodeResult.likely_glass`, which the window detector can
use as a prior.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DecodeConfig
from .patterns import Frame, Manifest, gray_to_binary

#: A function that returns one captured frame as float32 grayscale in [0, 1].
FrameLoader = Callable[[Frame], np.ndarray]


@dataclass
class DecodeResult:
    """Dense camera -> projector correspondence."""

    #: (H, W, 2) int32 projector (x, y) for each camera pixel. Meaningless
    #: where ``valid`` is False.
    proj_map: np.ndarray
    #: (H, W) bool -- this pixel's projector coordinate can be trusted.
    valid: np.ndarray
    #: (H, W) float32 -- the *weakest* bit's |pattern - inverse|. This is what
    #: validity is thresholded on.
    min_confidence: np.ndarray
    #: (H, W) float32 -- mean |pattern - inverse| across all bits.
    mean_confidence: np.ndarray
    #: (H, W) float32 -- white minus black, i.e. how much projector light
    #: actually lands here, with ambient and glare removed.
    illumination: np.ndarray
    #: (H, W) float32 -- the all-white capture.
    white: np.ndarray
    #: (H, W) float32 -- the all-black capture.
    black: np.ndarray
    #: (H, W) bool -- bright but unmodulated: almost certainly glazing.
    likely_glass: np.ndarray
    #: Projector resolution these coordinates address.
    projector_width: int
    projector_height: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.valid.shape  # type: ignore[return-value]

    @property
    def coverage(self) -> float:
        """Fraction of camera pixels successfully decoded."""
        return float(self.valid.mean())

    def stats(self) -> dict[str, float]:
        """Summary numbers worth printing after a scan and storing in scan.json."""
        lit = self.illumination > 0
        return {
            "camera_width": float(self.valid.shape[1]),
            "camera_height": float(self.valid.shape[0]),
            "projector_width": float(self.projector_width),
            "projector_height": float(self.projector_height),
            "coverage": self.coverage,
            "valid_pixels": float(self.valid.sum()),
            "likely_glass_pixels": float(self.likely_glass.sum()),
            "mean_confidence": float(self.mean_confidence[lit].mean()) if lit.any() else 0.0,
            "median_min_confidence": (
                float(np.median(self.min_confidence[self.valid])) if self.valid.any() else 0.0
            ),
        }

    def save(self, path: str | Path) -> Path:
        """Write the map and every mask to a single compressed .npz.

        Everything a later stage needs is in here, so a scan can be re-detected,
        re-transferred and re-exported without going back to the house.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            proj_map=self.proj_map.astype(np.int32),
            valid=self.valid,
            min_confidence=self.min_confidence.astype(np.float32),
            mean_confidence=self.mean_confidence.astype(np.float32),
            illumination=self.illumination.astype(np.float32),
            white=self.white.astype(np.float32),
            black=self.black.astype(np.float32),
            likely_glass=self.likely_glass,
            projector_size=np.array([self.projector_width, self.projector_height],
                                    dtype=np.int32),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> DecodeResult:
        with np.load(path) as d:
            size = d["projector_size"]
            return cls(
                proj_map=d["proj_map"], valid=d["valid"],
                min_confidence=d["min_confidence"], mean_confidence=d["mean_confidence"],
                illumination=d["illumination"], white=d["white"], black=d["black"],
                likely_glass=d["likely_glass"],
                projector_width=int(size[0]), projector_height=int(size[1]),
            )


# --------------------------------------------------------------------------- #
# Loading captures
# --------------------------------------------------------------------------- #
def image_loader(capture_dir: str | Path) -> FrameLoader:
    """A :data:`FrameLoader` reading ``<capture_dir>/<frame.filename>``."""
    import cv2

    root = Path(capture_dir)

    def load(frame: Frame) -> np.ndarray:
        path = root / frame.filename
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"capture missing for frame {frame.index}: {path}")
        return img.astype(np.float32) / 255.0

    return load


def array_loader(images: dict[int, np.ndarray]) -> FrameLoader:
    """A :data:`FrameLoader` over in-memory images keyed by frame index."""

    def load(frame: Frame) -> np.ndarray:
        img = images[frame.index]
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        return np.asarray(img, dtype=np.float32)

    return load


# --------------------------------------------------------------------------- #
# Median filter that respects depth discontinuities
# --------------------------------------------------------------------------- #
def median_filter_map(proj_map: np.ndarray, valid: np.ndarray, ksize: int,
                      max_jump: float, row_chunk: int = 128,
                      min_support: int = 4,
                      replace_threshold: float = 2.0) -> np.ndarray:
    """Remove isolated decode errors without smearing across depth steps.

    This is an outlier-rejection filter rather than a blanket median. The
    distinction matters. A plain median filter applied to a decoded map does two
    unwanted things:

    - it averages the garage bump-out's projector coordinate with the wall a
      metre behind it, inventing correspondences for points on neither surface;
    - even with a jump gate to prevent that, it *shifts* the pixels next to a
      depth step and along the frame border, because a decoded map is locally a
      ramp and the median of a ramp seen through a one-sided window is biased by
      half a window.

    So each pixel is classified first:

    ``support``
        how many valid neighbours lie within ``max_jump`` projector pixels of
        the centre -- that is, how many appear to be on the same surface.

    A pixel **with** support is on a surface, possibly right at its edge. Its
    gated median is computed from same-surface neighbours only, and it is
    replaced only if it differs from that median by more than
    ``replace_threshold``. Ramp bias is a fraction of a pixel, so edge and
    border pixels are left exactly alone while genuine small errors are fixed.

    A pixel **without** support disagrees with its entire neighbourhood, which
    is the signature of a gross decode error -- the kind that lands a hundred
    pixels away and throws light onto an unrelated part of the house. It is
    replaced by the median of all its valid neighbours. Note that a jump gate
    alone cannot fix these: the gate would reject every neighbour and leave the
    error in place, which is exactly backwards.

    Invalid centre pixels are left invalid. Nothing here inpaints; a missing
    correspondence stays missing.
    """
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError("median_ksize must be an odd number >= 3")

    from numpy.lib.stride_tricks import sliding_window_view

    h, w = valid.shape
    pad = ksize // 2
    centre_index = (ksize * ksize) // 2
    out = proj_map.copy()

    for channel in range(2):
        values = proj_map[..., channel].astype(np.float32)
        values = np.where(valid, values, np.nan)
        padded = np.pad(values, pad, mode="constant", constant_values=np.nan)

        for r0 in range(0, h, row_chunk):
            r1 = min(r0 + row_chunk, h)
            block = padded[r0:r1 + 2 * pad, :]
            win = sliding_window_view(block, (ksize, ksize))       # (rows, w, k, k)
            win = win.reshape(r1 - r0, w, ksize * ksize)
            centre = values[r0:r1, :, None]

            # NaN compares False everywhere, so invalid neighbours drop out.
            same_surface = np.abs(win - centre) <= max_jump
            support = same_surface.sum(axis=-1) - 1   # the centre matches itself

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                gated = np.nanmedian(np.where(same_surface, win, np.nan), axis=-1)
                neighbours = win.copy()
                neighbours[..., centre_index] = np.nan
                ungated = np.nanmedian(neighbours, axis=-1)

            centre2d = values[r0:r1]
            supported = support >= min_support
            nudged = np.where(np.abs(centre2d - gated) > replace_threshold, gated, centre2d)
            rescued = np.where(np.isfinite(ungated), ungated, centre2d)
            result = np.where(supported, nudged, rescued)

            good = valid[r0:r1] & np.isfinite(result)
            out[r0:r1, :, channel] = np.where(good, np.rint(result),
                                              proj_map[r0:r1, :, channel])
    return out


def inpaint_map(proj_map: np.ndarray, valid: np.ndarray, radius: int,
                agreement: float,
                allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fill undecoded gaps from the decoded pixels around them, where safe.

    Dark brick, a scratch of specular glare, a shadow rim: plenty of pixels sit
    in the middle of a perfectly ordinary wall and fail to decode. Left alone
    they become holes in the projected mask, and a hole in the mask is a patch
    of house left dark.

    They are filled one ring at a time from the outside in, and only where the
    already-decoded neighbours **agree with each other** to within ``agreement``
    projector pixels. That agreement test is what keeps the guarantee: a gap in
    the middle of a flat surface has neighbours that agree, so it fills; a gap
    straddling the edge of a bump-out has neighbours a depth step apart, so it
    does not. Nothing is ever interpolated across a discontinuity.

    ``allowed`` bounds where filling may happen at all, and should be the
    illumination mask: there has to be a *surface* there, merely one that
    failed to decode. Without it the fill leaks off the edge of the house into
    sky and into deep shadow, inventing correspondence for places the projector
    never reached -- which would then be lit.

    Returns the filled map and its new validity mask. Filling stops at
    ``radius``, so a large undecoded region keeps a genuinely unknown core
    rather than being invented wholesale.

    Filled values carry up to about a pixel of bias per ring, because a mean of
    one-sided neighbours lags a gradient. Over the few rings this runs for that
    stays well inside the project's accuracy bar, and the alternative is a hole
    in the mask.
    """
    if radius <= 0:
        return proj_map, valid

    import cv2

    filled = proj_map.copy()
    known = valid.copy()
    permitted = np.ones_like(valid) if allowed is None else allowed
    neighbourhood = np.ones((3, 3), np.uint8)

    for _ in range(int(radius)):
        frontier = cv2.dilate(known.astype(np.uint8), neighbourhood).astype(bool)
        frontier &= ~known & permitted
        if not frontier.any():
            break

        # Per-channel min, max and median of the known neighbours.
        accepted = np.ones(frontier.shape, dtype=bool) & frontier
        candidate = np.zeros_like(filled)
        for channel in range(2):
            values = np.where(known, filled[..., channel], 0).astype(np.float32)
            weights = known.astype(np.float32)
            total = cv2.filter2D(values, -1, neighbourhood.astype(np.float32))
            count = cv2.filter2D(weights, -1, neighbourhood.astype(np.float32))

            big = float(filled[..., channel].max()) + 1.0
            high = cv2.dilate(np.where(known, filled[..., channel], -big).astype(np.float32),
                              neighbourhood)
            low = -cv2.dilate(np.where(known, -filled[..., channel].astype(np.float32), -big),
                              neighbourhood)

            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(count > 0, total / np.maximum(count, 1e-6), 0.0)
            candidate[..., channel] = np.rint(mean)
            accepted &= (count >= 3) & ((high - low) <= agreement)

        if not accepted.any():
            break
        filled = np.where(accepted[..., None], candidate, filled)
        known |= accepted

    return filled, known


def discontinuity_mask(proj_map: np.ndarray, valid: np.ndarray,
                       jump_px: float, window: int = 1) -> np.ndarray:
    """Camera pixels sitting on a depth discontinuity in the decoded map.

    Adjacent camera pixels looking at the same surface differ by a fraction of a
    projector pixel. Adjacent camera pixels straddling the edge of a bump-out
    differ by many. Nothing may be interpolated across these.

    ``window`` is how far apart the compared pixels may be. The default of 1
    compares immediate neighbours, which is right on clean data. It is not
    enough on a real scan: the silhouette of an object is exactly where decoding
    is hardest, so a band of undecoded pixels usually sits *in* the
    discontinuity. Measured on one rig that band was 10 pixels wide typically
    and 116 at worst, so no pair of adjacent valid pixels ever straddled a
    22-pixel step and the edge went entirely undetected.

    A larger window looks across such a gap by taking the spread of decoded
    values within it. The cost is that a smooth gradient also spreads over a
    wider window, so ``jump_px`` has to grow with ``window`` -- see
    :meth:`~facade_scan.config.TransferConfig.resolve`, and prefer taking the
    union of a tight small-window pass and a loose large-window one.
    """
    if window <= 1:
        out = np.zeros(valid.shape, dtype=bool)
        coords = proj_map.astype(np.float32)
        for axis in (0, 1):
            d = np.abs(np.diff(coords, axis=axis)).max(axis=-1)
            both = np.logical_and(
                np.take(valid, np.arange(valid.shape[axis] - 1), axis=axis),
                np.take(valid, np.arange(1, valid.shape[axis]), axis=axis))
            big = (d > jump_px) & both
            lo = [slice(None)] * 2
            hi = [slice(None)] * 2
            lo[axis] = slice(0, valid.shape[axis] - 1)
            hi[axis] = slice(1, valid.shape[axis])
            out[tuple(lo)] |= big
            out[tuple(hi)] |= big
        return out

    import cv2

    size = int(window) | 1
    kernel = np.ones((size, size), np.uint8)
    spread = np.zeros(valid.shape, dtype=np.float32)
    for channel in range(2):
        values = proj_map[..., channel].astype(np.float32)
        sentinel = float(values.max()) + 1.0
        high = cv2.dilate(np.where(valid, values, -sentinel), kernel)
        low = -cv2.dilate(np.where(valid, -values, -sentinel), kernel)
        np.maximum(spread, high - low, out=spread)

    # Only meaningful where the window actually held valid pixels on both sides
    # of whatever it is straddling.
    support = cv2.filter2D(valid.astype(np.float32), -1,
                           kernel.astype(np.float32))
    return (spread > jump_px) & (support >= 2)


# --------------------------------------------------------------------------- #
# The decoder
# --------------------------------------------------------------------------- #
def decode(loader: FrameLoader, manifest: Manifest,
           cfg: DecodeConfig | None = None) -> DecodeResult:
    """Decode a captured Gray-code set into a camera -> projector map.

    Frames are loaded a pattern/inverse pair at a time rather than all at once,
    so peak memory stays at a handful of images regardless of how many bits the
    projector needs.
    """
    cfg = cfg or DecodeConfig()

    white = np.asarray(loader(manifest.frame_by_role("white")), dtype=np.float32)
    black = np.asarray(loader(manifest.frame_by_role("black")), dtype=np.float32)
    if white.shape != black.shape:
        raise ValueError("white and black captures differ in size")
    shape = white.shape
    illumination = white - black

    min_conf = np.full(shape, np.inf, dtype=np.float32)
    sum_conf = np.zeros(shape, dtype=np.float32)
    n_bits_total = 0
    decoded: dict[str, np.ndarray] = {}

    for axis, bits in (("x", manifest.bits_x), ("y", manifest.bits_y)):
        gray = np.zeros(shape, dtype=np.uint32)
        normals = manifest.frames_for(axis, inverted=False)
        inverses = manifest.frames_for(axis, inverted=True)
        for normal, inverse in zip(normals, inverses):
            if normal.bit is None or normal.bit != inverse.bit:
                raise ValueError(
                    f"manifest pairs {normal.filename} with {inverse.filename}, "
                    "which are not the same bit of the same axis"
                )
            pattern = np.asarray(loader(normal), dtype=np.float32)
            anti = np.asarray(loader(inverse), dtype=np.float32)
            if pattern.shape != shape or anti.shape != shape:
                raise ValueError(
                    f"capture {normal.filename} is {pattern.shape}, expected {shape}; "
                    "every frame in a scan must be the same size"
                )
            difference = pattern - anti
            bit = (difference > 0).astype(np.uint32)
            gray |= bit << np.uint32(bits - 1 - normal.bit)

            confidence = np.abs(difference)
            np.minimum(min_conf, confidence, out=min_conf)
            sum_conf += confidence
            n_bits_total += 1
        decoded[axis] = gray_to_binary(gray, bits).astype(np.int64)

    # Now that the camera resolution is known, turn the step-relative
    # thresholds into projector pixels.
    cfg = cfg.resolve(shape, (manifest.projector_width, manifest.projector_height))

    mean_conf = (sum_conf / max(n_bits_total, 1)).astype(np.float32)
    proj_map = np.stack([decoded["x"], decoded["y"]], axis=-1).astype(np.int32)

    # A Gray code over 11 bits addresses 2048 columns but the panel only has
    # 1920, so decodes can legitimately land outside the panel. Those are
    # errors, not coordinates.
    in_panel = (
        (proj_map[..., 0] >= 0) & (proj_map[..., 0] < manifest.projector_width)
        & (proj_map[..., 1] >= 0) & (proj_map[..., 1] < manifest.projector_height)
    )
    valid = (min_conf >= cfg.confidence_threshold) & in_panel
    valid &= illumination >= cfg.illumination_threshold

    if cfg.median_ksize:
        proj_map = median_filter_map(
            proj_map, valid, cfg.median_ksize, cfg.median_max_jump_px,
            cfg.median_row_chunk, cfg.median_min_support, cfg.median_replace_px,
        )

    # Only fill where the projector actually lit a surface. Anything else is
    # sky, or shadow the projector could not reach, and has no correspondence
    # to recover.
    proj_map, valid = inpaint_map(
        proj_map, valid, cfg.inpaint_radius_px, cfg.inpaint_agreement_px,
        allowed=illumination >= cfg.illumination_threshold,
    )

    # Bright, unmodulated, and *still bright with nothing projected*. The last
    # condition is what keeps shadow out: shadow is also unmodulated, but it
    # goes dark when the projector does.
    likely_glass = (
        (white >= cfg.glass_min_brightness)
        & (black >= cfg.glass_min_black)
        & (mean_conf < cfg.glass_confidence_threshold)
    )

    return DecodeResult(
        proj_map=proj_map, valid=valid, min_confidence=min_conf,
        mean_confidence=mean_conf, illumination=illumination,
        white=white, black=black, likely_glass=likely_glass,
        projector_width=manifest.projector_width,
        projector_height=manifest.projector_height,
    )


def decode_directory(capture_dir: str | Path, cfg: DecodeConfig | None = None,
                     manifest: Manifest | None = None) -> DecodeResult:
    """Decode a directory of captures that contains its own ``manifest.json``."""
    root = Path(capture_dir)
    if manifest is None:
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. The capture directory must contain the "
                "manifest written alongside the patterns, so the decoder knows which "
                "image is which bit."
            )
        manifest = Manifest.read(manifest_path)

    # Decoding with a manifest that does not describe these images produces a
    # plausible-looking map made of nonsense, which is far worse than an error.
    # A frame count mismatch is the cheap way to catch the common causes: a
    # dropped photo, a double exposure, or patterns generated for a different
    # projector resolution.
    suffixes = {ext.lower() for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")}
    found = [p for p in root.iterdir() if p.suffix.lower() in suffixes]
    if len(found) != manifest.num_frames:
        raise ValueError(
            f"{root} holds {len(found)} images but the manifest describes "
            f"{manifest.num_frames} frames. Every frame must be present and in "
            "capture order; re-run the capture or fix the folder."
        )
    return decode(image_loader(root), manifest, cfg)
