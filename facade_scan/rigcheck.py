"""Measure whether a physical rig can actually decode, before committing to a scan.

A full scan is forty-odd frames of holding still, and the failure it is most
likely to suffer -- projector defocus -- produces no error at all. It produces a
map that is simply empty in places, which you discover afterwards.

So this measures the thing that decides it, directly. For each Gray-code plane
it projects the pattern and its inverse, captures both, and reports the
**modulation**: the median ``|pattern - inverse|`` over the illuminated area.
That is exactly the quantity the decoder thresholds, so a plane whose modulation
sits below ``decode.confidence_threshold`` is a plane that will not decode, and
one bad plane means no decoded coordinate at all for those pixels.

Modulation falls off as stripes get finer, and where it falls off tells you
which knob to reach for:

- falls off only on the last plane or two -> coarsen the pattern grid slightly
- falls off from the middle -> the projector is out of focus, or cannot focus
  this close
- flat but low everywhere -> not enough light, or the camera is underexposed
- fine but the image is clipped -> too much light; shorten the exposure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .patterns import Frame, build_manifest, render_frame


@dataclass
class PlaneMeasurement:
    """How well one Gray-code plane survived the round trip."""

    axis: str
    bit: int
    #: Stripe width in logical projector pixels. Gray's finest plane is 2.
    stripe_px: int
    #: The same stripe on the physical panel, after any nearest-neighbour blowup.
    native_stripe_px: float
    #: Median |pattern - inverse| over the illuminated area, in [0, 1].
    modulation: float
    #: Fraction of illuminated pixels whose modulation clears the threshold.
    decodable_fraction: float


@dataclass
class RigCheck:
    """The verdict on a physical setup."""

    white_level: float
    black_level: float
    lit_fraction: float
    clipped_fraction: float
    planes: list[PlaneMeasurement] = field(default_factory=list)
    #: Fraction of the lit area where *every* measured plane cleared the
    #: threshold at the same pixel. This is what the decoder actually requires.
    joint_decodable: float = 0.0
    #: Which axis was measured. A full scan needs both, so real coverage is
    #: lower than a single axis suggests.
    measured_axis: str = "x"
    projector_grid: tuple[int, int] = (0, 0)
    native_grid: tuple[int, int] = (0, 0)
    camera_size: tuple[int, int] = (0, 0)
    threshold: float = 0.06

    @property
    def worst(self) -> PlaneMeasurement | None:
        return min(self.planes, key=lambda p: p.modulation) if self.planes else None

    @property
    def failing(self) -> list[PlaneMeasurement]:
        return [p for p in self.planes if p.decodable_fraction < 0.9]

    @property
    def predicted_coverage(self) -> float:
        """Fraction of the lit area expected to decode on the measured axis.

        A pixel needs every plane to clear the threshold *at that pixel*, so
        this is measured jointly rather than taken as the weakest plane's
        fraction. Those differ a great deal when failures are not aligned: on
        one real rig the weakest plane passed 62% of pixels while only 26%
        passed all of them, because different planes failed in different places.
        """
        return self.joint_decodable

    @property
    def predicted_scan_coverage(self) -> float:
        """Rough coverage for a full scan, which needs both axes.

        Only one axis is probed, so the other is assumed to fail independently
        in the same proportion. That is a guess -- failures are usually
        correlated, because they are mostly the same dark and specular pixels --
        so treat this as a lower bound.
        """
        return self.joint_decodable ** 2

    @property
    def oversampling(self) -> float:
        return self.camera_size[0] / max(self.projector_grid[0], 1)

    @property
    def baseline_loss(self) -> float:
        """Fraction of the lit area that fails even at the coarsest stripe.

        Nothing optical stops a stripe half the panel wide from being read, so
        whatever fails there fails for a reason that has nothing to do with
        resolution: the surface is too dark, or specular enough to throw the
        light somewhere other than the lens. Coarsening the grid will not
        recover it, and neither will refocusing.
        """
        if not self.planes:
            return 0.0
        coarsest = min(self.planes, key=lambda p: p.bit)
        return 1.0 - coarsest.decodable_fraction

    @property
    def resolution_limit_px(self) -> float | None:
        """Narrowest panel stripe still returning half the coarsest modulation.

        This is the rig's optical cutoff, and the pattern grid should be chosen
        so its finest stripe stays above it.
        """
        if len(self.planes) < 2:
            return None
        best = max(p.modulation for p in self.planes)
        good = [p for p in self.planes if p.modulation >= 0.5 * best]
        return min(p.native_stripe_px for p in good) if good else None

    def recommended_grid(self) -> tuple[int, int] | None:
        """A pattern grid whose finest stripe clears the optical cutoff."""
        limit = self.resolution_limit_px
        if limit is None or not self.planes:
            return None
        finest = min(p.native_stripe_px for p in self.planes)
        if finest >= limit:
            return None
        factor = 1
        while finest * factor < limit:
            factor *= 2
        width = self.projector_grid[0] // factor
        height = self.projector_grid[1] // factor
        return (width, height) if width >= 64 else None

    def advice(self) -> list[str]:
        """What to change, in the order worth trying."""
        notes: list[str] = []
        if not self.planes:
            return ["No planes were measured."]

        coarsest = min(self.planes, key=lambda p: p.bit)

        # --- is anything being seen at all? ---------------------------------
        if self.white_level <= self.black_level:
            return [
                "The white frame is no brighter than the black frame. The "
                "camera is not seeing the projection at all -- wrong camera, "
                "wrong display, or pointed the wrong way. Nothing downstream "
                "of this will mean anything."
            ]
        if coarsest.decodable_fraction < 0.5:
            return [
                "Even the COARSEST plane fails over most of the frame, and its "
                "stripes are half the panel wide. This is not focus and not "
                "resolution: the patterns are largely not reaching the camera. "
                "Check `facade-scan displays` and that the camera is pointed at "
                "the projected image."
            ]

        # --- exposure -------------------------------------------------------
        if self.clipped_fraction > 0.02:
            notes.append(
                f"{self.clipped_fraction:.1%} of the lit area is clipped to 255. "
                "Shorten the exposure: clipped pixels decode as ambiguous bits."
            )
        if self.white_level < 0.25:
            notes.append(
                f"The white frame only reaches {self.white_level:.2f}. Expose "
                "longer -- prefer exposure to gain, which amplifies the noise "
                "this method is reading through."
            )
        if self.black_level > 0.20:
            headroom = (self.white_level - self.black_level) / max(self.white_level, 1e-6)
            notes.append(
                f"The black frame sits at {self.black_level:.2f}, so only "
                f"{headroom:.0%} of your exposure is carrying pattern. Kill room "
                "lights, then expose longer -- the inverse frames cancel steady "
                "light, but it still eats the dynamic range you decode through."
            )

        # --- surface --------------------------------------------------------
        if self.baseline_loss > 0.05:
            notes.append(
                f"{self.baseline_loss:.0%} of the lit area fails even at the "
                "coarsest stripe. That is the surface, not the optics: too dark "
                "to return light, or specular enough to throw it past the lens. "
                "Brushed metal and glass do this. A matte backdrop -- plain "
                "cardboard or paper -- fixes it; nothing in software will."
            )

        # --- resolution -----------------------------------------------------
        if self.oversampling < 1.4:
            notes.append(
                f"The camera only out-resolves the pattern grid "
                f"{self.oversampling:.2f}x. Halve projector.width/height."
            )
        grid = self.recommended_grid()
        if grid is not None:
            limit = self.resolution_limit_px
            notes.append(
                f"Modulation halves below about {limit:.0f} panel pixels per "
                f"stripe, which is this rig's optical cutoff. Set "
                f"projector.width/height to {grid[0]}x{grid[1]} so the finest "
                f"stripe stays above it; keep display.native_* at "
                f"{self.native_grid[0]}x{self.native_grid[1]}."
            )

        if not notes:
            notes.append("Every plane decodes. Run the scan.")
        return notes

    def table(self) -> str:
        lines = [f"{'plane':>7}  {'stripe':>8}  {'panel px':>9}  "
                 f"{'modulation':>11}  {'decodable':>10}"]
        for plane in self.planes:
            flag = "" if plane.decodable_fraction >= 0.9 else "  <-- fails"
            lines.append(
                f"{plane.axis}{plane.bit:<6d}  {plane.stripe_px:>8d}  "
                f"{plane.native_stripe_px:>9.1f}  {plane.modulation:>11.3f}  "
                f"{plane.decodable_fraction:>9.1%}{flag}"
            )
        return "\n".join(lines)


def _stripe_width(size: int, bit: int, bits: int) -> int:
    """Narrowest stripe in the Gray plane for ``bit``, measured not derived.

    Gray code's stripe widths do not follow the obvious power of two: the two
    finest planes are 2 and 4 pixels wide, not 1 and 2, because a Gray bit is
    the XOR of two adjacent binary bits and so has twice the period. Deriving
    this by formula got it wrong; counting the runs cannot.
    """
    from .patterns import gray_plane

    plane = gray_plane(size, bit, bits).astype(np.int64)
    edges = np.flatnonzero(np.diff(plane))
    if edges.size == 0:
        return size
    runs = np.diff(edges)
    return int(runs.min()) if runs.size else int(max(edges[0] + 1, size - edges[0] - 1))


def check_rig(config: Config, backend, display, axis: str = "x",
              capture_dir: str | Path | None = None,
              on_frame=None) -> RigCheck:
    """Project each plane with its inverse and measure what comes back.

    ``backend`` and ``display`` are already-open capture and display objects, so
    this is testable against fakes and reuses whatever the CLI configured.
    """
    import cv2

    width, height = config.projector.width, config.projector.height
    manifest = build_manifest(width, height, config.patterns)
    native = (display.native_width, display.native_height)
    settle = config.capture.settle_ms / 1000.0

    out_dir = Path(capture_dir) if capture_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    def grab(frame: Frame) -> np.ndarray:
        import time

        image = render_frame(frame, width, height, manifest.bits_x,
                             manifest.bits_y, config.patterns)
        display.show(image)
        time.sleep(settle)
        captured = backend.capture(frame)
        if captured.ndim == 3:
            captured = cv2.cvtColor(captured, cv2.COLOR_BGR2GRAY)
        if out_dir:
            cv2.imwrite(str(out_dir / frame.filename), captured)
        if on_frame is not None:
            on_frame(frame)
        return captured.astype(np.float32) / 255.0

    white = grab(manifest.frame_by_role("white"))
    black = grab(manifest.frame_by_role("black"))
    illumination = white - black
    lit = illumination >= config.decode.illumination_threshold
    if not lit.any():
        raise RuntimeError(
            "The projector does not appear to be reaching the camera at all: "
            "the white and black frames are identical. Check that the patterns "
            "are on the projector (facade-scan displays) and that the camera is "
            "pointed at the projected image."
        )

    bits = manifest.bits_x if axis == "x" else manifest.bits_y
    scale = native[0] / width
    planes: list[PlaneMeasurement] = []
    # The decoder needs every plane to clear the threshold at the same pixel,
    # so carry a running per-pixel minimum rather than scoring planes apart.
    weakest = np.full(int(lit.sum()), np.inf, dtype=np.float32)
    for normal, inverse in zip(manifest.frames_for(axis, False),
                               manifest.frames_for(axis, True)):
        pattern = grab(normal)
        anti = grab(inverse)
        modulation = np.abs(pattern - anti)[lit]
        np.minimum(weakest, modulation, out=weakest)
        stripe = _stripe_width(width if axis == "x" else height,
                               normal.bit or 0, bits)
        planes.append(PlaneMeasurement(
            axis=axis, bit=normal.bit or 0, stripe_px=stripe,
            native_stripe_px=stripe * scale,
            modulation=float(np.median(modulation)),
            decodable_fraction=float(
                (modulation >= config.decode.confidence_threshold).mean()),
        ))

    return RigCheck(
        joint_decodable=float((weakest >= config.decode.confidence_threshold).mean()),
        measured_axis=axis,
        white_level=float(np.median(white[lit])),
        black_level=float(np.median(black[lit])),
        lit_fraction=float(lit.mean()),
        clipped_fraction=float((white[lit] >= 254 / 255).mean()),
        planes=planes,
        projector_grid=(width, height),
        native_grid=native,
        camera_size=(white.shape[1], white.shape[0]),
        threshold=config.decode.confidence_threshold,
    )
