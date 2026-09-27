"""Closed-loop verification: does the projector actually light what we meant?

Everything upstream of this is open-loop. The decoder measures a correspondence,
the transfer applies it, the exporter writes it out, and nothing ever checks
that light lands where the map says it should. That gap matters because the most
likely failure -- the rig shifting after the scan -- produces no error at all.
The mask still exports, the preview still projects, and it is simply in the
wrong place by a few pixels.

Two checks, cheap and thorough.

**Drift** (two frames). The scan stored the all-white capture it was decoded
from. Photograph the scene again and phase-correlate the two: if the camera's
view has moved, the correspondence describes a scene that no longer exists.
Measured on a real rig this caught a 10-pixel shift that had gone unnoticed
through four separate projections.

**Alignment** (a few frames). Draw fiducial squares at known *camera*
coordinates, push them through the correspondence into projector space, project
them, and see where the light actually lands. The offset between intended and
observed is the end-to-end error of the whole pipeline, in camera pixels,
measured rather than assumed. It is the only number here that owes nothing to
the decoder being right about itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .decode import DecodeResult


@dataclass
class Fiducial:
    """One probe: where we aimed, and where the light went."""

    camera_xy: tuple[float, float]
    observed_xy: tuple[float, float] | None = None

    @property
    def offset(self) -> tuple[float, float] | None:
        if self.observed_xy is None:
            return None
        return (self.observed_xy[0] - self.camera_xy[0],
                self.observed_xy[1] - self.camera_xy[1])

    @property
    def offset_px(self) -> float | None:
        delta = self.offset
        return None if delta is None else float(np.hypot(*delta))


@dataclass
class AlignmentReport:
    """End-to-end verification of a scan against the physical rig."""

    fiducials: list[Fiducial] = field(default_factory=list)
    #: How far the camera's view has moved since the scan, in camera pixels.
    drift: tuple[float, float] = (0.0, 0.0)
    drift_confidence: float = 0.0
    #: Tolerance the verdict is judged against, in camera pixels.
    #:
    #: There is a floor below which no scan can do better. The decoded map holds
    #: an integer *projector* coordinate per camera pixel, so on a coarse grid
    #: one projector pixel covers many camera pixels and half of that is pure
    #: quantization: at a 240-wide grid against a 1920-wide camera, a projector
    #: pixel is 8 camera pixels and +-4 px of scatter is the best achievable.
    #: Judging such a scan against a flat 3 px called it misaligned when it was
    #: perfect, so :func:`verify_alignment` raises this to suit the grid.
    tolerance_px: float = 3.0
    #: Camera pixels per projector pixel, for reporting that floor.
    quantisation_px: float = 0.0
    #: Phase-correlation response below which the drift figure is not believed.
    #: A weak peak means the two views had little in common to lock onto --
    #: a flat target, heavy colour-wheel banding, or a genuinely changed scene
    #: -- and the shift it reports is not evidence of anything.
    drift_confidence_floor: float = 0.35

    @property
    def drift_is_trustworthy(self) -> bool:
        return self.drift_confidence >= self.drift_confidence_floor

    @property
    def measured(self) -> list[Fiducial]:
        return [f for f in self.fiducials if f.observed_xy is not None]

    @property
    def offsets(self) -> np.ndarray:
        return np.array([f.offset_px for f in self.measured], dtype=float)

    @property
    def bias(self) -> tuple[float, float]:
        """Median signed offset: a shift affecting the whole projection."""
        found = self.measured
        if not found:
            return (0.0, 0.0)
        deltas = np.array([f.offset for f in found], dtype=float)
        return (float(np.median(deltas[:, 0])), float(np.median(deltas[:, 1])))

    @property
    def scatter_px(self) -> float:
        """Spread about the bias -- the part a rigid shift cannot explain."""
        found = self.measured
        if len(found) < 2:
            return 0.0
        deltas = np.array([f.offset for f in found], dtype=float)
        return float(np.median(np.linalg.norm(deltas - np.median(deltas, axis=0), axis=1)))

    @property
    def median_offset_px(self) -> float:
        return float(np.median(self.offsets)) if len(self.offsets) else 0.0

    @property
    def drift_px(self) -> float:
        return float(np.hypot(*self.drift))

    @property
    def aligned(self) -> bool:
        return bool(self.measured) and self.median_offset_px <= self.tolerance_px

    def verdict(self) -> list[str]:
        """What the numbers mean, and what to do about it."""
        if not self.measured:
            return ["No fiducial returned any light. The projector is not "
                    "reaching the camera at all -- check `facade-scan displays` "
                    "and that the camera is pointed at the projected image."]

        notes: list[str] = []
        bias = np.hypot(*self.bias)
        if self.aligned:
            note = (f"Aligned: light lands {self.median_offset_px:.2f} px from "
                    f"where the scan says it should, over "
                    f"{len(self.measured)} probes.")
            if self.quantisation_px > 1.0:
                note += (f" The floor for this pattern grid is "
                         f"{self.quantisation_px / 2:.1f} px, so this is close "
                         "to as good as it can measure.")
            notes.append(note)
            return notes

        notes.append(
            f"MISALIGNED by {self.median_offset_px:.1f} px "
            f"(bias {self.bias[0]:+.1f}, {self.bias[1]:+.1f}; "
            f"scatter {self.scatter_px:.1f} px)."
        )
        if (self.drift_is_trustworthy and bias > self.tolerance_px
                and 0.5 * bias < self.drift_px < 3.0 * bias):
            notes.append(
                f"The camera's view has moved {self.drift_px:.1f} px since the "
                "scan, which accounts for most of it. Something was knocked, or "
                "a tripod settled. The correspondence describes a scene that is "
                "no longer there: scan again."
            )
        elif not self.drift_is_trustworthy and self.drift_px > self.tolerance_px:
            notes.append(
                f"The drift check reported {self.drift_px:.1f} px but at only "
                f"{self.drift_confidence:.2f} confidence, so it is not evidence "
                "of anything -- the two views had too little in common to lock "
                "onto. Judge by the fiducials alone."
            )
        if bias > 2 * self.scatter_px and bias > self.tolerance_px:
            notes.append(
                "It is a near-rigid shift with little scatter, so the "
                "correspondence itself is sound and the rig has moved relative "
                "to where it was scanned. Scan again rather than adjusting "
                "anything in software."
            )
        elif self.scatter_px >= max(bias, self.tolerance_px):
            notes.append(
                f"The error scatters ({self.scatter_px:.1f} px) as much as it "
                f"shifts ({bias:.1f} px), so this is not simple movement. On a "
                "low-coverage scan with few probes that may be measurement "
                "noise; otherwise suspect the decode, and check coverage and "
                "the planarity residual before trusting it."
            )
        if len(self.measured) < 12:
            notes.append(
                f"Only {len(self.measured)} probes landed, so these figures are "
                "themselves uncertain. Coverage is probably too low to verify "
                "properly."
            )
        return notes


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
def measure_drift(reference: np.ndarray,
                  current: np.ndarray) -> tuple[tuple[float, float], float]:
    """Phase-correlate two views of the same scene.

    Returns ``((dx, dy), confidence)``. A non-zero shift means the camera, the
    projector or the target has moved since ``reference`` was captured, and any
    correspondence measured then no longer applies.
    """
    import cv2

    if reference.shape != current.shape:
        raise ValueError(
            f"cannot compare a {reference.shape} reference with a "
            f"{current.shape} capture; the camera resolution has changed"
        )
    a = np.ascontiguousarray(reference, dtype=np.float32)
    b = np.ascontiguousarray(current, dtype=np.float32)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    return (float(dx), float(dy)), float(response)


# --------------------------------------------------------------------------- #
# Fiducials
# --------------------------------------------------------------------------- #
def plan_fiducials(decoded: DecodeResult, spacing: int = 160, half: int = 12,
                   margin: int = 100, solidity: float = 0.97) -> list[Fiducial]:
    """Choose probe locations on ground that decoded solidly.

    A probe on a half-decoded patch measures the patchiness, not the alignment,
    so only fully-decoded neighbourhoods are used.
    """
    height, width = decoded.valid.shape
    chosen: list[Fiducial] = []
    for cy in range(margin, height - margin, spacing):
        for cx in range(margin, width - margin, spacing):
            window = decoded.valid[cy - half:cy + half + 1, cx - half:cx + half + 1]
            if window.size and window.mean() >= solidity:
                chosen.append(Fiducial(camera_xy=(float(cx), float(cy))))
    return chosen


def render_fiducials(decoded: DecodeResult, fiducials: list[Fiducial],
                     half: int = 12) -> np.ndarray:
    """Project-space image of the fiducials, via the scan's own correspondence."""
    out = np.zeros((decoded.projector_height, decoded.projector_width), np.uint8)
    for probe in fiducials:
        cx, cy = int(probe.camera_xy[0]), int(probe.camera_xy[1])
        ys, xs = np.mgrid[cy - half:cy + half + 1, cx - half:cx + half + 1]
        inside = ((ys >= 0) & (ys < decoded.valid.shape[0])
                  & (xs >= 0) & (xs < decoded.valid.shape[1]))
        ys, xs = ys[inside], xs[inside]
        usable = decoded.valid[ys, xs]
        out[decoded.proj_map[ys, xs, 1][usable],
            decoded.proj_map[ys, xs, 0][usable]] = 255
    return out


def locate_response(response: np.ndarray, fiducial: Fiducial, search: int = 45,
                    floor: float = 0.08) -> tuple[float, float] | None:
    """Intensity-weighted centroid of the light near where a probe was aimed.

    ``search`` must stay below half the fiducial spacing. A window that reaches
    the next probe pulls the centroid toward it, and the result measures the
    grid rather than the projection.
    """
    height, width = response.shape
    cx, cy = int(fiducial.camera_xy[0]), int(fiducial.camera_xy[1])
    y0, y1 = max(0, cy - search), min(height, cy + search + 1)
    x0, x1 = max(0, cx - search), min(width, cx + search + 1)
    patch = response[y0:y1, x0:x1]
    if patch.size == 0 or patch.max() < floor:
        return None

    # Half-maximum threshold, so a neighbouring probe's tail cannot drag the
    # centroid toward it.
    weight = np.clip(patch - 0.5 * patch.max(), 0.0, None)
    total = weight.sum()
    if total <= 0:
        return None
    gy, gx = np.mgrid[y0:y1, x0:x1]
    return (float((gx * weight).sum() / total), float((gy * weight).sum() / total))


def verify_alignment(decoded: DecodeResult, backend, display, config: Config,
                     half: int = 12, spacing: int = 160,
                     tolerance_px: float = 3.0,
                     on_step=None) -> AlignmentReport:
    """Project fiducials through the scan's correspondence and measure the error.

    ``backend`` and ``display`` are an already-open capture device and display,
    as the CLI builds them.
    """
    import time

    import cv2

    settle = config.capture.settle_ms / 1000.0

    def show_and_grab(image: np.ndarray) -> np.ndarray:
        display.show(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        time.sleep(settle)
        frame = backend.capture(_Frame())
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if on_step is not None:
            on_step()
        return frame.astype(np.float32) / 255.0

    size = (decoded.projector_height, decoded.projector_width)
    black = show_and_grab(np.zeros(size, np.uint8))
    white = show_and_grab(np.full(size, 255, np.uint8))

    drift, confidence = measure_drift(decoded.white, white)

    fiducials = plan_fiducials(decoded, spacing=spacing, half=half)
    if fiducials:
        lit = show_and_grab(render_fiducials(decoded, fiducials, half))
        response = lit - black
        # The search window must not reach the next fiducial, or a neighbour's
        # light drags the centroid and the measured offset is of the grid
        # rather than of the projection.
        search = max(half + 4, min(45, spacing // 2 - 2))
        for probe in fiducials:
            probe.observed_xy = locate_response(response, probe, search=search)

    # One projector pixel spans this many camera pixels; half of it is
    # unavoidable quantisation, so the tolerance can never sensibly sit below.
    quantisation = max(decoded.valid.shape[1] / max(decoded.projector_width, 1),
                       decoded.valid.shape[0] / max(decoded.projector_height, 1))
    return AlignmentReport(fiducials=fiducials, drift=drift,
                           drift_confidence=confidence,
                           tolerance_px=max(tolerance_px, 0.75 * quantisation),
                           quantisation_px=quantisation)


class _Frame:
    """Minimal stand-in for a manifest frame, for backends that want one."""

    index = 0
    filename = "verify.png"
    role = "white"
    axis = None
    bit = None
    inverted = False
