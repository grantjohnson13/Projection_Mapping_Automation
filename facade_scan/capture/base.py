"""Capture backend interface, the preflight checklist, and the scan loop."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from ..config import CaptureConfig, Config
from ..patterns import Frame, Manifest, render_frame

PREFLIGHT_CHECKLIST = """\
PREFLIGHT -- read this before you start, not after.

  1. Projector focus, zoom and keystone are set and LOCKED.
     Focus on the main wall, not the gable. Every stripe you cannot resolve is
     a bit you cannot decode.

  2. Keystone / lens shift correction is switched OFF if you can manage it.
     Digital keystone resamples the panel, which blurs the finest stripes.
     Square the projector up physically instead.

  3. Camera is mounted as close to the projector lens as physically possible.
     Every centimetre of separation is a centimetre of projector shadow beside
     the garage bump-out and under the eaves -- geometry no decoder can
     recover, because the projector genuinely cannot see it.

  4. Camera is fully manual: exposure, white balance, focus, ISO.
     Any 'auto' anything will track the patterns from frame to frame and
     silently destroy the pattern-versus-inverse comparison the whole method
     rests on.

  5. Exposure is set so the all-white frame is bright but NOT clipped.
     Check the histogram. Clipped highlights decode as ambiguous bits.

  6. Nothing is moving. No wind in the trees in frame, no flags, no
     inflatables, no cars arriving. A scan is a few minutes of stillness.

  7. Kill what ambient light you can. Porch lights and security floods off.
     Streetlights you cannot switch off are survivable -- the inverse frames
     cancel steady light -- but they cost you contrast.

  *** From this point on, do not touch the projector or the camera. ***
  Not the focus ring, not the zoom, not the tripod, not the keystone. Moving
  either one invalidates the entire scan and you start again.
"""


def preflight_text(config: Config, backend_name: str, display_description: str,
                   frame_count: int) -> str:
    """The checklist plus what this particular scan is about to do."""
    settle = config.capture.settle_ms
    estimate = frame_count * (settle + 250) / 1000.0
    return (
        f"{PREFLIGHT_CHECKLIST}\n"
        f"  This scan: {frame_count} frames, {settle} ms settle, "
        f"backend {backend_name!r}\n"
        f"  Projector: {config.projector.width}x{config.projector.height} via "
        f"{display_description}\n"
        f"  Rough duration: {estimate / 60:.1f} min of holding still\n"
    )


class CaptureBackend(ABC):
    """One captured image per projected frame.

    Backends are interchangeable. The scan loop only knows this interface, so
    adding a machine-vision camera later means adding one class.
    """

    #: Human-readable backend name.
    name: str = "backend"

    #: Whether the scan loop is responsible for putting patterns on the
    #: projector. False for backends where the operator has already done it.
    drives_display: bool = True

    def __enter__(self) -> CaptureBackend:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Acquire the device. Safe to call twice. Backends with nothing to
        acquire, such as ``folder``, do not override this."""

    def close(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release the device. Safe to call twice."""

    def prepare(self, manifest: Manifest) -> None:  # noqa: B027 - optional hook
        """Called once before the first frame, with the frame list to expect."""

    @abstractmethod
    def capture(self, frame: Frame) -> np.ndarray:
        """Return the camera's view of ``frame`` as a uint8 image."""

    def describe(self) -> str:
        return self.name


# --------------------------------------------------------------------------- #
# Backend registry
# --------------------------------------------------------------------------- #
def build_backend(cfg: CaptureConfig) -> CaptureBackend:
    """Construct the backend named by ``cfg.backend``."""
    from .folder import FolderBackend
    from .gphoto2 import GPhoto2Backend
    from .webcam import WebcamBackend

    backends: dict[str, Callable[[CaptureConfig], CaptureBackend]] = {
        "webcam": WebcamBackend,
        "gphoto2": GPhoto2Backend,
        "folder": FolderBackend,
    }
    try:
        factory = backends[cfg.backend]
    except KeyError:
        raise ValueError(
            f"unknown capture backend {cfg.backend!r}; "
            f"choose one of {sorted(backends)}"
        ) from None
    return factory(cfg)


# --------------------------------------------------------------------------- #
# The scan loop
# --------------------------------------------------------------------------- #
def run_scan(out_dir: str | Path, manifest: Manifest, backend: CaptureBackend,
             display: object, config: Config,
             on_frame: Callable[[Frame, np.ndarray], None] | None = None,
             patterns: Iterable[np.ndarray] | None = None) -> Path:
    """Project every frame, capture it, and write it to ``out_dir``.

    Each captured image is written as it arrives rather than at the end, so an
    interrupted scan leaves behind everything it managed to get. The manifest is
    copied alongside, which is what makes the directory self-describing and
    decodable later without remembering anything.
    """
    import cv2

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    settle_s = config.capture.settle_ms / 1000.0

    pattern_list = list(patterns) if patterns is not None else None
    backend.prepare(manifest)

    for i, frame in enumerate(manifest.frames):
        if backend.drives_display:
            if pattern_list is not None:
                image = pattern_list[i]
            else:
                image = render_frame(frame, manifest.projector_width,
                                     manifest.projector_height,
                                     manifest.bits_x, manifest.bits_y,
                                     config.patterns)
            display.show(image)              # type: ignore[attr-defined]
            # Cover projector panel refresh, any frame interpolation the
            # projector is doing, and the camera's own exposure time.
            time.sleep(settle_s)

        captured = backend.capture(frame)
        if not cv2.imwrite(str(out / frame.filename), captured):
            raise OSError(f"failed to write capture {out / frame.filename}")
        if on_frame is not None:
            on_frame(frame, captured)

    manifest.write(out / "manifest.json")
    return out
