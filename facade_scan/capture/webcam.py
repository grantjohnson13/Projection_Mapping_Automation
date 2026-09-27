"""Webcam backend, via OpenCV VideoCapture.

Two things here are not optional, and both silently ruin scans if skipped.

Automatic exposure, white balance and focus
-------------------------------------------
Every 'auto' feature on a camera is a feedback loop that reacts to the scene.
The scene here is a sequence of patterns that swing between mostly-white and
mostly-black many times. Auto-exposure will chase them, which means the gain
applied to a pattern differs from the gain applied to its inverse -- and the
whole method rests on those two exposures being comparable. Autofocus will hunt
on the stripes. Auto white balance will shift colour per frame.

So we turn them all off and set fixed values. OpenCV's property units are
backend- and driver-dependent, which is unavoidable; the values in
:class:`~facade_scan.config.CaptureConfig` are sane defaults for UVC webcams and
:meth:`WebcamBackend.report_settings` prints what actually took effect.

Buffer flushing
---------------
``VideoCapture.read()`` returns the oldest frame in the driver's queue, not the
newest. Call it right after changing what is on the projector and you get an
image of the *previous* pattern. Nothing errors, nothing looks wrong, and the
decoded map is garbage. So before keeping a frame we pull and discard several,
which drains the queue.
"""

from __future__ import annotations

import contextlib

import numpy as np

from ..config import CaptureConfig
from ..patterns import Frame
from .base import CaptureBackend


def list_cameras() -> list[str]:
    """Camera names as AVFoundation reports them, on macOS.

    .. warning::

       This order is **not guaranteed to match OpenCV's integer indices**. It
       was assumed to, and on at least one machine (OpenCV 5, macOS, a built-in
       camera plus a UVC webcam plus an iPhone on Continuity) it did not: the
       webcam AVFoundation listed second was the camera OpenCV opened at index
       zero. Two separate AVFoundation queries agreed with each other and both
       disagreed with reality.

       So treat this as a hint for a human, not as a lookup table. The only
       reliable way to identify the right camera is to check which one sees the
       projected image -- which is what ``facade-scan check`` reports as the lit
       area, and what :meth:`RigCheck.advice` complains about when it is tiny.
    """
    try:
        import AVFoundation as AV
    except ImportError:
        return []
    try:
        devices = AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)
        return [str(device.localizedName()) for device in devices]
    except Exception:
        return []


def resolve_camera_index(name: str) -> int | None:
    """Index of the first camera whose name contains ``name``.

    Subject to the caveat on :func:`list_cameras`: this maps names onto
    AVFoundation's ordering, which OpenCV does not always share. Prefer a
    verified ``webcam_index`` when the two disagree.
    """
    wanted = name.strip().lower()
    for index, found in enumerate(list_cameras()):
        if wanted in found.lower():
            return index
    return None


class WebcamBackend(CaptureBackend):
    name = "webcam"

    def __init__(self, cfg: CaptureConfig, capture_factory=None) -> None:
        self.cfg = cfg
        #: Injection point for tests: anything with read()/set()/get()/release().
        self._factory = capture_factory
        self._cap = None
        self._applied: dict[str, float] = {}
        #: The index actually opened, after any name lookup.
        self.resolved_index = cfg.webcam_index
        #: The device's own name, when we could find one out.
        self.resolved_name: str | None = None

    # ------------------------------------------------------------------ open --
    def open(self) -> None:
        if self._cap is not None:
            return
        import cv2

        index = self.cfg.webcam_index
        if self.cfg.webcam_name:
            matched = resolve_camera_index(self.cfg.webcam_name)
            if matched is None:
                available = list_cameras()
                raise OSError(
                    f"no camera matching {self.cfg.webcam_name!r}. "
                    + (f"Available: {available}" if available
                       else "Could not enumerate cameras; use capture.webcam_index.")
                )
            index = matched
        self.resolved_index = index
        names = list_cameras()
        self.resolved_name = names[index] if index < len(names) else None

        if self._factory is not None:
            cap = self._factory(index)
        else:
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise OSError(
                f"could not open camera {index}. Check capture.webcam_index / "
                "webcam_name, and that nothing else holds the camera."
            )
        self._cap = cap
        self._apply_settings()

    def _apply_settings(self) -> None:
        import cv2

        cap = self._cap
        assert cap is not None
        cfg = self.cfg

        def put(prop: int, value: float | None, label: str) -> None:
            if value is None:
                return
            cap.set(prop, float(value))
            self._applied[label] = float(cap.get(prop))

        if cfg.webcam_width:
            put(cv2.CAP_PROP_FRAME_WIDTH, cfg.webcam_width, "width")
        if cfg.webcam_height:
            put(cv2.CAP_PROP_FRAME_HEIGHT, cfg.webcam_height, "height")

        if cfg.webcam_disable_auto:
            # 0.25 is "manual" on V4L2-backed UVC cameras; 0 on some others.
            # Both are written, harmlessly, because getting this wrong is worse
            # than setting a property twice.
            put(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "auto_exposure")
            put(cv2.CAP_PROP_AUTO_WB, 0, "auto_wb")
            put(cv2.CAP_PROP_AUTOFOCUS, 0, "autofocus")

        put(cv2.CAP_PROP_EXPOSURE, cfg.webcam_exposure, "exposure")
        put(cv2.CAP_PROP_GAIN, cfg.webcam_gain, "gain")
        put(cv2.CAP_PROP_WB_TEMPERATURE, cfg.webcam_wb_temperature, "wb_temperature")
        put(cv2.CAP_PROP_FOCUS, cfg.webcam_focus, "focus")

        # A short buffer helps, where the driver honours it. We still flush,
        # because plenty of drivers quietly ignore this.
        with contextlib.suppress(Exception):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def report_settings(self) -> dict[str, float]:
        """What the driver actually accepted, which is often not what we asked."""
        return dict(self._applied)

    # --------------------------------------------------------------- capture --
    def capture(self, frame: Frame) -> np.ndarray:
        cap = self._cap
        if cap is None:
            raise RuntimeError("open() must be called before capture()")

        # Drain the driver's queue. Without this we get an image of whatever
        # was on the projector several frames ago.
        for _ in range(max(0, self.cfg.flush_frames)):
            cap.read()

        ok, image = cap.read()
        if not ok or image is None:
            raise OSError(f"webcam returned no frame for {frame.filename}")
        return image

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> str:
        label = self.resolved_name or self.cfg.webcam_name or f"#{self.resolved_index}"
        return f"webcam {label} (index {self.resolved_index}, flush {self.cfg.flush_frames})"
