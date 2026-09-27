"""Interchangeable capture backends behind a single interface."""

from .base import (
    PREFLIGHT_CHECKLIST,
    CaptureBackend,
    build_backend,
    preflight_text,
    run_scan,
)
from .folder import FolderBackend
from .gphoto2 import GPhoto2Backend
from .webcam import WebcamBackend

__all__ = [
    "PREFLIGHT_CHECKLIST",
    "CaptureBackend",
    "FolderBackend",
    "GPhoto2Backend",
    "WebcamBackend",
    "build_backend",
    "preflight_text",
    "run_scan",
]
