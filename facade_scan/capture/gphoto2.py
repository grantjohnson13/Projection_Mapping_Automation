"""gphoto2 backend: tethered DSLR capture by shelling out to the CLI.

gphoto2 is an optional dependency and deliberately not a Python one -- the
libgphoto2 bindings are painful to install and the command-line tool is both
easier to get hold of and easier to debug when a camera misbehaves. If the
binary is not on PATH, this backend refuses to start and tells you to use the
``folder`` backend instead.

A DSLR is the right tool here: full manual control, a real sensor, and enough
resolution to out-sample the projector several times over, which is what makes
the finest Gray planes decode cleanly.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..config import CaptureConfig
from ..patterns import Frame, Manifest
from .base import CaptureBackend


class GPhoto2Backend(CaptureBackend):
    name = "gphoto2"

    def __init__(self, cfg: CaptureConfig, runner=None) -> None:
        self.cfg = cfg
        #: Injection point for tests: a callable(args, cwd) -> CompletedProcess.
        self._injected = runner is not None
        self._runner = runner or self._run
        self._tmp: tempfile.TemporaryDirectory | None = None

    # ------------------------------------------------------------------ open --
    def open(self) -> None:
        if self._tmp is not None:
            return
        if not self._injected and shutil.which(self.cfg.gphoto2_binary) is None:
            raise OSError(
                f"{self.cfg.gphoto2_binary!r} is not on PATH. Install gphoto2, or "
                "switch to capture.backend = 'folder' and shoot the scan manually."
            )
        self._tmp = tempfile.TemporaryDirectory(prefix="facade-scan-")

    def prepare(self, manifest: Manifest) -> None:
        self.open()

    def _run(self, args: list[str], cwd: str) -> subprocess.CompletedProcess:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                              timeout=self.cfg.gphoto2_timeout_s)

    # --------------------------------------------------------------- capture --
    def capture(self, frame: Frame) -> np.ndarray:
        import cv2

        self.open()
        assert self._tmp is not None
        workdir = Path(self._tmp.name)
        for stale in workdir.iterdir():
            stale.unlink()

        target = f"capture_{frame.index:04d}.%C"
        args = [self.cfg.gphoto2_binary, "--capture-image-and-download",
                "--filename", target, "--force-overwrite"]
        if not self.cfg.gphoto2_keep_on_camera:
            args.append("--keep=no")
        args.extend(self.cfg.gphoto2_extra_args)

        result = self._runner(args, str(workdir))
        if result.returncode != 0:
            raise OSError(
                f"gphoto2 failed on frame {frame.index} ({frame.filename}):\n"
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        produced = sorted(p for p in workdir.iterdir() if p.is_file())
        if not produced:
            raise OSError(
                f"gphoto2 reported success for frame {frame.index} but downloaded "
                "no file. Check that the camera is not set to RAW-only with a "
                "format OpenCV cannot read."
            )
        image = cv2.imread(str(produced[0]), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(
                f"could not decode {produced[0].name} from the camera. Shoot JPEG, "
                "or convert the RAW files and use the 'folder' backend."
            )
        return image

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def describe(self) -> str:
        return f"gphoto2 ({self.cfg.gphoto2_binary})"
