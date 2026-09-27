"""Folder backend: the operator projects the patterns and photographs them.

This one must always work. It is the fallback for when the webcam is unusable,
gphoto2 does not speak to your camera, or you simply want to shoot the scan on a
DSLR with an intervalometer and deal with the files afterwards. It has no
dependencies beyond reading images off disk.

Images are matched to the manifest by **sorted filename order**. Cameras name
files sequentially, so as long as the folder contains exactly the scan and
nothing else, this is reliable. It also means the directory produced by
``facade-scan simulate`` can be fed straight in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import CaptureConfig
from ..patterns import Frame, Manifest
from .base import CaptureBackend


class FolderBackend(CaptureBackend):
    name = "folder"
    #: The operator has already projected the patterns by hand.
    drives_display = False

    def __init__(self, cfg: CaptureConfig) -> None:
        self.cfg = cfg
        self._files: list[Path] = []
        self._by_index: dict[int, Path] = {}

    @property
    def root(self) -> Path:
        if not self.cfg.folder_path:
            raise ValueError(
                "the folder backend needs capture.folder_path set to the "
                "directory holding your photographs"
            )
        return Path(self.cfg.folder_path)

    def prepare(self, manifest: Manifest) -> None:
        root = self.root
        if not root.is_dir():
            raise NotADirectoryError(f"capture folder {root} does not exist")

        wanted = {ext.lower() for ext in self.cfg.folder_extensions}
        self._files = sorted(
            (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in wanted),
            key=lambda p: p.name,
        )
        if len(self._files) != manifest.num_frames:
            raise ValueError(
                f"{root} holds {len(self._files)} images but the scan needs "
                f"{manifest.num_frames}. Every projected frame must be "
                "photographed exactly once, and the folder must contain nothing "
                "else. Found: "
                + ", ".join(p.name for p in self._files[:6])
                + (" ..." if len(self._files) > 6 else "")
            )
        self._by_index = {frame.index: path
                          for frame, path in zip(manifest.frames, self._files)}

    def capture(self, frame: Frame) -> np.ndarray:
        import cv2

        try:
            path = self._by_index[frame.index]
        except KeyError:
            raise RuntimeError("prepare() must be called before capture()") from None
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(f"could not read {path}")
        return image

    def describe(self) -> str:
        return f"folder ({self.cfg.folder_path})"
