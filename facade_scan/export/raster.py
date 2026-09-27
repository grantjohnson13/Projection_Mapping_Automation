"""Projector-resolution mask PNG: white means project here."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ExportConfig
from .base import Exporter, ScanExport, register_exporter


class MaskExporter(Exporter):
    """Writes ``scan.mask`` as a 1-bit-looking 8-bit PNG at projector resolution.

    If the scan carries no raster mask, one is rasterised from the regions
    instead, so this exporter always produces something usable.
    """

    name = "mask"
    extension = ".png"

    def __init__(self, cfg: ExportConfig | None = None) -> None:
        self.cfg = cfg or ExportConfig()

    def export(self, scan: ScanExport, path: str | Path) -> Path:
        import cv2

        scan.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        mask = scan.mask
        if mask is None:
            mask = np.zeros(scan.shape, dtype=bool)
            for region in scan.regions:
                mask |= region.rasterize(scan.shape)

        image = np.where(mask, 255, 0).astype(np.uint8)
        if not cv2.imwrite(str(path), image):
            raise OSError(f"failed to write {path}")
        return path


register_exporter(MaskExporter())
