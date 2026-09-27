"""scan.json: the whole result as data.

Projector resolution, every region polygon and label, and the decode confidence
statistics. Enough to drive your own renderer, and enough to tell after the fact
whether a scan was any good.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import Exporter, ScanExport, register_exporter

SCHEMA_VERSION = 1


def _ring(points: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in points]


def scan_to_dict(scan: ScanExport) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "projector": {"width": scan.projector_width, "height": scan.projector_height},
        "coordinate_space": "projector_pixels",
        "regions": [
            {
                "label": region.label,
                "polygon": _ring(region.polygon),
                "holes": [_ring(h) for h in region.holes],
                "area_px": round(float(region.area), 2),
                "centroid": [round(float(c), 2) for c in region.centroid],
                "attributes": {k: round(float(v), 4)
                               for k, v in sorted(region.attributes.items())},
            }
            for region in scan.regions
        ],
        "confidence": {k: round(float(v), 6) for k, v in sorted(scan.stats.items())},
        "source": dict(sorted(scan.source.items())),
        "notes": list(scan.notes),
    }


class JsonExporter(Exporter):
    name = "json"
    extension = ".json"

    def export(self, scan: ScanExport, path: str | Path) -> Path:
        scan.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scan_to_dict(scan), indent=2) + "\n",
                        encoding="utf-8")
        return path


register_exporter(JsonExporter())
