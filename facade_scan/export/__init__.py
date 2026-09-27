"""Export: unambiguous artifacts, and an interface for adding real adapters."""

from .base import (
    Exporter,
    ScanExport,
    available_exporters,
    get_exporter,
    register_exporter,
)
from .raster import MaskExporter
from .scanjson import SCHEMA_VERSION, JsonExporter, scan_to_dict
from .svg import SvgExporter

__all__ = [
    "SCHEMA_VERSION",
    "Exporter",
    "JsonExporter",
    "MaskExporter",
    "ScanExport",
    "SvgExporter",
    "available_exporters",
    "export_all",
    "get_exporter",
    "register_exporter",
    "scan_to_dict",
]


def export_all(scan: ScanExport, out_dir, cfg=None):
    """Write the three standard artifacts, returning the paths written."""
    from pathlib import Path

    from ..config import ExportConfig

    cfg = cfg or ExportConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "mask": MaskExporter(cfg).export(scan, out / cfg.mask_filename),
        "svg": SvgExporter(cfg).export(scan, out / cfg.svg_filename),
        "json": JsonExporter().export(scan, out / cfg.json_filename),
    }
