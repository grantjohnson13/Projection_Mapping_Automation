"""SVG export: the reference :class:`~facade_scan.export.base.Exporter`.

The canvas is exactly the projector panel, and every coordinate is a projector
pixel, so there is no scaling to get wrong. A region with openings in it becomes
a single path with one subpath per ring and ``fill-rule="evenodd"``, which is
how a facade keeps its windows unlit.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from ..config import ExportConfig
from .base import Exporter, ScanExport, register_exporter

#: escape() leaves quotes alone, which is fine for element text and fatal
#: inside a double-quoted attribute. Region labels reach both.
_ATTRIBUTE_ENTITIES = {'"': "&quot;", "'": "&apos;"}


def _attr(value: object) -> str:
    return escape(str(value), _ATTRIBUTE_ENTITIES)


def _text(value: object) -> str:
    return escape(str(value))


def _path_data(rings: list[np.ndarray]) -> str:
    parts = []
    for ring in rings:
        if len(ring) < 3:
            continue
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in ring)
        parts.append(f"M {points} Z")
    return " ".join(parts)


class SvgExporter(Exporter):
    """One labelled path per region, in projector pixel coordinates."""

    name = "svg"
    extension = ".svg"

    def __init__(self, cfg: ExportConfig | None = None) -> None:
        self.cfg = cfg or ExportConfig()

    def export(self, scan: ScanExport, path: str | Path) -> Path:
        scan.validate()
        cfg = self.cfg
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        width, height = scan.projector_width, scan.projector_height
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'  <title>facade-scan regions ({width}x{height} projector pixels)</title>',
        ]
        if scan.source:
            described = ", ".join(f"{k}={v}" for k, v in sorted(scan.source.items()))
            lines.append(f"  <desc>{_text(described)}</desc>")

        lines.append('  <g id="regions">')
        for region in scan.regions:
            data = _path_data(region.rings)
            if not data:
                continue
            lines.append(f'    <g id="{_attr(region.label)}">')
            lines.append(
                f'      <path d="{data}" fill-rule="evenodd" '
                f'fill="{_attr(cfg.svg_fill)}" fill-opacity="{cfg.svg_fill_opacity}" '
                f'stroke="{_attr(cfg.svg_stroke)}" stroke-width="{cfg.svg_stroke_width}">'
                f"<title>{_text(region.label)}</title></path>"
            )
            if cfg.svg_label_regions:
                centre = region.centroid
                lines.append(
                    f'      <text x="{centre[0]:.1f}" y="{centre[1]:.1f}" '
                    f'font-size="{cfg.svg_font_size}" fill="{_attr(cfg.svg_stroke)}" '
                    f'text-anchor="middle">{_text(region.label)}</text>'
                )
            lines.append("    </g>")
        lines.append("  </g>")
        lines.append("</svg>")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


register_exporter(SvgExporter())
