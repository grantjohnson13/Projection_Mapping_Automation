"""Export interface.

A deliberate non-goal
---------------------
This package does **not** write MadMapper ``.mmp``, Resolume ``.avc``/XML, or
any other proprietary project file. Those formats are undocumented, versioned
without notice, and a plausible-looking file with a guessed-at schema is worse
than no file at all: it opens, it looks fine, and the geometry is subtly wrong
in a way you only discover on the night.

So what is written instead is three things that are completely unambiguous:

``mask.png``
    Projector-resolution bitmap. White means project here. Drop it into any
    tool as a layer mask, or feed it to your own renderer.
``regions.svg``
    One labelled path per region, in projector pixel coordinates, on a canvas
    the exact size of the projector panel. Every mapping tool imports SVG, and
    so does every vector editor, so this is the format to hand-adjust.
``scan.json``
    The same geometry as data, plus the confidence statistics, for anything you
    want to script.

Adding a format adapter
-----------------------
When you do know a target schema, subclass :class:`Exporter`::

    class MyToolExporter(Exporter):
        name = "mytool"
        extension = ".mytool"

        def export(self, scan: ScanExport, path: Path) -> Path:
            path.write_text(serialise(scan.regions, scan.projector_width, ...))
            return path

    register_exporter(MyToolExporter())

Everything an adapter could need is on :class:`ScanExport`, and its geometry is
already in projector pixels, so an adapter is a serialiser and nothing more.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..detect.regions import Region


@dataclass
class ScanExport:
    """Everything an exporter is allowed to need, all in projector pixels."""

    projector_width: int
    projector_height: int
    #: Regions in **projector** pixel coordinates, holes included.
    regions: list[Region] = field(default_factory=list)
    #: Projector-resolution boolean mask: True means project here.
    mask: np.ndarray | None = None
    #: Decode quality numbers, for the record.
    stats: dict[str, float] = field(default_factory=dict)
    #: Provenance: what produced this, from what, when.
    source: dict[str, str] = field(default_factory=dict)
    #: Anything the pipeline wants to warn the operator about.
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.projector_height, self.projector_width

    def validate(self) -> None:
        """Fail loudly on geometry that is not in projector space.

        An exporter handed camera-space coordinates would write a file that
        looks perfectly reasonable and is silently useless, so this is checked
        rather than assumed.
        """
        if self.projector_width <= 0 or self.projector_height <= 0:
            raise ValueError("projector resolution must be positive")
        if self.mask is not None and self.mask.shape != self.shape:
            raise ValueError(
                f"mask is {self.mask.shape} but the projector is {self.shape}; "
                "the mask must already be in projector space"
            )
        for region in self.regions:
            for ring in region.rings:
                if len(ring) and (ring.ndim != 2 or ring.shape[1] != 2):
                    raise ValueError(f"region {region.label!r} has a malformed ring")


class Exporter(ABC):
    """Writes a :class:`ScanExport` to one file in one format."""

    #: Short name, used to select this exporter from the CLI.
    name: str = "exporter"
    #: File extension including the dot.
    extension: str = ".out"

    @abstractmethod
    def export(self, scan: ScanExport, path: str | Path) -> Path:
        """Write ``scan`` to ``path`` and return the path written."""

    def default_filename(self) -> str:
        return f"{self.name}{self.extension}"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


_REGISTRY: dict[str, Exporter] = {}


def register_exporter(exporter: Exporter) -> Exporter:
    """Make an exporter selectable by name."""
    _REGISTRY[exporter.name] = exporter
    return exporter


def get_exporter(name: str) -> Exporter:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown export format {name!r}; available: {sorted(_REGISTRY)}"
        ) from None


def available_exporters() -> list[str]:
    return sorted(_REGISTRY)
