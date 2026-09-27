"""Turning a set of line segments into closed, labelled regions.

Architectural edges are detected as segments, but what you actually want to
project onto is a *face*: a window opening, a garage door, a gable end. The
trick is that a set of segments that visually enclose a window usually do not
quite touch -- the detector stops a few pixels short at each corner where the
contrast fades.

So each segment is extended slightly at both ends, the whole lot is noded
against itself with shapely's ``unary_union`` (which splits every line at every
crossing), and ``polygonize`` reads off the faces of the resulting planar graph.
The extension is what closes the near-misses; without it almost nothing encloses.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import polygonize, unary_union

from ..config import DetectConfig
from .lines import Segments, directions


@dataclass
class Region:
    """A closed face in camera space, possibly with openings in it.

    Holes are not a detail. Polygonizing a facade produces one big face whose
    interior rings are the window openings, and a region that has dropped its
    holes is a region that will project light straight through the glass. So
    they are carried all the way through transfer and export.
    """

    #: (N, 2) exterior ring, in camera pixels.
    polygon: np.ndarray
    label: str = "region"
    #: Extra facts worth carrying into scan.json.
    attributes: dict[str, float] = field(default_factory=dict)
    #: Interior rings -- openings within this face, in camera pixels.
    holes: list[np.ndarray] = field(default_factory=list)

    @property
    def shapely(self) -> Polygon:
        return Polygon(self.polygon, [h for h in self.holes if len(h) >= 3])

    @property
    def rings(self) -> list[np.ndarray]:
        """Exterior ring followed by every hole."""
        return [self.polygon, *self.holes]

    def rasterize(self, shape: tuple[int, int]) -> np.ndarray:
        """Fill the region into a boolean mask of ``shape``, holes excluded."""
        import cv2

        mask = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(self.polygon).astype(np.int32)], 1)
        for hole in self.holes:
            if len(hole) >= 3:
                cv2.fillPoly(mask, [np.round(hole).astype(np.int32)], 0)
        return mask.astype(bool)

    @property
    def area(self) -> float:
        return float(self.shapely.area)

    @property
    def centroid(self) -> np.ndarray:
        c = self.shapely.centroid
        return np.array([c.x, c.y])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.shapely.bounds


def extend_segments(segments: Segments, amount: float) -> Segments:
    """Grow every segment by ``amount`` pixels at both ends."""
    if len(segments) == 0:
        return segments.copy()
    d = directions(segments) * amount
    out = segments.copy()
    out[:, 0:2] -= d
    out[:, 2:4] += d
    return out


def polygonize_segments(segments: Segments, shape: tuple[int, int],
                        cfg: DetectConfig | None = None) -> list[Polygon]:
    """Extend, node and polygonize a segment set into candidate faces."""
    cfg = cfg or DetectConfig()
    if len(segments) < 3:
        return []

    grown = extend_segments(segments, cfg.extend_px)
    lines = [LineString([(s[0], s[1]), (s[2], s[3])]) for s in grown
             if not np.allclose(s[0:2], s[2:4])]
    if len(lines) < 3:
        return []

    # unary_union splits every line at every intersection, which is what turns a
    # pile of overlapping segments into a planar graph polygonize can read.
    noded = unary_union(MultiLineString(lines))
    faces = [p for p in polygonize(noded) if p.is_valid and not p.is_empty]

    height, width = shape[:2]
    frame_area = float(width * height)
    keep = []
    for face in faces:
        if face.area < cfg.min_region_area_px:
            continue
        if face.area > cfg.max_region_area_frac * frame_area:
            # Almost always the outer boundary face rather than a real feature.
            continue
        keep.append(face)
    keep.sort(key=lambda p: -p.area)
    return keep


def label_regions(faces: Iterable[Polygon], shape: tuple[int, int],
                  glass_mask: np.ndarray | None = None,
                  glass_fraction: float = 0.5) -> list[Region]:
    """Name each face, using the decoder's glass mask where one is available.

    ``likely_glass`` comes out of the Gray-code decode for free -- a pixel that
    is bright but carries no pattern modulation is almost certainly glazing --
    so a face that is mostly glass is almost certainly a window. This is why the
    decoder bothers to publish that mask.
    """
    regions: list[Region] = []
    for i, face in enumerate(faces):
        ring = np.asarray(face.exterior.coords, dtype=np.float64)[:-1]
        holes = [np.asarray(interior.coords, dtype=np.float64)[:-1]
                 for interior in face.interiors]
        region = Region(polygon=ring, label=f"region_{i:02d}", holes=holes,
                        attributes={"area_px": float(face.area)})

        if glass_mask is not None:
            fraction = _mask_fraction(region, glass_mask, shape)
            region.attributes["glass_fraction"] = fraction
            if fraction >= glass_fraction:
                region.label = f"window_{i:02d}"

        regions.append(region)
    return regions


def _mask_fraction(region: Region, mask: np.ndarray, shape: tuple[int, int]) -> float:
    """Fraction of a region's pixels that are set in a boolean mask.

    Holes are excluded, so a facade riddled with window openings is not judged
    to be glass just because its openings are.
    """
    inside = region.rasterize(shape)
    if not inside.any():
        return 0.0
    return float(mask[inside].mean())


def extract_regions(segments: Segments, shape: tuple[int, int],
                    cfg: DetectConfig | None = None,
                    glass_mask: np.ndarray | None = None) -> list[Region]:
    """Segments in, labelled closed regions out."""
    cfg = (cfg or DetectConfig()).resolve(shape)
    faces = polygonize_segments(segments, shape, cfg)
    return label_regions(faces, shape, glass_mask, cfg.window_glass_fraction)


def house_region(illumination: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of everything the projector is actually landing on.

    ``illumination`` is white minus black, so ambient light and glare are
    already gone and what remains is projector light on a real surface.
    """
    return illumination >= threshold
