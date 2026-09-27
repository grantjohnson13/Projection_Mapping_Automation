"""Detection: architectural lines and regions, found in camera space."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import DecodeConfig, DetectConfig
from .lines import (
    Segments,
    VanishingPoint,
    detect_depth_segments,
    detect_lines,
    detect_segments,
    estimate_vanishing_points,
    filter_short,
    merge_collinear,
    snap_endpoints,
    snap_to_vanishing_points,
)
from .regions import (
    Region,
    extend_segments,
    extract_regions,
    house_region,
    label_regions,
    polygonize_segments,
)
from .sam import SamUnavailable, sam_available, segment_with_sam

__all__ = [
    "Detection",
    "Region",
    "SamUnavailable",
    "Segments",
    "VanishingPoint",
    "detect",
    "detect_depth_segments",
    "detect_lines",
    "detect_segments",
    "estimate_vanishing_points",
    "extend_segments",
    "extract_regions",
    "filter_short",
    "house_region",
    "label_regions",
    "merge_collinear",
    "polygonize_segments",
    "sam_available",
    "segment_with_sam",
    "snap_endpoints",
    "snap_to_vanishing_points",
]


@dataclass
class Detection:
    """Everything found in the all-white capture, all in camera pixels."""

    segments: Segments
    vanishing_points: list[VanishingPoint]
    regions: list[Region]
    #: Camera-space mask of illuminated surface, if one was supplied.
    house_mask: np.ndarray | None = None
    raw_segment_count: int = 0
    notes: list[str] = field(default_factory=list)


def detect(white: np.ndarray, cfg: DetectConfig | None = None,
           glass_mask: np.ndarray | None = None,
           illumination: np.ndarray | None = None,
           illumination_threshold: float | None = None,
           depth_edges: np.ndarray | None = None) -> Detection:
    """Run the full detection pipeline on an all-white capture.

    ``glass_mask`` and ``illumination`` come from the decoder. Neither is
    required, but both make the result better: glass labels windows, and
    illumination gives the house mask.

    ``depth_edges`` is the one that matters on a difficult surface. Line
    detection on a photograph can only find an edge where two surfaces differ
    in *brightness*, and adjacent surfaces often do not. A cardboard box
    against a stainless door shows a crisp edge on the two sides that happen to
    fall in shadow and no edge at all on the other two, so the rectangle never
    closes and no region is found.

    But the box is a hundred millimetres nearer than the door, and that is a
    step in the decoded correspondence whatever the two surfaces look like. A
    structured-light scan measures geometry, so it can see boundaries a
    photograph cannot, and passing that edge map in lets the detector use them.
    """
    # Resolve size-relative thresholds once, here, so every function below
    # works in plain pixels.
    cfg = (cfg or DetectConfig()).resolve(white.shape)
    raw = detect_segments(white, cfg)
    if depth_edges is not None and depth_edges.any():
        raw = np.vstack([raw, detect_depth_segments(depth_edges, cfg)])
    long_enough = filter_short(raw, cfg.min_segment_length_px)
    vanishing_points = estimate_vanishing_points(long_enough, cfg)
    snapped = snap_to_vanishing_points(long_enough, vanishing_points, cfg)
    segments = snap_endpoints(merge_collinear(snapped, cfg), cfg)

    regions = extract_regions(segments, white.shape[:2], cfg, glass_mask)

    notes: list[str] = []
    if cfg.use_sam:
        try:
            regions = regions + segment_with_sam(white, cfg)
        except SamUnavailable as exc:
            # Never fail a scan over an optional extra.
            notes.append(f"Segment Anything skipped: {exc}")

    mask = None
    if illumination is not None:
        if illumination_threshold is None:
            # Defer to the one documented default rather than keeping a second
            # copy of it here that could drift.
            illumination_threshold = DecodeConfig().illumination_threshold
        mask = house_region(illumination, illumination_threshold)

    return Detection(segments=segments, vanishing_points=vanishing_points,
                     regions=regions, house_mask=mask,
                     raw_segment_count=len(raw), notes=notes)
