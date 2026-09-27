"""Optional Segment Anything integration.

This is behind a feature flag and is *not* required. ``torch`` and
``segment-anything`` are optional extras; nothing in the main pipeline imports
them, and a scan produces usable windows, doors and a garage door from line
detection plus the decoder's ``likely_glass`` mask alone.

Install with::

    pip install 'facade-scan[sam]'

and point ``detect.sam_checkpoint`` at a downloaded checkpoint.
"""

from __future__ import annotations

import numpy as np

from ..config import DetectConfig
from .regions import Region


class SamUnavailable(RuntimeError):
    """Raised when SAM is asked for but not installed or not configured."""


def sam_available() -> bool:
    """True if both optional packages import. Does not load any weights."""
    try:
        import segment_anything  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def segment_with_sam(image: np.ndarray, cfg: DetectConfig) -> list[Region]:
    """Propose regions with Segment Anything.

    Returns regions in the same form as the line-based detector, so the two can
    simply be concatenated.
    """
    if not sam_available():
        raise SamUnavailable(
            "segment-anything and torch are not installed. They are optional "
            "extras: pip install 'facade-scan[sam]'. The pipeline works without "
            "them -- leave detect.use_sam = false."
        )
    if not cfg.sam_checkpoint:
        raise SamUnavailable(
            "detect.sam_checkpoint is not set. Download a Segment Anything "
            "checkpoint and point that setting at it."
        )

    import cv2
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    model = sam_model_registry[cfg.sam_model_type](checkpoint=cfg.sam_checkpoint)
    model.to(cfg.sam_device)
    generator = SamAutomaticMaskGenerator(model, points_per_side=cfg.sam_points_per_side)

    regions: list[Region] = []
    for i, proposal in enumerate(generator.generate(rgb)):
        mask = proposal["segmentation"].astype(np.uint8)
        if mask.sum() < cfg.sam_min_area_px:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        biggest = max(contours, key=cv2.contourArea)
        simplified = cv2.approxPolyDP(biggest, epsilon=3.0, closed=True)
        if len(simplified) < 3:
            continue
        regions.append(Region(
            polygon=simplified.reshape(-1, 2).astype(np.float64),
            label=f"sam_{i:02d}",
            attributes={"area_px": float(mask.sum()),
                        "predicted_iou": float(proposal.get("predicted_iou", 0.0))},
        ))
    return regions
