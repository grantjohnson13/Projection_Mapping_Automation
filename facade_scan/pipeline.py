"""The stages wired together, and the artifacts each one leaves behind.

Every stage writes its output to disk before the next one starts. That is not
tidiness, it is the difference between a scan you can debug and a scan you have
to repeat. Repeating a scan means going back out to the house at night, setting
the projector up again, and hoping for the same weather -- so if detection
throws an exception, the captures and the decoded map are still sitting there
and you can iterate on them indoors at a desk.

Layout of a scan directory::

    scan/
      config.toml          the effective config, so the run is reproducible
      patterns/            the projected frames + manifest.json
      captures/            one photograph per frame + manifest.json
      decoded.npz          the camera -> projector map and every mask
      detection.json       camera-space segments and regions
      white.png            the all-white capture, for eyeballing
      export/
        mask.png
        regions.svg
        scan.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import __version__
from .config import Config
from .decode import DecodeResult, decode_directory, discontinuity_mask
from .detect import Detection, Region, VanishingPoint, detect
from .export import ScanExport, export_all
from .transfer import foreground_projector_mask, house_mask, transfer_regions


def scan_layout(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    return {
        "root": root,
        "config": root / "config.toml",
        "patterns": root / "patterns",
        "captures": root / "captures",
        "decoded": root / "decoded.npz",
        "detection": root / "detection.json",
        "white": root / "white.png",
        "export": root / "export",
    }


def write_config(root: str | Path, config: Config) -> Path:
    paths = scan_layout(root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["config"].write_text(
        f"# facade-scan {__version__} effective config, written "
        f"{datetime.now(UTC).isoformat(timespec='seconds')}\n"
        + config.to_toml()
    )
    return paths["config"]


# --------------------------------------------------------------------------- #
# Decode
# --------------------------------------------------------------------------- #
def run_decode(root: str | Path, config: Config) -> DecodeResult:
    """Decode ``captures/`` and write ``decoded.npz`` plus ``white.png``."""
    import cv2

    paths = scan_layout(root)
    result = decode_directory(paths["captures"], config.decode)
    result.save(paths["decoded"])
    cv2.imwrite(str(paths["white"]),
                np.clip(result.white * 255.0, 0, 255).astype(np.uint8))
    return result


def load_decode(root: str | Path) -> DecodeResult:
    paths = scan_layout(root)
    if not paths["decoded"].exists():
        raise FileNotFoundError(
            f"{paths['decoded']} not found -- run `facade-scan decode` first."
        )
    return DecodeResult.load(paths["decoded"])


# --------------------------------------------------------------------------- #
# Detect
# --------------------------------------------------------------------------- #
def run_detect(root: str | Path, config: Config,
               decoded: DecodeResult | None = None) -> Detection:
    """Detect in camera space and write ``detection.json``."""
    paths = scan_layout(root)
    decoded = decoded or load_decode(root)
    white = np.clip(decoded.white * 255.0, 0, 255).astype(np.uint8)

    # A surface boundary is a step in the correspondence, whether or not the
    # two surfaces differ in brightness. Hand those to the detector alongside
    # the photograph.
    transfer_cfg = config.transfer.resolve(
        decoded.valid.shape, (decoded.projector_width, decoded.projector_height))
    # Two passes: a tight one on adjacent pixels, which is exact on clean data,
    # and a wider one that can see across the undecoded band that usually sits
    # inside a real silhouette.
    depth_edges = discontinuity_mask(decoded.proj_map, decoded.valid,
                                     transfer_cfg.discontinuity_jump_px)
    if transfer_cfg.discontinuity_window_px > 1:
        depth_edges = depth_edges | discontinuity_mask(
            decoded.proj_map, decoded.valid, transfer_cfg.discontinuity_bridge_px,
            window=transfer_cfg.discontinuity_window_px)

    found = detect(white, config.detect,
                   glass_mask=decoded.likely_glass,
                   illumination=decoded.illumination,
                   illumination_threshold=config.decode.illumination_threshold,
                   depth_edges=depth_edges)
    paths["detection"].write_text(json.dumps(detection_to_dict(found), indent=2))
    return found


def _vanishing_point_to_dict(vp: VanishingPoint) -> dict:
    point = vp.image_point
    return {
        "homogeneous": [float(v) for v in vp.point],
        "image_point": None if point is None else [round(float(c), 2) for c in point],
        "at_infinity": bool(vp.at_infinity),
        "inlier_count": len(vp.inliers),
        "support_px": round(float(vp.support), 1),
    }


def detection_to_dict(found: Detection) -> dict:
    return {
        "coordinate_space": "camera_pixels",
        "raw_segment_count": found.raw_segment_count,
        "segments": [[round(float(v), 2) for v in s] for s in found.segments],
        "vanishing_points": [_vanishing_point_to_dict(vp)
                             for vp in found.vanishing_points],
        "regions": [
            {
                "label": r.label,
                "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in r.polygon],
                "holes": [[[round(float(x), 2), round(float(y), 2)] for x, y in h]
                          for h in r.holes],
                "attributes": {k: round(float(v), 4) for k, v in sorted(r.attributes.items())},
            }
            for r in found.regions
        ],
        "notes": list(found.notes),
    }


def load_detection(root: str | Path) -> list[Region]:
    paths = scan_layout(root)
    if not paths["detection"].exists():
        raise FileNotFoundError(
            f"{paths['detection']} not found -- run `facade-scan detect` first."
        )
    data = json.loads(paths["detection"].read_text())
    return [
        Region(polygon=np.array(r["polygon"], dtype=np.float64),
               label=r["label"],
               holes=[np.array(h, dtype=np.float64) for h in r.get("holes", [])],
               attributes=dict(r.get("attributes", {})))
        for r in data["regions"]
    ]


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
@dataclass
class ExportOutcome:
    scan: ScanExport
    paths: dict[str, Path]


def run_export(root: str | Path, config: Config,
               decoded: DecodeResult | None = None,
               regions: list[Region] | None = None,
               mask_only: bool = False,
               foreground: bool = False) -> ExportOutcome:
    """Transfer everything to projector space and write the artifacts."""
    paths = scan_layout(root)
    decoded = decoded or load_decode(root)

    mask = (foreground_projector_mask(decoded, config.transfer) if foreground
            else house_mask(decoded, config.transfer, config.decode))

    transferred: list[Region] = []
    notes: list[str] = []
    if not mask_only:
        if regions is None:
            try:
                regions = load_detection(root)
            except FileNotFoundError:
                regions = []
                notes.append("no detection found; exported the house mask only")
        result = transfer_regions(decoded, regions, config.transfer)
        transferred = result.regions
        notes.extend(result.dropped)

    scan = ScanExport(
        projector_width=decoded.projector_width,
        projector_height=decoded.projector_height,
        regions=transferred,
        mask=mask,
        stats=decoded.stats(),
        source={
            "tool": f"facade-scan {__version__}",
            "scan": str(Path(root).resolve()),
            "exported": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        notes=notes,
    )
    written = export_all(scan, paths["export"], config.export)
    return ExportOutcome(scan=scan, paths=written)
