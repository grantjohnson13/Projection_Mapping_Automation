"""Turn a finished scan directory into a video and a summary.

A scan produces a pile of numpy arrays. That is fine for a machine and useless
for deciding whether the test rig on the kitchen table actually worked. This
module renders the whole thing as a short video -- what the camera saw, what
decoded, what was detected, and what comes out the other end -- plus a summary
you can diff between runs.

Everything here works on a real scan and a simulated one alike. When a
``ground_truth.npz`` is present (simulated scans have one) the summary also
scores the decode against it; when it is not, the summary falls back to the
measures that need no ground truth at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import __version__
from .config import Config, ReportConfig
from .decode import DecodeResult
from .detect.regions import Region
from .pipeline import load_decode, load_detection, scan_layout
from .preview import render_preview
from .transfer import house_mask, transfer_regions

BACKGROUND = (18, 18, 20)
TEXT = (235, 235, 235)
ACCENT = (90, 220, 120)
WARN = (60, 180, 250)


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def _to_bgr(image: np.ndarray) -> np.ndarray:
    import cv2

    if image.dtype != np.uint8:
        top = float(image.max()) if image.size else 1.0
        scale = 255.0 if top <= 1.0 else 1.0
        image = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def fit_into(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Letterbox an image into ``(width, height)`` without distorting it."""
    import cv2

    width, height = size
    image = _to_bgr(image)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)),
                                 max(1, int(image.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), BACKGROUND, np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def caption(frame: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """Draw a translucent caption bar along the bottom."""
    import cv2

    out = frame.copy()
    height, width = out.shape[:2]
    bar_height = 86 if subtitle else 58
    overlay = out.copy()
    cv2.rectangle(overlay, (0, height - bar_height), (width, height), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.62, out, 0.38, 0)
    cv2.putText(out, title, (28, height - bar_height + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, TEXT, 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (28, height - bar_height + 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, ACCENT, 1, cv2.LINE_AA)
    return out


def card(lines: list[tuple[str, float, tuple[int, int, int]]],
         size: tuple[int, int]) -> np.ndarray:
    """A plain text card: list of (text, scale, colour)."""
    import cv2

    width, height = size
    canvas = np.full((height, width, 3), BACKGROUND, np.uint8)
    total = sum(int(46 * scale) + 16 for _, scale, _ in lines)
    y = max(60, (height - total) // 2 + 30)
    for text, scale, colour in lines:
        cv2.putText(canvas, text, (70, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    colour, 2 if scale > 0.7 else 1, cv2.LINE_AA)
        y += int(46 * scale) + 16
    return canvas


def colourise_correspondence(decoded: DecodeResult) -> np.ndarray:
    """The decoded map as an image: red encodes projector x, green projector y.

    A smooth two-way gradient means a clean decode. Speckle means noise, and a
    hard discontinuity across an object edge is a depth step -- which is the
    thing a homography could not have produced.
    """
    height, width = decoded.valid.shape
    out = np.zeros((height, width, 3), np.uint8)
    x = np.clip(decoded.proj_map[..., 0] / max(decoded.projector_width - 1, 1), 0, 1)
    y = np.clip(decoded.proj_map[..., 1] / max(decoded.projector_height - 1, 1), 0, 1)
    out[..., 2] = (x * 255).astype(np.uint8)      # red   = projector x
    out[..., 1] = (y * 255).astype(np.uint8)      # green = projector y
    out[..., 0] = 90
    out[~decoded.valid] = (30, 30, 34)
    return out


def colourise_confidence(decoded: DecodeResult) -> np.ndarray:
    import cv2

    scaled = np.clip(decoded.min_confidence / 0.5, 0, 1)
    return cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def draw_detection(white: np.ndarray, segments: np.ndarray,
                   regions: list[Region]) -> np.ndarray:
    import cv2

    out = _to_bgr(white)
    for segment in segments:
        cv2.line(out, (int(segment[0]), int(segment[1])),
                 (int(segment[2]), int(segment[3])), (60, 90, 255), 2, cv2.LINE_AA)
    for region in regions:
        colour = (0, 255, 120) if not region.label.startswith("window") else (255, 180, 0)
        for ring in region.rings:
            if len(ring) >= 3:
                cv2.polylines(out, [np.round(ring).astype(np.int32)], True, colour,
                              2, cv2.LINE_AA)
        centre = np.round(region.centroid).astype(int)
        cv2.putText(out, region.label, tuple(centre), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colour, 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Ground-truth-free quality measures
# --------------------------------------------------------------------------- #
def planarity_residual(decoded: DecodeResult, region: np.ndarray | None = None,
                       sample: int = 40_000, seed: int = 0,
                       with_inliers: bool = False):
    """RMS residual, in projector pixels, of a homography fitted to a flat area.

    The decoded map over any *planar* surface is exactly a homography. So on a
    scan of a flat wall, the residual of a fitted homography is not modelling
    error -- it is the measurement noise of the scan itself, in projector
    pixels, obtained with no ground truth whatsoever.

    This is the number to watch on a physical rig, where nothing knows the
    right answer. Returns None if there are too few valid pixels to fit.

    .. important::

       It only means "measurement noise" if the region really is one plane.
       Fit it to a scene containing two surfaces at different depths and the
       parallax between them lands in the residual, making a *good* scan look
       bad. RANSAC rejects the minority surface, so pass ``with_inliers=True``
       and watch the inlier fraction: well below 1.0 means the fit found one
       plane among several and the residual describes only that one.

       Measured on one real rig: 1.41 px over a whole scene, but 0.42 px over
       the flat cardboard box within it and 0.88 px over the specular fridge
       door behind it. The mixed figure described neither surface.
    """
    import cv2

    usable = decoded.valid if region is None else (decoded.valid & region)
    ys, xs = np.nonzero(usable)
    if len(ys) < 500:
        return None

    rng = np.random.default_rng(seed)
    if len(ys) > sample:
        pick = rng.choice(len(ys), sample, replace=False)
        ys, xs = ys[pick], xs[pick]

    camera = np.stack([xs, ys], -1).astype(np.float32)
    projector = decoded.proj_map[ys, xs].astype(np.float32)
    homography, inliers = cv2.findHomography(camera, projector, cv2.RANSAC, 3.0)
    if homography is None:
        return None
    predicted = cv2.perspectiveTransform(camera.reshape(-1, 1, 2),
                                         homography).reshape(-1, 2)
    keep = (inliers.ravel().astype(bool) if inliers is not None
            else np.ones(len(camera), dtype=bool))
    if keep.sum() < 50:
        return None
    residual = np.linalg.norm(predicted[keep] - projector[keep], axis=1)
    value = float(np.sqrt((residual ** 2).mean()))
    return (value, float(keep.mean())) if with_inliers else value


def per_bit_confidence(decoded: DecodeResult) -> dict[str, float]:
    lit = decoded.illumination > 0.02
    if not lit.any():
        return {}
    return {
        "p05": float(np.percentile(decoded.min_confidence[lit], 5)),
        "median": float(np.median(decoded.min_confidence[lit])),
        "p95": float(np.percentile(decoded.min_confidence[lit], 95)),
    }


def score_against_ground_truth(decoded: DecodeResult,
                               path: str | Path) -> dict[str, float]:
    """Median and p95 projector-pixel error, for simulated scans."""
    with np.load(path) as truth:
        valid = decoded.valid & truth["valid"]
        if not valid.any():
            return {}
        error = np.linalg.norm(
            decoded.proj_map.astype(np.float64) - truth["proj_uv"], axis=-1
        )[valid]
        decodable = truth["valid"]
        return {
            "median_error_px": float(np.median(error)),
            "p95_error_px": float(np.percentile(error, 95)),
            "max_error_px": float(error.max()),
            "coverage_of_decodable": float(decoded.valid[decodable].mean()),
            "scored_pixels": float(valid.sum()),
        }


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
@dataclass
class Report:
    summary: dict = field(default_factory=dict)
    video_path: Path | None = None
    still_paths: list[Path] = field(default_factory=list)
    summary_paths: list[Path] = field(default_factory=list)


def write_video(path: str | Path, frames: list[np.ndarray],
                cfg: ReportConfig) -> Path:
    """Encode frames, preferring H.264 and falling back if unavailable."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = (cfg.width, cfg.height)
    for codec in (cfg.codec, cfg.codec_fallback):
        # VideoWriter_fourcc moved around between OpenCV versions; the stubs
        # only know one spelling.
        fourcc = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
        writer = cv2.VideoWriter(str(path), fourcc(*codec), cfg.fps, size)
        if writer.isOpened():
            for frame in frames:
                writer.write(frame)
            writer.release()
            if path.exists() and path.stat().st_size > 1024:
                return path
    raise OSError(
        f"could not encode {path}: neither {cfg.codec!r} nor "
        f"{cfg.codec_fallback!r} is available in this OpenCV build"
    )


def build_report(scan_dir: str | Path, out_dir: str | Path, config: Config,
                 title: str = "facade-scan", notes: str = "") -> Report:
    """Render a scan directory into a video, stills and a summary."""
    import cv2

    scan_dir = Path(scan_dir)
    out_dir = Path(out_dir)
    (out_dir / "stills").mkdir(parents=True, exist_ok=True)
    cfg = config.report
    size = (cfg.width, cfg.height)
    paths = scan_layout(scan_dir)

    decoded = load_decode(scan_dir)
    try:
        camera_regions = load_detection(scan_dir)
    except FileNotFoundError:
        camera_regions = []

    white = np.clip(decoded.white * 255.0, 0, 255).astype(np.uint8)
    mask = house_mask(decoded, config.transfer, config.decode)
    projector_regions = transfer_regions(decoded, camera_regions,
                                         config.transfer).regions

    segments = np.zeros((0, 4))
    detection_file = paths["detection"]
    if detection_file.exists():
        segments = np.array(json.loads(detection_file.read_text())["segments"],
                            dtype=np.float64).reshape(-1, 4)

    # ----------------------------------------------------------- summary --
    stats = decoded.stats()
    summary: dict = {
        "title": title,
        "notes": notes,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "tool": f"facade-scan {__version__}",
        "scan": str(scan_dir.resolve()),
        "camera": {"width": decoded.valid.shape[1],
                   "height": decoded.valid.shape[0]},
        "projector": {"width": decoded.projector_width,
                      "height": decoded.projector_height},
        "decode": {k: round(float(v), 6) for k, v in stats.items()},
        "per_bit_confidence": {k: round(v, 4)
                               for k, v in per_bit_confidence(decoded).items()},
        "detection": {"segments": len(segments),
                      "camera_regions": len(camera_regions),
                      "projector_regions": len(projector_regions)},
        "projector_mask": {
            "lit_pixels": int(mask.sum()),
            "lit_fraction": round(float(mask.mean()), 4),
        },
    }
    measured = planarity_residual(decoded, with_inliers=True)
    residual = None
    if measured is not None:
        residual, inlier_fraction = measured
        summary["planarity_residual_px"] = round(residual, 3)
        summary["planarity_inlier_fraction"] = round(inlier_fraction, 3)
        if inlier_fraction < 0.9:
            summary.setdefault("notes_auto", []).append(
                f"Only {inlier_fraction:.0%} of decoded pixels fitted one plane, "
                "so the scene holds more than one surface and this residual "
                "describes the largest of them, not the whole scan."
            )

    ground_truth = scan_dir / "ground_truth.npz"
    if ground_truth.exists():
        scored = score_against_ground_truth(decoded, ground_truth)
        if scored:
            summary["ground_truth"] = {k: round(v, 4) for k, v in scored.items()}

    # ------------------------------------------------------------- video --
    frames: list[np.ndarray] = []
    stills: list[tuple[str, np.ndarray]] = []

    def hold(image: np.ndarray, seconds: float) -> None:
        frames.extend([image] * max(1, round(seconds * cfg.fps)))

    def section(name: str, image: np.ndarray, heading: str, sub: str) -> None:
        framed = caption(fit_into(image, size), heading, sub)
        stills.append((name, framed))
        hold(framed, cfg.section_seconds)

    camera_px = f"{decoded.valid.shape[1]}x{decoded.valid.shape[0]}"
    projector_px = f"{decoded.projector_width}x{decoded.projector_height}"
    oversampling = decoded.valid.shape[1] / max(decoded.projector_width, 1)

    hold(card([
        (title, 1.15, TEXT),
        (datetime.now().strftime("%Y-%m-%d %H:%M"), 0.6, ACCENT),
        ("", 0.4, TEXT),
        (f"projector {projector_px}    camera {camera_px}", 0.62, TEXT),
        (f"camera/projector width ratio {oversampling:.2f}x", 0.62, TEXT),
        (notes or "", 0.55, WARN),
    ], size), cfg.card_seconds)

    # The capture sequence, replayed slowly enough to see.
    capture_files = sorted(paths["captures"].glob("*.png"))
    if capture_files:
        repeats = max(1, round(cfg.fps / cfg.capture_replay_fps))
        for i, capture in enumerate(capture_files):
            image = cv2.imread(str(capture), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            framed = caption(fit_into(image, size), "1. What the camera saw",
                             f"frame {i + 1} of {len(capture_files)}   "
                             f"{capture.name}")
            frames.extend([framed] * repeats)
        stills.append(("01_capture_white", caption(
            fit_into(white, size), "1. What the camera saw",
            "the all-white frame: the projector as a floodlight")))

    section("02_correspondence", colourise_correspondence(decoded),
            "2. Decoded correspondence",
            "red = projector x, green = projector y. Smooth means clean; "
            "a hard step means depth.")

    section("03_confidence", colourise_confidence(decoded),
            "3. Per-pixel confidence",
            f"|pattern - inverse|, weakest bit.   coverage "
            f"{stats['coverage']:.1%} of frame")

    if len(segments) or camera_regions:
        section("04_detection", draw_detection(white, segments, camera_regions),
                "4. Detected geometry (camera space)",
                f"{len(segments)} merged edges, {len(camera_regions)} regions")

    section("05_projector_mask", np.where(mask, 255, 0).astype(np.uint8),
            "5. Projector-space mask",
            f"white = project here.   {int(mask.sum()):,} of "
            f"{mask.size:,} projector pixels")

    # The preview, blinking, because that is how it is actually used on site.
    preview_shape = (decoded.projector_height, decoded.projector_width)
    blink_frames = max(1, round(cfg.preview_blink_ms / 1000.0 * cfg.fps))
    lit_frame = caption(fit_into(render_preview(
        mask, projector_regions, preview_shape, config.preview, "outline"), size),
        "6. Preview: projected back at the target",
        "the edge should sit exactly on the edge of the object")
    dark_frame = caption(fit_into(
        np.zeros((*preview_shape, 3), np.uint8), size),
        "6. Preview: projected back at the target",
        "blinking makes a few pixels of misalignment obvious")
    for _ in range(3):
        frames.extend([lit_frame] * blink_frames)
        frames.extend([dark_frame] * blink_frames)
    stills.append(("06_preview", lit_frame))

    summary_lines: list[tuple[str, float, tuple[int, int, int]]] = [
        ("Result", 1.0, TEXT),
        ("", 0.3, TEXT),
        (f"coverage            {stats['coverage']:.1%} of camera frame", 0.62, TEXT),
        (f"projector mask      {mask.mean():.1%} of panel lit", 0.62, TEXT),
        (f"regions found       {len(camera_regions)}", 0.62, TEXT),
    ]
    if residual is not None:
        summary_lines.append(
            (f"planarity residual  {residual:.2f} px  "
             f"({summary['planarity_inlier_fraction']:.0%} of pixels on one plane)",
             0.62, ACCENT))
    if "ground_truth" in summary:
        g = summary["ground_truth"]
        summary_lines.extend([
            ("", 0.3, TEXT),
            (f"vs ground truth     median {g['median_error_px']:.2f} px, "
             f"p95 {g['p95_error_px']:.2f} px", 0.62, ACCENT),
            (f"                    coverage of decodable "
             f"{g['coverage_of_decodable']:.1%}", 0.62, ACCENT),
        ])
    hold(card(summary_lines, size), cfg.card_seconds)

    video_path = write_video(out_dir / "result.mp4", frames, cfg)

    still_paths: list[Path] = []
    if cfg.write_stills:
        for name, image in stills:
            still = out_dir / "stills" / f"{name}.png"
            cv2.imwrite(str(still), image)
            still_paths.append(still)

    summary["video"] = {"path": video_path.name,
                        "frames": len(frames),
                        "seconds": round(len(frames) / cfg.fps, 1)}

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    md_path = out_dir / "summary.md"
    md_path.write_text(_summary_markdown(summary))

    return Report(summary=summary, video_path=video_path,
                  still_paths=still_paths, summary_paths=[json_path, md_path])


def _summary_markdown(summary: dict) -> str:
    lines = [f"# {summary['title']}", "",
             f"*{summary['generated']} — {summary['tool']}*", ""]
    if summary.get("notes"):
        lines += [summary["notes"], ""]
    lines += [
        "| | |",
        "|---|---|",
        f"| projector | {summary['projector']['width']}x{summary['projector']['height']} |",
        f"| camera | {summary['camera']['width']}x{summary['camera']['height']} |",
        f"| coverage | {summary['decode']['coverage']:.1%} of camera frame |",
        f"| projector mask | {summary['projector_mask']['lit_fraction']:.1%} of panel |",
        f"| regions | {summary['detection']['camera_regions']} camera, "
        f"{summary['detection']['projector_regions']} transferred |",
    ]
    if "planarity_residual_px" in summary:
        lines.append(f"| planarity residual | {summary['planarity_residual_px']:.2f} px |")
    if "ground_truth" in summary:
        g = summary["ground_truth"]
        lines += [
            f"| median error | {g['median_error_px']:.2f} px |",
            f"| p95 error | {g['p95_error_px']:.2f} px |",
            f"| coverage of decodable | {g['coverage_of_decodable']:.1%} |",
        ]
    lines += ["", f"Video: `{summary['video']['path']}` "
                  f"({summary['video']['seconds']}s)", ""]
    return "\n".join(lines)
