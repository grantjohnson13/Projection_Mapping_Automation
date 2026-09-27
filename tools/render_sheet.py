"""Render review sheets of the animation for judging.

Two sheets, because they answer different questions:

``variety.png``  twelve frames spread across the whole song -- does the show go
                anywhere, or is every moment the same moment?
``motion.png``   eight frames two thirds of a second apart -- does it move well,
                or is it a still image with flicker on top?

Usage:  python tools/render_sheet.py <out-dir> [config.toml]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from facade_scan.animate import analyse
from facade_scan.animate.layers import Layers
from facade_scan.config import Config, LayerConfig
from facade_scan.pipeline import load_decode
from facade_scan.preview import upscale_smooth
from facade_scan.transfer import foreground_projector_mask

SCAN = "results/2026-09-26_foreground-run/scan"
AUDIO = "assets/audio/most-wonderful-time.mp3"
RIG = "results/2026-09-26_animation-wonderful-time/config.toml"


def build(out_dir: str, config_path: str = RIG) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config.from_toml(config_path)
    # The look comes from the code's current defaults, not from the snapshot
    # saved next to the run: a saved config pins every value it contains, so
    # editing a default silently had no effect on what got rendered or
    # measured. Only the rig-specific margin is carried over.
    margin = cfg.animate.edge_margin_px
    cfg.animate = LayerConfig()
    cfg.animate.edge_margin_px = margin

    decoded = load_decode(SCAN)
    mask = upscale_smooth(foreground_projector_mask(decoded, cfg.transfer), (1920, 1080))
    track = analyse(AUDIO)
    layers = Layers(mask=mask, audio=track, cfg=cfg.animate)

    top, bottom, left, right = layers.box
    pad = 24
    crop = (slice(max(top - pad, 0), min(bottom + pad, 1080)),
            slice(max(left - pad, 0), min(right + pad, 1920)))

    def tile(time: float, label: str) -> np.ndarray:
        frame = layers.frame(float(time))[crop].copy()
        frame = cv2.copyMakeBorder(frame, 22, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(frame, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1, cv2.LINE_AA)
        return frame

    # Spread across the song, and deliberately including the last seconds:
    # sampling only up to duration-8 meant the ending was never once looked
    # at, so there was no way to tell a finale from its absence.
    times = list(np.linspace(6.0, track.duration - 9.0, 9))
    times += [track.duration - 6.0, track.duration - 3.0, track.duration - 1.0]
    tiles = [tile(t, f"{int(t // 60)}:{int(t % 60):02d}") for t in times]
    cv2.imwrite(str(out / "variety.png"),
                np.vstack([np.hstack(tiles[i:i + 4]) for i in (0, 4, 8)]))

    # A continuous run, to judge movement rather than composition.
    start = track.duration * 0.42
    tiles = [tile(start + i * 0.66, f"+{i * 0.66:.1f}s") for i in range(8)]
    cv2.imwrite(str(out / "motion.png"),
                np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))

    print(f"wrote {out}/variety.png and {out}/motion.png")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else RIG)
