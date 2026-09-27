"""Rendering the animation: to a file, and live to the projector."""

from __future__ import annotations

import shutil
import subprocess
import time as clock
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import AudioAnalysis
from .layers import LayerConfig, Layers


@dataclass
class RenderResult:
    path: Path
    frames: int
    fps: int
    seconds: float
    has_audio: bool


def render_video(mask: np.ndarray, audio: AudioAnalysis, out_path: str | Path,
                 cfg: LayerConfig | None = None, fps: int = 30,
                 start: float = 0.0, duration: float | None = None,
                 mux_audio: bool = True, on_frame=None) -> RenderResult:
    """Render the animation to an MP4 at the mask's own resolution.

    The audio is muxed back in when ffmpeg is available, so the result is a
    single file you can open fullscreen on the projector and loop. That matters
    on the night: one file, one player, nothing to synchronise by hand.
    """
    import cv2

    cfg = cfg or LayerConfig()
    layers = Layers(mask=mask, audio=audio, cfg=cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    length = audio.duration - start if duration is None else duration
    total = max(1, round(length * fps))
    size = (layers.width, layers.height)

    silent = out_path.with_name(out_path.stem + "_silent.mp4") if mux_audio else out_path
    writer = None
    for codec in ("avc1", "mp4v"):
        fourcc = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
        candidate = cv2.VideoWriter(str(silent), fourcc(*codec), fps, size)
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()
    if writer is None:
        raise OSError(f"could not open a video writer for {silent}")

    for index in range(total):
        writer.write(layers.frame(start + index / fps))
        if on_frame is not None:
            on_frame(index, total)
    writer.release()

    has_audio = False
    if mux_audio and shutil.which("ffmpeg"):
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(silent),
             "-ss", f"{start}", "-t", f"{length}", "-i", str(audio.path),
             "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
             "-shortest", str(out_path)],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and out_path.exists():
            has_audio = True
            silent.unlink(missing_ok=True)
        else:
            # Keep the silent render rather than losing the work.
            silent.replace(out_path)
    elif mux_audio:
        silent.replace(out_path)

    return RenderResult(path=out_path, frames=total, fps=fps,
                        seconds=total / fps, has_audio=has_audio)


def play_live(display, mask: np.ndarray, audio: AudioAnalysis,
              cfg: LayerConfig | None = None, fps: int = 30,
              seconds: float | None = None, start: float = 0.0,
              native_size: tuple[int, int] | None = None,
              clock_fn=clock.monotonic) -> int:
    """Play the animation straight to the projector.

    Frames are generated against the wall clock rather than counted, so if a
    frame takes too long the animation stays in time with the music by skipping
    rather than drifting. Drifting is what you would notice.
    """
    import cv2

    cfg = cfg or LayerConfig()
    layers = Layers(mask=mask, audio=audio, cfg=cfg)
    length = (audio.duration - start) if seconds is None else seconds
    began = clock_fn()
    drawn = 0
    last_index = -1

    while True:
        elapsed = clock_fn() - began
        if elapsed >= length:
            break
        index = int(elapsed * fps)
        if index == last_index:
            display.poll(1)
            continue
        last_index = index

        frame = layers.frame(start + elapsed)
        if native_size is not None and native_size != (layers.width, layers.height):
            frame = cv2.resize(frame, native_size, interpolation=cv2.INTER_NEAREST)
        key = display.show(frame, wait_ms=1)
        drawn += 1
        if key in (27, ord("q")):
            break
    return drawn
