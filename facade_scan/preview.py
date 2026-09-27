"""Project the computed mask back onto the house.

This is the feedback loop that makes the tool usable on site. You have just
spent several minutes holding still in the cold; the question you need answered
immediately, before packing anything away, is "did that work?". Opening a
mapping tool to find out is the wrong answer, so the preview projects the result
straight back at the house and you look at the wall.

What to look for
----------------
The mask edge should sit exactly on the edge of the house. If it is out by a
consistent few pixels everywhere, the projector has moved since the scan and you
need to scan again. If it is out only around the garage or under the eaves, the
decode was poor there -- check ``decoded.npz`` coverage rather than rescanning.

Blink mode is on for a reason: a static edge a few pixels off the roofline is
genuinely hard to see from the driveway at night, and a blinking one is obvious.

Controls
--------
=========  ====================================================================
space      next region (in cycle mode)
a          show everything at once
b          toggle blinking
o          cycle mask -> outline -> fill
q / esc    quit
=========  ====================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import PreviewConfig
from .detect.regions import Region

MODES = ("mask", "outline", "fill", "cycle")


def upscale_smooth(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Enlarge a boolean mask by redrawing its outline, not its pixels.

    The correspondence is only known at pattern-grid resolution, so a mask
    computed from it is coarse -- 480x270 against a 1920x1080 panel. Blowing
    that up pixel-wise, which is right for a Gray-code stripe, puts a visible
    four-pixel staircase along every edge of the projected result.

    The underlying edge is not a staircase; it is a smooth boundary sampled
    coarsely. So the contours are extracted, scaled, and refilled at panel
    resolution, which recovers a straight edge from a straight edge.
    """
    import cv2

    width, height = size
    if (mask.shape[1], mask.shape[0]) == (width, height):
        return mask
    scale_x = width / mask.shape[1]
    scale_y = height / mask.shape[0]

    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros((height, width), np.uint8)
    if not contours:
        return out.astype(bool)

    for index, contour in enumerate(contours):
        scaled = contour.astype(np.float64)
        scaled[:, :, 0] *= scale_x
        scaled[:, :, 1] *= scale_y
        # A hole in RETR_CCOMP has a parent; fill it back out.
        is_hole = hierarchy is not None and hierarchy[0][index][3] >= 0
        cv2.fillPoly(out, [np.round(scaled).astype(np.int32)], 0 if is_hole else 1)
    return out.astype(bool)


def render_preview(mask: np.ndarray | None, regions: list[Region],
                   shape: tuple[int, int], cfg: PreviewConfig,
                   mode: str, region_index: int = -1,
                   lit: bool = True,
                   native_size: tuple[int, int] | None = None) -> np.ndarray:
    """Build one projector-resolution BGR frame of the preview.

    Pure function of its arguments, so it is testable without a projector.
    """
    import cv2

    height, width = shape
    if native_size is not None and native_size != (width, height):
        # Render at panel resolution so edges come out straight rather than
        # stepped. Geometry is scaled, not resampled.
        scale_x = native_size[0] / width
        scale_y = native_size[1] / height
        scaled_regions = [
            Region(polygon=r.polygon * (scale_x, scale_y), label=r.label,
                   holes=[h * (scale_x, scale_y) for h in r.holes],
                   attributes=dict(r.attributes))
            for r in regions
        ]
        scaled_mask = upscale_smooth(mask, native_size) if mask is not None else None
        return render_preview(scaled_mask, scaled_regions,
                              (native_size[1], native_size[0]), cfg, mode,
                              region_index, lit)

    frame = np.zeros((height, width, 3), np.uint8)
    if not lit:
        return frame

    selected = regions
    if mode == "cycle" and regions:
        selected = [regions[region_index % len(regions)]]

    if mode == "mask":
        if mask is not None:
            frame[mask] = (255, 255, 255)
        return frame

    if mode == "fill":
        overlay = np.zeros_like(frame)
        if mask is not None:
            overlay[mask] = (60, 60, 60)
        for i, region in enumerate(selected):
            colour = _colour(i if mode != "cycle" else region_index)
            filled = region.rasterize(shape)
            overlay[filled] = colour
        frame = np.asarray(
            cv2.addWeighted(frame, 1.0 - cfg.fill_alpha, overlay, cfg.fill_alpha, 0),
            dtype=np.uint8,
        )

    # Outlines go on last so they stay crisp on top of any fill.
    if mask is not None and mode in ("outline", "cycle"):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # Bright and thick enough to read off a wall from across the room. A
        # hairline in dark grey is legible on a monitor and invisible once it
        # is a few metres away on brick.
        cv2.drawContours(frame, contours, -1, cfg.surface_outline_colour,
                         cfg.surface_outline_thickness)

    for i, region in enumerate(selected):
        colour = _colour(region_index if mode == "cycle" else i)
        for ring in region.rings:
            if len(ring) >= 3:
                cv2.polylines(frame, [np.round(ring).astype(np.int32)], True,
                              colour, cfg.outline_thickness)
        if cfg.show_labels:
            centre = np.round(region.centroid).astype(int)
            cv2.putText(frame, region.label, tuple(centre), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, colour, 2, cv2.LINE_AA)
    return frame


def _colour(index: int) -> tuple[int, int, int]:
    """Distinguishable colours that all read clearly when projected."""
    palette = [
        (0, 255, 0), (0, 200, 255), (255, 160, 0), (255, 0, 200),
        (255, 255, 0), (120, 120, 255), (0, 255, 160),
    ]
    return palette[index % len(palette)]


@dataclass
class PreviewState:
    """Interactive state, separated out so key handling can be unit tested."""

    mode: str
    region_index: int = 0
    blink_ms: int = 0
    running: bool = True
    region_count: int = 0

    def handle_key(self, key: int) -> None:
        if key in (27, ord("q")):
            self.running = False
        elif key == ord(" "):
            if self.region_count:
                self.region_index = (self.region_index + 1) % self.region_count
            if self.mode != "cycle":
                self.mode = "cycle"
        elif key == ord("a"):
            self.mode = "outline"
        elif key == ord("b"):
            self.blink_ms = 0 if self.blink_ms else 600
        elif key == ord("o"):
            order = ["mask", "outline", "fill", "cycle"]
            self.mode = order[(order.index(self.mode) + 1) % len(order)]

    def lit(self, now: float) -> bool:
        """Blink phase: True when the preview should be showing."""
        if not self.blink_ms:
            return True
        return int(now * 1000.0 / self.blink_ms) % 2 == 0


def run_preview(display: object, mask: np.ndarray | None, regions: list[Region],
                shape: tuple[int, int], cfg: PreviewConfig | None = None,
                max_seconds: float | None = None,
                clock: object = time.monotonic,
                native_size: tuple[int, int] | None = None) -> PreviewState:
    """Show the preview until the operator quits.

    ``max_seconds`` and ``clock`` exist so the loop itself can be tested
    headlessly; on site you just leave it running and look at the house.
    """
    cfg = cfg or PreviewConfig()
    if cfg.mode not in MODES:
        raise ValueError(f"preview mode must be one of {MODES}, got {cfg.mode!r}")

    state = PreviewState(mode=cfg.mode, blink_ms=cfg.blink_ms,
                         region_count=len(regions))
    started = clock()          # type: ignore[operator]
    last_advance = started
    shown: tuple | None = None

    while state.running:
        now = clock()          # type: ignore[operator]
        if max_seconds is not None and now - started >= max_seconds:
            break
        if (state.mode == "cycle" and regions and cfg.cycle_dwell_s > 0
                and now - last_advance >= cfg.cycle_dwell_s):
            state.region_index = (state.region_index + 1) % len(regions)
            last_advance = now

        # Everything the rendered frame depends on. Re-uploading an identical
        # image thirty times a second makes the projection flicker, so a
        # preview that is not changing is simply left on screen.
        wanted = (state.mode, state.region_index, state.lit(now))
        if wanted != shown:
            frame = render_preview(mask, regions, shape, cfg, state.mode,
                                   state.region_index, state.lit(now),
                                   native_size=native_size)
            key = display.show(frame, wait_ms=30)   # type: ignore[attr-defined]
            shown = wanted
        else:
            key = display.poll(30)                  # type: ignore[attr-defined]
        if key not in (-1, 255):
            state.handle_key(key)
            shown = None                            # force a redraw next pass
    return state
