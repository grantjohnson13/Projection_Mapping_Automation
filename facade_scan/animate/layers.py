"""Visual layers, composited additively in projector space.

Everything here is written for a projector rather than a screen. Three
consequences shape the whole design:

**Black is invisible.** There is no "dark grey" on a wall at night; there is lit
and unlit. So layers *add* light and are never composited over a background.
Anything that should recede is simply not drawn.

**Detail does not survive.** Between the projector's optics, the throw distance
and a matte surface, fine texture is gone. Large shapes and motion survive, so
the layers are built from broad washes and things that move.

**Saturation carries.** Pale colours wash out into "white-ish". The palettes
here stay deep and saturated, and brightness is carried by how much of the
frame is lit rather than by how pale it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import ICE, LayerConfig
from .audio import AudioAnalysis


def _resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    """Evenly space ``count`` points around a closed path by arc length."""
    if len(points) < 2:
        return np.repeat(points, count, axis=0) if len(points) else points
    closed = np.vstack([points, points[:1]])
    steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(steps)])
    total = distance[-1]
    if total < 1e-6:
        return np.repeat(closed[:1], count, axis=0)
    wanted = np.linspace(0.0, total, count, endpoint=False)
    return np.stack([np.interp(wanted, distance, closed[:, 0]),
                     np.interp(wanted, distance, closed[:, 1])], axis=-1)


def _load_sprite(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a PNG as (BGR in 0..1, alpha in 0..1), or None if unset.

    Raises rather than silently drawing the built-in figure: someone who set a
    path and got the drawn Santa would have no idea why.
    """
    if not path:
        return None

    import cv2

    file = Path(path).expanduser()
    if not file.parent.name and not file.is_absolute():
        # A bare filename means the bundled assets, so the default works from
        # any working directory and from an installed package.
        bundled = Path(__file__).resolve().parent.parent / "assets" / file.name
        if bundled.exists():
            file = bundled
    if not file.exists():
        raise FileNotFoundError(f"no Santa image at {file}")
    image = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not read {file} as an image")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(
            f"{file.name} has no alpha channel, so it would be composited as a "
            "rectangle. Use a PNG with transparency."
        )
    colour = image[:, :, :3].astype(np.float32) / 255.0
    alpha = image[:, :, 3].astype(np.float32) / 255.0
    return colour, alpha


def _scaled_sprite(sprite: tuple[np.ndarray, np.ndarray] | None,
                   height: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Resize a sprite to `height`, keeping its own aspect ratio."""
    if sprite is None:
        return None

    import cv2

    colour, alpha = sprite
    rows = max(1, round(height))
    cols = max(1, round(height * colour.shape[1] / colour.shape[0]))
    return (cv2.resize(colour, (cols, rows), interpolation=cv2.INTER_AREA),
            cv2.resize(alpha, (cols, rows), interpolation=cv2.INTER_AREA))


def roof_profile(mask: np.ndarray) -> np.ndarray:
    """Topmost lit row per column, or -1 where the column is empty."""
    lit = mask.any(axis=0)
    first = np.argmax(mask, axis=0)
    return np.where(lit, first, -1)


def find_zones(mask: np.ndarray, min_width_frac: float = 0.12) -> list[tuple[int, int]]:
    """Split the silhouette into architectural masses at its roof valleys.

    A facade is not one shape: this one is a central gable, a lower peak to
    the right, and a wing below left. Lighting all of it with a single
    gradient flattens it into a cookie cutter, so the masses are found and can
    be lit separately -- which is the difference between a video clipped to a
    shape and projection mapping.

    The valleys in the roofline are where one mass ends and the next begins,
    so the profile's local maxima (lowest roof points) are the dividers.
    """
    profile = roof_profile(mask)
    columns = np.nonzero(profile >= 0)[0]
    if columns.size < 8:
        return [(0, mask.shape[1])]
    left, right = int(columns[0]), int(columns[-1]) + 1
    inner = profile[left:right].astype(np.float32)

    # Smooth, so cardboard nicks and decode noise are not read as valleys.
    window = max(3, (right - left) // 25 | 1)
    kernel = np.ones(window, np.float32) / window
    smooth = np.convolve(inner, kernel, mode="same")

    span = right - left
    guard = max(2, int(span * min_width_frac))
    dividers: list[int] = []
    for i in range(guard, span - guard):
        lo, hi = max(i - guard, 0), min(i + guard + 1, span)
        # A valley: the roof is at its lowest here relative to its neighbours.
        if smooth[i] == smooth[lo:hi].max() and smooth[i] > smooth[lo:hi].min() + 1.0:
            if all(abs(i - d) >= guard for d in dividers):
                dividers.append(i)

    edges = [0] + sorted(dividers) + [span]
    return [(left + a, left + b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def verge_points(mask: np.ndarray, count: int, tolerance: float = 3.0) -> np.ndarray:
    """Points along the roof edges only -- eaves and verges, not the walls.

    Tracing the whole silhouette puts a lit chain down both vertical walls and
    straight across the ground, which reads as a neon sign in the shape of a
    house and makes the brightest thing in the frame a band along the floor.
    Real outline lighting follows the roof and stops, so only outline points
    sitting on their column's top edge are kept.
    """
    outline = mask_outline(mask, count * 3)
    if not len(outline):
        return outline
    profile = roof_profile(mask)
    columns = np.clip(outline[:, 0].astype(int), 0, mask.shape[1] - 1)
    tops = profile[columns]
    keep = (tops >= 0) & (np.abs(outline[:, 1] - tops) <= tolerance)
    kept = outline[keep]
    if len(kept) < 4:
        return outline[::3]
    # Re-space along the kept run so the lights sit evenly on the roof.
    order = np.argsort(kept[:, 0])
    return _resample_open(kept[order], count)


def _resample_open(points: np.ndarray, count: int) -> np.ndarray:
    """Evenly space `count` points along an open path by arc length."""
    if len(points) < 2:
        return points
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(steps)])
    total = distance[-1]
    if total < 1e-6:
        return np.repeat(points[:1], count, axis=0)
    wanted = np.linspace(0.0, total, count)
    return np.stack([np.interp(wanted, distance, points[:, 0]),
                     np.interp(wanted, distance, points[:, 1])], axis=-1)


def mask_outline(mask: np.ndarray, count: int = 160) -> np.ndarray:
    """Evenly spaced points around the mask's outer edge.

    This is the scan's most valuable product for decoration: the exact
    silhouette of the subject, which is the single most tedious thing to trace
    by hand and the line everyone hangs lights along.
    """
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), np.float32)
    biggest = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    return _resample_closed(biggest, count)


@dataclass
class Layers:
    """Composites the layers for one scan's mask, at panel resolution."""

    mask: np.ndarray
    audio: AudioAnalysis
    cfg: LayerConfig = field(default_factory=LayerConfig)
    seed: int = 7

    def __post_init__(self) -> None:
        height, width = self.mask.shape
        self.height, self.width = height, width
        rng = np.random.default_rng(self.seed)

        # Hold the animation clear of the silhouette's edge. Done here rather
        # than by the caller so every layer, and the outline the bulbs sit on,
        # agrees on where the subject stops.
        if self.cfg.edge_margin_px > 0 and self.mask.any():
            from ..transfer import inset_mask
            pulled = inset_mask(self.mask, self.cfg.edge_margin_px)
            if pulled.any():          # never erode the subject out of existence
                self.mask = pulled

        # Everything is composited inside the subject's bounding box, not
        # across the panel.
        #
        # The projector almost always covers far more than the thing being lit
        # -- on this rig the subject is 5% of the panel -- and every pixel
        # outside it is multiplied away by the stencil at the end. Rendering
        # the full frame spent 28 of every 38 ms computing light that was
        # discarded. The box is the whole frame when the subject fills it, so
        # nothing is lost in the case this is not helping.
        rows = np.nonzero(self.mask.any(axis=1))[0]
        cols = np.nonzero(self.mask.any(axis=0))[0]
        if rows.size and cols.size:
            top, bottom = int(rows[0]), int(rows[-1]) + 1
            left, right = int(cols[0]), int(cols[-1]) + 1
        else:
            top, bottom, left, right = 0, height, 0, width
        self.box = (top, bottom, left, right)
        self.box_h, self.box_w = bottom - top, right - left

        # Where light may land at all.
        self.stencil = self.mask[top:bottom, left:right].astype(np.float32)[..., None]

        # 1 at the bottom of the subject, 0 at the top, for the graded wash.
        span = max(self.box_h - 1, 1)
        vertical = np.arange(self.box_h, dtype=np.float32) / span
        self.vertical = np.repeat(vertical[:, None], self.box_w, axis=1)[..., None]

        # Where the light actually falls: full at the base, fading to ambient
        # well before the roofline, so most of the subject stays dark.
        #
        # Measured per column, against that column's own base and roofline,
        # not against the bounding box. Measuring it globally makes the light
        # a set of horizontal bands, and where those cross the zones' vertical
        # bands the result is visible rectangles on the facade -- which read
        # as a rendering fault. Per column it follows the building instead:
        # the light climbs the gable where the gable is tall and stops lower
        # on the wing, which is what light on a real facade does.
        rows_index = np.arange(self.box_h, dtype=np.float32)[:, None]
        sub = self.mask[top:bottom, left:right]
        any_column = sub.any(axis=0)
        first = np.argmax(sub, axis=0).astype(np.float32)
        last = (self.box_h - 1 - np.argmax(sub[::-1], axis=0)).astype(np.float32)
        # Empty columns get a harmless unit span rather than a divide by zero.
        first = np.where(any_column, first, 0.0)
        last = np.where(any_column, last, self.box_h - 1.0)
        # Smooth the profile across columns before shaping light to it. The
        # raw profile steps wherever the roofline does, and a step becomes a
        # hard vertical edge in the wash -- which on a facade reads as a
        # blend seam rather than as anything intentional.
        width = int(max(3, self.cfg.wash_profile_smooth * self.box_w)) | 1
        kernel = np.ones(width, np.float32) / width
        first = np.convolve(np.pad(first, width, mode="edge"), kernel,
                            mode="same")[width:-width].astype(np.float32)
        last = np.convolve(np.pad(last, width, mode="edge"), kernel,
                           mode="same")[width:-width].astype(np.float32)

        self.column_base = last[None, :]
        self.column_span = np.maximum(last - first, 1.0)[None, :]
        height_above_base = np.clip(
            (self.column_base - rows_index) / self.column_span, 0.0, 1.0)
        self.vertical_up = height_above_base[..., None]
        self.height_above_base = height_above_base

        # The window: a soft-edged rectangle, precomputed as a slice plus a
        # falloff so it costs almost nothing per frame.
        wx = self.cfg.window_x_frac * self.box_w
        wy = self.cfg.window_y_frac * self.box_h
        hw = max(2.0, self.cfg.window_w_frac * self.box_w / 2.0)
        hh = max(2.0, self.cfg.window_h_frac * self.box_h / 2.0)
        x0, x1 = int(max(wx - hw, 0)), int(min(wx + hw, self.box_w))
        y0, y1 = int(max(wy - hh, 0)), int(min(wy + hh, self.box_h))
        self.window_slice = (slice(y0, max(y1, y0 + 1)), slice(x0, max(x1, x0 + 1)))
        gy = np.linspace(-1.0, 1.0, max(y1 - y0, 1), dtype=np.float32)[:, None]
        gx = np.linspace(-1.0, 1.0, max(x1 - x0, 1), dtype=np.float32)[None, :]
        # A high exponent keeps the middle flat and the edge quick, so it is a
        # lit rectangle with a soft border rather than a fuzzy blob.
        edge = max(self.cfg.window_edge, 1.0)
        shape = np.clip(1.0 - np.maximum(np.abs(gx), np.abs(gy)) ** edge, 0.0, 1.0)
        if self.cfg.window_panes:
            bar = self.cfg.window_bar_frac * edge
            cross = ((np.abs(gx) < bar / edge) | (np.abs(gy) < bar / edge))
            shape = shape * np.where(cross, 0.25, 1.0)
        self.window_falloff = shape.astype(np.float32)[..., None]

        # Which architectural mass each column belongs to, so they can be lit
        # separately rather than as one flat shape.
        self.zone_spans = find_zones(self.mask)
        column_zone = np.zeros(self.box_w, np.int32)
        for index, (a, b) in enumerate(self.zone_spans):
            lo, hi = max(a - left, 0), min(b - left, self.box_w)
            if hi > lo:
                column_zone[lo:hi] = index
        self.column_zone = column_zone
        self.zone_blur = int(max(1, self.cfg.zone_feather * self.box_w)) | 1
        # A feathered version of "which mass am I", for blending colour across
        # the boundary. Unfeathered, a red mass against a green one meets on a
        # single column and draws a hard vertical line down the facade -- the
        # same seam that makes the gain look like a projector fault.
        # How high the roof is over each column, 0 at the lowest eave and 1 at
        # the apex. Used to blend two colours along the building's own pitch
        # rather than across an invented vertical line.
        roof = self.column_base[0] - self.column_span[0]
        eave, apex = float(roof.min()), float(roof.max())
        pitch = (roof - eave) / max(apex - eave, 1e-6)
        self.zone_side = self._feather(1.0 - pitch.astype(np.float32))

        # Outline in box coordinates, like everything else drawn here.
        self.outline = (verge_points(self.mask, self.cfg.bulb_count)
                        if self.cfg.bulb_verges_only
                        else mask_outline(self.mask, self.cfg.bulb_count))
        if len(self.outline):
            self.outline = self.outline - np.array([left, top], np.float32)
        self.bulb_phase = rng.uniform(0.0, 2.0 * np.pi, len(self.outline))
        self.bulb_rate = rng.uniform(*self.cfg.bulb_twinkle_hz, len(self.outline))

        # Size everything against the subject, so the look survives being
        # pointed at a cutout on a stand or at the side of a house.
        self.bulb_radius = self.cfg.bulb_radius_px
        self.snow_radius = self.cfg.snow_radius_px
        if self.cfg.auto_scale and len(self.outline) > 2:
            closed = np.vstack([self.outline, self.outline[:1]])
            perimeter = float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())
            spacing = perimeter / max(len(self.outline), 1)
            self.bulb_radius = max(self.cfg.bulb_radius_px,
                                   round(spacing * self.cfg.bulb_fill / 2))
            extent = self.outline.max(axis=0) - self.outline.min(axis=0)
            self.snow_radius = max(1, round(float(extent.min())
                                            * self.cfg.snow_size_frac))

        # Snow is seeded over a band taller than the frame so it is already
        # falling when the animation starts.
        count = self.cfg.snow_count
        self.snow_x = rng.uniform(0, self.box_w, count).astype(np.float32)
        self.snow_y0 = rng.uniform(-self.box_h, self.box_h, count).astype(np.float32)
        self.snow_speed = (self.cfg.snow_speed_px_s
                           * rng.uniform(*self.cfg.snow_speed_spread,
                                         count)).astype(np.float32)
        self.snow_phase = rng.uniform(0, 2 * np.pi, count).astype(np.float32)
        self.snow_size = rng.integers(1, max(2, self.snow_radius + 1), count)

        # Sprites are scaled once here, not per frame. Their display size is
        # fixed -- only where they are drawn changes -- and resizing the
        # reindeer's 1970x525 original every frame cost 13.5 ms against a
        # 2.7 ms baseline, a spike landing exactly during the crossing, which
        # is the most visible motion in the whole show.
        if self.cfg.santa_in_window:
            # Sized to the window he is framed by, not to the whole subject.
            santa_h = max(8.0, self.cfg.window_h_frac * self.box_h
                          * self.cfg.santa_window_fill)
        else:
            santa_h = max(8.0, self.box_h * self.cfg.santa_height_frac)
        # How loud each section is relative to the others, which becomes how
        # bright and busy it gets. Taken from the recording so the arc belongs
        # to the song rather than to a hand-written cue sheet.
        edges = list(self.audio.sections) + [self.audio.duration]
        levels = []
        for start, end in zip(edges[:-1], edges[1:]):
            lo = int(max(0.0, start - self.audio.frame_offset) * self.audio.frame_rate)
            hi = int(max(0.0, end - self.audio.frame_offset) * self.audio.frame_rate)
            chunk = self.audio.energy[lo:max(hi, lo + 1)]
            levels.append(float(chunk.mean()) if chunk.size else 0.5)
        weights = np.array(levels, np.float32)
        # Ranked, not min-max scaled. Raw loudness is dominated by whichever
        # section happens to be loudest: on this track min-max put five of
        # seven sections below 0.1, so most of the song rendered at the floor
        # and the show visibly sagged right before its peak. Ranking keeps
        # which section is louder than which, but spreads them evenly so every
        # section gets a distinct level worth looking at.
        if weights.size > 1:
            order = np.argsort(np.argsort(weights)).astype(np.float32)
            self.section_weight = order / float(weights.size - 1)
        else:
            self.section_weight = np.full_like(weights, 0.5)

        self.santa_sprite = _scaled_sprite(
            _load_sprite(self.cfg.santa_image), santa_h)

        flyer_h = max(6.0, self.box_h * self.cfg.flyer_height_frac)
        flyer = _scaled_sprite(_load_sprite(self.cfg.flyer_image), flyer_h)
        if flyer is not None:
            # The art is a black silhouette, and black is the one colour a
            # projector cannot make: composited it would be a sleigh-shaped
            # hole in the wash. The alpha becomes a stencil filled with light.
            colour, alpha = flyer
            tint = np.array(self.cfg.flyer_colour, np.float32) / 255.0
            flyer = (np.broadcast_to(tint, colour.shape).copy(), alpha)
        self.flyer_sprite = flyer

        # Icicles hang from the silhouette's own top edge, so they follow the
        # gables and eaves exactly. Geometry is fixed; only the shimmer moves.
        self.icicles: list[tuple[np.ndarray, float]] = []
        if self.cfg.icicle_count > 0:
            columns = np.nonzero(self.mask[:, :].any(axis=0))[0]
            if columns.size:
                picks = np.linspace(columns[0], columns[-1],
                                    self.cfg.icicle_count).astype(int)
                short, long_ = self.cfg.icicle_length_frac
                half = max(1.0, self.box_w * self.cfg.icicle_width_frac / 2.0)
                for column in picks:
                    rows = np.nonzero(self.mask[:, column])[0]
                    if not rows.size:
                        continue
                    x = float(column - left)
                    y = float(rows[0] - top)
                    length = float(rng.uniform(short, long_)) * self.box_h
                    self.icicles.append((
                        np.array([[x - half, y], [x + half, y],
                                  [x, y + length]], np.float32),
                        float(rng.uniform(0.0, 2.0 * np.pi)),
                    ))

        # The apex: the highest point of the silhouette, which on a house is
        # the peak of the main gable.
        lit_rows = np.nonzero(self.mask.any(axis=1))[0]
        if lit_rows.size:
            apex_row = int(lit_rows[0])
            apex_cols = np.nonzero(self.mask[apex_row])[0]
            self.apex = (float(apex_cols.mean() - left), float(apex_row - top))
        else:
            self.apex = (self.box_w / 2.0, 0.0)

        # Sparkles are scattered once in the sweep's own coordinates: `along`
        # runs 0..1 across the subject, `across` is the drift off the line.
        sparks = self.cfg.sparkle_count
        self.spark_along = rng.uniform(0.0, 1.0, sparks).astype(np.float32)
        self.spark_across = rng.normal(0.0, 0.32, sparks).astype(np.float32)
        self.spark_phase = rng.uniform(0, 2 * np.pi, sparks).astype(np.float32)
        self.spark_scale = rng.uniform(0.45, 1.0, sparks).astype(np.float32)
        self.spark_radius = max(
            2, round(min(self.box_w, self.box_h) * self.cfg.sparkle_size_frac))

    def intensity(self, time: float) -> float:
        """How bright and busy the show should be, 0 to 1.

        Layers scale themselves by this rather than each running flat out, so
        a quiet verse is genuinely quiet and the last chorus genuinely arrives.
        """
        cfg = self.cfg
        if not cfg.arc or self.audio.sections.size < 2:
            return 1.0
        index = min(self.section_index_now(time), self.section_weight.size - 1)
        level = cfg.arc_floor + (cfg.arc_ceiling - cfg.arc_floor) * float(
            self.section_weight[index])

        # Ease in, so a section swells rather than stepping.
        start = float(self.audio.sections[index])
        ease = self.audio.beat_period * max(cfg.arc_ease_beats, 1e-6)
        since = time - start
        if since < ease:
            previous = (cfg.arc_floor + (cfg.arc_ceiling - cfg.arc_floor)
                        * float(self.section_weight[max(index - 1, 0)]))
            blend = since / ease
            level = previous + (level - previous) * (blend * blend * (3 - 2 * blend))
            # Dip through darkness on the way, so the two colour states are
            # never both on screen and so the change reads as a cut.
            level *= 1.0 - cfg.arc_cut_dip * float(np.sin(np.pi * blend) ** 2)
        return float(np.clip(level, 0.0, 1.0))

    def section_index_now(self, time: float) -> int:
        return self.audio.section_index(time)

    def active(self, layer: str, time: float) -> bool:
        """Whether `layer` has entered yet, under the build schedule.

        A layer not in the order is always on; one that is enters at its own
        section and stays.
        """
        cfg = self.cfg
        if not cfg.build or layer not in cfg.build_order:
            return True
        sections = max(int(self.audio.sections.size), 1)
        position = cfg.build_order.index(layer)
        # Spread the order across however many sections the song turned out to
        # have, so the schedule works on a two-minute carol and a five-minute
        # one alike.
        if position < cfg.build_opening:
            return True
        enters = int(position * sections / max(len(cfg.build_order), 1))
        return self.section_index_now(time) >= max(enters, 1)

    def _feather(self, per_column: np.ndarray) -> np.ndarray:
        """Blur a per-column value smoothly across the mass boundaries.

        Blurred twice, which matters more than it sounds. A single box blur
        turns a step into a straight ramp, and the kinks where that ramp
        starts and stops are themselves discontinuities -- faint but dead
        straight, and a dead straight vertical line on a facade that has none
        reads as a projector out of registration. Two passes approximate a
        Gaussian, whose ends are smooth, and the edge stops being findable.
        """
        if self.zone_blur <= 1:
            return per_column.astype(np.float32)
        kernel = np.ones(self.zone_blur, np.float32) / self.zone_blur
        out = per_column.astype(np.float32)
        for _ in range(2):
            padded = np.pad(out, self.zone_blur, mode="edge")
            out = np.convolve(padded, kernel, mode="same")[
                self.zone_blur:-self.zone_blur].astype(np.float32)
        return out

    def falloff_at(self, time: float) -> np.ndarray:
        """The lit shape now, as ``(h, w, 1)``.

        Recomputed per frame because the reach climbs with the arc: the light
        creeps up the building as the song builds and floods the gable at the
        climax.
        """
        cfg = self.cfg
        level = self.intensity(time) if cfg.arc else 1.0
        # Mixed with how far through the song we are, so the climb is monotone.
        # Driven by loudness alone it oscillates: a loud verse floods the gable
        # early and the actual climax has nowhere left to go.
        progress = float(np.clip(time / max(self.audio.duration, 1e-6), 0.0, 1.0))
        weight = float(np.clip(cfg.arc_progress_weight, 0.0, 1.0))
        climb = level * (1.0 - weight) + progress * weight
        reach = cfg.wash_reach_low + (cfg.wash_reach_high - cfg.wash_reach_low) * climb
        height = self.height_above_base
        mode = (cfg.wash_modes[self.section_index_now(time) % len(cfg.wash_modes)]
                if cfg.wash_modes else "base")
        if cfg.statement and self.statement_gain(time) > 1.05:
            # The early statement is full width, and lit off the ridge so it
            # is modelled rather than a flat sheet -- the loudest moment
            # should be the most legible, not the least.
            reach = max(reach, cfg.wash_reach_high * 0.95)
            mode = "ridge"
        if cfg.finale and self.in_finale(time):
            # The climax is a different picture, not a brighter one: the whole
            # silhouette lit edge to edge, top-down off the ridge -- the one
            # composition the show has not used since its second section, and
            # the only moment the right peak and the wing are ever at full.
            mode = "ridge"
            reach = max(reach, cfg.wash_reach_high)

        if mode == "ridge":
            # Falling from the roofline instead of rising from the ground, so
            # the gable is the lit surface and the base goes dark.
            shape = np.clip(1.0 - (1.0 - height) / max(reach, 1e-6),
                            0.0, 1.0) ** cfg.wash_falloff
        elif mode == "band":
            # A lit band travelling up the pitch: an edge the eye can follow,
            # which a level change does not give it.
            beats = self.audio.beat_index(time) + self.audio.beat_phase(time)
            centre = (beats / max(cfg.wash_band_beats, 1e-6)) % 1.0
            width = max(cfg.wash_band_width, 1e-6)
            band = np.clip(1.0 - np.abs(height - centre) / width, 0.0, 1.0) ** 1.5
            base = np.clip(1.0 - height / max(reach, 1e-6), 0.0, 1.0) ** cfg.wash_falloff
            shape = np.maximum(band, base * cfg.wash_partial_floor)
        elif mode == "mass":
            # One mass lit, the rest dark: the building read as separate
            # planes rather than as one silhouette.
            shape = np.clip(1.0 - height / max(reach * 1.4, 1e-6),
                            0.0, 1.0) ** cfg.wash_falloff
            shape = shape * self.mass_gate(time)[:, :, 0]
        else:
            shape = np.clip(1.0 - height / max(reach, 1e-6),
                            0.0, 1.0) ** cfg.wash_falloff
        # Smoothstep the result for the same reason: where a clipped gradient
        # reaches zero it leaves a hard curvature edge along a contour.
        shape = shape * shape * (3.0 - 2.0 * shape)

        if cfg.roof_plane:
            # The roof read as its own surface. The boundary follows the pitch
            # because the height is measured against each column's own eave
            # and ridge, so it runs diagonally up the gable and steps down
            # over the wing without any of that being described anywhere.
            edge = (height - cfg.roof_plane_from) / max(cfg.roof_plane_soft, 1e-6)
            t = np.clip(edge, 0.0, 1.0)
            roof = t * t * (3.0 - 2.0 * t)
            shape = (shape * (1.0 + (cfg.roof_plane_lift - 1.0) * roof)
                     + roof * cfg.roof_plane_floor)

        return (cfg.wash_ambient + (1.0 - cfg.wash_ambient) * shape)[..., None]

    def mass_gate(self, time: float) -> np.ndarray:
        """A hard-ish gate selecting one architectural mass at a time."""
        count = len(self.zone_spans)
        if count < 2:
            return np.ones((1, 1, 1), np.float32)
        beats = self.audio.beat_index(time) + self.audio.beat_phase(time)
        lead = int(beats / max(self.cfg.zone_beats * 2.0, 1e-6)) % count
        gains = np.full(count, self.cfg.wash_partial_floor, np.float32)
        gains[lead] = 1.0
        per_column = gains[self.column_zone]
        # A much narrower blend than the wash uses: this boundary is meant to
        # be seen, and it sits on the valley, which is a line the building
        # already has.
        narrow = max(1, int(self.zone_blur * self.cfg.wash_mass_edge)) | 1
        kernel = np.ones(narrow, np.float32) / narrow
        padded = np.pad(per_column, narrow, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="same")[
            narrow:-narrow].astype(np.float32)

        # Released with height, so the division never runs the full height of
        # the facade as one straight line. A hard vertical edge crossing the
        # whole building has no counterpart in the architecture and reads as
        # two projectors out of registration -- the most amateur thing a
        # mapping show can look like. Fading it out towards the roof keeps the
        # separation where the light actually is and lets the masses rejoin
        # where the eye would see the seam.
        gate = smoothed[None, :, None]
        # Smoothstep, not a clipped ramp: a clip leaves a kink along a contour
        # of constant height, which is the horizontal half of the same
        # straight-edge artifact.
        raw = np.clip(self.height_above_base / 0.7, 0.0, 1.0)
        release = (raw * raw * (3.0 - 2.0 * raw))[..., None]
        return (gate + (1.0 - gate) * release).astype(np.float32)

    def zone_gain(self, time: float) -> np.ndarray:
        """Per-column brightness, so the masses answer each other on the beat.

        Shaped ``(1, box_w, 1)`` so it broadcasts straight onto a frame.
        """
        count = len(self.zone_spans)
        if not self.cfg.zones or count < 2 or not self.active("zones", time):
            return np.ones((1, 1, 1), np.float32)
        if self.cfg.finale and self.in_finale(time):
            # Edge to edge at the climax: the one moment the right peak and
            # the wing are lit at full.
            return np.ones((1, 1, 1), np.float32)
        beats = self.audio.beat_index(time) + self.audio.beat_phase(time)
        lead = int(beats / max(self.cfg.zone_beats, 1e-6)) % count
        gains = np.full(count, 1.0 - self.cfg.zone_depth, np.float32)
        gains[lead] = 1.0
        return self._feather(gains[self.column_zone])[None, :, None]

    def state_colours(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """The (primary, accent) pair for whichever state is running."""
        states = self.cfg.wash_states
        if self.cfg.finale and self.in_finale(time):
            # The last section is pinned to the hottest state rather than
            # taking its turn in the rotation: a climax that lands on the icy
            # one by accident is not a climax.
            index = 1 % len(states)
        else:
            index = self.section_index_now(time) % len(states)
        primary, accent = states[index]
        return (np.array(primary, np.float32), np.array(accent, np.float32))

    def statement_gain(self, time: float) -> float:
        """A short full-value flare early in the song, 1.0 the rest of the time."""
        cfg = self.cfg
        if not cfg.statement:
            return 1.0
        start = self.audio.duration * cfg.statement_at
        span = self.audio.beat_period * max(cfg.statement_beats, 1e-6)
        if not (start <= time < start + span):
            return 1.0
        # A trapezoid: up fast, HOLD, then away. The previous curve was a
        # half sine over the whole span, which is zero at both ends -- so the
        # statement spent most of its own duration fading and was already
        # gone by the time anyone sampled it. A statement has to sit still
        # long enough to be seen.
        phase = (time - start) / span
        rise, fall = 0.18, 0.72
        if phase < rise:
            shape = phase / rise
        elif phase < fall:
            shape = 1.0
        else:
            shape = max(0.0, (1.0 - phase) / max(1.0 - fall, 1e-6))
        shape = shape * shape * (3.0 - 2.0 * shape)     # smooth the corners
        return 1.0 + (cfg.statement_gain - 1.0) * float(shape)

    def in_finale(self, time: float) -> bool:
        """True through the song's last section."""
        if self.audio.sections.size < 2:
            return False
        return self.section_index_now(time) >= self.audio.sections.size - 1

    def blackout(self, time: float) -> float:
        """1.0 normally, falling to 0.0 for the held ending.

        Everything the facade is doing drops away and the window and the star
        are left burning. An ending needs a full stop; fading out on a pale
        wash is a trailing off.
        """
        cfg = self.cfg
        if not cfg.finale:
            return 1.0
        # Never let the ending eat the piece: on a short clip a fixed five
        # seconds would be most of it, and the whole thing would render as
        # blackout.
        hold = min(cfg.finale_hold_s, self.audio.duration * 0.08)
        start = self.audio.duration - hold
        if time < start:
            return 1.0
        cut = self.audio.beat_period * max(cfg.finale_cut_beats, 1e-6)
        return float(np.clip(1.0 - (time - start) / cut, 0.0, 1.0))

    # ---------------------------------------------------------------- wash --
    def _wash(self, canvas: np.ndarray, time: float) -> None:
        """Light shaped onto the subject, not poured over it.

        The wash is a gradient anchored at the base that falls away to near
        darkness well before the roofline. What that buys is everything the
        flat fill gave away: the lit area is a shape, the top of the building
        stays dark so the icicles and the star have ground to read against,
        and the eye has somewhere to go.
        """
        cfg = self.cfg
        colour, _accent = self.state_colours(time)

        loudness = self.audio.at(self.audio.energy, time) ** cfg.loudness_gamma
        level = cfg.wash_floor + (cfg.wash_ceiling - cfg.wash_floor) * loudness
        level += cfg.wash_beat_lift * (1.0 - self.audio.beat_phase(time)) ** 3
        level *= self.intensity(time) * self.statement_gain(time)

        shaped = self.falloff_at(time) * self.zone_gain(time)
        if (self.section_index_now(time) % len(cfg.wash_states)
                in cfg.wash_split_states and len(self.zone_spans) > 1):
            # One colour per mass rather than one across the building.
            _base, other = self.state_colours(time)
            # Hardened to nearly binary. A continuous blend from red to green
            # does not put red and green on the building -- it puts every
            # yellow between them on it, because that is what the midpoints
            # of an RGB lerp between complementaries are. Each part of the
            # facade takes one colour or the other, and only a narrow seam
            # mixes at all.
            raw = np.clip((self.zone_side - 0.5) / 0.14 * 0.5 + 0.5, 0.0, 1.0)
            side = (raw * raw * (3.0 - 2.0 * raw))[None, :, None]
            tint = ((colour / 255.0)[None, None, :] * (1.0 - side)
                    + (other / 255.0)[None, None, :] * side)
            # Dip through the crossover so the two meet in shadow, not in the
            # ochre their RGB midpoint would give.
            # A narrow band around the crossover, not a broad falloff. The
            # blend runs on the roof pitch, which sits near the middle over
            # most of the building, so a wide dip darkens the whole facade
            # instead of just the seam where the two hues mix.
            mix = 1.0 - cfg.wash_split_dip * np.exp(
                -((side - 0.5) / 0.1) ** 2)
            canvas += tint * (shaped * level * mix)
            return
        if cfg.wash_sweep > 0.0:
            # A defined edge travelling up the subject, so the wash moves
            # rather than merely changing level. A full-field brightness
            # change is the least legible motion there is at distance,
            # because nothing has an edge the eye can follow.
            beats = self.audio.beat_index(time) + self.audio.beat_phase(time)
            head = (beats / max(cfg.wash_sweep_beats, 1e-6)) % 1.0
            band = np.clip(1.0 - np.abs(self.vertical_up - head) * 5.0, 0.0, 1.0)
            shaped = shaped * (1.0 + cfg.wash_sweep * band ** 2)
        canvas += (colour / 255.0)[None, None, :] * (shaped * level)

        if cfg.heat:
            warmth = loudness ** max(cfg.heat_curve, 1e-6)
            if cfg.heat_follow_state:
                _base, glow = self.state_colours(time)
            else:
                glow = np.array(cfg.heat_colour, np.float32)
            canvas += ((glow / 255.0)[None, None, :]
                       * self.falloff_at(time)
                       * (cfg.heat_gain * warmth * self.intensity(time)))

    # -------------------------------------------------------------- window --
    def _window(self, canvas: np.ndarray, time: float) -> None:
        """A warm lit window, low on the main mass.

        Deliberately the one thing that holds still. Everything else changes
        colour, chases, falls or flies, and with no fixed point the eye has
        nothing to measure any of it against. A steady warm rectangle also
        does something no amount of wash can: it implies an inside.
        """
        cfg = self.cfg
        flicker = 1.0 + cfg.window_flicker * float(
            np.sin(time * 2.3) * 0.6 + np.sin(time * 5.7) * 0.4)
        pulse = 1.0 + cfg.window_pulse * (
            self.audio.at(self.audio.energy, time) ** cfg.loudness_gamma
            * (1.0 - self.audio.beat_phase(time)) ** 2)
        # Grows across the song, and flares once the facade has been cut away.
        progress = float(np.clip(time / max(self.audio.duration, 1e-6), 0.0, 1.0))
        grown = cfg.window_open + (cfg.window_close - cfg.window_open) * progress
        if self.blackout(time) < 1.0:
            grown *= cfg.window_finale_flare
        warm = np.array(cfg.window_colour, np.float32) / 255.0
        level = (cfg.window_brightness * flicker * pulse * grown
                 * (0.55 + 0.45 * self.intensity(time)))
        level = min(level, cfg.window_ceiling)
        canvas[self.window_slice] += (warm * level) * self.window_falloff

        if cfg.window_strike > 0.0:
            # Snapped on the beat and gone before the next. Struck toward
            # white so it is a change of value, not merely more amber.
            every = max(cfg.window_strike_beats, 1e-6)
            beats = self.audio.beat_index(time) + self.audio.beat_phase(time)
            since = beats % every
            hit = max(0.0, 1.0 - since / 0.55) ** 3
            if hit > 0.0:
                cold = np.array(cfg.window_strike_colour, np.float32) / 255.0
                strike = min(cfg.window_strike * hit * level,
                             cfg.window_ceiling - level * 0.35)
                if strike > 0.0:
                    canvas[self.window_slice] += (
                        cold * strike) * self.window_falloff

    # --------------------------------------------------------------- bulbs --
    def _bulbs(self, canvas: np.ndarray, time: float) -> None:
        import cv2

        cfg = self.cfg
        if len(self.outline) == 0:
            return
        count = len(self.outline)
        lap = max(cfg.bulb_lap_beats, 1e-6)
        # Chase position advances a lap every `lap` beats, so it stays locked
        # to the music however the tempo was estimated.
        head = (self.audio.beat_index(time) + self.audio.beat_phase(time)) / lap
        offset = (np.arange(count) / count - head) % 1.0
        comet = np.clip(1.0 - offset / max(cfg.bulb_comet_frac, 1e-6), 0.0, 1.0) ** 2

        twinkle = 1.0 + cfg.bulb_twinkle * np.sin(
            time * self.bulb_rate * 2.0 * np.pi + self.bulb_phase)
        # Struck on the beat and decaying across it, so the string carries the
        # rhythm rather than merely existing.
        flash = 1.0 + cfg.bulb_beat_flash * (1.0 - self.audio.beat_phase(time)) ** 3
        if cfg.bulb_strike > 0.0:
            # On the window's beat, not its own, so the two land together.
            every = max(cfg.window_strike_beats, 1e-6)
            beats_now = self.audio.beat_index(time) + self.audio.beat_phase(time)
            since = beats_now % every
            flash += cfg.bulb_strike * max(0.0, 1.0 - since / 0.55) ** 3
        brightness = np.clip((cfg.bulb_base + comet) * twinkle * flash, 0.0, 2.2)

        warm = np.array(cfg.bulb_colour, np.float32) / 255.0
        if cfg.bulb_follow_state:
            _primary, accent = self.state_colours(time)
            accent = accent / 255.0
        else:
            accent = np.array(cfg.bulb_accent_colour, np.float32) / 255.0
        layer = np.zeros_like(canvas)
        for index, (x, y) in enumerate(self.outline):
            colour = accent if (cfg.bulb_accent_every
                                and index % cfg.bulb_accent_every == 0) else warm
            value = colour * brightness[index]
            cv2.circle(layer, (round(x), round(y)),
                       self.bulb_radius, value.tolist(), -1, cv2.LINE_AA)
        canvas += layer

    # ---------------------------------------------------------------- snow --
    def _snow(self, canvas: np.ndarray, time: float) -> None:
        import cv2

        cfg = self.cfg
        y = (self.snow_y0 + self.snow_speed * time) % (self.box_h + 40) - 20
        x = (self.snow_x + cfg.snow_drift_px_s
             * np.sin(time * 0.4 + self.snow_phase)) % self.box_w
        value = (np.array(ICE, np.float32) / 255.0) * cfg.snow_brightness
        layer = np.zeros_like(canvas)
        for px, py, size in zip(x.astype(int), y.astype(int), self.snow_size):
            cv2.circle(layer, (int(px), int(py)), int(size), value.tolist(),
                       -1, cv2.LINE_AA)
        canvas += layer

    # ------------------------------------------------------------ sparkles --
    def _sparkles(self, canvas: np.ndarray, time: float) -> None:
        import cv2

        cfg = self.cfg
        if not len(self.spark_along):
            return
        # Position in the sweep cycle, measured in beats so it arrives with the
        # music however the tempo came out.
        beats = self.audio.beat_index(time) + self.audio.beat_phase(time)
        phase = (beats % max(cfg.sparkle_every_beats, 1e-6)) / max(
            cfg.sparkle_cross_beats, 1e-6)
        if phase > 1.0 + cfg.sparkle_tail:
            return                      # between sweeps: nothing on screen

        # Brightness falls off behind the head and is zero in front of it.
        behind = phase - self.spark_along
        tail = max(cfg.sparkle_tail, 1e-6)
        alive = (behind >= 0.0) & (behind <= tail)
        if not alive.any():
            return
        strength = (1.0 - behind / tail) ** 2
        twinkle = 0.55 + 0.45 * np.sin(time * 9.0 + self.spark_phase)
        strength = strength * twinkle * self.spark_scale * cfg.sparkle_brightness

        warm = np.array(cfg.sparkle_colour, np.float32) / 255.0
        cool = np.array(cfg.sparkle_accent_colour, np.float32) / 255.0
        # The sweep runs left to right and tilts upward, so it crosses the
        # gables rather than running along the wall below them.
        layer = np.zeros_like(canvas)
        for index in np.nonzero(alive)[0]:
            along = float(self.spark_along[index])
            x = along * self.box_w
            y = (0.5 + self.spark_across[index] * cfg.sparkle_spread * 0.5
                 - (along - 0.5) * 0.25) * self.box_h
            value = (cool if (cfg.sparkle_accent_every
                              and index % cfg.sparkle_accent_every == 0) else warm)
            value = value * float(strength[index])
            size = max(1, round(self.spark_radius * float(self.spark_scale[index])))
            centre = (round(x), round(y))
            if cfg.sparkle_star:
                # Four points: a dot plus a cross reads as a sparkle, where a
                # dot on its own reads as snow.
                cv2.line(layer, (centre[0] - size * 2, centre[1]),
                         (centre[0] + size * 2, centre[1]), value.tolist(), 1,
                         cv2.LINE_AA)
                cv2.line(layer, (centre[0], centre[1] - size * 2),
                         (centre[0], centre[1] + size * 2), value.tolist(), 1,
                         cv2.LINE_AA)
            cv2.circle(layer, centre, size, value.tolist(), -1, cv2.LINE_AA)
        canvas += layer

    # --------------------------------------------------------------- santa --
    def _santa_progress(self, time: float) -> float:
        """How far up he is, 0 hidden to 1 fully up."""
        cfg = self.cfg
        beats = self.audio.beat_index(time) + self.audio.beat_phase(time)
        phase = beats % max(cfg.santa_every_beats, 1e-6)
        rise, hold, duck = (max(cfg.santa_rise_beats, 1e-6), cfg.santa_hold_beats,
                            max(cfg.santa_duck_beats, 1e-6))
        if phase < rise:
            eased = phase / rise
        elif phase < rise + hold:
            return 1.0
        elif phase < rise + hold + duck:
            eased = 1.0 - (phase - rise - hold) / duck
        else:
            return 0.0
        # Smoothstep, so he does not start and stop dead.
        return float(eased * eased * (3.0 - 2.0 * eased))

    def _santa(self, canvas: np.ndarray, time: float) -> None:
        import cv2

        cfg = self.cfg
        progress = self._santa_progress(time)
        if progress <= 0.0:
            return

        if self.santa_sprite is not None:
            colour, sprite_alpha = self.santa_sprite
            height, width = float(sprite_alpha.shape[0]), float(sprite_alpha.shape[1])
        else:
            height = max(8.0, self.box_h * cfg.santa_height_frac)
            width = height * 0.8
        sway = np.sin(time * 1.6) * cfg.santa_sway * width * progress
        if cfg.santa_in_window:
            # Framed by the window: he rises past the sill and is clipped to
            # the opening, so what you see is a face at a window.
            rows, cols = self.window_slice
            left = (cols.start + cols.stop) / 2.0 - width / 2.0 + sway
            top = rows.stop - progress * height * cfg.santa_window_rise
            clip = (rows.start, rows.stop, cols.start, cols.stop)
        else:
            left = self.box_w * cfg.santa_x_frac - width / 2.0 + sway
            # Fully up he clears the bottom edge by his own height; hidden, he
            # sits entirely below it.
            top = self.box_h - progress * height
            clip = None

        def point(fx: float, fy: float) -> tuple[int, int]:
            return (round(left + fx * width), round(top + fy * height))

        def scaled(f: float) -> int:
            return max(1, round(f * height))

        if self.santa_sprite is not None:
            self._blit(canvas, colour, sprite_alpha, round(left), round(top),
                       clip=clip)
            return

        hat = np.array(cfg.santa_hat, np.float32)
        trim = np.array(cfg.santa_trim, np.float32)
        skin = np.array(cfg.santa_skin, np.float32)
        eyes = np.array(cfg.santa_eyes, np.float32)

        # Drawn opaque into his own layer, then composited over the wash, so he
        # occludes it rather than glowing through it.
        figure = np.zeros_like(canvas)
        alpha = np.zeros(canvas.shape[:2], np.float32)

        def draw(fn, colour) -> None:
            fn(figure, (colour / 255.0).tolist())
            fn(alpha, 1.0)

        draw(lambda img, c: cv2.fillPoly(
            img, [np.array([point(0.5, 0.02), point(0.12, 0.44),
                            point(0.88, 0.44)], np.int32)], c), hat)
        draw(lambda img, c: cv2.ellipse(
            img, point(0.5, 0.45), (scaled(0.34), scaled(0.07)), 0, 0, 360, c, -1), trim)
        draw(lambda img, c: cv2.circle(
            img, point(0.5, 0.06), scaled(0.08), c, -1), trim)
        draw(lambda img, c: cv2.circle(
            img, point(0.5, 0.64), scaled(0.23), c, -1), skin)
        draw(lambda img, c: cv2.ellipse(
            img, point(0.5, 0.84), (scaled(0.30), scaled(0.22)), 0, 0, 360, c, -1), trim)
        draw(lambda img, c: cv2.ellipse(
            img, point(0.5, 0.74), (scaled(0.22), scaled(0.07)), 0, 0, 360, c, -1), trim)
        draw(lambda img, c: cv2.circle(
            img, point(0.39, 0.60), scaled(0.035), c, -1), eyes)
        draw(lambda img, c: cv2.circle(
            img, point(0.61, 0.60), scaled(0.035), c, -1), eyes)

        mask = alpha[..., None]
        np.multiply(canvas, 1.0 - mask, out=canvas)
        canvas += figure * mask

    @staticmethod
    def _blit(canvas: np.ndarray, colour: np.ndarray, alpha: np.ndarray,
              x: int, y: int, add: bool = False,
              clip: tuple[int, int, int, int] | None = None) -> None:
        """Draw an already-scaled sprite at (x, y), clipped to the canvas.

        Clipping matters here rather than being defensive: Santa spends most of
        his entrance partly below the bottom edge, which is the whole point of
        him, and the flyer enters and leaves off both sides, so the off-canvas
        case is the normal one.
        """
        height, width = alpha.shape
        limit_top, limit_bottom, limit_left, limit_right = (
            clip if clip is not None else (0, canvas.shape[0], 0, canvas.shape[1]))
        left, top = max(x, limit_left), max(y, limit_top)
        right = min(x + width, canvas.shape[1], limit_right)
        bottom = min(y + height, canvas.shape[0], limit_bottom)
        if right <= left or bottom <= top:
            return

        patch_c = colour[top - y:bottom - y, left - x:right - x]
        patch_a = alpha[top - y:bottom - y, left - x:right - x][..., None]

        region = canvas[top:bottom, left:right]
        if not add:
            np.multiply(region, 1.0 - patch_a, out=region)
        region += patch_c * patch_a

    # --------------------------------------------------------------- flyer --
    def _flyer(self, canvas: np.ndarray, time: float) -> None:
        cfg = self.cfg
        if self.flyer_sprite is None:
            return
        beats = self.audio.beat_index(time) + self.audio.beat_phase(time)
        phase = ((beats + cfg.flyer_phase_beats) % max(cfg.flyer_every_beats, 1e-6)
                 ) / max(cfg.flyer_cross_beats, 1e-6)
        if phase > 1.0:
            return                                  # between crossings

        lit, alpha = self.flyer_sprite
        width = alpha.shape[1]

        travel = phase if not cfg.flyer_reverse else 1.0 - phase
        x = -width + travel * (self.box_w + 2.0 * width)
        # An arc, so he rises into the middle of the crossing and dips away.
        y = (self.box_h * cfg.flyer_lane_frac
             - np.sin(np.pi * phase) * self.box_h * cfg.flyer_arc_frac)

        self._blit(canvas, lit, alpha, round(x), round(y), add=True)

    # ------------------------------------------------------------- icicles --
    def _icicles(self, canvas: np.ndarray, time: float) -> None:
        import cv2

        cfg = self.cfg
        if not self.icicles:
            return
        base = np.array(cfg.icicle_colour, np.float32) / 255.0
        # A highlight travelling along the roofline, so they glint in sequence.
        beats = self.audio.beat_index(time) + self.audio.beat_phase(time)
        head = (beats / max(cfg.icicle_glint_beats, 1e-6)) % 1.0
        layer = np.zeros_like(canvas)
        for index, (points, phase) in enumerate(self.icicles):
            shimmer = 1.0 + cfg.icicle_shimmer * np.sin(time * 1.3 + phase)
            along = index / max(len(self.icicles), 1)
            gap = abs(along - head)
            near = 1.0 - min(gap, 1.0 - gap) * cfg.icicle_glint_focus
            glint = 1.0 + cfg.icicle_glint * max(near, 0.0) ** 2
            # The strike: a hard bright front crossing the roofline, snapping
            # on the beat and gone before the next one.
            strike_head = (beats / max(cfg.icicle_strike_beats, 1e-6)) % 1.0
            sgap = abs(along - strike_head)
            close = 1.0 - min(sgap, 1.0 - sgap) * cfg.icicle_strike_focus
            glint += cfg.icicle_strike * max(close, 0.0) ** 4
            burn = cfg.finale_flare if (cfg.finale and self.in_finale(time)) else 1.0
            value = base * (cfg.icicle_brightness * shimmer * glint * burn)
            cv2.fillPoly(layer, [np.round(points).astype(np.int32)],
                         value.tolist(), cv2.LINE_AA)
        canvas += layer

    # ---------------------------------------------------------------- star --
    def _star(self, canvas: np.ndarray, time: float) -> None:
        import cv2

        cfg = self.cfg
        ignite = 1.0
        if cfg.star_from_frac > 0.0:
            start = self.audio.duration * cfg.star_from_frac
            if time < start:
                return
            span = self.audio.beat_period * max(cfg.star_ignite_beats, 1e-6)
            ignite = float(np.clip((time - start) / span, 0.0, 1.0))
        beats = self.audio.beat_index(time) + self.audio.beat_phase(time)
        pulse = 0.72 + 0.28 * np.sin(
            2.0 * np.pi * beats / max(cfg.star_twinkle_beats, 1e-6))
        radius = max(2.0, min(self.box_w, self.box_h) * cfg.star_size_frac / 2.0)
        flare_now = cfg.finale_flare if (cfg.finale and self.in_finale(time)) else 1.0
        reach = radius * cfg.star_ray_scale * pulse * (flare_now ** 0.6)
        value = (np.array(cfg.star_colour, np.float32) / 255.0
                 * cfg.star_brightness * pulse * ignite * flare_now)

        x, y = self.apex
        # Dropped below the apex so the whole star lands on the subject. At
        # the apex itself the upward rays fall outside the silhouette and are
        # clipped, which leaves a downward spike rather than a star.
        y += reach * cfg.star_drop
        layer = np.zeros_like(canvas)
        # Rays drawn as tapering triangles rather than lines: a one-pixel line
        # is gone at throw distance, and alternating long and short points is
        # what separates a star from a cross.
        points = max(int(cfg.star_points), 4)
        for index in range(points):
            # Start at the top and go round, so ray 0 points up and the
            # downward one can be the long tail.
            angle = -np.pi / 2.0 + 2.0 * np.pi * index / points
            fraction = index / points
            if abs(fraction - 0.5) < 1e-6:
                span = reach * cfg.star_tail          # the tail
            elif index % 2 == 0:
                span = reach
            else:
                span = reach * 0.4
            tip = (x + np.cos(angle) * span, y + np.sin(angle) * span)
            side = angle + np.pi / 2.0
            half = radius * cfg.star_ray_width
            cv2.fillPoly(layer, [np.round(np.array([
                [x + np.cos(side) * half, y + np.sin(side) * half],
                [x - np.cos(side) * half, y - np.sin(side) * half],
                [tip[0], tip[1]],
            ])).astype(np.int32)], value.tolist(), cv2.LINE_AA)
        cv2.circle(layer, (round(x), round(y)), max(1, round(radius * 0.72)),
                   (value * 1.5).tolist(), -1, cv2.LINE_AA)
        canvas += layer

    # ------------------------------------------------------------- accents --
    def _accents(self, canvas: np.ndarray, time: float) -> None:
        cfg = self.cfg
        strength = self.audio.at(self.audio.onset, time)
        if strength < cfg.accent_threshold:
            return
        # How far into this beat we are decides how much of the flash is left.
        remaining = max(0.0, 1.0 - self.audio.beat_phase(time)
                        * self.audio.beat_period / max(cfg.accent_decay_s, 1e-6))
        if remaining <= 0.0:
            return
        amount = cfg.accent_gain * remaining * (strength - cfg.accent_threshold) / (
            1.0 - cfg.accent_threshold + 1e-9)
        canvas += (np.array(cfg.accent_colour, np.float32) / 255.0)[None, None, :] * amount

    # --------------------------------------------------------------- frame --
    def frame(self, time: float) -> np.ndarray:
        """Render one composited, stencilled frame as BGR uint8."""
        canvas = np.zeros((self.box_h, self.box_w, 3), np.float32)
        held = self.blackout(time)
        on = self.active
        if self.cfg.wash:
            self._wash(canvas, time)
        if self.cfg.snow and on("snow", time):
            self._snow(canvas, time)
        if self.cfg.bulbs and on("bulbs", time):
            self._bulbs(canvas, time)
        if self.cfg.icicles and on("icicles", time):
            self._icicles(canvas, time)
        if self.cfg.star and on("star", time):
            self._star(canvas, time)
        if self.cfg.sparkles and on("sparkles", time):
            self._sparkles(canvas, time)
        if self.cfg.flyer and on("flyer", time):
            self._flyer(canvas, time)
        if self.cfg.accents:
            self._accents(canvas, time)
        # Santa last: he stands in front of everything else.
        if self.cfg.santa and on("santa", time):
            self._santa(canvas, time)

        # The ending: the facade drops away and these two are left burning.
        # Applied here so everything above it is what gets cut, and the window
        # and star are drawn afterwards at full.
        if held < 1.0:
            # Not to nothing: the silhouette is the thing the whole show has
            # been mapped onto, and deleting it on the final beat reads as the
            # projector failing rather than as an ending.
            canvas *= held + (1.0 - held) * self.cfg.finale_wall_rest
        # Drawn after the cut, so these are what is left burning. The roof
        # lights come back faintly with them so the house is still there.
        outline_held = held
        if held < 1.0 and self.cfg.finale_outline_s > 0.0:
            # The outline outlives the facade, then goes too.
            gone = self.audio.duration - (self.cfg.finale_hold_s
                                          - self.cfg.finale_outline_s)
            cut = self.audio.beat_period * max(self.cfg.finale_cut_beats, 1e-6)
            fade = float(np.clip((gone - time) / cut, 0.0, 1.0))
            rest = self.cfg.finale_outline_rest
            outline_held = rest + (1.0 - rest) * fade
        if held < 1.0 and self.cfg.bulbs and self.cfg.finale_outline > 0.0:
            trace = np.zeros_like(canvas)
            self._bulbs(trace, time)
            # Recoloured cool: held dim, the warm accent reads as dirty brown
            # rather than as embers.
            cool = np.array(self.cfg.finale_outline_colour, np.float32) / 255.0
            strength = trace.max(axis=2, keepdims=True)
            canvas += (strength * cool[None, None, :]
                       * self.cfg.finale_outline * (1.0 - held) * outline_held)
        if self.cfg.window:
            self._window(canvas, time)
        if self.cfg.star and held < 1.0:
            self._star(canvas, time)
        # Light only ever lands on the subject.
        canvas *= self.stencil
        canvas *= self.cfg.master_gain

        top, bottom, left, right = self.box
        frame = np.zeros((self.height, self.width, 3), np.uint8)
        frame[top:bottom, left:right] = (np.clip(canvas, 0.0, 1.0)
                                         * 255.0).astype(np.uint8)
        return frame
