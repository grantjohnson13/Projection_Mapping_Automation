"""Central configuration for facade-scan.

Every tunable threshold in the codebase lives here. Nothing that a user might
reasonably want to change is hardcoded inside a function body; functions take a
config object (or the specific sub-config they need) and read values from it.

Load from TOML::

    cfg = Config.from_toml("myscan.toml")

The TOML layout mirrors the dataclass layout, one table per sub-config::

    [projector]
    width = 1920
    height = 1080

    [decode]
    confidence_threshold = 0.06

Any table or key that is omitted keeps its documented default.
"""

from __future__ import annotations

import dataclasses
import re
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Scale helpers
# --------------------------------------------------------------------------- #
def projector_px_per_camera_px(camera_shape: tuple[int, ...],
                               projector_size: tuple[int, int]) -> float:
    """How far the decoded coordinate moves between adjacent camera pixels.

    On a flat wall this is set purely by how much the camera out-resolves the
    projector, and it is the natural unit for every threshold that asks "is this
    change between neighbouring pixels a gradient or a depth step?".
    """
    height, width = int(camera_shape[0]), int(camera_shape[1])
    return max(projector_size[0] / max(width, 1), projector_size[1] / max(height, 1))


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class ProjectorConfig:
    """The projector grid that patterns are generated for.

    Normally this is the projector's native panel resolution, and patterns must
    reach the panel with no OS scaling, or the finest stripes get resampled
    away and the low-order bits decode as noise.

    Coarser patterns on a fine panel
    --------------------------------
    Decoding needs the camera to out-resolve the projector over the target --
    roughly 1.5 camera pixels per projector pixel. A 1080p camera pointed at a
    1080p projector gives about 1.0, and the finest Gray planes will not decode.

    The fix is not to scale the display, which resamples and blurs. Instead set
    ``width``/``height`` to a coarser grid (say 960x540) and set
    ``display.native_width``/``native_height`` to the real panel. Patterns are
    then generated coarse and blown up by an integer factor with
    nearest-neighbour, so every logical pixel is a hard-edged block of native
    pixels: no resampling, no blur, and half the spatial frequency to resolve.

    Everything downstream works in these logical projector pixels, so the
    exported mask comes out at this size and is upscaled the same way when
    projected back.
    """

    width: int = 1920
    height: int = 1080


@dataclass
class PatternConfig:
    """Gray-code pattern generation."""

    #: Value written for a "lit" stripe. Lower this if the projector clips
    #: highlights or the camera saturates on white walls.
    white_level: int = 255
    #: Value written for an "unlit" stripe. Raise slightly (e.g. 8) if your
    #: projector has poor black level and banding confuses the decoder.
    black_level: int = 0
    #: Filename template for written frames. Must sort in capture order.
    filename_template: str = "{index:04d}_{label}.png"


@dataclass
class DisplayConfig:
    """How pattern frames get onto the projector's output."""

    #: Select the display by name instead of by index, e.g. "OTM" (Optoma) or
    #: "Epson". Matched case-insensitively as a substring against the EDID
    #: vendor code. Preferred over an index, because macOS renumbers and
    #: rearranges displays when any of them sleeps or is unplugged.
    display_name: str | None = None
    #: Which physical display to go fullscreen on. 0 is normally the laptop
    #: panel, so the projector is usually 1.
    display_index: int = 1
    #: Explicit desktop x of the top-left corner of that display. Leave as None
    #: to auto-detect (macOS ``system_profiler`` / Linux ``xrandr``), falling
    #: back to ``display_index * projector.width``. Set both of these if the
    #: patterns come up on the wrong screen -- detection is best-effort, and on
    #: macOS in particular the desktop origin is not reported at all.
    origin_x: int | None = None
    #: Explicit desktop y of that display's top-left corner. See ``origin_x``.
    origin_y: int | None = None
    #: The projector's real panel width, when it differs from the pattern grid
    #: in ``projector``. Frames are blown up to this with nearest-neighbour
    #: before display. Leave as None to send patterns at their own size.
    native_width: int | None = None
    #: The panel height to match ``native_width``.
    native_height: int | None = None
    window_name: str = "facade-scan"


@dataclass
class CaptureConfig:
    """Capture backend selection and camera control."""

    #: One of "webcam", "gphoto2", "folder".
    backend: str = "folder"

    #: Milliseconds to wait between putting a frame on the projector and
    #: triggering the camera. Covers projector panel refresh, any frame
    #: interpolation the projector is doing, and camera exposure. 400ms is
    #: conservative; do not drop below ~150ms without testing.
    settle_ms: int = 400

    #: Frames to pull and discard before keeping one. OpenCV's VideoCapture
    #: buffers frames, so a naive read() returns an image from several frames
    #: ago -- i.e. the *previous* pattern. This silently ruins scans.
    flush_frames: int = 5

    # --- webcam backend -----------------------------------------------------
    #: Select the camera by name instead of by index, e.g. "USB Webcam".
    #: Matched case-insensitively as a substring. Strongly preferred over an
    #: index: plugging in a phone, or Continuity Camera waking up, renumbers
    #: every device and you will silently scan with the wrong camera.
    webcam_name: str | None = None
    webcam_index: int = 0
    webcam_width: int | None = None
    webcam_height: int | None = None
    #: Manual exposure. OpenCV's units are backend-dependent; on most UVC
    #: cameras this is log2(seconds), so -6 is 1/64s.
    webcam_exposure: float | None = -6.0
    #: Manual sensor gain. Leave as None on cameras that do not expose it;
    #: prefer a longer exposure to gain, since gain amplifies the noise that
    #: eats into per-bit confidence.
    webcam_gain: float | None = None
    #: Manual white balance in Kelvin (UVC cameras).
    webcam_wb_temperature: float | None = 4600.0
    #: Manual focus, 0 = infinity on most UVC cameras.
    webcam_focus: float | None = 0.0
    #: Set auto-exposure/AWB/AF off. Leave True; auto anything will track the
    #: patterns and destroy the pattern-vs-inverse comparison.
    webcam_disable_auto: bool = True

    # --- gphoto2 backend ----------------------------------------------------
    gphoto2_binary: str = "gphoto2"
    #: Extra CLI args, e.g. ["--set-config", "iso=400"].
    gphoto2_extra_args: list[str] = field(default_factory=list)
    gphoto2_timeout_s: float = 30.0
    #: Delete the image from the camera card after downloading.
    gphoto2_keep_on_camera: bool = False

    # --- folder backend -----------------------------------------------------
    #: Directory of manually-shot photographs, matched to the manifest by
    #: sorted filename order.
    folder_path: str | None = None
    folder_extensions: list[str] = field(
        default_factory=lambda: [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    )


@dataclass
class DecodeConfig:
    """Gray-code decoding thresholds."""

    #: Per-bit confidence floor, as |pattern - inverse| in normalised [0,1]
    #: intensity. A pixel is decoded only if *every* bit clears this, because a
    #: single wrong bit yields a wrong coordinate.
    #:
    #: Lowering it buys coverage on dark brick, and the cost is not a gentle
    #: loss of precision -- it is a scattering of rare, *gross* errors, pixels
    #: whose coordinate is wrong by a hundred pixels rather than one. Those
    #: throw light onto an unrelated part of the house. Raise it if windows or
    #: wet surfaces produce garbage.
    confidence_threshold: float = 0.06

    #: Window size for the depth-discontinuity-preserving outlier filter.
    #: Set to 0 to skip filtering entirely, which is worth doing once when
    #: diagnosing a bad scan so you can see the raw decode.
    median_ksize: int = 5
    #: Neighbours whose decoded coordinate differs from the centre by more than
    #: this many projector pixels are excluded from the median, so the filter
    #: cannot smear across a depth step (garage bump-out, window reveal).
    #:
    #: The natural unit here is not pixels but *steps*: how much the decoded
    #: coordinate changes between adjacent camera pixels looking at the same
    #: flat wall, which is set by how much the camera out-resolves the
    #: projector. A DSLR at 6000 px against a 1920 px projector moves a third of
    #: a projector pixel per camera pixel; a 1280 px webcam moves one and a
    #: half. A fixed pixel threshold means opposite things in those two cases,
    #: so the resolved value is ``max(median_max_jump_px, steps * step)``.
    median_max_jump_steps: float = 8.0
    median_max_jump_px: float = 4.0

    #: white - black, normalised. Below this the pixel is not lit by the
    #: projector at all (sky, shadow, beyond throw).
    illumination_threshold: float = 0.08

    #: A pixel whose mean per-bit confidence is below this is a candidate for
    #: `likely_glass`. See decode.py for why this correlates with windows.
    glass_confidence_threshold: float = 0.15
    #: ...but only if the *white* frame is at least this bright there. Note this
    #: is absolute brightness, not white-minus-black: glass is bright (it throws
    #: back glare and streetlights) while carrying almost no pattern
    #: modulation, and that combination is the whole signature.
    glass_min_brightness: float = 0.12
    #: ...and only if the *black* frame is at least this bright there, which is
    #: what separates glazing from shadow.
    #:
    #: Both return no pattern modulation, so low confidence alone cannot tell
    #: them apart, and a dimly-lit shadow was being labelled a window. The
    #: difference is what they do when nothing is projected at all: glass still
    #: throws back glare, the projector's own body and whatever the room is
    #: doing, while a shadow is a shadow and goes dark. So the black frame is
    #: the discriminator.
    glass_min_black: float = 0.08

    #: How many same-surface neighbours a pixel needs before it is treated as
    #: sitting on a surface at all. A pixel with fewer disagrees with its whole
    #: neighbourhood and is treated as a gross decode error.
    median_min_support: int = 4
    #: A supported pixel is only overwritten if it differs from its local median
    #: by more than this. Keeping it above the sub-pixel ramp bias is what stops
    #: the filter shifting pixels along depth steps and the frame border.
    median_replace_px: float = 2.0

    #: How far, in camera pixels, an undecoded gap may be filled from the
    #: decoded pixels around it. 0 disables inpainting entirely.
    #:
    #: This does *not* invent correspondence across a depth step: a gap is only
    #: filled where the decoded pixels bordering it agree with each other to
    #: within `inpaint_agreement_px`, which is exactly the test for "these are
    #: all on one smooth surface". A gap straddling the edge of a bump-out has
    #: disagreeing neighbours and is left alone.
    #:
    #: Worth having because the alternative is worse: an undecoded pixel in the
    #: middle of a wall becomes a hole in the projected mask, and a hole in the
    #: mask is a patch of house left dark.
    inpaint_radius_px: int = 6
    #: Spread the bordering decoded pixels must agree within, in projector
    #: pixels, before a gap is filled from them.
    #:
    #: Has to sit above the honest variation across a 3x3 window on a smooth
    #: surface -- twice the per-camera-pixel step, plus the bias the fill itself
    #: accumulates ring by ring -- and below the disparity of a real depth step,
    #: which is several pixels at minimum. 4 px separates them comfortably on
    #: every rig measured so far.
    inpaint_agreement_px: float = 4.0

    #: Rows to process at a time in the median filter, to bound peak memory.
    median_row_chunk: int = 128

    def resolve(self, camera_shape: tuple[int, ...],
                projector_size: tuple[int, int]) -> DecodeConfig:
        """Convert step-relative thresholds into projector pixels."""
        step = projector_px_per_camera_px(camera_shape, projector_size)
        return dataclasses.replace(
            self,
            median_max_jump_px=max(self.median_max_jump_px,
                                   self.median_max_jump_steps * step),
        )


@dataclass
class DetectConfig:
    """Line and region detection on the all-white capture.

    Size-relative thresholds
    ------------------------
    A phone camera and a 45 MP DSLR photograph the same house at wildly
    different pixel scales, and so does the same camera framed tight or wide.
    A threshold like "ignore segments under 40 pixels" therefore means something
    completely different from one scan to the next: on a small frame it throws
    away every window, and on a large one it keeps every brick course.

    So the length, gap and area thresholds are expressed as a fraction of the
    image diagonal (or, for areas, of the frame), with an absolute floor for
    sanity on very small images. :meth:`resolve` turns them into pixel values
    for a given capture, and that resolved config is what the detector actually
    reads. Set a fraction to 0 to pin a threshold to its absolute value instead.
    """

    # --- FastLineDetector ---------------------------------------------------
    fld_length_threshold: int = 20
    fld_distance_threshold: float = 1.414
    fld_canny_th1: float = 50.0
    fld_canny_th2: float = 150.0
    fld_canny_aperture_size: int = 3
    fld_do_merge: bool = False

    # --- minimum segment length ---------------------------------------------
    #: Segments shorter than this are dropped. This is the main knob that
    #: removes siding, shingle and brick-course texture, all of which scales
    #: with image resolution exactly as architectural edges do.
    min_segment_length_frac: float = 0.02
    #: Absolute floor, in pixels.
    min_segment_length_px: float = 8.0

    # --- vanishing points ---------------------------------------------------
    vp_ransac_iterations: int = 2000
    #: A segment votes for a vanishing point if the direction from its midpoint
    #: to the VP is within this angle of the segment's own direction.
    vp_inlier_angle_deg: float = 2.0
    vp_min_inliers: int = 6
    #: A typical house shows 2-3 dominant directions (vertical, and one or two
    #: horizontal families depending on how oblique the view is).
    max_vanishing_points: int = 3
    #: Snap inlier segments to point exactly at their vanishing point.
    vp_snap: bool = True
    vp_random_seed: int = 0

    # --- collinear merging --------------------------------------------------
    merge_angle_deg: float = 2.5
    #: Perpendicular separation allowed between two segments being merged.
    merge_perp_frac: float = 0.0025
    merge_perp_px: float = 2.0
    #: Collinear segments further apart than this along their shared direction
    #: are left separate. Without it, a window sill and a roofline that happen
    #: to be collinear get welded into one edge spanning the whole house.
    merge_gap_frac: float = 0.02
    merge_gap_px: float = 8.0

    # --- regions ------------------------------------------------------------
    #: How far a loose segment end may be moved onto the corner it was reaching
    #: for. Set to 0 to disable snapping and rely on blanket extension alone.
    #:
    #: Snapping does precisely what extension does bluntly, and the difference
    #: matters: the extension needed to close the worst corner in a frame also
    #: manufactures faces everywhere else. Measured on one scan, closing a box
    #: needed 220 px of extension and produced 53 faces to recover 1.
    #: Hard cap on how far an end may move, as a fraction of the image diagonal.
    snap_radius_frac: float = 0.12
    #: Absolute floor for that cap, in pixels.
    snap_radius_px: float = 12.0
    #: ...and no further than this fraction of the segment's own length.
    #:
    #: This is what makes a single setting work on very different scans. How far
    #: an end may credibly reach for its corner is set by how long the edge is,
    #: not by how big the photograph is: a 600 px roofline stopping 200 px short
    #: is an obvious near-miss, while a 30 px fragment reaching 200 px is
    #: inventing a corner. Capping by image size alone had to choose between
    #: closing a box on one rig and destroying a window on another.
    snap_length_frac: float = 0.5
    #: Pairs closer to parallel than this are not snapped together. Their
    #: intersection moves wildly for a fraction of a degree of noise, so it is
    #: not evidence of a corner.
    snap_min_angle_deg: float = 20.0

    #: Grow each merged segment by this much at both ends, after snapping, to
    #: close what snapping could not -- corners where no neighbouring line was
    #: found at all, and near-parallel pairs snapping refuses to touch.
    #: Measured on one scan: snapping alone found 3 faces, extension alone
    #: needed 220 px and produced 53, and the two together found the box among
    #: 6.
    extend_frac: float = 0.03
    extend_px: float = 6.0
    #: Faces smaller than this are texture, not architecture.
    min_region_area_frac: float = 0.0007
    min_region_area_px: float = 64.0
    #: Discard regions covering more than this fraction of the frame; these are
    #: normally the outer boundary face rather than a real architectural region.
    max_region_area_frac: float = 0.9
    #: A face at least this much covered by the decoder's `likely_glass` mask is
    #: labelled a window.
    window_glass_fraction: float = 0.5

    # --- optional Segment Anything -----------------------------------------
    use_sam: bool = False
    sam_checkpoint: str | None = None
    sam_model_type: str = "vit_b"
    sam_points_per_side: int = 16
    sam_min_area_px: float = 2000.0
    sam_device: str = "cpu"

    def resolve(self, shape: tuple[int, ...]) -> DetectConfig:
        """Return a copy with size-relative thresholds converted to pixels.

        Detection functions take an already-resolved config, so nothing
        downstream has to know about the image size.
        """
        height, width = int(shape[0]), int(shape[1])
        diagonal = float(np.hypot(width, height))
        area = float(width * height)
        return dataclasses.replace(
            self,
            min_segment_length_px=max(self.min_segment_length_px,
                                      self.min_segment_length_frac * diagonal),
            merge_perp_px=max(self.merge_perp_px, self.merge_perp_frac * diagonal),
            merge_gap_px=max(self.merge_gap_px, self.merge_gap_frac * diagonal),
            extend_px=max(self.extend_px, self.extend_frac * diagonal),
            snap_radius_px=max(self.snap_radius_px, self.snap_radius_frac * diagonal),
            min_region_area_px=max(self.min_region_area_px,
                                   self.min_region_area_frac * area),
        )


@dataclass
class TransferConfig:
    """Camera-space -> projector-space transfer."""

    #: Spacing, in camera pixels, at which polygon edges are resampled before
    #: transfer. Straight lines in camera space are not straight in projector
    #: space once they cross a depth discontinuity, so edges must be carried
    #: point-by-point rather than vertex-to-vertex.
    densify_step_px: float = 4.0

    #: If a sample lands on an invalid (undecoded) camera pixel, search this
    #: many pixels around it for a valid neighbour before giving up on it.
    lookup_radius_px: int = 4

    #: A jump larger than this between adjacent decoded camera pixels is a
    #: depth discontinuity, not a gradient. Nothing is interpolated across one.
    #: Expressed, like the decoder's, as a multiple of the per-camera-pixel step
    #: with an absolute floor -- see DecodeConfig.median_max_jump_steps.
    discontinuity_jump_steps: float = 4.0
    discontinuity_jump_px: float = 2.0

    #: A second, wider pass that can see a discontinuity across the band of
    #: undecoded pixels that usually sits inside one. Set to 0 to disable.
    #: Its threshold scales with the window, because a smooth gradient spreads
    #: over a wider window too.
    discontinuity_window_px: int = 11
    #: Resolved threshold for that wider pass.
    discontinuity_bridge_px: float = 8.0

    #: Scattering a camera mask into projector space leaves gaps wherever the
    #: projector out-resolves the camera. Close them morphologically...
    close_kernel_px: int = 5
    #: ...and fill any remaining interior hole smaller than this.
    max_hole_area_px: float = 2000.0
    #: Drop transferred polygons smaller than this in projector space.
    min_polygon_area_px: float = 100.0

    # --- foreground isolation ----------------------------------------------
    #: Residual, in projector pixels, above which a camera pixel is taken to be
    #: off the scene's dominant plane. 0 chooses the threshold automatically
    #: from the residual histogram, which is what you want: the two surfaces
    #: separate cleanly and the gap between them is obvious in the data.
    foreground_residual_px: float = 0.0
    #: Fraction of decoded pixels to sample when fitting the dominant plane.
    foreground_sample: int = 60000
    #: Morphological cleanup of the foreground mask, in camera pixels.
    foreground_close_px: int = 9
    #: Drop foreground blobs smaller than this fraction of the largest one.
    foreground_min_blob_frac: float = 0.05
    #: Keep only the single largest connected piece of the projector-space mask.
    #:
    #: Scattering a camera mask into projector space leaves a scatter of stray
    #: pixels around the subject -- pixels whose decode was marginal, or which
    #: sit on the rim where the two surfaces blur into each other. They project
    #: as a spray of dots around the object. When the subject is one object,
    #: keeping one piece removes them entirely.
    foreground_single_blob: bool = True
    #: Shrink the finished mask by this many projector pixels.
    #:
    #: The mask boundary sits *on* the silhouette, and between grid
    #: quantisation and the parallax between camera and projector at a silhouette
    #: edge, "on" means a little light lands on whatever is behind. Pulling the
    #: edge in by a pixel or two keeps the projection on the subject, at the
    #: cost of an equally thin unlit margin around it.
    foreground_inset_px: float = 1.0

    #: A densified ring point is dropped when it sits this far from *both* its
    #: neighbours while they sit close to each other -- an isolated spike.
    #:
    #: The distinction matters. A genuine depth step is close to one neighbour
    #: and far from the other, and must be kept: it is the bend that lets a
    #: straight camera-space edge follow a real surface. A bad lookup, where the
    #: radius search grabbed a valid pixel belonging to another surface, jumps
    #: away and straight back. Projected, that is a beam of light thrown across
    #: the room.
    ring_outlier_steps: float = 6.0
    #: Absolute floor for the above, in projector pixels.
    ring_outlier_px: float = 6.0

    #: Reject a transferred ring whose perimeter exceeds this multiple of its
    #: own bounding-box diagonal.
    #:
    #: Spike rejection removes a *lone* bad lookup, but not a systematic zigzag:
    #: when a ring runs along a surface boundary, consecutive densified points
    #: can land alternately on the near and far surface, and every point is then
    #: far from both its neighbours while the neighbours are far from each other
    #: too. Nothing local distinguishes that from a real bend.
    #:
    #: Its global shape does. A plausible outline has a perimeter a few times
    #: its diagonal; a zigzag has tens of times. Projected, such a ring is a
    #: bundle of parallel beams thrown across the target, so it is better
    #: dropped than shown.
    ring_max_perimeter_ratio: float = 8.0

    def resolve(self, camera_shape: tuple[int, ...],
                projector_size: tuple[int, int]) -> TransferConfig:
        """Convert step-relative thresholds into projector pixels."""
        step = projector_px_per_camera_px(camera_shape, projector_size)
        span = max(1, self.discontinuity_window_px)
        return dataclasses.replace(
            self,
            discontinuity_jump_px=max(self.discontinuity_jump_px,
                                      self.discontinuity_jump_steps * step),
            discontinuity_bridge_px=max(self.discontinuity_bridge_px,
                                        self.discontinuity_jump_steps * step * span),
            ring_outlier_px=max(self.ring_outlier_px,
                                self.ring_outlier_steps * step * self.densify_step_px),
        )


@dataclass
class ExportConfig:
    """Output artifacts."""

    svg_stroke: str = "#00ff00"
    svg_stroke_width: float = 2.0
    svg_fill: str = "none"
    svg_fill_opacity: float = 0.15
    svg_label_regions: bool = True
    svg_font_size: float = 16.0
    mask_filename: str = "mask.png"
    svg_filename: str = "regions.svg"
    json_filename: str = "scan.json"


@dataclass
class PreviewConfig:
    """On-site visual feedback loop."""

    #: "mask", "outline", "fill", "cycle"
    mode: str = "mask"
    #: Blink period in ms for the mask preview; 0 disables blinking. Blinking
    #: makes a misalignment of a few pixels far easier to see than a static
    #: mask does.
    blink_ms: int = 0
    outline_thickness: int = 3
    #: Colour and weight of the *surface* outline -- the edge of everything the
    #: projector is landing on. Distinct from the per-region outlines, and
    #: bright because it is read off a wall from several metres away.
    surface_outline_colour: tuple[float, float, float] = (255.0, 255.0, 255.0)
    #: Weight of that surface outline, in projector pixels.
    surface_outline_thickness: int = 3
    fill_alpha: float = 0.5
    #: Seconds each region is shown for in "cycle" mode.
    cycle_dwell_s: float = 1.5
    show_labels: bool = True


@dataclass
class ReportConfig:
    """Video and summary generation for a finished scan."""

    #: Output frame size for the video. Everything is letterboxed into this.
    width: int = 1280
    height: int = 720
    fps: int = 24
    #: Preferred codec, with a fallback if the build cannot open it. avc1 is
    #: H.264 and plays everywhere; mp4v is larger but universally available.
    codec: str = "avc1"
    codec_fallback: str = "mp4v"

    #: Seconds each still section holds for.
    section_seconds: float = 3.0
    #: Seconds the title and summary cards hold for.
    card_seconds: float = 4.0
    #: Frames per second to replay the capture sequence at. Slower than the
    #: video frame rate, so each projected pattern is actually visible.
    capture_replay_fps: float = 8.0
    #: Blink period for the preview section, in milliseconds.
    preview_blink_ms: int = 500
    #: Also write each section's still frame as a PNG.
    write_stills: bool = True


#: Deep, saturated, and recognisably Christmas. BGR, as OpenCV wants.
WARM_WHITE = (150, 205, 255)
HOLLY = (40, 170, 40)
CRIMSON = (40, 40, 220)
GOLD = (40, 180, 255)
ICE = (255, 200, 120)
#: A deeper blue for the wash. ICE is pale on purpose -- it is the colour of a
#: snowflake, which should read as near-white -- but a pale colour spread over
#: the whole subject comes out grey rather than blue, which is the one thing
#: this palette is meant to avoid. Large areas get the saturated version.
FROST = (255, 150, 60)
#: Deep emerald, for the pine state. Holly is a mid green and goes grey when
#: it is dimmed; this holds its hue down at low levels.
PINE = (30, 110, 25)
#: A warm interior amber, for window light.
LAMP = (60, 150, 245)

#: The three colour states the show lives in, as (primary, accent).
#:
#: Three states, hard-cut, rather than a palette swept continuously. A sweep
#: through saturated hues spends most of its running time on the midpoint
#: between two of them -- crimson easing into holly passes through olive and
#: dusty mauve, colours nobody chose and which read as a crossfade caught
#: half-way. Christmas has a small, strong vocabulary; these are three pairs
#: from it, and the show stays inside one pair at a time.
#: Each state is a deep base plus one accent, with nothing in the middle.
#:
#: Mud is a luminance problem, not a hue problem. A saturated colour held at
#: mid brightness on a dark ground desaturates: yellow-green becomes khaki,
#: orange becomes rust, red becomes dried blood. So each base sits low and
#: dark and its accent sits high and hot, and the range between them is where
#: no pixel should linger.
ICE_STATE = ((190, 70, 20), (255, 225, 170))      # deep navy -> ice cyan-white
#: Warmed off pure red on purpose. A saturated crimson has a luma of 67 at
#: full scale, so a wall painted in it cannot exceed mid-grey however bright
#: the projector is driven -- the colour itself is the ceiling, and on a
#: facade it lands as maroon-to-black at throw distance. Pushed towards
#: red-orange it carries 121 and still reads unambiguously as Christmas red.
FIRE_STATE = ((30, 70, 255), (90, 205, 255))      # hot red-orange -> gold
#: Pine keeps a green accent rather than a gold one. Gold is the natural
#: partner for green on paper, but here the accent is also what the heat glow
#: is drawn in, and warm light over green lands on hue 77 -- yellow-green, the
#: olive-khaki that made a third of the song look like army surplus. A bright
#: spring green holds the state where it belongs.
PINE_STATE = ((25, 120, 20), (150, 255, 120))     # saturated pine -> true green
#: Red on one mass, green on the other. Red and green together is the one
#: colour signature everyone reads as Christmas instantly, and a palette that
#: never puts them on the building at the same time is a winter palette, not a
#: Christmas one.
HOLLY_SPLIT = ((30, 60, 240), (40, 185, 40))


@dataclass
class LayerConfig:
    """Everything the look is made of."""

    # --- base wash ----------------------------------------------------------
    wash: bool = True
    #: Colours cycled through, slowly. Deep enough to stay coloured on brick.
    wash_palette: tuple = (CRIMSON, HOLLY, GOLD, CRIMSON, FROST)
    #: Seconds for one pass through the palette.
    wash_cycle_s: float = 26.0
    #: The colour states to move between, as (primary, accent) pairs. One is
    #: held for a whole section and then cut, never blended across.
    #: Five, not four, and deliberately alternating warm and cool.
    #:
    #: With four states over seven sections the cycle put pine next to the
    #: red/green split, so a third of the song ran through one desaturated
    #: green -- the least festive colour available, three sections in a row.
    #: Five breaks that adjacency.
    wash_states: tuple = (ICE_STATE, FIRE_STATE, PINE_STATE, ICE_STATE,
                          HOLLY_SPLIT)
    #: Give each mass its own colour from the running state -- the base on one,
    #: the accent on the next -- instead of washing the whole building in one.
    #: This is what lets red and green share the facade.
    wash_split_states: tuple = (4,)
    #: How far the level dips where the two colours meet.
    #:
    #: Red and green are complementary, so interpolating between them in RGB
    #: runs straight through ochre and khaki -- the mud shows up only in the
    #: band where they mix, which is why measuring the two ends finds nothing
    #: wrong. Dropping the level through the crossover means they meet in
    #: shadow instead of in mud, which is also what happens when two coloured
    #: lights overlap on a real building.
    wash_split_dip: float = 0.75

    #: Advance the palette at the song's own section boundaries rather than on
    #: a timer, and hold each colour steady in between.
    #:
    #: Continuously crossfading between saturated palette entries means most of
    #: the running time is spent on an intermediate blend rather than on a
    #: colour anyone chose: crimson easing into holly spends twenty seconds
    #: passing through olive and mauve. Holding a colour and changing it
    #: quickly, on a boundary the music already has, is both cleaner and reads
    #: as a decision rather than as a slow drift.
    wash_follow_sections: bool = True
    #: Beats spent crossing from one palette colour to the next at a boundary.
    #: Short: this is the only time mud is on screen.
    wash_blend_beats: float = 2.0
    #: Wash brightness at silence, and at the loudest moment.
    #: Wash level at silence and at the song's loudest, before the uplight
    #: grade. The ceiling is short of 1.0 deliberately: the wash covers the
    #: whole subject, so when it clips it clips everywhere at once and the
    #: colour goes white. The bulbs are the layer allowed to reach full
    #: brightness, because they are small and are meant to read as lights.
    wash_floor: float = 0.85
    wash_ceiling: float = 1.3

    # --- where the light falls ----------------------------------------------
    #: How far the wash reaches up the subject before it falls away, and how
    #: sharply.
    #:
    #: A wash that fills the whole silhouette evenly is the one mistake that
    #: costs the most: it leaves a coloured panel with a decorated fringe, no
    #: focal point, and nowhere dark for a bright element to read against. A
    #: projection show is built out of the difference between lit and unlit,
    #: so most of the subject is deliberately left near black and the light is
    #: a shape rather than a fill.
    wash_reach: float = 0.75
    wash_falloff: float = 1.1
    #: The shapes the light takes, one per section, cycled in this order.
    #:
    #: Changing only the colour of a single bottom-anchored gradient means the
    #: *picture* never changes: a viewer on the lawn sees one image for three
    #: minutes with the hue cycling. These are different shapes, so a section
    #: is a different composition rather than a repaint.
    #:
    #: - ``base``   light pooled at the foot of the building, climbing
    #: - ``ridge``  light falling from the roofline down, gable first
    #: - ``band``   a lit band travelling up the pitch
    #: - ``mass``   one architectural mass lit, the others left dark
    wash_modes: tuple = ("base", "ridge", "mass", "band")
    #: How hard the mass-isolation boundary is, as a multiple of the normal
    #: feather. Below 1 it is sharper: a gradient fading toward the right peak
    #: reads as uneven lamp coverage, not as "that peak is off". Mass
    #: isolation only becomes a shape when its edge is the architecture's own.
    wash_mass_edge: float = 0.28
    #: Light the roof planes at a different level from the wall beneath them.
    #:
    #: The one gesture that makes this projection *mapping* rather than a wash
    #: aimed at a house. Because the height is measured per column against
    #: that column's own eave and ridge, the dividing line follows the pitch
    #: automatically -- diagonal along the gable, stepping down over the wing,
    #: dipping through the valley -- which is exactly the line a vertical mask
    #: can never draw.
    roof_plane: bool = True
    #: Where the roof starts, as a fraction of each column's height, and how
    #: much brighter it is than the wall.
    roof_plane_from: float = 0.62
    #: Strong enough that the roof reads as the lit surface even in cues
    #: whose wash rises from the ground. A bright base under a dark gable is
    #: ground fog, not lighting; the roof carrying more light than the wall
    #: beneath it is what makes the building look lit from outside.
    roof_plane_lift: float = 2.1
    #: How soft the transition is. Feathered, never hard.
    roof_plane_soft: float = 0.3
    #: Light given to the roof regardless of what the wash is doing below.
    #:
    #: A multiplier alone cannot rescue a cue whose wash rises from the
    #: ground: at the roofline the wash is already zero, and any multiple of
    #: zero is zero, so the gable stays dark and the frame reads as fog off
    #: the lawn. A small additive term means the roof is always a lit surface
    #: and the building always reads as a building.
    roof_plane_floor: float = 0.22

    #: Thickness of the travelling band, as a fraction of the subject.
    wash_band_width: float = 0.42
    #: Light left on the rest of the building outside the travelling band, and
    #: on the masses that are not the lit one.
    #:
    #: Not near-zero. Lighting one part is a composition; switching the rest
    #: off is an absence, and it takes the roof modelling with it -- a
    #: gesture worth 90 points of luma across a row needs a wall bright
    #: enough to carry it, or it is present in the numbers and invisible on
    #: the house.
    wash_partial_floor: float = 0.42
    #: Beats for the band to make one pass.
    wash_band_beats: float = 16.0

    #: How far the light reaches at the quietest point and at the loudest.
    #:
    #: The reach climbs with the arc instead of being fixed. A pool that sits
    #: on the base for the whole song lights the least interesting part of the
    #: building and leaves the gable -- the one thing a facade has that a
    #: rectangle does not -- black from start to finish. Letting it climb
    #: means the light reaching the apex is itself the climax, and the
    #: composition genuinely reorganises rather than just collecting more
    #: objects on the same layout.
    wash_reach_low: float = 0.85
    wash_reach_high: float = 1.25
    #: How much the roofline profile is smoothed before the light is shaped to
    #: it. The profile steps at the valley and at the wing, and an unsmoothed
    #: step puts a hard vertical discontinuity down the facade that reads as a
    #: projector alignment fault. Fraction of the subject's width.
    wash_profile_smooth: float = 0.22
    #: Light left in the darkest part. Not zero: a silhouette that vanishes
    #: entirely stops reading as a building.
    #:
    #: Note what does *not* control darkness: the level. Turning a saturated
    #: colour down does not make it dark, it makes it grey -- crimson at forty
    #: percent is brown, amber is khaki. The dark comes from the light landing
    #: on less of the building, so the level stays high enough for the colour
    #: to be a colour and the reach is what is kept small.
    wash_ambient: float = 0.04

    # --- window -------------------------------------------------------------
    #: A warm lit window low on the main mass.
    #:
    #: It does the work the wash cannot: a small, steady, warm shape that holds
    #: still while everything else changes, so the eye has somewhere to land
    #: and the silhouette reads as a building with someone inside rather than
    #: as a coloured panel.
    window: bool = True
    window_colour: tuple[float, float, float] = LAMP
    #: Centre, as a fraction of the subject's width and height.
    window_x_frac: float = 0.3
    window_y_frac: float = 0.62
    window_w_frac: float = 0.2
    window_h_frac: float = 0.21
    window_brightness: float = 0.95
    #: Depth of the slow flicker, as if it were firelight.
    window_flicker: float = 0.12
    #: How hard the window pulses with the music. A focal point that never
    #: responds to anything is not a focal point, it is wallpaper.
    window_pulse: float = 0.6
    #: Level at the start of the song and by the end.
    #:
    #: It is the only warm mass on the building and the only thing lit in the
    #: first bar and the last, so it is the one element that can carry the
    #: whole arc on its own. Held at one value for three minutes it is
    #: wallpaper; brought up across the song it is the thing that has been
    #: there all along, getting warmer.
    window_open: float = 0.55
    window_close: float = 1.35
    #: How much brighter it goes on the very last beat before the blackout, so
    #: the held window is a deliberate note rather than a leftover.
    window_finale_flare: float = 1.5
    #: How hard the window snaps on the downbeat.
    #:
    #: The window is the hardest-edged, brightest, highest-contrast object on
    #: the facade at every moment, so it is where the eye already is -- which
    #: makes it the only element whose change is guaranteed to read at throw
    #: distance. Soft light pulsing behind it reads as flicker; this reads as
    #: a hit. A constant-value hotspot is also what stops any compositional
    #: change behind it registering at all, because the anchor never moves.
    window_strike: float = 1.9
    #: Beats between strikes. 1 is every beat; 2 puts it on the backbeat.
    window_strike_beats: float = 2.0
    #: Colour it snaps to -- near white, so the hit is a value change and not
    #: just more amber.
    window_strike_colour: tuple[float, float, float] = (235, 245, 255)
    #: Ceiling on the window's total output.
    #:
    #: Held below full so the glazing bars stay dark. Blown to white the cross
    #: disappears and the window stops being a window -- which matters most in
    #: the last seconds, where it is one of only two things on screen and the
    #: shape is the entire point of the shot.
    window_ceiling: float = 0.88
    #: How sharp the edges are. A soft blob reads as a smudge on the wall; a
    #: window is a made thing and wants a real edge, with the glazing bars that
    #: are the only reason it reads as a window rather than a lit rectangle.
    window_edge: float = 6.0
    window_panes: bool = True
    window_bar_frac: float = 0.09
    #: How much the wash breathes on each beat.
    #: How much the whole lit field lifts on each beat.
    #:
    #: This is where the rhythm has to live. A chase running through
    #: four-pixel bulbs merges into a continuous line at throw distance and
    #: takes the beat with it; the mass of light is the only element big
    #: enough to carry it.
    wash_beat_lift: float = 0.3
    #: Power applied to the loudness envelope before it drives brightness.
    #:
    #: The envelope is min-max normalised, so a handful of loud bars set the
    #: top of the scale and the median lands near 0.27 -- the song spends most
    #: of its length in the bottom third of the range, and the wash sits near
    #: its floor throughout. Perceived loudness is closer to a power law than a
    #: linear one, so a square root puts a typical bar near the middle of the
    #: range where it belongs, without touching the peaks (1.0 stays 1.0).
    loudness_gamma: float = 0.5
    #: Grade the wash from top to bottom instead of filling flat. A single
    #: colour across a whole facade reads as a coloured gel; a gradient reads
    #: as light with a direction to it.
    wash_two_tone: bool = True
    #: How much brighter the bottom of the wash is than the top.
    #:
    #: The grade is a gain on the colour, never a blend towards another one.
    #: Blending saturated hues in RGB muds: crimson into holly lays a band of
    #: olive-brown across the middle, and lifting holly towards gold turns the
    #: whole subject olive. A gain leaves the hue untouched, and where it
    #: clips it desaturates towards white -- which is what more light actually
    #: does to a surface, so it reads as uplighting rather than as a mistake.
    #:
    #: This is measured across the subject, not the panel. There used to be a
    #: second, panel-relative falloff as well, which meant the two pulled in
    #: opposite directions and the look changed when the projector moved and
    #: the subject landed somewhere else in the frame. One control, anchored to
    #: the thing being lit.
    wash_uplight: float = 1.25

    # --- heat ---------------------------------------------------------------
    #: A warm glow layered over the wash, growing with how loud the music is.
    #:
    #: The palette cycles on a timer, so without this the loudest bar of the
    #: song can land on icy blue and the finale feels like an interlude. Heat
    #: is tied to the music rather than the clock, so a climax always warms up
    #: whatever colour the wash happens to be passing through.
    heat: bool = True
    heat_colour: tuple[float, float, float] = GOLD
    #: Take the glow from the running state's accent instead of that fixed
    #: gold.
    #:
    #: A fixed warm glow poured over every state is the single largest source
    #: of mud in the show: gold added to the deep navy of the ice state drops
    #: its saturation from 0.90 to 0.21, and what should be a cold blue
    #: facade renders as a grey-beige haze. Each state's own accent keeps the
    #: climax warming *within* the colour it is already in.
    heat_follow_state: bool = True
    heat_gain: float = 0.45
    #: Raising energy to this power keeps the glow out of quiet passages and
    #: concentrates it on genuine peaks.
    heat_curve: float = 1.8

    #: How far inside the scanned silhouette the animation stays, in panel
    #: pixels.
    #:
    #: Separate from `transfer.foreground_inset_px`, and at a different
    #: resolution: that one shrinks the mask at the scan grid, where one unit
    #: can be eight panel pixels, which is far too blunt to trim an edge by eye.
    #: This one works at panel resolution, where the animation is actually
    #: drawn.
    #:
    #: A margin is needed at all because the subject stands proud of whatever
    #: is behind it. A ray aimed just inside the silhouette still clears the
    #: edge and lands on the wall beyond, so the lit area has to stop slightly
    #: short of the true outline. How far short depends on how far the subject
    #: stands out and how oblique the projector is -- raise it until the wall
    #: behind goes dark.
    #:
    #: 10 px is what the cardboard rig needed, measured by photographing the
    #: result: at 6 px the edge lights still sat on the boundary itself.
    edge_margin_px: float = 10.0

    # --- the build ----------------------------------------------------------
    #: Bring the elements in one at a time as the song moves through its
    #: sections, and never take one away until the finale.
    #:
    #: This is what stops every frame being the same composition in a
    #: different colour. Cycling hue over a fixed layout means a viewer who
    #: looks away and back sees the same picture; adding a *new element* each
    #: section means the picture is visibly further along than it was, which
    #: is the only thing that actually reads as a build. It also gives the
    #: opening somewhere to start from -- one window in the dark -- instead of
    #: spending everything in the first eight seconds.
    build: bool = True
    #: The order things arrive in, spread evenly across the song's sections.
    #: Earlier means quieter and more fundamental.
    build_order: tuple = ("window", "bulbs", "snow", "zones", "santa",
                          "icicles", "flyer", "sparkles", "star")
    #: How many of the order are lit from the very first bar, so the
    #: opening has some texture rather than being a single window in an
    #: otherwise empty frame for twenty seconds.
    build_opening: int = 3

    # --- the arc ------------------------------------------------------------
    #: Let each section's own loudness set how bright and busy it is.
    #:
    #: Without this every section runs at full tilt and the show has no
    #: beginning, middle or end -- shuffle the frames and nothing tells you
    #: which came first. Scoring each section against the others means a quiet
    #: verse actually drops away and the last chorus actually arrives, and the
    #: schedule comes from the recording instead of being hand-typed per song.
    arc: bool = True
    #: Intensity at the quietest section and at the loudest.
    #: Raised deliberately: at 0.55 the quiet sections rendered as black
    #: houses with a lit window, which is not a hush, it is an absence. A
    #: verse should be quieter than a chorus, not invisible.
    #: Raised again after measuring: the roof-plane gesture swings 94 points
    #: of luma across a row, but only where the wall is bright enough to carry
    #: it. Below about a quarter of pixels over luma 60 the mapping is there
    #: in the numbers and invisible on the house.
    arc_floor: float = 0.86
    #: Deliberately short of 1.0. The loudest section is where every layer is
    #: already at its brightest, so letting the wash reach full as well pushes
    #: the whole subject to white and the snow, icicles, star and beard -- all
    #: the white elements -- disappear into it. The climax should be the frame
    #: with the most contrast, not the least.
    arc_ceiling: float = 1.0
    #: How much of the light's climb comes from simply being further through
    #: the song, rather than from how loud the current section happens to be.
    #:
    #: Loudness alone oscillates: a loud verse floods the gable two thirds of
    #: the way in and leaves the actual climax with nowhere left to go, so the
    #: show peaks in the middle and arrives nowhere. Mixing in progress makes
    #: the climb monotone -- the light creeps up the building across the whole
    #: song and only tops out at the end.
    arc_progress_weight: float = 0.72
    #: Beats spent easing into a new section's level, so it swells rather than
    #: stepping.
    arc_ease_beats: float = 4.0
    #: How far the level dips at a section boundary before the new colour
    #: arrives. Cutting through darkness keeps the two states from ever being
    #: on screen together, which is what would mix them into mud.
    #: Deep enough to read as a cut, shallow enough that the building does
    #: not vanish: at 0.7 the facade went fully black at every boundary, which
    #: reads as a dropout rather than as an edit.
    arc_cut_dip: float = 0.45

    #: Put the last section into the fire state whatever the rotation says,
    #: and end on a held blackout.
    #:
    #: A climax has to be the most saturated frame in the show, not the
    #: palest. Letting every layer run to full turns the last chorus white,
    #: and white on a projector is the one colour with no identity -- the
    #: dimmest frame ends up with more presence than the finale. So the finale
    #: is capped in luminance, pinned to the hottest state, and then cut: the
    #: whole facade drops away leaving the window and the star burning, held.
    finale: bool = True
    #: At the climax, invert: the fringe goes to full white and the star
    #: throws its rays wide. Something has to *arrive* at the peak that has
    #: not been seen before, or the climax is only a brighter version of the
    #: verse and the show builds to nothing.
    finale_flare: float = 2.2
    #: One full-value, edge-to-edge moment early on, and where to put it as a
    #: fraction through the song.
    #:
    #: Without it the show opens on its two dimmest pictures and only reaches
    #: full light in the last twenty seconds, which is a step, not a build:
    #: the audience is never shown that the house *can* light up, so the
    #: climax arrives to settle a question nobody knew was open. One early
    #: statement sets the ceiling, and everything after it is measured
    #: against what they have already seen.
    statement: bool = True
    statement_at: float = 0.24
    statement_beats: float = 9.0
    #: Reduced from a level that simply clipped everything. Driven hard
    #: enough to saturate, a statement stops being lit *form* and becomes a
    #: flat sheet with no falloff, no direction and no separation between the
    #: wall and the trim -- the loudest moment ends up the least legible. The
    #: brightness now comes mostly from reach, which keeps the modelling.
    statement_gain: float = 2.6

    #: Seconds of the held ending given to the outline before it too drops.
    #:
    #: Two identical held frames is a stop, not an ending. Dropping the
    #: roofline a beat after the facade gives the last seconds a second event
    #: to land on: full house, then the outline alone, then the window and the
    #: star. The song ends on a button; so should this.
    finale_outline_s: float = 2.6
    #: Wall light left on under the held ending, so the last frame is still a
    #: house rather than two objects in a void.
    #: Tuned so the last frame's wall sits around 60 -- lit enough that the
    #: silhouette is plainly still there, dim enough that the cut still reads
    #: as the light going out of the house rather than as a dimmer.
    finale_wall_rest: float = 0.45
    #: How much of the roofline is left burning at the very end.
    #:
    #: Not zero. Dropping it entirely leaves a star and a window floating in
    #: a void, and the audience has spent the whole show learning that shape
    #: -- deleting it on the last beat reads as the projector failing rather
    #: than as an ending. A trace of it keeps the house present while the
    #: light goes out of it.
    finale_outline_rest: float = 0.42
    #: Seconds of held blackout at the end, with only window and star lit.
    finale_hold_s: float = 5.0
    #: Beats the blackout takes to arrive.
    finale_cut_beats: float = 1.0
    #: How much of the roof lights survives the blackout.
    #:
    #: Not zero. Holding a window and a star in a void leaves two objects
    #: floating with no relation to each other or to anything else; a faint
    #: roofline keeps the house present, so it reads as the house going quiet
    #: rather than as the house disappearing.
    finale_outline: float = 0.55
    #: Colour the held roofline is recoloured to. Cool, not the running warm
    #: accent: held dim, a warm amber reads as dirty brown rather than embers.
    finale_outline_colour: tuple[float, float, float] = (255, 225, 175)

    #: Overall output level, applied last to every layer.
    #:
    #: Throw distance and surface make an enormous difference to how much light
    #: actually arrives: a projector a metre from a cardboard cutout is wildly
    #: over-powered compared with the same unit lighting a house from across a
    #: driveway. The look is designed at 1.0 for the latter; bring this down
    #: when the subject is close, or when the projector is brighter than the
    #: scene needs and the colours are washing out towards white.
    master_gain: float = 1.0

    #: Scale feature sizes to the subject rather than the panel.
    #:
    #: The projector usually covers far more than the thing being lit, so a
    #: bulb "6 pixels across" is huge on a facade filling the frame and a speck
    #: on a cutout occupying a tenth of it. Sizes are therefore derived from the
    #: silhouette's own perimeter, and the same settings look right on both.
    auto_scale: bool = True

    # --- edge lights --------------------------------------------------------
    bulbs: bool = True
    bulb_count: int = 96
    #: Hang the lights on the roof edges only, not right round the silhouette.
    #:
    #: A chain round the whole outline turns a facade with a gable, a lower
    #: peak and a wing into one flat outlined shape -- a neon sign shaped like
    #: a house -- and puts its brightest band along the ground, which inverts
    #: the focal hierarchy completely. Real outline lighting follows the eaves
    #: and verges and stops there.
    bulb_verges_only: bool = True
    #: Used directly when auto_scale is off; otherwise only a floor, and kept
    #: small so the computed size is what actually governs.
    bulb_radius_px: int = 5
    #: Bulb diameter as a fraction of the spacing between bulbs. Below 1 they
    #: read as separate lights; above, as a continuous rope.
    bulb_fill: float = 0.8
    #: Beats for one full lap of the chase around the silhouette.
    bulb_lap_beats: float = 8.0
    #: Width of the travelling bright band, as a fraction of the perimeter.
    bulb_comet_frac: float = 0.22
    #: Brightness of a bulb outside the travelling band.
    bulb_base: float = 0.8
    bulb_colour: tuple[float, float, float] = WARM_WHITE
    #: Every Nth bulb takes the accent colour, the way a real string alternates.
    bulb_accent_every: int = 5
    bulb_accent_colour: tuple[float, float, float] = CRIMSON
    #: Take the accent from whichever colour state is running, rather than the
    #: fixed colour above.
    #:
    #: A multicolour string is the most arbitrary choice available: it contains
    #: every hue, so by definition it clashes with whatever the wash is doing,
    #: and at throw distance it collapses into an iridescent fringe on the one
    #: contour that should read most cleanly. Warm white plus the running
    #: state's accent keeps the outline crisp and part of the same scheme.
    bulb_follow_state: bool = True
    #: How hard the whole string flashes on a beat.
    #:
    #: The one element that is literally "Christmas lights on a house" was
    #: doing nothing for the entire song. Ten beats would pass with no visual
    #: accent anywhere in the frame -- movement, but no rhythm.
    bulb_beat_flash: float = 0.85
    #: Extra kick the roofline takes on the same beat the window snaps.
    #:
    #: One element pulsing alone reads as a blinking light; the roofline and
    #: the window hitting together reads as the house breathing. They have to
    #: land on the same frame to do that, so this shares the window's cadence
    #: rather than running on its own.
    bulb_strike: float = 1.3
    #: Per-bulb random twinkle depth.
    bulb_twinkle: float = 0.25
    #: Range the per-bulb twinkle rates are drawn from, in Hz. A spread rather
    #: than one rate, so the string shimmers instead of pulsing in unison.
    bulb_twinkle_hz: tuple[float, float] = (1.5, 4.0)

    # --- snow ---------------------------------------------------------------
    snow: bool = True
    #: Flakes across the subject's bounding box.
    #:
    #: Counted over the subject rather than the panel, which is what anyone
    #: setting it actually means. Over the panel it was badly misleading: with
    #: the subject at 5% of the frame, 420 flakes put about 22 where they could
    #: be seen, and the same number would read completely differently the
    #: moment the projector was moved closer.
    snow_count: int = 70
    snow_speed_px_s: float = 70.0
    #: Multipliers on the fall speed, giving near flakes and far ones.
    snow_speed_spread: tuple[float, float] = (0.6, 1.5)
    snow_drift_px_s: float = 18.0
    snow_radius_px: int = 3
    #: Snowflake diameter as a fraction of the silhouette's smaller dimension.
    snow_size_frac: float = 0.012
    snow_brightness: float = 0.85

    # --- sparkle sweep ------------------------------------------------------
    #: A trail of sparkles that flies across the subject and fades behind it.
    #:
    #: The wash, bulbs and snow all sit still or repeat on a short cycle. This
    #: is the layer that crosses the whole facade, so it is what gives the show
    #: somewhere to go, and it is worth having it arrive on the music rather
    #: than on a timer.
    sparkles: bool = True
    sparkle_count: int = 90
    #: Beats between one sweep starting and the next.
    sparkle_every_beats: float = 8.0
    #: Beats a sweep takes to cross. Shorter than the gap, or they overlap.
    sparkle_cross_beats: float = 3.0
    #: Length of the fading tail, as a fraction of the crossing.
    sparkle_tail: float = 0.34
    #: How far sparkles scatter off the sweep line, as a fraction of the height.
    sparkle_spread: float = 0.9
    sparkle_colour: tuple[float, float, float] = GOLD
    #: Every Nth sparkle takes the cooler colour, so the trail is not one flat
    #: gold. 0 disables.
    sparkle_accent_every: int = 3
    sparkle_accent_colour: tuple[float, float, float] = ICE
    sparkle_size_frac: float = 0.02
    sparkle_brightness: float = 1.25
    #: Draw each sparkle as a four-pointed star rather than a dot. Costs a
    #: little and is most of what makes it read as magic rather than as rain.
    sparkle_star: bool = True

    # --- santa --------------------------------------------------------------
    #: Santa rises from the bottom edge, looks around, and ducks back down.
    #:
    #: Drawn opaque rather than added, unlike every other layer here. He is a
    #: character standing in front of the wash, not light falling on it, and
    #: adding him would mean his eyes and the shadow under his hat brim -- the
    #: things that make him read as a face at all -- simply would not appear.
    santa: bool = True
    #: Beats between appearances. He is a surprise; too often and he is wallpaper.
    santa_every_beats: float = 48.0
    #: Beats spent rising, looking around, and ducking back.
    santa_rise_beats: float = 2.0
    santa_hold_beats: float = 6.0
    santa_duck_beats: float = 1.5
    #: His height, as a fraction of the subject's.
    santa_height_frac: float = 0.42
    #: Show him at the window rather than rising from the bottom edge.
    #:
    #: A head with no body floating in the middle of a facade has no scale and
    #: no reason to be there, and it is the one element whose drawing style
    #: does not match the flat silhouettes elsewhere. Framed by the lit window
    #: all three problems go away at once: a head is exactly what you see at a
    #: window, the frame gives it scale, and it becomes part of the building
    #: instead of a sticker on top of it. He still rises from below -- the
    #: sill is simply the edge he rises past.
    santa_in_window: bool = True
    #: How much of the window's height he fills when fully up.
    santa_window_fill: float = 1.35
    #: How far he rises, as a fraction of his own height.
    #:
    #: Short of 1.0 on purpose. Rising his full height carries his face up
    #: past the head of the window and leaves only the beard framed, so the
    #: moment lands on a white blob instead of on a face. This stops him with
    #: his eyes in the opening, which is the whole point of him.
    santa_window_rise: float = 0.87

    #: Where along the bottom edge he comes up, 0 left to 1 right.
    santa_x_frac: float = 0.5
    #: How far he leans side to side while up, as a fraction of his width.
    santa_sway: float = 0.16
    #: Path to a PNG to use for Santa instead of the drawn one.
    #:
    #: Needs an alpha channel, since it is composited over the wash by that
    #: alpha -- a JPEG, or a PNG on a white background, appears in a white box.
    #: Anything else about it is free: it is scaled to `santa_height_frac` and
    #: positioned exactly like the drawn figure, so swapping it changes nothing
    #: about the timing. Empty falls back to drawing him.
    #:
    #: Defaults to the public-domain head shipped in `facade_scan/assets/`.
    #: A bare name with no directory is looked up there, so the default keeps
    #: working from any directory and from an installed package; anything with
    #: a path is resolved normally. Empty uses the drawn figure instead.
    #:
    #: It is bundled only because it is CC0. Redistribution is a stricter
    #: question than whether it is fine to point this at a file on your own
    #: disk, so anything not clearly licensed for it belongs outside the repo.
    santa_image: str = "santa-head.png"

    santa_hat: tuple[float, float, float] = (40, 40, 205)
    santa_trim: tuple[float, float, float] = (245, 245, 250)
    santa_skin: tuple[float, float, float] = (150, 190, 235)
    santa_eyes: tuple[float, float, float] = (25, 25, 35)

    # --- flyer --------------------------------------------------------------
    #: Santa's sleigh and reindeer, flying across the subject on an arc.
    #:
    #: Tinted rather than composited: the bundled art is a black silhouette,
    #: and black is the one colour a projector cannot produce -- it would come
    #: out as a sleigh-shaped hole in the wash. The alpha is used as a stencil
    #: and filled with `flyer_colour`, so it flies as light.
    flyer: bool = True
    flyer_image: str = "sleigh-reindeer.png"
    #: Beats between crossings, and beats spent crossing.
    flyer_every_beats: float = 48.0
    flyer_cross_beats: float = 10.0
    #: Offset into the cycle, so the flyer and Santa take turns instead of
    #: sometimes landing on top of each other. Half of `flyer_every_beats`
    #: puts one exactly between two of the other.
    flyer_phase_beats: float = 24.0
    #: His height, as a fraction of the subject's.
    flyer_height_frac: float = 0.3
    #: Height of the arc he flies, as a fraction of the subject's height, and
    #: where the arc sits vertically (0 top, 1 bottom).
    flyer_arc_frac: float = 0.22
    flyer_lane_frac: float = 0.42
    flyer_colour: tuple[float, float, float] = GOLD
    #: Right to left instead of left to right.
    flyer_reverse: bool = False

    # --- zones --------------------------------------------------------------
    #: Light the building's masses separately instead of as one shape.
    #:
    #: The silhouette has real structure -- a central gable, a lower peak, the
    #: valley between them. One gradient across the whole thing ignores every
    #: plane break, which is what makes a show look like a poster with
    #: animation on it rather than like light on a building. Lighting the
    #: masses independently, and letting them answer each other on the beat,
    #: is the difference.
    zones: bool = True
    #: How far a zone falls below full when it is not the one being lit.
    zone_depth: float = 0.62
    #: Beats each zone holds before the light moves to the next.
    zone_beats: float = 4.0
    #: How wide the blend between zones is, as a fraction of the subject's
    #: width. Without it the change in level lands on a single column and
    #: draws a hard vertical seam straight down the facade, which reads as a
    #: rendering fault rather than as light. Real light has an edge, but not
    #: one that is one pixel wide and perfectly plumb.
    zone_feather: float = 0.22
    #: A lit edge that rakes up each mass, so the wash travels rather than
    #: simply changing level. Depth of the band, and beats per pass.
    wash_sweep: float = 0.5
    wash_sweep_beats: float = 8.0

    # --- icicles ------------------------------------------------------------
    #: Icicles hanging from the roofline.
    #:
    #: This is the layer that most repays having scanned the house: it hangs
    #: them from the real edge, following the gables and the eaves exactly,
    #: which is the one thing that cannot be faked with a generic loop.
    icicles: bool = True
    #: Roughly how many across the subject; spacing follows from its width.
    icicle_count: int = 34
    #: Length as a fraction of the subject's height, shortest to longest. They
    #: are drawn at random lengths in this range, because even icicles are.
    icicle_length_frac: tuple[float, float] = (0.04, 0.13)
    icicle_width_frac: float = 0.016
    icicle_colour: tuple[float, float, float] = ICE
    #: Held below full so the strike has somewhere to go.
    #:
    #: At 0.85 the fringe already sat at white, so the strike that was
    #: supposed to run along it could not make it any brighter and was
    #: invisible -- the whole roofline measured within 5% across a beat.
    #: Dimmer at rest, the strike reads.
    icicle_brightness: float = 0.52
    #: Depth of the slow shimmer along them.
    icicle_shimmer: float = 0.35
    #: A highlight that travels along the roofline, so the icicles glint in
    #: sequence instead of sitting there. They are the highest-contrast thing
    #: on the subject, so leaving them completely static spends the strongest
    #: read in the frame on furniture.
    #: A hard white strike that runs along the fringe in sequence.
    #:
    #: A soft luminance pump across the whole mass reads as a flicker, not as
    #: a hit -- if the on-beat frames cannot be picked out by eye, nobody
    #: thirty feet away will feel the beat either. This is the one crisp,
    #: high-contrast element on the building, so this is what gets snapped.
    icicle_strike: float = 2.6
    #: Beats between strikes, and how tightly the strike is focused.
    icicle_strike_beats: float = 4.0
    icicle_strike_focus: float = 3.5
    icicle_glint: float = 1.6
    #: Beats for the highlight to cross the whole roofline. Fast and hard: a
    #: shader subtlety that cannot be seen from thirty feet is not worth
    #: drawing, and this is the highest-contrast element on the building.
    icicle_glint_beats: float = 4.0
    #: How tight the travelling highlight is. Higher is sharper.
    icicle_glint_focus: float = 9.0

    # --- star ---------------------------------------------------------------
    #: A star at the highest point of the silhouette -- the apex of the gable
    #: on a house, which the scan already knows.
    star: bool = True
    #: Hold the star back until this fraction through the song.
    #:
    #: An element that is present from the first bar can never become an
    #: event. The apex is the most privileged point on the building, so it is
    #: worth withholding and then igniting for the last chorus, after which it
    #: is the last thing still lit.
    star_from_frac: float = 0.55
    star_ignite_beats: float = 6.0
    star_size_frac: float = 0.06
    star_colour: tuple[float, float, float] = (120, 225, 255)
    star_brightness: float = 1.9
    #: Beats per full twinkle.
    star_twinkle_beats: float = 4.0
    #: How many points it throws. Four reads as a cross and as antennae at
    #: small scale; more, with a tight core, reads as a star.
    star_points: int = 8
    #: Width of each ray at its base, as a fraction of the core radius.
    #:
    #: Thin. Wide rays merge with the core into a blunt four-lobed polygon
    #: that reads as a mouse cursor -- which is a poor last image to end a
    #: show on, and it is the last thing anyone sees.
    star_ray_width: float = 0.34
    #: A long vertical tail below the core, the way a Christmas star is drawn.
    star_tail: float = 1.6
    #: How far below the apex the star's centre sits, as a multiple of its ray
    #: reach.
    #:
    #: It has to clear the roofline by its own radius or more. Centred on the
    #: apex, every ray pointing upward falls outside the silhouette and is
    #: clipped away, leaving only the downward half -- which is why it read as
    #: a rocket rather than a star however the points were drawn.
    star_drop: float = 1.15
    #: Length of the rays, as a multiple of the star's radius.
    star_ray_scale: float = 4.5

    # --- accents ------------------------------------------------------------
    accents: bool = True
    #: Onset strength above which an accent fires.
    accent_threshold: float = 0.62
    #: Seconds an accent takes to fall away.
    accent_decay_s: float = 0.35
    accent_gain: float = 0.5
    accent_colour: tuple[float, float, float] = WARM_WHITE


@dataclass
class SimPoseConfig:
    """A pinhole device (camera or projector) in the synthetic scene."""

    width: int = 960
    height: int = 600
    #: Horizontal field of view in degrees. A projector's throw ratio T maps to
    #: hfov = 2*atan(1/(2*T)); T=1.4 gives 39.3 degrees.
    hfov_deg: float = 39.3
    #: Where the device sits, in world metres. The camera should be as close to
    #: the projector as it can physically be: every centimetre between them is
    #: projector shadow beside the bump-out that no decoder can recover.
    position: tuple[float, float, float] = (0.0, 1.20, 0.0)
    #: Point the device is aimed at, in world coordinates.
    target: tuple[float, float, float] = (0.0, 2.90, 14.0)


@dataclass
class SimConfig:
    """Synthetic simulator.

    The simulator resolutions are deliberately independent of
    :class:`ProjectorConfig`; they only need to be representative. What matters
    is that the camera out-resolves the projector over the facade (as a real
    DSLR massively does), because a camera that undersamples the finest stripe
    cannot decode the low-order bits no matter how good the decoder is.
    """

    #: The simulated projector. Its 43-degree default field of view is a throw
    #: ratio of about 1.3, which covers an 11 m width from 14 m away.
    projector: SimPoseConfig = field(
        default_factory=lambda: SimPoseConfig(
            width=960, height=600, hfov_deg=43.0,
            position=(0.0, 1.20, 0.0), target=(0.0, 2.85, 14.0),
        )
    )
    #: The simulated camera, framed slightly wider than the projector so it sees
    #: the whole lit area, and at 1.5x the resolution so it out-samples the
    #: finest Gray stripe.
    camera: SimPoseConfig = field(
        default_factory=lambda: SimPoseConfig(
            width=1440, height=900, hfov_deg=41.5,
            position=(0.60, 1.10, 0.05), target=(0.0, 2.85, 14.0),
        )
    )
    #: Path to the house geometry TOML. None uses the bundled default house.
    house_path: str | None = None
    #: Fraction of projector light that reaches an unlit surface anyway
    #: (projector black level + streetlights + moon).
    ambient: float = 0.06
    #: Gaussian sensor noise standard deviation in normalised [0,1] intensity.
    #: 0.008 is about 2/255, typical of a decent sensor at moderate ISO.
    noise_sigma: float = 0.008
    #: Optical defocus of the projected pattern, as a gaussian sigma in *camera*
    #: pixels. Real projectors are never perfectly focused across a facade that
    #: spans a metre of depth.
    blur_px: float = 0.5
    #: Lambertian shading term weight. 0 = flat albedo only, 1 = full cosine.
    lambert: float = 0.7
    #: Inverse-square falloff normalisation distance in metres.
    falloff_ref_m: float = 14.0
    #: Enable projector shadow casting (surfaces occluding each other from the
    #: projector's point of view). Turning this off is useful for debugging.
    shadows: bool = True
    random_seed: int = 1234


@dataclass
class Config:
    """Top-level configuration."""

    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    patterns: PatternConfig = field(default_factory=PatternConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    animate: LayerConfig = field(default_factory=LayerConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    # ----------------------------------------------------------------- load --
    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Load a config, filling anything unspecified with defaults."""
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        return _build(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_toml(self) -> str:
        """Serialise back to TOML. Useful for dumping the effective config
        next to a scan so a result is reproducible."""
        return _to_toml(self.to_dict())


# --------------------------------------------------------------------------- #
# Generic dataclass <- dict construction
# --------------------------------------------------------------------------- #
def _build(cls: type, data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        nested = _resolve(f.type)
        if nested is not None and isinstance(value, dict):
            kwargs[name] = _build(nested, value)
        elif _is_float_tuple(f.type) and isinstance(value, list):
            kwargs[name] = tuple(float(v) for v in value)
        elif _is_tuple(f.type) and isinstance(value, list):
            # A bare `tuple` field: palettes, mode names, the build order.
            # TOML has only lists, so without this they come back as lists and
            # a loaded config is subtly not the same object as a default one.
            kwargs[name] = _as_tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


_DATACLASS_REGISTRY: dict[str, type] = {}


def _resolve(annotation: Any) -> type | None:
    """Resolve a (possibly string) annotation to a dataclass type, if it is one."""
    if isinstance(annotation, type):
        return annotation if is_dataclass(annotation) else None
    if isinstance(annotation, str):
        return _DATACLASS_REGISTRY.get(annotation.strip())
    return None


def _as_tuple(value: Any) -> Any:
    """Lists to tuples, all the way down."""
    if isinstance(value, list):
        return tuple(_as_tuple(v) for v in value)
    return value


def _is_tuple(annotation: Any) -> bool:
    """A bare ``tuple`` annotation, with no element types given."""
    return isinstance(annotation, str) and annotation.strip() == "tuple"


def _is_float_tuple(annotation: Any) -> bool:
    """A fixed-arity tuple of floats, which TOML can only give us as a list.

    Any arity, not just three: colours are triples but ranges like
    ``bulb_twinkle_hz`` are pairs, and both have to survive a round trip
    through TOML as tuples rather than lists.
    """
    if not isinstance(annotation, str):
        return False
    text = annotation.replace(" ", "")
    return bool(re.fullmatch(r"tuple\[(?:float,)*float\]", text))


for _obj in list(globals().values()):
    if isinstance(_obj, type) and is_dataclass(_obj):
        _DATACLASS_REGISTRY[_obj.__name__] = _obj


# --------------------------------------------------------------------------- #
# Minimal TOML writer (stdlib has a reader but no writer)
# --------------------------------------------------------------------------- #
def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    raise TypeError(f"cannot serialise {type(value)!r} to TOML")


def _to_toml(data: dict[str, Any], prefix: str = "") -> str:
    scalars: list[str] = []
    tables: list[str] = []
    for key, value in data.items():
        if value is None:
            scalars.append(f"# {key} = (unset)")
        elif isinstance(value, dict):
            name = f"{prefix}{key}"
            tables.append(f"\n[{name}]\n" + _to_toml(value, prefix=f"{name}."))
        else:
            scalars.append(f"{key} = {_fmt(value)}")
    return "\n".join(scalars) + ("\n" if scalars else "") + "".join(tables)
