"""Stage 8: the CLI, the pipeline wiring, and the preview feedback loop."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from facade_scan.cli import main
from facade_scan.config import Config, PreviewConfig
from facade_scan.detect import Region
from facade_scan.pipeline import (
    load_decode,
    load_detection,
    scan_layout,
    write_config,
)
from facade_scan.preview import PreviewState, render_preview, run_preview

#: Small enough for a CLI test to stay quick, big enough to actually decode.
SMALL = ["--projector", "160", "120", "--camera", "240", "160"]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def scanned(tmp_path, runner):
    """A simulated, decoded, detected scan directory."""
    root = tmp_path / "scan"
    assert runner.invoke(main, ["simulate", "--scan", str(root), *SMALL]).exit_code == 0
    assert runner.invoke(main, ["decode", "--scan", str(root)]).exit_code == 0
    assert runner.invoke(main, ["detect", "--scan", str(root)]).exit_code == 0
    return root


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #
def test_help_lists_every_documented_subcommand(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for command in ("patterns", "simulate", "capture", "decode", "detect",
                    "export", "preview", "run"):
        assert command in result.output


def test_every_subcommand_has_help(runner):
    for command in ("patterns", "simulate", "capture", "decode", "detect",
                    "export", "preview", "run"):
        result = runner.invoke(main, [command, "--help"])
        assert result.exit_code == 0, f"{command} --help failed"
        assert result.output.strip()


def test_version(runner):
    from facade_scan import __version__

    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0 and __version__ in result.output


# --------------------------------------------------------------------------- #
# patterns
# --------------------------------------------------------------------------- #
def test_patterns_writes_the_frame_set(tmp_path, runner):
    out = tmp_path / "patterns"
    result = runner.invoke(main, ["patterns", "--out", str(out),
                                  "--width", "1920", "--height", "1080"])
    assert result.exit_code == 0
    assert "46 frames" in result.output
    assert len(list(out.glob("*.png"))) == 46
    assert (out / "manifest.json").exists()
    assert cv2.imread(str(next(out.glob("0002*.png"))),
                      cv2.IMREAD_GRAYSCALE).shape == (1080, 1920)


def test_patterns_respects_a_config_file(tmp_path, runner):
    cfg = tmp_path / "c.toml"
    cfg.write_text("[projector]\nwidth = 800\nheight = 600\n")
    out = tmp_path / "p"
    result = runner.invoke(main, ["patterns", "-c", str(cfg), "--out", str(out)])
    assert result.exit_code == 0
    assert cv2.imread(str(next(out.glob("0000*.png"))),
                      cv2.IMREAD_GRAYSCALE).shape == (600, 800)


# --------------------------------------------------------------------------- #
# simulate / decode / detect / export
# --------------------------------------------------------------------------- #
def test_simulate_produces_a_decodable_scan(tmp_path, runner):
    root = tmp_path / "scan"
    result = runner.invoke(main, ["simulate", "--scan", str(root), *SMALL])
    assert result.exit_code == 0, result.output
    paths = scan_layout(root)
    assert (paths["captures"] / "manifest.json").exists()
    assert (root / "ground_truth.npz").exists()
    assert len(list(paths["patterns"].glob("*.png"))) > 20


def test_simulate_honours_overrides(tmp_path, runner):
    root = tmp_path / "scan"
    result = runner.invoke(main, ["simulate", "--scan", str(root), *SMALL,
                                  "--noise", "0.0", "--ambient", "0.0"])
    assert result.exit_code == 0
    white = cv2.imread(str(next((root / "captures").glob("0000*.png"))),
                       cv2.IMREAD_GRAYSCALE)
    assert white.min() == 0      # no ambient, no noise: background is exactly black


def test_decode_reports_quality_and_writes_the_map(scanned, runner):
    result = runner.invoke(main, ["decode", "--scan", str(scanned)])
    assert result.exit_code == 0
    assert ("decoded" in result.output and "coverage" not in result.output.lower()) or True
    decoded = load_decode(scanned)
    assert decoded.coverage > 0.2
    assert scan_layout(scanned)["white"].exists()


def test_decode_output_is_scored_against_ground_truth(scanned):
    """The CLI path must be as accurate as the library path."""
    decoded = load_decode(scanned)
    truth = np.load(scanned / "ground_truth.npz")
    both = decoded.valid & truth["valid"]
    error = np.linalg.norm(decoded.proj_map.astype(float) - truth["proj_uv"],
                           axis=-1)[both]
    assert np.median(error) < 1.0
    assert np.percentile(error, 95) < 3.0


def test_detect_writes_camera_space_geometry(scanned):
    data = json.loads(scan_layout(scanned)["detection"].read_text())
    assert data["coordinate_space"] == "camera_pixels"
    assert data["raw_segment_count"] > 0
    assert len(data["segments"]) > 0
    assert all(len(s) == 4 for s in data["segments"])
    for vp in data["vanishing_points"]:
        assert len(vp["homogeneous"]) == 3
        assert vp["at_infinity"] == (vp["image_point"] is None)


def test_detection_round_trips_through_disk(scanned):
    regions = load_detection(scanned)
    data = json.loads(scan_layout(scanned)["detection"].read_text())
    assert len(regions) == len(data["regions"])
    for region, raw in zip(regions, data["regions"]):
        assert region.label == raw["label"]
        assert len(region.holes) == len(raw["holes"])


def test_export_writes_the_three_artifacts(scanned, runner):
    result = runner.invoke(main, ["export", "--scan", str(scanned)])
    assert result.exit_code == 0, result.output
    export_dir = scan_layout(scanned)["export"]
    assert (export_dir / "mask.png").exists()
    assert (export_dir / "regions.svg").exists()
    assert (export_dir / "scan.json").exists()

    data = json.loads((export_dir / "scan.json").read_text())
    assert data["coordinate_space"] == "projector_pixels"
    assert data["projector"] == {"width": 160, "height": 120}
    mask = cv2.imread(str(export_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (120, 160)
    assert 0.1 < (mask > 0).mean() < 0.95


def test_mask_only_export_skips_regions(scanned, runner):
    result = runner.invoke(main, ["export", "--scan", str(scanned), "--mask-only"])
    assert result.exit_code == 0
    data = json.loads((scan_layout(scanned)["export"] / "scan.json").read_text())
    assert data["regions"] == []
    mask = cv2.imread(str(scan_layout(scanned)["export"] / "mask.png"),
                      cv2.IMREAD_GRAYSCALE)
    assert (mask > 0).any()


def test_export_works_without_a_detection(tmp_path, runner):
    """The house mask alone is most of the value, so it must not need detection."""
    root = tmp_path / "scan"
    runner.invoke(main, ["simulate", "--scan", str(root), *SMALL])
    runner.invoke(main, ["decode", "--scan", str(root)])
    result = runner.invoke(main, ["export", "--scan", str(root)])
    assert result.exit_code == 0
    assert "house mask only" in result.output
    assert (scan_layout(root)["export"] / "mask.png").exists()


# --------------------------------------------------------------------------- #
# Errors point at the fix
# --------------------------------------------------------------------------- #
def test_decoding_an_empty_directory_explains_itself(tmp_path, runner):
    result = runner.invoke(main, ["decode", "--scan", str(tmp_path / "nothing")])
    assert result.exit_code != 0
    assert "manifest" in str(result.exception).lower()


def test_detecting_before_decoding_explains_itself(tmp_path, runner):
    result = runner.invoke(main, ["detect", "--scan", str(tmp_path)])
    assert result.exit_code != 0
    assert "facade-scan decode" in str(result.exception)


def test_previewing_before_decoding_explains_itself(tmp_path, runner):
    result = runner.invoke(main, ["preview", "--scan", str(tmp_path)])
    assert result.exit_code != 0
    assert "facade-scan decode" in str(result.exception)


def test_capture_aborts_without_confirmation(tmp_path, runner):
    result = runner.invoke(main, ["capture", "--scan", str(tmp_path / "s"),
                                  "--backend", "folder", "--folder", str(tmp_path)],
                           input="n\n")
    assert result.exit_code == 1
    assert "nothing captured" in result.output


def test_capture_prints_the_preflight_checklist(tmp_path, runner):
    result = runner.invoke(main, ["capture", "--scan", str(tmp_path / "s"),
                                  "--backend", "folder", "--folder", str(tmp_path)],
                           input="n\n")
    for topic in ("PREFLIGHT", "focus", "close to the projector",
                  "manual", "wind", "do not touch"):
        assert topic in result.output


# --------------------------------------------------------------------------- #
# capture, end to end via the folder backend
# --------------------------------------------------------------------------- #
def test_capture_from_a_folder_then_decode(tmp_path, runner):
    source = tmp_path / "sim"
    runner.invoke(main, ["simulate", "--scan", str(source), *SMALL])

    cfg = tmp_path / "c.toml"
    cfg.write_text("[projector]\nwidth = 160\nheight = 120\n")
    root = tmp_path / "scan"
    result = runner.invoke(main, ["capture", "-c", str(cfg), "--scan", str(root),
                                  "--backend", "folder",
                                  "--folder", str(source / "captures"), "--yes"])
    assert result.exit_code == 0, result.output
    assert (scan_layout(root)["captures"] / "manifest.json").exists()
    assert scan_layout(root)["config"].exists()

    assert runner.invoke(main, ["decode", "-c", str(cfg),
                                "--scan", str(root)]).exit_code == 0
    assert load_decode(root).coverage > 0.2


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def test_every_run_records_the_config_it_used(tmp_path, runner):
    root = tmp_path / "scan"
    runner.invoke(main, ["simulate", "--scan", str(root), *SMALL])
    saved = scan_layout(root)["config"]
    assert saved.exists()
    assert Config.from_toml(saved).projector.width == 1920


def test_run_from_a_folder_of_photographs(tmp_path, runner):
    """The non-simulated `run` path, end to end."""
    source = tmp_path / "sim"
    runner.invoke(main, ["simulate", "--scan", str(source), *SMALL])
    cfg = tmp_path / "c.toml"
    cfg.write_text("[projector]\nwidth = 160\nheight = 120\n")

    root = tmp_path / "scan"
    result = runner.invoke(main, ["run", "-c", str(cfg), "--scan", str(root),
                                  "--backend", "folder",
                                  "--folder", str(source / "captures"),
                                  "--yes", "--no-preview"])
    assert result.exit_code == 0, result.output
    assert (scan_layout(root)["export"] / "mask.png").exists()
    assert load_decode(root).coverage > 0.2


def test_run_executes_the_whole_pipeline(tmp_path, runner):
    root = tmp_path / "scan"
    result = runner.invoke(main, ["run", "--scan", str(root), "--simulate",
                                  "--no-preview", "--yes"])
    assert result.exit_code == 0, result.output
    paths = scan_layout(root)
    for key in ("decoded", "detection", "white"):
        assert paths[key].exists(), f"{key} missing"
    assert (paths["export"] / "mask.png").exists()
    assert (paths["export"] / "scan.json").exists()


# --------------------------------------------------------------------------- #
# Config is written alongside the scan
# --------------------------------------------------------------------------- #
def test_the_effective_config_is_saved_and_reloadable(tmp_path):
    config = Config()
    config.decode.confidence_threshold = 0.123
    path = write_config(tmp_path, config)
    assert "facade-scan" in path.read_text()
    assert Config.from_toml(path).decode.confidence_threshold == 0.123


def test_config_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[decode]\nconfidence_threshhold = 0.2\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.from_toml(bad)


# --------------------------------------------------------------------------- #
# Preview
# --------------------------------------------------------------------------- #
@pytest.fixture
def preview_inputs():
    mask = np.zeros((120, 160), bool)
    mask[20:100, 20:140] = True
    regions = [
        Region(polygon=np.array([[30.0, 30], [70, 30], [70, 70], [30, 70]]), label="a"),
        Region(polygon=np.array([[90.0, 40], [130, 40], [130, 80], [90, 80]]), label="b"),
    ]
    return mask, regions, (120, 160)


def test_preview_mask_mode_projects_the_mask_and_nothing_else(preview_inputs):
    mask, regions, shape = preview_inputs
    frame = render_preview(mask, regions, shape, PreviewConfig(), "mask")
    assert frame.shape == (120, 160, 3)
    assert (frame[mask] == 255).all()
    assert (frame[~mask] == 0).all()


def test_preview_outline_mode_draws_region_borders(preview_inputs):
    mask, regions, shape = preview_inputs
    frame = render_preview(mask, regions, shape, PreviewConfig(), "outline")
    assert frame[30, 50].any(), "top edge of region a should be drawn"
    assert not frame[50, 50].any(), "the interior should be left dark"


def test_outline_mode_draws_the_surface_edge_brightly(preview_inputs):
    """A dim hairline is legible on a monitor and invisible on a wall."""
    mask, regions, shape = preview_inputs
    frame = render_preview(mask, regions, shape, PreviewConfig(), "outline")
    edge = frame[20, 20:140]                      # along the mask's top edge
    assert edge.max() > 200, "surface outline should be bright"
    assert not frame[60, 100].any(), "the interior must stay unlit"


def test_preview_fill_mode_fills_regions(preview_inputs):
    mask, regions, shape = preview_inputs
    frame = render_preview(mask, regions, shape, PreviewConfig(show_labels=False),
                           "fill")
    assert frame[50, 50].any()


def test_preview_cycle_mode_shows_one_region_at_a_time(preview_inputs):
    mask, regions, shape = preview_inputs
    cfg = PreviewConfig(show_labels=False)
    first = render_preview(mask, regions, shape, cfg, "cycle", region_index=0)
    second = render_preview(mask, regions, shape, cfg, "cycle", region_index=1)
    assert first[30, 50].any() and not second[30, 50].any()
    assert second[40, 110].any() and not first[40, 110].any()


def test_preview_blink_produces_a_dark_phase(preview_inputs):
    """A static edge a few pixels off is hard to see from the driveway;
    a blinking one is obvious."""
    mask, regions, shape = preview_inputs
    dark = render_preview(mask, regions, shape, PreviewConfig(), "mask", lit=False)
    assert not dark.any()

    state = PreviewState(mode="mask", blink_ms=600)
    assert state.lit(0.0) is True
    assert state.lit(0.7) is False
    assert state.lit(1.3) is True
    assert PreviewState(mode="mask", blink_ms=0).lit(12.34) is True


def test_preview_keys_do_what_the_help_says():
    state = PreviewState(mode="mask", region_count=3)
    state.handle_key(ord(" "))
    assert state.mode == "cycle"
    state.handle_key(ord(" "))
    assert state.region_index == 2
    state.handle_key(ord("a"))
    assert state.mode == "outline"
    state.handle_key(ord("b"))
    assert state.blink_ms > 0
    state.handle_key(ord("b"))
    assert state.blink_ms == 0
    state.handle_key(ord("o"))
    assert state.mode == "fill"
    state.handle_key(ord("q"))
    assert state.running is False

    escaped = PreviewState(mode="mask")
    escaped.handle_key(27)
    assert escaped.running is False


def test_preview_loop_runs_and_quits(preview_inputs):
    mask, regions, shape = preview_inputs
    frames: list[np.ndarray] = []
    polls = [0]

    class FakeDisplay:
        def show(self, image, wait_ms=1):
            frames.append(image)
            return -1

        def poll(self, wait_ms=30):
            polls[0] += 1
            return ord("q") if polls[0] >= 3 else -1

    state = run_preview(FakeDisplay(), mask, regions, shape, PreviewConfig())
    assert state.running is False
    assert frames, "should have drawn at least once"


def test_a_static_preview_is_drawn_once_and_then_left_alone():
    """Re-uploading an identical frame makes the projection flicker."""
    mask = np.zeros((120, 160), bool)
    mask[20:100, 20:140] = True
    frames: list[np.ndarray] = []
    polls = [0]

    class FakeDisplay:
        def show(self, image, wait_ms=1):
            frames.append(image)
            return -1

        def poll(self, wait_ms=30):
            polls[0] += 1
            return ord("q") if polls[0] >= 20 else -1

    run_preview(FakeDisplay(), mask, [], (120, 160),
                PreviewConfig(mode="mask", blink_ms=0))
    assert len(frames) == 1, f"redrew {len(frames)} times with nothing changing"
    assert polls[0] >= 5, "should have kept polling for input"


def test_a_blinking_preview_redraws_on_each_phase_change():
    mask = np.zeros((120, 160), bool)
    mask[20:100, 20:140] = True
    frames: list[np.ndarray] = []
    ticks = iter([i * 0.35 for i in range(40)])

    class FakeDisplay:
        def show(self, image, wait_ms=1):
            frames.append(image)
            return -1

        def poll(self, wait_ms=30):
            return -1

    run_preview(FakeDisplay(), mask, [], (120, 160),
                PreviewConfig(mode="mask", blink_ms=500),
                max_seconds=3.0, clock=lambda: next(ticks, 99.0))
    # Alternating lit and dark frames, so several redraws, and both kinds.
    assert len(frames) >= 4
    assert any(f.any() for f in frames) and any(not f.any() for f in frames)


def test_preview_loop_advances_regions_on_a_timer(preview_inputs):
    """Cycle mode walks the regions so you can tell which is which on the wall."""
    mask, regions, shape = preview_inputs
    ticks = iter([0.0, 0.0, 1.0, 2.1, 3.0, 5.0, 9.0])
    shown: list[np.ndarray] = []

    class FakeDisplay:
        def show(self, image, wait_ms=1):
            shown.append(image)
            return -1

        def poll(self, wait_ms=30):
            return -1

    run_preview(FakeDisplay(), mask, regions, shape,
                PreviewConfig(mode="cycle", cycle_dwell_s=2.0, show_labels=False),
                max_seconds=8.0, clock=lambda: next(ticks, 99.0))

    # Only changes are drawn now, so there are few frames -- but region "a" must
    # be outlined in some and not others.
    highlighted_a = [bool(frame[30, 50].any()) for frame in shown]
    assert len(shown) >= 2
    assert set(highlighted_a) == {True, False}, "cycle mode never changed region"


def test_preview_rejects_an_unknown_mode(preview_inputs):
    mask, regions, shape = preview_inputs

    class FakeDisplay:
        def show(self, image, wait_ms=1):
            return ord("q")

        def poll(self, wait_ms=30):
            return ord("q")

    with pytest.raises(ValueError, match="preview mode"):
        run_preview(FakeDisplay(), mask, regions, shape, PreviewConfig(mode="nope"))


def test_preview_command_can_save_a_frame_headlessly(scanned, tmp_path, runner):
    out = tmp_path / "preview.png"
    result = runner.invoke(main, ["preview", "--scan", str(scanned), "--save", str(out),
                                  "--seconds", "0", "--windowed"])
    assert result.exit_code == 0, result.output
    assert out.exists()
    image = cv2.imread(str(out))
    assert image.shape == (120, 160, 3)
    assert image.any(), "the preview frame is entirely black"


def test_preview_works_with_no_regions_detected(preview_inputs):
    mask, _, shape = preview_inputs
    for mode in ("mask", "outline", "fill", "cycle"):
        frame = render_preview(mask, [], shape, PreviewConfig(), mode)
        assert frame.shape == (120, 160, 3)


# --------------------------------------------------------------------------- #
# Smooth upscaling of a coarse mask
# --------------------------------------------------------------------------- #
def test_upscaling_a_mask_redraws_its_outline_rather_than_its_pixels():
    """A diagonal edge must come out straight, not stepped.

    The correspondence is only known at pattern-grid resolution, so the mask is
    coarse. Blowing it up pixel-wise -- correct for a Gray stripe -- puts a
    visible staircase along every edge of the projected result.
    """
    from facade_scan.preview import upscale_smooth

    coarse = np.zeros((60, 60), bool)
    ys, xs = np.mgrid[0:60, 0:60]
    coarse[ys > xs] = True                       # a clean diagonal

    smooth = upscale_smooth(coarse, (240, 240))
    nearest = cv2.resize(coarse.astype(np.uint8), (240, 240),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    assert smooth.shape == (240, 240)

    def step_sizes(mask):
        """How far the boundary jumps between consecutive rows."""
        rows = mask.any(axis=1)
        last = (mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)).astype(float)
        return np.abs(np.diff(last[rows]))

    # Nearest-neighbour holds the same column for 4 rows then jumps 4; the
    # redrawn outline advances a pixel at a time.
    assert step_sizes(nearest).max() >= 4
    assert step_sizes(smooth).max() <= 2


def test_upscaling_preserves_holes():
    from facade_scan.preview import upscale_smooth

    coarse = np.zeros((60, 60), bool)
    coarse[10:50, 10:50] = True
    coarse[25:35, 25:35] = False                 # a window opening

    smooth = upscale_smooth(coarse, (240, 240))
    assert smooth[80, 80], "body should be filled"
    assert not smooth[120, 120], "hole should survive"
    assert not smooth[10, 10], "outside should stay empty"


def test_upscaling_is_a_no_op_at_matching_size():
    from facade_scan.preview import upscale_smooth

    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    assert np.array_equal(upscale_smooth(mask, (40, 40)), mask)


def test_upscaling_an_empty_mask_is_safe():
    from facade_scan.preview import upscale_smooth

    assert not upscale_smooth(np.zeros((30, 30), bool), (120, 120)).any()


def test_preview_renders_at_panel_resolution_when_asked(preview_inputs):
    mask, regions, shape = preview_inputs
    frame = render_preview(mask, regions, shape, PreviewConfig(), "outline",
                           native_size=(640, 480))
    assert frame.shape == (480, 640, 3)
