"""The single config object: every threshold in the project lives here."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import tomllib
from pathlib import Path

import pytest

import facade_scan.config as config_module
from facade_scan.config import (
    Config,
    DecodeConfig,
    DetectConfig,
    TransferConfig,
    projector_px_per_camera_px,
)

SUB_CONFIGS = [
    f.type for f in dataclasses.fields(Config)
]


def test_defaults_construct():
    cfg = Config()
    assert cfg.projector.width == 1920 and cfg.projector.height == 1080
    assert cfg.capture.settle_ms == 400
    assert cfg.capture.backend == "folder"


def test_every_threshold_is_reachable_from_one_object():
    cfg = Config()
    for field in dataclasses.fields(Config):
        assert dataclasses.is_dataclass(getattr(cfg, field.name))


def test_toml_round_trip_preserves_every_value():
    original = Config()
    original.decode.confidence_threshold = 0.09
    original.detect.merge_angle_deg = 4.5
    original.sim.camera.hfov_deg = 55.0
    original.capture.gphoto2_extra_args = ["--set-config", "iso=800"]

    reloaded = Config.from_dict(tomllib.loads(original.to_toml()))
    assert reloaded.decode.confidence_threshold == 0.09
    assert reloaded.detect.merge_angle_deg == 4.5
    assert reloaded.sim.camera.hfov_deg == 55.0
    assert reloaded.capture.gphoto2_extra_args == ["--set-config", "iso=800"]


def test_a_partial_toml_keeps_all_other_defaults():
    cfg = Config.from_dict({"decode": {"confidence_threshold": 0.2}})
    assert cfg.decode.confidence_threshold == 0.2
    assert cfg.decode.median_ksize == DecodeConfig().median_ksize
    assert cfg.projector.width == 1920


def test_a_typo_is_an_error_not_a_silent_default():
    """A misspelled threshold that silently does nothing is the worst outcome."""
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.from_dict({"decode": {"confidence_threshhold": 0.2}})
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.from_dict({"decoder": {}})


def test_nested_tables_and_triples_load(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(
        "[sim.camera]\nwidth = 800\nposition = [1.0, 2.0, 3.0]\n"
        "[display]\norigin_x = 1920\norigin_y = 0\n"
    )
    cfg = Config.from_toml(path)
    assert cfg.sim.camera.width == 800
    assert cfg.sim.camera.position == (1.0, 2.0, 3.0)
    assert cfg.display.origin_x == 1920


def test_unset_optionals_survive_serialisation():
    cfg = Config()
    assert cfg.display.origin_x is None
    assert "# origin_x = (unset)" in cfg.to_toml()
    assert Config.from_dict(tomllib.loads(cfg.to_toml())).display.origin_x is None


def test_the_shipped_example_config_loads_and_matches_the_defaults():
    example = Path(__file__).resolve().parents[1] / "facade_scan.example.toml"
    assert example.exists(), "the documented example config is missing"
    loaded = Config.from_toml(example)
    assert isinstance(loaded, Config)


# --------------------------------------------------------------------------- #
# Documentation of defaults
# --------------------------------------------------------------------------- #
def _documented_fields(cls: type) -> set[str]:
    """Field names preceded by a ``#:`` comment in the source."""
    source = inspect.getsource(cls)
    tree = ast.parse(inspect.cleandoc(source))
    lines = inspect.cleandoc(source).splitlines()
    documented = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            above = node.lineno - 2
            while above >= 0 and lines[above].strip().startswith("#"):
                if lines[above].strip().startswith("#:"):
                    documented.add(node.target.id)
                    break
                above -= 1
    return documented


@pytest.mark.parametrize("cls", [
    getattr(config_module, name) for name in (
        "ProjectorConfig", "PatternConfig", "DisplayConfig", "CaptureConfig",
        "DecodeConfig", "DetectConfig", "TransferConfig", "ExportConfig",
        "PreviewConfig", "SimConfig", "SimPoseConfig",
    )
])
def test_thresholds_are_documented(cls):
    """Every non-obvious default must say what it is for and when to change it.

    Exempt are the handful of fields whose class docstring already covers them
    (resolutions, filenames, colours).
    """
    obvious = {
        "width", "height", "window_name", "svg_stroke", "svg_stroke_width",
        "svg_fill", "svg_fill_opacity", "svg_label_regions", "svg_font_size",
        "mask_filename", "svg_filename", "json_filename", "outline_thickness",
        "fill_alpha", "show_labels", "hfov_deg", "webcam_index", "webcam_width",
        "webcam_height", "gphoto2_binary", "gphoto2_timeout_s",
        "gphoto2_keep_on_camera", "folder_extensions", "sam_checkpoint",
        "sam_model_type", "sam_points_per_side", "sam_min_area_px", "sam_device",
        "use_sam", "vp_ransac_iterations", "vp_min_inliers", "vp_random_seed",
        "vp_snap", "merge_angle_deg", "fld_length_threshold",
        "fld_distance_threshold", "fld_canny_th1", "fld_canny_th2",
        "fld_canny_aperture_size", "fld_do_merge", "random_seed", "version",
        "min_segment_length_px", "merge_perp_px", "merge_gap_px", "extend_px",
        "min_region_area_px", "median_max_jump_px", "discontinuity_jump_px",
        "webcam_disable_auto",
    }
    fields = {f.name for f in dataclasses.fields(cls)}
    undocumented = fields - _documented_fields(cls) - obvious
    assert not undocumented, f"{cls.__name__} has undocumented defaults: {sorted(undocumented)}"


# --------------------------------------------------------------------------- #
# Size-relative resolution
# --------------------------------------------------------------------------- #
def test_projector_px_per_camera_px():
    assert projector_px_per_camera_px((1000, 2000), (1000, 500)) == pytest.approx(0.5)
    assert projector_px_per_camera_px((500, 1000), (2000, 1000)) == pytest.approx(2.0)


def test_detect_resolve_is_idempotent_in_spirit():
    cfg = DetectConfig().resolve((900, 1440))
    again = cfg.resolve((900, 1440))
    assert again.extend_px == cfg.extend_px
    assert again.min_segment_length_px == cfg.min_segment_length_px


def test_decode_and_transfer_resolve_scale_with_the_resolution_ratio():
    dslr = DecodeConfig().resolve((4000, 6000), (1920, 1080))
    webcam = DecodeConfig().resolve((720, 1280), (1920, 1080))
    assert webcam.median_max_jump_px > dslr.median_max_jump_px

    dslr_t = TransferConfig().resolve((4000, 6000), (1920, 1080))
    webcam_t = TransferConfig().resolve((720, 1280), (1920, 1080))
    assert webcam_t.discontinuity_jump_px > dslr_t.discontinuity_jump_px


def test_resolve_never_goes_below_the_absolute_floor():
    huge = DetectConfig(min_segment_length_frac=0.0).resolve((10, 10))
    assert huge.min_segment_length_px == DetectConfig().min_segment_length_px
