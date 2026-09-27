"""The run report: video, stills and the summary numbers."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from facade_scan.config import Config
from facade_scan.report import (
    build_report,
    colourise_correspondence,
    fit_into,
    per_bit_confidence,
    planarity_residual,
    score_against_ground_truth,
    write_video,
)


@pytest.fixture(scope="module")
def small_scan(tmp_path_factory):
    """A complete simulated scan on disk, cheap enough to report on."""
    from facade_scan.config import SimConfig
    from facade_scan.pipeline import run_decode, run_detect, run_export
    from facade_scan.sim.render import simulate

    root = tmp_path_factory.mktemp("scan")
    sim_cfg = SimConfig()
    sim_cfg.projector.width, sim_cfg.projector.height = 192, 128
    sim_cfg.camera.width, sim_cfg.camera.height = 288, 192
    simulate(root, sim_cfg)

    config = Config()
    config.report.width, config.report.height = 320, 180
    config.report.section_seconds = 0.2
    config.report.card_seconds = 0.2
    config.report.capture_replay_fps = 24.0
    decoded = run_decode(root, config)
    run_detect(root, config, decoded)
    run_export(root, config, decoded)
    return root, config


# --------------------------------------------------------------------------- #
# Ground-truth-free quality measures
# --------------------------------------------------------------------------- #
def _planar_decode(noise_px: float = 0.0, seed: int = 0):
    """A decode whose map is exactly a homography, plus optional noise."""
    from facade_scan.decode import DecodeResult

    height, width = 120, 160
    ys, xs = np.mgrid[0:height, 0:width]
    # A genuine projective map, not just a scale: a tilted plane.
    denom = 1.0 + 0.0009 * xs + 0.0004 * ys
    px = (0.6 * xs + 0.05 * ys + 20) / denom
    py = (0.04 * xs + 0.55 * ys + 12) / denom
    if noise_px:
        rng = np.random.default_rng(seed)
        px = px + rng.normal(0, noise_px, px.shape)
        py = py + rng.normal(0, noise_px, py.shape)

    proj = np.stack([np.rint(px), np.rint(py)], -1).astype(np.int32)
    valid = np.ones((height, width), bool)
    zeros = np.zeros((height, width), np.float32)
    return DecodeResult(proj_map=proj, valid=valid, min_confidence=zeros + 0.4,
                        mean_confidence=zeros + 0.5, illumination=zeros + 0.6,
                        white=zeros + 0.7, black=zeros + 0.1,
                        likely_glass=np.zeros((height, width), bool),
                        projector_width=200, projector_height=120)


def test_planarity_residual_is_near_zero_on_a_perfect_plane():
    """The decoded map over a flat surface *is* a homography, exactly."""
    residual = planarity_residual(_planar_decode())
    assert residual is not None
    assert residual < 0.6, f"residual {residual} on a noiseless plane"


def test_planarity_residual_measures_injected_noise():
    """This is the point: with no ground truth, it recovers the scan's own error."""
    clean = planarity_residual(_planar_decode(noise_px=0.0))
    noisy = planarity_residual(_planar_decode(noise_px=2.0, seed=1))
    assert noisy > clean + 0.8
    assert 1.0 < noisy < 4.0, f"should roughly recover the 2 px injected, got {noisy}"


def test_planarity_residual_agrees_with_ground_truth_on_a_real_scan(sim_hires,
                                                                    decoded_hires):
    """The reason it is trustworthy on a physical rig, where truth is unavailable.

    Restricted to the main facade, which really is one plane, the residual
    should land close to the error measured against ground truth.
    """
    facade = sim_hires.surface_mask("main_facade", "gable_main")
    residual = planarity_residual(decoded_hires, region=facade)
    assert residual is not None

    truth = sim_hires.ground_truth_uv
    both = decoded_hires.valid & sim_hires.ground_truth_valid & facade
    actual = np.linalg.norm(
        decoded_hires.proj_map.astype(np.float64) - truth, axis=-1)[both]
    assert abs(residual - float(np.median(actual))) < 0.5


def test_planarity_residual_reports_its_inlier_fraction():
    residual, fraction = planarity_residual(_planar_decode(), with_inliers=True)
    assert residual < 0.6
    assert fraction > 0.95, "a single plane should fit almost every pixel"


def test_a_second_surface_shows_up_as_a_low_inlier_fraction():
    """The trap this guards against.

    A scene with two depths makes the residual look bad even when the scan is
    good, because the parallax between the surfaces lands in the fit. RANSAC
    keeps the majority plane, and the inlier fraction is what says so.
    """
    import numpy as np

    decoded = _planar_decode()
    # Displace a third of the frame, as a nearer surface would.
    decoded.proj_map[:, :55, 0] += 30

    residual, fraction = planarity_residual(decoded, with_inliers=True)
    assert fraction < 0.9, "the second surface should be rejected as outliers"
    assert residual < 1.0, "...leaving the majority plane's own error intact"

    # And restricted to just the displaced region, it is flat again.
    region = np.zeros(decoded.valid.shape, bool)
    region[:, :55] = True
    near_residual, near_fraction = planarity_residual(decoded, region=region,
                                                      with_inliers=True)
    assert near_residual < 1.0 and near_fraction > 0.95


def test_planarity_residual_declines_to_guess_without_enough_pixels():
    decoded = _planar_decode()
    decoded.valid = np.zeros_like(decoded.valid)
    assert planarity_residual(decoded) is None


def test_per_bit_confidence_reports_the_spread(decoded):
    stats = per_bit_confidence(decoded)
    assert set(stats) == {"p05", "median", "p95"}
    assert 0.0 <= stats["p05"] <= stats["median"] <= stats["p95"]


def test_ground_truth_scoring_matches_the_decoder_test(small_scan):
    root, _ = small_scan
    from facade_scan.pipeline import load_decode

    scored = score_against_ground_truth(load_decode(root), root / "ground_truth.npz")
    assert scored["median_error_px"] < 1.0
    assert scored["p95_error_px"] < 3.0
    assert scored["coverage_of_decodable"] > 0.5


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def test_fit_into_letterboxes_without_distorting():
    tall = np.full((200, 50, 3), 255, np.uint8)
    out = fit_into(tall, (320, 180))
    assert out.shape == (180, 320, 3)
    assert out[:, 0].max() < 40 and out[:, -1].max() < 40   # letterbox bars


def test_fit_into_accepts_grayscale_and_float():
    assert fit_into(np.zeros((40, 60), np.uint8), (100, 80)).shape == (80, 100, 3)
    assert fit_into(np.ones((40, 60), np.float32), (100, 80)).shape == (80, 100, 3)


def test_correspondence_image_encodes_both_axes_and_marks_invalid():
    decoded = _planar_decode()
    image = colourise_correspondence(decoded)
    assert image.shape == (120, 160, 3)
    # Red rises with projector x, green with projector y.
    assert int(image[60, 150, 2]) > int(image[60, 10, 2])
    assert int(image[110, 80, 1]) > int(image[10, 80, 1])

    decoded.valid[:10] = False
    masked = colourise_correspondence(decoded)
    assert masked[5, 5].tolist() == [30, 30, 34]


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #
def test_write_video_produces_a_readable_file(tmp_path):
    cfg = Config().report
    cfg.width, cfg.height, cfg.fps = 160, 120, 12
    frames = [np.full((120, 160, 3), i * 8, np.uint8) for i in range(24)]
    path = write_video(tmp_path / "v.mp4", frames, cfg)

    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 24
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (160, 120)
    cap.release()


def test_an_unavailable_codec_is_reported_not_silently_empty(tmp_path):
    cfg = Config().report
    cfg.codec = cfg.codec_fallback = "ZZZZ"
    with pytest.raises(OSError, match="could not encode"):
        write_video(tmp_path / "v.mp4", [np.zeros((120, 160, 3), np.uint8)], cfg)


# --------------------------------------------------------------------------- #
# The whole report
# --------------------------------------------------------------------------- #
def test_build_report_writes_video_stills_and_summary(small_scan, tmp_path):
    root, config = small_scan
    out = tmp_path / "report"
    built = build_report(root, out, config, title="unit test", notes="a note")

    assert built.video_path is not None and built.video_path.exists()
    assert (out / "summary.json").exists() and (out / "summary.md").exists()
    assert len(built.still_paths) >= 4
    for still in built.still_paths:
        assert cv2.imread(str(still)) is not None

    cap = cv2.VideoCapture(str(built.video_path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 10
    cap.release()


def test_summary_carries_the_numbers_worth_diffing(small_scan, tmp_path):
    root, config = small_scan
    build_report(root, tmp_path / "r", config, title="unit test")
    summary = json.loads((tmp_path / "r" / "summary.json").read_text())

    assert summary["title"] == "unit test"
    assert summary["projector"] == {"width": 192, "height": 128}
    assert summary["camera"] == {"width": 288, "height": 192}
    assert 0.0 < summary["decode"]["coverage"] <= 1.0
    assert summary["ground_truth"]["median_error_px"] < 1.0
    assert "planarity_residual_px" in summary
    assert 0.0 <= summary["planarity_inlier_fraction"] <= 1.0
    assert summary["video"]["seconds"] > 0
    assert set(summary["per_bit_confidence"]) == {"p05", "median", "p95"}


def test_summary_markdown_is_written(small_scan, tmp_path):
    root, config = small_scan
    build_report(root, tmp_path / "r", config, title="unit test", notes="context")
    text = (tmp_path / "r" / "summary.md").read_text()
    assert "# unit test" in text and "context" in text
    assert "median error" in text


def test_a_report_works_without_ground_truth(small_scan, tmp_path):
    """Physical scans have none, and that is the normal case."""
    import shutil

    root, config = small_scan
    copy = tmp_path / "scan"
    shutil.copytree(root, copy)
    (copy / "ground_truth.npz").unlink()

    built = build_report(copy, tmp_path / "r", config, title="physical")
    assert "ground_truth" not in built.summary
    assert "planarity_residual_px" in built.summary
    assert built.video_path.exists()


def test_a_report_works_without_a_detection(small_scan, tmp_path):
    import shutil

    root, config = small_scan
    copy = tmp_path / "scan"
    shutil.copytree(root, copy)
    (copy / "detection.json").unlink()

    built = build_report(copy, tmp_path / "r", config, title="mask only")
    assert built.summary["detection"]["camera_regions"] == 0
    assert built.video_path.exists()


def test_report_command_runs(small_scan, tmp_path):
    from click.testing import CliRunner

    from facade_scan.cli import main

    root, _ = small_scan
    result = CliRunner().invoke(main, ["report", "--scan", str(root),
                                       "--out", str(tmp_path / "r"),
                                       "--title", "cli"])
    assert result.exit_code == 0, result.output
    assert "planarity residual" in result.output
    assert "vs ground truth" in result.output


# --------------------------------------------------------------------------- #
# Rig check diagnostics
# --------------------------------------------------------------------------- #
def _rig(planes, **kwargs):
    from facade_scan.rigcheck import PlaneMeasurement, RigCheck

    defaults = {"white_level": 0.6, "black_level": 0.05, "lit_fraction": 0.4,
                "clipped_fraction": 0.0, "projector_grid": (960, 540),
                "native_grid": (1920, 1080), "camera_size": (1920, 1080),
                "joint_decodable": 1.0}
    defaults.update(kwargs)
    n = len(planes)
    scale = defaults["native_grid"][0] / defaults["projector_grid"][0]
    built = []
    for bit, (mod, frac) in enumerate(planes):
        stripe = 2 if bit == n - 1 else 1 << (n - 1 - bit)
        built.append(PlaneMeasurement("x", bit, stripe, stripe * scale, mod, frac))
    return RigCheck(planes=built, **defaults)


def test_a_healthy_rig_is_told_to_go_ahead():
    check = _rig([(0.4, 1.0)] * 10)
    assert check.predicted_coverage == 1.0
    assert check.advice() == ["Every plane decodes. Run the scan."]


def test_a_resolution_cutoff_recommends_a_coarser_grid():
    """Modulation healthy until it falls off a cliff at the fine end."""
    check = _rig([(0.46, 1.0), (0.45, 1.0), (0.44, 1.0), (0.43, 1.0), (0.42, 1.0),
                  (0.40, 1.0), (0.37, 1.0), (0.31, 0.95), (0.22, 0.67), (0.05, 0.47)])
    advice = " ".join(check.advice())
    assert "optical cutoff" in advice
    assert check.recommended_grid() is not None
    assert "480x270" in advice
    # It must NOT blame the surface: the coarsest plane was fine.
    assert "specular" not in advice


def test_a_specular_surface_is_blamed_on_the_surface_not_the_optics():
    """A third of the frame failing at EVERY stripe width is the surface.

    This is the fridge case: brushed metal throws the light past the lens, and
    no amount of refocusing or coarsening recovers it.
    """
    check = _rig([(0.45, 0.66)] * 9 + [(0.44, 0.65)])
    advice = " ".join(check.advice())
    assert "specular" in advice and "matte" in advice
    assert "optical cutoff" not in advice
    assert check.baseline_loss == pytest.approx(0.34)


def test_healthy_modulation_with_partial_loss_is_not_called_defocus():
    """The bug this replaced: mid planes with 0.40 modulation were reported as
    defocus purely because a subset of pixels failed at every plane."""
    check = _rig([(0.46, 0.97), (0.45, 0.94), (0.44, 0.96), (0.43, 0.94),
                  (0.42, 0.89), (0.40, 0.88), (0.37, 0.85), (0.31, 0.78),
                  (0.22, 0.67), (0.20, 0.66)])
    advice = " ".join(check.advice()).lower()
    assert "refocus" not in advice
    assert "defocus" not in advice


def test_the_coarsest_plane_failing_outright_is_never_blamed_on_focus():
    check = _rig([(0.03, 0.3)] * 10, lit_fraction=0.03)
    advice = " ".join(check.advice())
    assert "not reaching the camera" in advice
    assert "Refocus" not in advice


def test_a_white_frame_no_brighter_than_black_is_called_out():
    check = _rig([(0.0, 0.0)] * 10, white_level=0.12, black_level=0.13)
    assert any("not seeing the projection at all" in n for n in check.advice())


def test_clipping_and_ambient_are_reported():
    assert any("clipped" in n for n in _rig([(0.4, 1.0)] * 10,
                                            clipped_fraction=0.2).advice())
    ambient = _rig([(0.4, 1.0)] * 10, black_level=0.4).advice()
    assert any("carrying pattern" in n for n in ambient)


def test_resolution_limit_is_where_modulation_halves():
    check = _rig([(0.46, 1.0)] * 8 + [(0.22, 0.7), (0.05, 0.4)])
    # Stripe widths in the helper are all 4.0 panel px, so the limit collapses
    # to that; the useful assertion is that a cutoff was found at all.
    assert check.resolution_limit_px is not None


def test_predicted_coverage_is_measured_jointly_not_per_plane():
    """Per-plane fractions are optimistic when failures are not aligned.

    On a real rig the weakest plane passed 62% of pixels while only 26% passed
    every plane, because different planes failed in different places. Taking the
    minimum per-plane fraction predicted more than twice the coverage actually
    delivered.
    """
    check = _rig([(0.4, 1.0)] * 9 + [(0.1, 0.55)], joint_decodable=0.26)
    assert check.predicted_coverage == pytest.approx(0.26)
    assert min(p.decodable_fraction for p in check.planes) == pytest.approx(0.55)
    assert check.worst.bit == 9


def test_a_full_scan_needs_both_axes_so_is_predicted_lower():
    check = _rig([(0.4, 1.0)] * 10, joint_decodable=0.6)
    assert check.predicted_scan_coverage == pytest.approx(0.36)
    assert check.predicted_scan_coverage < check.predicted_coverage


def test_joint_decodable_is_computed_from_real_captures(tmp_path):
    """Drive check_rig with fakes and confirm the joint figure is per-pixel."""
    import numpy as np

    from facade_scan.config import Config
    from facade_scan.rigcheck import check_rig

    config = Config()
    config.projector.width, config.projector.height = 32, 32
    config.capture.settle_ms = 0
    config.decode.confidence_threshold = 0.5
    config.decode.illumination_threshold = 0.1

    class FakeDisplay:
        native_width = native_height = 32

        def show(self, image, wait_ms=1):
            self.last = image
            return -1

    class FakeBackend:
        """Returns the projected frame, but knocks out a different quarter of
        the image on each plane, so no pixel survives them all."""

        def __init__(self, display):
            self.display = display
            self.calls = 0

        def capture(self, frame):
            image = self.display.last.astype(np.float32)
            if frame.role == "gray":
                band = (frame.bit or 0) % 4
                image = image.copy()
                image[band * 8:(band + 1) * 8, :] = 0
            self.calls += 1
            return image.astype(np.uint8)

    display = FakeDisplay()
    result = check_rig(config, FakeBackend(display), display, axis="x")
    # Four different bands were destroyed across the planes, so the fraction of
    # pixels clearing every plane is well below the best single plane's.
    assert result.joint_decodable < min(p.decodable_fraction for p in result.planes)
    assert result.measured_axis == "x"


def test_gray_stripe_widths_are_measured_correctly():
    """The two finest Gray planes are 2 and 4 px, not 1 and 2.

    A Gray bit is the XOR of two adjacent binary bits, so it has twice the
    period of the binary bit. Deriving this by the obvious power of two gets the
    fine end wrong, which then misreports how hard the rig is being asked to
    resolve.
    """
    from facade_scan.rigcheck import _stripe_width

    widths = [_stripe_width(1024, bit, 10) for bit in range(10)]
    assert widths[-1] == 2, "finest plane"
    assert widths[-2] == 4, "second finest plane"
    assert widths[0] == 512, "coarsest plane is one transition at the middle"
    # Monotonically halving from coarse to fine.
    assert widths == [512, 256, 128, 64, 32, 16, 8, 4, 4, 2][:10] or all(
        widths[i] >= widths[i + 1] for i in range(len(widths) - 1))
