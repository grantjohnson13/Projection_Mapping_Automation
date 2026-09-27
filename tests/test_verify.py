"""Closed-loop verification, tested against a simulated optical round trip.

The verifier is checked the only way that means anything: by building a fake
rig whose true alignment is known, including one that has deliberately moved,
and confirming the numbers come back right.
"""

from __future__ import annotations

import numpy as np
import pytest

from facade_scan.config import Config
from facade_scan.decode import DecodeResult
from facade_scan.verify import (
    AlignmentReport,
    Fiducial,
    locate_response,
    measure_drift,
    plan_fiducials,
    render_fiducials,
    verify_alignment,
)

CAMERA = (400, 600)          # height, width
#: Scene texture shared by the fake rig and the decodes built against it.
FAKE_ALBEDO = np.random.default_rng(7).random((400, 600)).astype(np.float32) * 0.5 + 0.5
PROJECTOR = (200, 300)       # height, width
SCALE = 2                    # camera pixels per projector pixel


def make_decode(shift: tuple[int, int] = (0, 0)) -> DecodeResult:
    """A camera->projector map that is a clean 2x downscale, optionally offset."""
    height, width = CAMERA
    ys, xs = np.mgrid[0:height, 0:width]
    px = np.clip((xs + shift[0]) // SCALE, 0, PROJECTOR[1] - 1).astype(np.int32)
    py = np.clip((ys + shift[1]) // SCALE, 0, PROJECTOR[0] - 1).astype(np.int32)
    zeros = np.zeros((height, width), np.float32)
    white = np.zeros((height, width), np.float32)
    # Give the reference frame some texture so phase correlation has something
    # to lock onto, as a real scene does.
    white[:] = FAKE_ALBEDO
    return DecodeResult(
        proj_map=np.stack([px, py], -1), valid=np.ones((height, width), bool),
        min_confidence=zeros + 0.4, mean_confidence=zeros + 0.5,
        illumination=zeros + 0.6, white=white, black=zeros + 0.05,
        likely_glass=np.zeros((height, width), bool),
        projector_width=PROJECTOR[1], projector_height=PROJECTOR[0],
    )


class FakeRig:
    """A projector and camera whose optics are exactly a known map.

    ``truth`` is the correspondence the *physical* rig has right now. Feeding
    the verifier a decode built from a different map is how a moved rig is
    simulated.
    """

    #: A fixed camera-space texture, standing in for the scene's own markings.
    def __init__(self, truth: DecodeResult, noise: float = 0.0):
        self.truth = truth
        self.noise = noise
        self.albedo = FAKE_ALBEDO
        self.last: np.ndarray | None = None
        self.native_width = truth.projector_width
        self.native_height = truth.projector_height

    # display half
    def show(self, image, wait_ms: int = 1):
        import cv2

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.last = image
        return -1

    def poll(self, wait_ms: int = 30):
        return -1

    def describe(self) -> str:
        return "fake rig"

    # capture half: form the camera image the projected frame would produce
    def capture(self, frame):
        """Form the camera image the projected frame would produce.

        Multiplied by a fixed scene texture, so an all-white projection comes
        back looking like a scene rather than a blank field -- which is what
        drift detection needs something to lock onto.
        """
        assert self.last is not None
        seen = self.last[self.truth.proj_map[..., 1], self.truth.proj_map[..., 0]]
        seen = seen.astype(np.float32) * self.albedo
        if self.noise:
            seen = seen + np.random.default_rng(1).normal(0, self.noise * 255, seen.shape)
        return np.clip(seen, 0, 255).astype(np.uint8)


@pytest.fixture
def config():
    cfg = Config()
    cfg.capture.settle_ms = 0
    return cfg


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
def test_drift_is_zero_for_an_unmoved_scene():
    reference = make_decode().white
    (dx, dy), confidence = measure_drift(reference, reference)
    assert abs(dx) < 0.5 and abs(dy) < 0.5
    assert confidence > 0.5


def test_drift_recovers_a_known_shift():
    reference = make_decode().white
    moved = np.roll(np.roll(reference, 7, axis=1), -4, axis=0)
    (dx, dy), _ = measure_drift(reference, moved)
    assert dx == pytest.approx(7, abs=1.0)
    assert dy == pytest.approx(-4, abs=1.0)


def test_drift_rejects_a_resolution_change():
    reference = make_decode().white
    with pytest.raises(ValueError, match="camera resolution has changed"):
        measure_drift(reference, reference[::2, ::2])


# --------------------------------------------------------------------------- #
# Fiducial planning and rendering
# --------------------------------------------------------------------------- #
def test_fiducials_avoid_undecoded_ground():
    decoded = make_decode()
    decoded.valid[:, :200] = False
    probes = plan_fiducials(decoded, spacing=40, half=10, margin=30)
    assert probes
    assert all(p.camera_xy[0] > 190 for p in probes)


def test_fiducials_render_into_projector_space():
    decoded = make_decode()
    probes = [Fiducial(camera_xy=(200.0, 150.0))]
    image = render_fiducials(decoded, probes, half=10)
    assert image.shape == PROJECTOR
    assert image.any()
    ys, xs = np.nonzero(image)
    # 200,150 in camera maps to 100,75 in projector at a 2x downscale.
    assert abs(xs.mean() - 100) < 2 and abs(ys.mean() - 75) < 2


def test_locating_a_response_finds_the_centroid():
    response = np.zeros(CAMERA, np.float32)
    response[145:156, 195:206] = 1.0
    found = locate_response(response, Fiducial(camera_xy=(200.0, 150.0)))
    assert found is not None
    assert found[0] == pytest.approx(200, abs=0.6)
    assert found[1] == pytest.approx(150, abs=0.6)


def test_no_light_reports_nothing_rather_than_guessing():
    assert locate_response(np.zeros(CAMERA, np.float32),
                           Fiducial(camera_xy=(200.0, 150.0))) is None


# --------------------------------------------------------------------------- #
# End to end, against a rig whose truth is known
# --------------------------------------------------------------------------- #
def test_an_unmoved_rig_verifies_as_aligned(config):
    decoded = make_decode()
    rig = FakeRig(truth=decoded)
    report = verify_alignment(decoded, rig, rig, config, spacing=120)

    assert len(report.measured) >= 4
    assert report.aligned, report.verdict()
    assert report.median_offset_px < 3.0
    assert "Aligned" in report.verdict()[0]


def test_a_moved_rig_is_caught_and_the_shift_is_measured(config):
    """The failure that is otherwise silent.

    The scan describes the rig as it was; the rig has since shifted 12 camera
    pixels. Nothing else in the tool would notice.
    """
    scanned = make_decode()
    moved = make_decode(shift=(12, -6))
    rig = FakeRig(truth=moved)

    report = verify_alignment(scanned, rig, rig, config, spacing=120)
    assert not report.aligned
    assert report.median_offset_px > 5.0
    # A rigid shift: large bias, small scatter.
    assert np.hypot(*report.bias) > 3 * max(report.scatter_px, 0.5)
    assert any("scan again" in note.lower() for note in report.verdict())


def test_a_dark_rig_says_so_rather_than_reporting_alignment(config):
    decoded = make_decode()

    class Dark(FakeRig):
        def capture(self, frame):
            return np.zeros(CAMERA, np.uint8)

    report = verify_alignment(decoded, Dark(decoded), Dark(decoded), config,
                              spacing=120)
    assert not report.aligned
    assert "not reaching the camera" in report.verdict()[0]


def test_verification_survives_sensor_noise(config):
    decoded = make_decode()
    rig = FakeRig(truth=decoded, noise=0.02)
    report = verify_alignment(decoded, rig, rig, config, spacing=120)
    assert report.aligned, report.verdict()


# --------------------------------------------------------------------------- #
# Report arithmetic
# --------------------------------------------------------------------------- #
def test_bias_and_scatter_separate_a_shift_from_a_mess():
    rigid = AlignmentReport(fiducials=[
        Fiducial((float(x), 100.0), (float(x) + 10.0, 105.0)) for x in range(0, 200, 20)
    ])
    assert np.hypot(*rigid.bias) == pytest.approx(np.hypot(10, 5), abs=0.1)
    assert rigid.scatter_px < 0.5

    rng = np.random.default_rng(3)
    messy = AlignmentReport(fiducials=[
        Fiducial((float(x), 100.0),
                 (float(x) + rng.normal(0, 9), 100.0 + rng.normal(0, 9)))
        for x in range(0, 400, 20)
    ])
    assert np.hypot(*messy.bias) < messy.scatter_px


def test_a_report_with_no_measurements_is_not_aligned():
    report = AlignmentReport(fiducials=[Fiducial((10.0, 10.0))])
    assert not report.aligned
    assert report.median_offset_px == 0.0
    assert "No fiducial returned any light" in report.verdict()[0]


def test_a_low_confidence_drift_reading_is_not_used_to_explain_anything():
    """A weak phase-correlation peak is not evidence.

    Measured on a real low-coverage scan: drift reported 151 px at confidence
    0.20 while the fiducials showed 6 px. Believing the drift figure produced a
    confident, wrong explanation.
    """
    report = AlignmentReport(
        fiducials=[Fiducial((float(x), 100.0), (float(x) + 5.0, 101.0))
                   for x in range(0, 300, 20)],
        drift=(12.8, -150.5), drift_confidence=0.20,
    )
    assert not report.drift_is_trustworthy
    joined = " ".join(report.verdict())
    assert "not evidence of anything" in joined
    assert "no longer there" not in joined


def test_a_confident_drift_reading_that_matches_the_bias_is_used():
    report = AlignmentReport(
        fiducials=[Fiducial((float(x), 100.0), (float(x) - 10.0, 105.0))
                   for x in range(0, 400, 20)],
        drift=(-10.5, 5.4), drift_confidence=0.75,
    )
    assert report.drift_is_trustworthy
    assert any("no longer there" in n for n in report.verdict())


def test_a_confident_drift_far_larger_than_the_bias_is_not_blamed():
    """Drift and bias must agree in magnitude to be the same phenomenon."""
    report = AlignmentReport(
        fiducials=[Fiducial((float(x), 100.0), (float(x) + 4.0, 100.0))
                   for x in range(0, 400, 20)],
        drift=(200.0, 0.0), drift_confidence=0.9,
    )
    assert not any("no longer there" in n for n in report.verdict())


def test_too_few_probes_is_called_out():
    report = AlignmentReport(
        fiducials=[Fiducial((float(x), 100.0), (float(x) + 9.0, 100.0))
                   for x in range(0, 60, 20)],
    )
    assert any("too low to verify" in n for n in report.verdict())


def test_the_tolerance_respects_the_grid_quantisation(config):
    """A coarse pattern grid has a floor no scan can beat.

    The decoded map holds an integer projector coordinate per camera pixel, so
    on a 240-wide grid against a 1920-wide camera one projector pixel spans 8
    camera pixels. Judging that against a flat 3 px reported a perfect scan as
    misaligned.
    """
    decoded = make_decode()
    rig = FakeRig(truth=decoded)
    report = verify_alignment(decoded, rig, rig, config, spacing=120,
                              tolerance_px=3.0)
    expected = CAMERA[1] / PROJECTOR[1]
    assert report.quantisation_px == pytest.approx(expected)
    assert report.tolerance_px >= 0.75 * expected
    assert report.tolerance_px >= 3.0, "never below the caller's floor"


def test_a_scan_at_the_quantisation_floor_is_called_aligned():
    quantisation = 8.0
    report = AlignmentReport(
        fiducials=[Fiducial((float(x), 100.0),
                            (float(x) + (3.5 if x % 40 else -3.5), 100.0))
                   for x in range(0, 400, 20)],
        tolerance_px=max(3.0, 0.75 * quantisation),
        quantisation_px=quantisation,
    )
    assert report.aligned
    assert "floor for this pattern grid" in report.verdict()[0]


def test_a_fine_grid_keeps_the_tight_tolerance():
    report = AlignmentReport(fiducials=[], tolerance_px=3.0, quantisation_px=0.4)
    assert report.tolerance_px == 3.0
