"""Stage 3: the decoder, scored end to end against simulator ground truth.

The headline test is :func:`test_decoded_map_matches_ground_truth`. Everything
downstream -- detection, transfer, export -- is only as good as this number.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from facade_scan.config import DecodeConfig
from facade_scan.decode import (
    DecodeResult,
    array_loader,
    decode,
    decode_directory,
    discontinuity_mask,
    inpaint_map,
    median_filter_map,
)
from facade_scan.patterns import build_manifest

from .conftest import build_sim


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run_decode(sim, cfg: DecodeConfig | None = None) -> DecodeResult:
    return decode(array_loader(sim.images), sim.manifest, cfg or DecodeConfig())


def errors_against_truth(sim, result: DecodeResult) -> np.ndarray:
    """Euclidean projector-pixel error where decoder and ground truth agree a
    pixel is decodable."""
    both = result.valid & sim.ground_truth_valid
    return np.linalg.norm(
        result.proj_map.astype(np.float64) - sim.ground_truth_uv, axis=-1
    )[both]


# --------------------------------------------------------------------------- #
# The headline accuracy test
# --------------------------------------------------------------------------- #
def test_decoded_map_matches_ground_truth(sim, decoded):
    """Median error under 1 projector pixel, 95th percentile under 3."""
    err = errors_against_truth(sim, decoded)
    assert err.size > 10_000
    assert np.median(err) < 1.0, f"median error {np.median(err):.3f} px"
    assert np.percentile(err, 95) < 3.0, f"p95 error {np.percentile(err, 95):.3f} px"


def test_accuracy_also_holds_with_real_projector_shadows(sim_wide_baseline):
    """Same bar, on a scene where the bump-out casts genuine shadows."""
    result = run_decode(sim_wide_baseline)
    err = errors_against_truth(sim_wide_baseline, result)
    assert sim_wide_baseline.cache.shadowed.sum() > 200
    assert np.median(err) < 1.0
    assert np.percentile(err, 95) < 3.0


def test_the_accuracy_test_cannot_be_passed_by_decoding_nothing(sim, decoded):
    """Guard the guard: near-total coverage on every surface that can decode.

    Without this, a decoder that marked all but a handful of easy pixels invalid
    would sail through the error thresholds above.
    """
    decodable = sim.ground_truth_valid & ~sim.glass_mask
    assert decoded.valid[decodable].mean() > 0.97
    assert decoded.coverage > 0.4  # the house does not fill the whole frame


def test_error_is_dominated_by_quantisation_not_by_mistakes(sim, decoded):
    """A perfect decoder still has ~0.38 px median error, because it returns
    integer projector pixels and the truth is continuous. We should be near it."""
    err = errors_against_truth(sim, decoded)
    assert np.median(err) < 0.6
    assert (err < 2.0).mean() > 0.999


# --------------------------------------------------------------------------- #
# Validity: the decoder must refuse rather than invent
# --------------------------------------------------------------------------- #
def test_nothing_is_decoded_where_there_is_no_house(sim, decoded):
    assert not decoded.valid[~sim.cache.hit].any()


def test_deep_shadow_is_marked_invalid_not_decoded(sim_wide_baseline):
    """Erode to the shadow's interior first.

    Projector defocus spills real light a pixel or two past a shadow edge, so
    the rim genuinely is partly lit and can legitimately decode or be filled in
    from its neighbours. Only the interior is unambiguous, and the shadows in
    this scene are only a few pixels wide.
    """
    import cv2

    result = run_decode(sim_wide_baseline)
    shadow = (sim_wide_baseline.cache.shadowed & sim_wide_baseline.cache.hit)
    assert shadow.sum() > 200
    interior = cv2.erode(shadow.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert interior.sum() > 50
    assert result.valid[interior].mean() < 0.05


def test_inpainting_does_not_leak_into_shadow_or_sky(sim_wide_baseline):
    """The illumination guard, checked on a rendered scene rather than a fixture."""
    off = run_decode(sim_wide_baseline, DecodeConfig(inpaint_radius_px=0))
    on = run_decode(sim_wide_baseline, DecodeConfig(inpaint_radius_px=8))
    background = ~sim_wide_baseline.cache.hit
    assert not on.valid[background].any(), "filled into empty space"
    assert on.coverage > off.coverage, "inpainting should still gain coverage"


def test_every_valid_coordinate_is_inside_the_projector_panel(sim, decoded):
    """11 Gray bits address 2048 columns but a 1920 panel has 1920."""
    xs = decoded.proj_map[..., 0][decoded.valid]
    ys = decoded.proj_map[..., 1][decoded.valid]
    assert xs.min() >= 0 and xs.max() < decoded.projector_width
    assert ys.min() >= 0 and ys.max() < decoded.projector_height


def test_dark_surfaces_lose_coverage_but_not_accuracy():
    """The documented trade-off on dark brick: fewer pixels, not worse ones."""
    dark = build_sim()
    for surface in dark.scene.surfaces:
        surface.albedo *= 0.18
    dark = build_sim(scene=dark.scene)

    result = run_decode(dark)
    decodable = dark.ground_truth_valid & ~dark.glass_mask
    coverage = result.valid[decodable].mean()
    err = errors_against_truth(dark, result)

    assert 0.2 < coverage < 0.95, f"expected partial coverage on dark brick, got {coverage}"
    assert np.median(err) < 1.0
    assert np.percentile(err, 95) < 3.0


@pytest.fixture(scope="module")
def dark_house():
    """The same house with everything scaled to near-black brick."""
    base = build_sim()
    for surface in base.scene.surfaces:
        surface.albedo *= 0.15
    return build_sim(scene=base.scene)


def test_lowering_the_confidence_threshold_buys_coverage_at_a_price(dark_house):
    """A low threshold decodes far more pixels, and some of them are wrong.

    A pixel decoded from noise does not land *near* the right projector pixel;
    it lands somewhere else entirely, and throws light onto an unrelated part of
    the house. That is why the default threshold is conservative.
    """
    strict = run_decode(dark_house, DecodeConfig(confidence_threshold=0.06))
    loose = run_decode(dark_house, DecodeConfig(confidence_threshold=0.005,
                                                illumination_threshold=0.005))
    strict_err = errors_against_truth(dark_house, strict)
    loose_err = errors_against_truth(dark_house, loose)

    assert loose.coverage > 2.0 * strict.coverage
    # The price is in the tail, not the typical case: the worst pixel gets worse.
    assert loose_err.max() > 2.0 * strict_err.max()
    # ...but both still clear the project-wide accuracy bar, because the outlier
    # filter contains the damage. See the next test.
    for err in (strict_err, loose_err):
        assert np.median(err) < 1.0
        assert np.percentile(err, 95) < 3.0


def test_the_outlier_filter_is_what_contains_gross_errors(dark_house):
    """Measure what the filter actually buys, on real decoded data.

    Unfiltered, a permissive decode on near-black brick scatters coordinates
    that are wrong by a hundred pixels or more. Those are the ones that matter:
    they do not blur the projection, they light up the wrong wall.
    """
    permissive = DecodeConfig(confidence_threshold=0.005, illumination_threshold=0.005,
                              inpaint_radius_px=0)
    raw = run_decode(dark_house, replace(permissive, median_ksize=0))
    filtered = run_decode(dark_house, permissive)

    raw_err = errors_against_truth(dark_house, raw)
    filtered_err = errors_against_truth(dark_house, filtered)

    assert raw_err.max() > 50.0, "no gross errors present; test is not exercising anything"
    assert filtered_err.max() < 0.1 * raw_err.max()
    assert filtered_err.max() < 10.0
    # Coverage is untouched: the filter corrects coordinates, it does not drop pixels.
    assert filtered.coverage == raw.coverage
    assert np.percentile(raw_err, 99.9) > np.percentile(filtered_err, 99.9)


def test_badly_defocused_projector_fails_loudly_rather_than_silently():
    """Losing focus should collapse coverage, not quietly return wrong pixels."""
    blurry = build_sim(blur_px=3.0)
    result = run_decode(blurry)
    decodable = blurry.ground_truth_valid & ~blurry.glass_mask
    assert result.valid[decodable].mean() < 0.3
    err = errors_against_truth(blurry, result)
    if err.size:
        assert np.percentile(err, 95) < 3.0


# --------------------------------------------------------------------------- #
# Why the inverse frames exist
# --------------------------------------------------------------------------- #
def test_streetlight_ambient_does_not_move_the_decode(sim, decoded):
    """Flooding the scene with ambient light must not change a single coordinate.

    This is the payoff for doubling the frame count: pattern-minus-inverse is
    blind to anything that lights both exposures equally.
    """
    flooded = build_sim(ambient=0.55)
    result = run_decode(flooded)
    shared = result.valid & decoded.valid
    assert shared.sum() > 10_000
    agree = (result.proj_map[shared] == decoded.proj_map[shared]).all(axis=-1)
    assert agree.mean() > 0.999


def test_heavy_sensor_noise_barely_moves_the_decode(sim, decoded):
    noisy = build_sim(noise_sigma=0.05)
    result = run_decode(noisy)
    shared = result.valid & decoded.valid
    agree = (result.proj_map[shared] == decoded.proj_map[shared]).all(axis=-1)
    assert agree.mean() > 0.99


# --------------------------------------------------------------------------- #
# likely_glass
# --------------------------------------------------------------------------- #
def test_likely_glass_finds_the_windows(sim, decoded):
    """Bright but unmodulated is glazing. Free signal, straight out of decode."""
    glass = sim.glass_mask
    flagged = decoded.likely_glass
    assert flagged.sum() > 1000
    recall = flagged[glass & sim.cache.hit].mean()
    precision = glass[flagged].mean()
    assert recall > 0.9, f"glass recall {recall:.3f}"
    assert precision > 0.9, f"glass precision {precision:.3f}"


def test_likely_glass_does_not_fire_on_shadow_or_background(sim_wide_baseline):
    import cv2

    result = run_decode(sim_wide_baseline)
    cache = sim_wide_baseline.cache
    assert result.likely_glass[~cache.hit].mean() < 0.01

    # Erode away the penumbra first. Projector defocus spills real light a pixel
    # or two past a shadow edge, so the rim of a shadow genuinely is partly lit
    # and partly modulated; only the interior is unambiguous.
    shadow = (cache.shadowed & cache.hit & ~sim_wide_baseline.glass_mask).astype(np.uint8)
    interior = cv2.erode(shadow, np.ones((3, 3), np.uint8)).astype(bool)
    assert interior.sum() > 50
    assert result.likely_glass[interior].mean() < 0.05


def test_likely_glass_does_not_fire_on_merely_dark_masonry():
    """Dark brick decodes badly too, but it is *dark*; glass is bright."""
    sim = build_sim()
    for surface in sim.scene.surfaces:
        if "window" not in surface.name:
            surface.albedo = 0.10
    sim = build_sim(scene=sim.scene)
    result = run_decode(sim)
    masonry = sim.surface_mask("main_facade") & sim.cache.lit
    assert result.likely_glass[masonry].mean() < 0.15


# --------------------------------------------------------------------------- #
# Median filter
# --------------------------------------------------------------------------- #
def test_median_filter_removes_isolated_errors():
    """Errors spaced far enough apart to be genuinely isolated are erased exactly."""
    truth = np.tile(np.arange(40, dtype=np.int32), (40, 1))
    proj = np.stack([truth, truth], axis=-1)
    valid = np.ones((40, 40), dtype=bool)

    corrupted = proj.copy()
    ys, xs = np.meshgrid(np.arange(4, 37, 6), np.arange(4, 37, 6))
    corrupted[ys, xs, 0] += 25

    cleaned = median_filter_map(corrupted, valid, ksize=5, max_jump=6.0)
    assert np.abs(corrupted - proj).max() == 25
    assert np.abs(cleaned - proj).max() == 0


def test_median_filter_removes_gross_errors_a_jump_gate_alone_could_not():
    """A +5000 px error has no neighbour within max_jump, so the gate rejects
    them all. The support test is what catches it."""
    truth = np.tile(np.arange(40, dtype=np.int32), (40, 1))
    proj = np.stack([truth, truth], axis=-1)
    valid = np.ones((40, 40), dtype=bool)
    corrupted = proj.copy()
    corrupted[20, 20, 0] += 5000

    cleaned = median_filter_map(corrupted, valid, ksize=5, max_jump=6.0)
    assert cleaned[20, 20, 0] == truth[20, 20]


def test_median_filter_leaves_a_clean_map_completely_alone():
    """No shifting at depth steps, at the frame border, or anywhere else."""
    truth = np.tile(np.arange(40, dtype=np.int32), (40, 1))
    proj = np.stack([truth, truth], axis=-1)
    valid = np.ones((40, 40), dtype=bool)
    assert np.array_equal(median_filter_map(proj, valid, 5, 6.0), proj)


def test_median_filter_preserves_a_thin_surface_at_another_depth():
    """The reason for max_jump.

    A garage return seen nearly edge-on, or a deep window reveal, is only a
    couple of camera pixels wide but sits at a completely different projector
    coordinate. Without the jump gate the surrounding wall outvotes it and it is
    erased -- and erasing it means projecting that strip of light onto the wrong
    surface.
    """
    truth = np.tile(np.arange(30, dtype=np.int32), (40, 1))
    truth[:, 14:16] += 60                   # a two-pixel-wide strip, 60 px away
    proj = np.stack([truth, truth], axis=-1)
    valid = np.ones_like(truth, dtype=bool)

    cleaned = median_filter_map(proj, valid, ksize=5, max_jump=6.0)
    assert np.array_equal(cleaned, proj), "the thin surface was modified"

    erased = median_filter_map(proj, valid, ksize=5, max_jump=1e9)
    assert not np.array_equal(erased[:, 14:16], proj[:, 14:16]), \
        "without the jump gate the strip should be overwritten, proving the test bites"


def test_median_filter_ignores_invalid_neighbours_and_leaves_invalid_pixels_alone():
    truth = np.tile(np.arange(30, dtype=np.int32), (30, 1))
    proj = np.stack([truth, truth], axis=-1)
    valid = np.ones((30, 30), dtype=bool)
    valid[10:20, 10:20] = False
    proj[10:20, 10:20] = -999            # garbage under the invalid mask

    cleaned = median_filter_map(proj, valid, ksize=5, max_jump=6.0)
    assert np.array_equal(cleaned[10:20, 10:20], proj[10:20, 10:20])  # untouched
    border = cleaned[8:10, 5:9, 0]
    assert np.array_equal(border, truth[8:10, 5:9])                   # uncontaminated


def test_median_filter_rejects_even_or_tiny_kernels():
    proj = np.zeros((5, 5, 2), dtype=np.int32)
    valid = np.ones((5, 5), dtype=bool)
    for bad in (2, 4, 1):
        with pytest.raises(ValueError, match="odd number"):
            median_filter_map(proj, valid, ksize=bad, max_jump=1.0)


def test_row_chunking_does_not_change_the_result():
    rng = np.random.default_rng(3)
    truth = np.tile(np.arange(64, dtype=np.int32), (70, 1))
    proj = np.stack([truth, truth], axis=-1)
    proj[rng.integers(0, 70, 40), rng.integers(0, 64, 40), 0] += 30
    valid = np.ones((70, 64), dtype=bool)
    a = median_filter_map(proj, valid, 5, 6.0, row_chunk=8)
    b = median_filter_map(proj, valid, 5, 6.0, row_chunk=1000)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Discontinuity mask
# --------------------------------------------------------------------------- #
def test_discontinuity_mask_finds_the_bump_out_edge(sim_wide_baseline):
    result = run_decode(sim_wide_baseline)
    mask = discontinuity_mask(result.proj_map, result.valid, jump_px=3.0)
    assert mask.sum() > 50

    surface = sim_wide_baseline.cache.surface
    near_boundary = np.zeros_like(mask)
    for axis in (0, 1):
        d = np.diff(surface, axis=axis) != 0
        pad = [(0, 0), (0, 0)]
        pad[axis] = (0, 1)
        near_boundary |= np.pad(d, pad)
        pad[axis] = (1, 0)
        near_boundary |= np.pad(d, pad)
    import cv2
    near_boundary = cv2.dilate(near_boundary.astype(np.uint8),
                               np.ones((5, 5), np.uint8)).astype(bool)
    assert near_boundary[mask].mean() > 0.9


def test_discontinuity_mask_is_empty_on_a_smooth_map():
    truth = np.tile(np.arange(40, dtype=np.int32), (40, 1))
    proj = np.stack([truth, truth], axis=-1)
    valid = np.ones((40, 40), dtype=bool)
    assert not discontinuity_mask(proj, valid, jump_px=5.0).any()


# --------------------------------------------------------------------------- #
# Persistence and I/O
# --------------------------------------------------------------------------- #
def test_decode_result_round_trips_through_disk(tmp_path, decoded):
    path = decoded.save(tmp_path / "decoded.npz")
    again = DecodeResult.load(path)
    assert np.array_equal(again.proj_map, decoded.proj_map)
    assert np.array_equal(again.valid, decoded.valid)
    assert np.array_equal(again.likely_glass, decoded.likely_glass)
    assert (again.projector_width, again.projector_height) == \
           (decoded.projector_width, decoded.projector_height)


def test_stats_are_reportable(decoded):
    stats = decoded.stats()
    assert 0.0 < stats["coverage"] <= 1.0
    assert stats["valid_pixels"] > 0
    assert stats["projector_width"] == decoded.projector_width


def test_decode_directory_reads_a_simulated_scan_from_disk(tmp_path):
    from facade_scan.config import SimConfig
    from facade_scan.sim.render import simulate

    cfg = SimConfig()
    cfg.projector.width, cfg.projector.height = 160, 120
    cfg.camera.width, cfg.camera.height = 240, 160
    result = simulate(tmp_path, cfg)

    decoded = decode_directory(result.capture_dir)
    assert decoded.shape == (160, 240)
    assert decoded.coverage > 0.3

    gt = np.load(result.ground_truth_path)
    both = decoded.valid & gt["valid"]
    err = np.linalg.norm(decoded.proj_map.astype(float) - gt["proj_uv"], axis=-1)[both]
    assert np.median(err) < 1.0
    assert np.percentile(err, 95) < 3.0


def test_missing_manifest_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        decode_directory(tmp_path)


def test_mismatched_capture_size_is_rejected(sim):
    images = dict(sim.images)
    bad = sim.manifest.frames_for("x", False)[0]
    images[bad.index] = images[bad.index][:-5, :]
    with pytest.raises(ValueError, match="every frame in a scan must be the same size"):
        decode(array_loader(images), sim.manifest)


def test_a_capture_set_that_does_not_match_its_manifest_is_rejected(tmp_path):
    """A dropped photo must be an error, not a plausible-looking wrong map."""
    from facade_scan.config import SimConfig
    from facade_scan.sim.render import simulate

    cfg = SimConfig()
    cfg.projector.width, cfg.projector.height = 64, 64
    cfg.camera.width, cfg.camera.height = 120, 90
    result = simulate(tmp_path, cfg)

    decode_directory(result.capture_dir)  # intact: fine
    next(result.capture_dir.glob("00*.png")).unlink()
    with pytest.raises(ValueError, match="manifest describes"):
        decode_directory(result.capture_dir)


def test_decoding_with_a_manifest_for_another_projector_is_caught(tmp_path):
    """Patterns generated for a different panel must not decode silently."""
    from facade_scan.config import SimConfig
    from facade_scan.sim.render import simulate

    cfg = SimConfig()
    cfg.projector.width, cfg.projector.height = 64, 64
    cfg.camera.width, cfg.camera.height = 120, 90
    result = simulate(tmp_path, cfg)

    with pytest.raises(ValueError, match="manifest describes"):
        decode_directory(result.capture_dir, manifest=build_manifest(320, 200))


# --------------------------------------------------------------------------- #
# Guarded inpainting
# --------------------------------------------------------------------------- #
def _ramp(size=60, gradient=0.5):
    """A decoded map with a realistic gradient: half a projector pixel per
    camera pixel, which is what a camera out-resolving the projector gives."""
    ys, xs = np.mgrid[0:size, 0:size]
    truth = np.rint(gradient * xs + 0.25 * gradient * ys).astype(np.int32)
    return np.stack([truth, truth], axis=-1)


def test_inpainting_fills_a_hole_in_a_smooth_surface():
    proj = _ramp()
    valid = np.ones((60, 60), bool)
    valid[25:31, 25:31] = False          # a 6x6 dropout mid-wall

    filled, known = inpaint_map(proj, valid, radius=6, agreement=4.0)
    assert known[25:31, 25:31].all(), "the hole should be filled"
    assert np.abs(filled[25:31, 25:31] - proj[25:31, 25:31]).max() <= 2


def test_inpainting_never_blends_two_surfaces_together():
    """The guarantee that matters at a depth step.

    A gap at the edge of a bump-out may legitimately be filled by *extending*
    one surface into it. What must never happen is a value blended between the
    two, belonging to neither -- that would throw light into the gap between
    them, landing on nothing.
    """
    proj = _ramp()
    proj[:, 30:] += 80                    # a hard step down the middle
    valid = np.ones((60, 60), bool)
    valid[:, 28:32] = False               # gap straddling the step

    filled, known = inpaint_map(proj, valid, radius=6, agreement=4.0)
    gap = np.zeros((60, 60), bool)
    gap[:, 28:32] = True
    newly = gap & known
    assert newly.any(), "the rim of the gap should still fill from one side"

    row = 30
    left_value = float(proj[row, 20, 0])
    right_value = float(proj[row, 45, 0])
    for column in range(28, 32):
        if not known[row, column]:
            continue
        value = float(filled[row, column, 0])
        near_left = abs(value - left_value) < 12
        near_right = abs(value - right_value) < 12
        assert near_left or near_right, (
            f"column {column} filled to {value}, between the surfaces at "
            f"{left_value} and {right_value}"
        )


def test_inpainting_stays_inside_the_illuminated_area():
    """Without this it leaks off the house into the sky."""
    proj = _ramp()
    valid = np.ones((60, 60), bool)
    valid[:, 40:] = False                 # undecoded
    allowed = np.ones((60, 60), bool)
    allowed[:, 45:] = False               # ...and unlit beyond column 45

    _, known = inpaint_map(proj, valid, radius=10, agreement=4.0, allowed=allowed)
    assert known[:, 40:44].any(), "lit-but-undecoded area should fill"
    assert not known[:, 45:].any(), "filled into unlit space"


def test_inpainting_is_bounded_by_its_radius():
    """A large undecoded region keeps an honestly unknown core."""
    proj = _ramp(80)
    valid = np.ones((80, 80), bool)
    valid[20:60, 20:60] = False           # 40x40 dropout

    _, known = inpaint_map(proj, valid, radius=3, agreement=4.0)
    assert known[22, 22], "the rim should fill"
    assert not known[40, 40], "the centre is too far in to invent"


def test_inpainting_can_be_switched_off():
    proj = np.zeros((20, 20, 2), np.int32)
    valid = np.ones((20, 20), bool)
    valid[5:8, 5:8] = False
    _, known = inpaint_map(proj, valid, radius=0, agreement=4.0)
    assert np.array_equal(known, valid)


def test_inpainting_leaves_a_fully_decoded_map_alone():
    truth = np.tile(np.arange(40, dtype=np.int32), (40, 1))
    proj = np.stack([truth, truth], axis=-1)
    valid = np.ones((40, 40), bool)
    filled, known = inpaint_map(proj, valid, radius=6, agreement=4.0)
    assert np.array_equal(filled, proj) and np.array_equal(known, valid)


def test_inpainting_raises_coverage_without_hurting_accuracy(sim):
    """On the simulator, where the right answer is known."""
    off = decode(array_loader(sim.images), sim.manifest,
                 DecodeConfig(inpaint_radius_px=0))
    on = decode(array_loader(sim.images), sim.manifest,
                DecodeConfig(inpaint_radius_px=6))
    assert on.coverage > off.coverage

    for result in (off, on):
        err = errors_against_truth(sim, result)
        assert np.median(err) < 1.0
        assert np.percentile(err, 95) < 3.0


def test_glass_is_distinguished_from_shadow_by_the_black_frame(sim_wide_baseline):
    """Both are unmodulated; only one is still bright with nothing projected.

    A dimly-lit shadow was being labelled a window, because low confidence alone
    cannot tell the two apart.
    """
    import cv2

    result = run_decode(sim_wide_baseline)
    cache = sim_wide_baseline.cache
    shadow = (cache.shadowed & cache.hit & ~sim_wide_baseline.glass_mask)
    interior = cv2.erode(shadow.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    glass = sim_wide_baseline.glass_mask & cache.hit

    assert interior.sum() > 50 and glass.sum() > 500
    assert np.median(result.black[glass]) > np.median(result.black[interior]) + 0.04
    assert result.likely_glass[glass].mean() > 0.85
    assert result.likely_glass[interior].mean() < 0.05


def test_a_dark_unmodulated_region_is_not_called_glass():
    """Directly: same confidence, different black level."""
    from facade_scan.config import DecodeConfig as DC

    cfg = DC()
    bright_unmodulated = (0.30, 0.20, 0.02)     # white, black, confidence
    dark_unmodulated = (0.30, 0.01, 0.02)       # lit by something, but no glare
    for (white, black, conf), expected in ((bright_unmodulated, True),
                                           (dark_unmodulated, False)):
        flagged = (white >= cfg.glass_min_brightness
                   and black >= cfg.glass_min_black
                   and conf < cfg.glass_confidence_threshold)
        assert flagged is expected
