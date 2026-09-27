"""Stage 5: line and region detection, judged against the simulator's geometry.

Detection is scored by asking whether the regions it finds actually coincide
with the surfaces the simulator drew -- the windows, the doors, the gables --
rather than by counting how many shapes came out.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from facade_scan.config import DetectConfig
from facade_scan.detect import (
    Region,
    SamUnavailable,
    detect,
    detect_lines,
    detect_segments,
    estimate_vanishing_points,
    extend_segments,
    filter_short,
    house_region,
    merge_collinear,
    polygonize_segments,
    sam_available,
    segment_with_sam,
    snap_to_vanishing_points,
)
from facade_scan.detect.lines import (
    directions,
    lengths,
    midpoints,
)

from .conftest import best_match, build_sim, region_iou

#: Surfaces a competent detector should find on the synthetic house.
EXPECTED_FEATURES = [
    "window_lower_left", "window_lower_mid", "window_upper_left",
    "window_upper_mid", "window_upper_right", "front_door", "garage_door",
    "gable_main", "main_facade",
]


@pytest.fixture(scope="module")
def detection(sim_hires, decoded_hires):
    return detect(sim_hires.white, DetectConfig(),
                  glass_mask=decoded_hires.likely_glass,
                  illumination=decoded_hires.illumination)


# --------------------------------------------------------------------------- #
# Segment detection
# --------------------------------------------------------------------------- #
def test_the_contrib_detector_is_the_one_being_used():
    """cv2.createLineSegmentDetector was removed from mainline OpenCV; this
    project must be using the ximgproc one."""
    import cv2

    assert hasattr(cv2, "ximgproc")
    assert hasattr(cv2.ximgproc, "createFastLineDetector")


def test_segments_are_found_on_the_white_capture(sim_hires):
    segments = detect_segments(sim_hires.white, DetectConfig())
    assert len(segments) > 30
    assert segments.shape[1] == 4
    height, width = sim_hires.white.shape
    assert segments[:, [0, 2]].min() >= -1 and segments[:, [0, 2]].max() <= width + 1
    assert segments[:, [1, 3]].min() >= -1 and segments[:, [1, 3]].max() <= height + 1


def test_length_filter_removes_short_segments():
    segments = np.array([[0, 0, 100, 0], [0, 0, 5, 0], [0, 0, 0, 60]], dtype=float)
    kept = filter_short(segments, 50.0)
    assert len(kept) == 2
    assert lengths(kept).min() >= 50.0


def test_detector_accepts_colour_and_float_images(sim_hires):
    import cv2

    gray = sim_hires.white
    colour = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    as_float = gray.astype(np.float32) / 255.0
    n = len(detect_segments(gray, DetectConfig()))
    assert len(detect_segments(colour, DetectConfig())) == n
    assert len(detect_segments(as_float, DetectConfig())) == n


def test_empty_image_yields_no_segments():
    blank = np.zeros((120, 160), np.uint8)
    assert len(detect_segments(blank, DetectConfig())) == 0


# --------------------------------------------------------------------------- #
# Vanishing points
# --------------------------------------------------------------------------- #
def test_a_synthetic_pencil_of_lines_recovers_its_vanishing_point():
    """Exact test with a known answer, before trusting it on a photograph."""
    target = np.array([900.0, 140.0])
    rng = np.random.default_rng(0)
    segments = []
    for _ in range(14):
        start = rng.uniform([40, 300], [400, 620])
        direction = target - start
        direction /= np.linalg.norm(direction)
        segments.append(np.concatenate([start, start + direction * 160.0]))
    segments = np.array(segments)

    vps = estimate_vanishing_points(segments, DetectConfig(max_vanishing_points=1))
    assert len(vps) == 1
    assert not vps[0].at_infinity
    assert np.allclose(vps[0].image_point, target, atol=1.0)
    assert len(vps[0].inliers) == len(segments)


def test_parallel_lines_give_a_vanishing_point_at_infinity():
    """A facade shot dead-on has no finite vanishing point, and that must not
    be an overflow or a division by zero."""
    segments = np.array([[10.0, y, 500.0, y] for y in range(20, 300, 20)])
    vps = estimate_vanishing_points(segments, DetectConfig(max_vanishing_points=1))
    assert len(vps) == 1
    assert vps[0].at_infinity
    assert vps[0].image_point is None
    toward = vps[0].direction_from(midpoints(segments))
    assert np.allclose(np.abs(toward @ np.array([1.0, 0.0])), 1.0, atol=1e-6)


def test_two_families_are_separated():
    horizontal = np.array([[10.0, y, 400.0, y + 6] for y in range(20, 260, 20)])
    vertical = np.array([[x, 10.0, x + 6, 400.0] for x in range(20, 260, 20)])
    vps = estimate_vanishing_points(np.vstack([horizontal, vertical]),
                                    DetectConfig(max_vanishing_points=3))
    assert len(vps) == 2
    assert {len(v.inliers) for v in vps} == {len(horizontal), len(vertical)}
    assert all(v.inliers.size > 0 for v in vps)


def test_a_house_shows_two_or_three_dominant_directions(detection):
    assert 2 <= len(detection.vanishing_points) <= 3
    assert detection.vanishing_points[0].support >= detection.vanishing_points[-1].support


def test_vanishing_point_estimation_is_deterministic(sim_hires):
    cfg = DetectConfig()
    segments = filter_short(detect_segments(sim_hires.white, cfg),
                            cfg.resolve(sim_hires.white.shape).min_segment_length_px)
    first = estimate_vanishing_points(segments, cfg)
    second = estimate_vanishing_points(segments, cfg)
    assert [v.point.tolist() for v in first] == [v.point.tolist() for v in second]


def test_too_few_segments_yields_no_vanishing_points():
    assert estimate_vanishing_points(np.zeros((0, 4)), DetectConfig()) == []
    assert estimate_vanishing_points(np.array([[0.0, 0, 10, 10]]), DetectConfig()) == []


# --------------------------------------------------------------------------- #
# Snapping
# --------------------------------------------------------------------------- #
def test_snapping_aligns_segments_without_moving_or_resizing_them():
    target = np.array([1200.0, 200.0])
    rng = np.random.default_rng(1)
    segments = []
    for _ in range(10):
        start = rng.uniform([40, 300], [400, 620])
        d = target - start
        d /= np.linalg.norm(d)
        # a degree or so of detector noise, which is enough to block merging
        angle = np.radians(rng.uniform(-1.5, 1.5))
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        segments.append(np.concatenate([start, start + (rot @ d) * 200.0]))
    segments = np.array(segments)

    vps = estimate_vanishing_points(segments, DetectConfig(vp_inlier_angle_deg=4.0,
                                                          max_vanishing_points=1))
    snapped = snap_to_vanishing_points(segments, vps, DetectConfig())

    assert np.allclose(midpoints(snapped), midpoints(segments), atol=1e-6)
    assert np.allclose(lengths(snapped), lengths(segments), atol=1e-6)
    # Every snapped segment now points exactly at the vanishing point.
    toward = vps[0].direction_from(midpoints(snapped))
    assert np.allclose(np.abs(np.sum(toward * directions(snapped), axis=-1)), 1.0, atol=1e-9)
    # ...and still points the way it did before, so endpoint order is stable.
    assert (np.sum(directions(snapped) * directions(segments), axis=-1) > 0).all()


def test_snapping_can_be_switched_off():
    segments = np.array([[0.0, 0, 100, 3], [0.0, 50, 100, 54]])
    vps = estimate_vanishing_points(segments, DetectConfig(vp_min_inliers=2,
                                                          max_vanishing_points=1))
    untouched = snap_to_vanishing_points(segments, vps, DetectConfig(vp_snap=False))
    assert np.array_equal(untouched, segments)


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #
def test_collinear_pieces_of_one_edge_merge_into_it():
    """A roofline broken by a downpipe should come back as one roofline."""
    segments = np.array([
        [10.0, 100.0, 200.0, 100.0],
        [215.0, 100.5, 400.0, 100.0],   # small gap, half a pixel off
        [10.0, 300.0, 400.0, 300.0],    # a different edge entirely
    ])
    cfg = DetectConfig(merge_angle_deg=2.5, merge_perp_px=4.0, merge_gap_px=30.0,
                       merge_perp_frac=0.0, merge_gap_frac=0.0)
    merged = merge_collinear(segments, cfg)
    assert len(merged) == 2
    longest = merged[np.argmax(lengths(merged))]
    assert min(longest[0], longest[2]) == pytest.approx(10.0, abs=1.0)
    assert max(longest[0], longest[2]) == pytest.approx(400.0, abs=1.0)


def test_parallel_but_separated_edges_do_not_merge():
    """Two window heads, one above the other. Perpendicular distance separates
    them even though they are perfectly parallel."""
    segments = np.array([[10.0, 100.0, 200.0, 100.0], [10.0, 140.0, 200.0, 140.0]])
    cfg = DetectConfig(merge_perp_px=4.0, merge_perp_frac=0.0, merge_gap_frac=0.0)
    assert len(merge_collinear(segments, cfg)) == 2


def test_distant_collinear_edges_do_not_merge():
    """A window sill and a roofline that happen to line up must stay separate."""
    segments = np.array([[10.0, 100.0, 120.0, 100.0], [900.0, 100.0, 1100.0, 100.0]])
    cfg = DetectConfig(merge_gap_px=30.0, merge_gap_frac=0.0, merge_perp_frac=0.0)
    assert len(merge_collinear(segments, cfg)) == 2
    generous = replace(cfg, merge_gap_px=2000.0)
    assert len(merge_collinear(segments, generous)) == 1


def test_perpendicular_segments_never_merge():
    segments = np.array([[0.0, 100.0, 200.0, 100.0], [100.0, 0.0, 100.0, 200.0]])
    assert len(merge_collinear(segments, DetectConfig())) == 2


def test_merging_reduces_the_segment_count_on_a_real_house(sim_hires):
    cfg = DetectConfig().resolve(sim_hires.white.shape)
    kept = filter_short(detect_segments(sim_hires.white, cfg), cfg.min_segment_length_px)
    vps = estimate_vanishing_points(kept, cfg)
    merged = merge_collinear(snap_to_vanishing_points(kept, vps, cfg), cfg)
    assert 0 < len(merged) < len(kept)


def test_merge_handles_trivial_inputs():
    assert len(merge_collinear(np.zeros((0, 4)), DetectConfig())) == 0
    one = np.array([[0.0, 0.0, 10.0, 0.0]])
    assert np.array_equal(merge_collinear(one, DetectConfig()), one)


# --------------------------------------------------------------------------- #
# Regions
# --------------------------------------------------------------------------- #
def test_extending_segments_preserves_direction_and_grows_both_ends():
    segments = np.array([[10.0, 10.0, 10.0, 110.0]])
    grown = extend_segments(segments, 5.0)
    assert grown[0].tolist() == [10.0, 5.0, 10.0, 115.0]
    assert np.allclose(directions(grown), directions(segments))


def test_four_segments_that_nearly_meet_become_one_region():
    """The whole point of the extension step."""
    gap = 6.0
    segments = np.array([
        [100.0, 100.0, 300.0 - gap, 100.0],
        [300.0, 100.0, 300.0, 300.0 - gap],
        [300.0, 300.0, 100.0 + gap, 300.0],
        [100.0, 300.0, 100.0, 100.0 + gap],
    ])
    cfg = DetectConfig(extend_px=12.0, extend_frac=0.0, min_region_area_px=100.0,
                       min_region_area_frac=0.0)
    faces = polygonize_segments(segments, (400, 400), cfg)
    assert len(faces) == 1
    assert faces[0].area == pytest.approx(200 * 200, rel=0.15)

    # ...and without the extension, nothing closes.
    no_extension = replace(cfg, extend_px=0.0)
    assert polygonize_segments(segments, (400, 400), no_extension) == []


def test_tiny_and_frame_filling_faces_are_dropped():
    cfg = DetectConfig(extend_px=4.0, extend_frac=0.0, min_region_area_px=5000.0,
                       min_region_area_frac=0.0, max_region_area_frac=0.5)
    small = np.array([[10.0, 10, 40, 10], [40.0, 10, 40, 40],
                      [40.0, 40, 10, 40], [10.0, 40, 10, 10]])
    assert polygonize_segments(small, (400, 400), cfg) == []

    huge = np.array([[5.0, 5, 395, 5], [395.0, 5, 395, 395],
                     [395.0, 395, 5, 395], [5.0, 395, 5, 5]])
    assert polygonize_segments(huge, (400, 400), cfg) == []


def test_regions_are_found_for_the_architectural_features(sim_hires, detection):
    """The headline detection test: do the regions land on the real features?"""
    missed = []
    for name in EXPECTED_FEATURES:
        truth = sim_hires.surface_mask(name)
        assert truth.sum() > 200, f"{name} is not visible in the render"
        _, iou = best_match(detection.regions, truth)
        if iou < 0.6:
            missed.append(f"{name} (best IoU {iou:.2f})")
    assert not missed, "features not recovered: " + ", ".join(missed)


def test_every_window_is_labelled_as_a_window(sim_hires, detection):
    """The decoder's glass mask is free signal; this is it paying off."""
    from .conftest import GLASS_SURFACES

    for name in GLASS_SURFACES:
        truth = sim_hires.surface_mask(name)
        region, iou = best_match(detection.regions, truth)
        assert iou > 0.6, f"{name} not found at all"
        assert region.label.startswith("window"), f"{name} labelled {region.label!r}"
        assert region.attributes["glass_fraction"] > 0.8


def test_solid_surfaces_are_not_labelled_as_windows(sim_hires, detection):
    for name in ("front_door", "garage_door", "gable_main"):
        region, iou = best_match(detection.regions, sim_hires.surface_mask(name))
        assert iou > 0.6 and region is not None
        assert not region.label.startswith("window"), f"{name} mislabelled"
        assert region.attributes["glass_fraction"] < 0.5


def test_regions_without_a_glass_mask_still_work(sim_hires):
    plain = detect(sim_hires.white, DetectConfig())
    assert len(plain.regions) >= 5
    assert all(r.label.startswith("region") for r in plain.regions)
    _, iou = best_match(plain.regions, sim_hires.surface_mask("garage_door"))
    assert iou > 0.6


def test_regions_carry_usable_geometry(detection):
    for region in detection.regions:
        assert region.polygon.ndim == 2 and region.polygon.shape[1] == 2
        assert len(region.polygon) >= 3
        assert region.area > 0
        assert region.shapely.is_valid
        assert np.isfinite(region.centroid).all()
        assert region.attributes["area_px"] == pytest.approx(region.area, rel=1e-6)
        for hole in region.holes:
            assert hole.ndim == 2 and hole.shape[1] == 2 and len(hole) >= 3


def test_region_labels_are_unique(detection):
    labels = [r.label for r in detection.regions]
    assert len(labels) == len(set(labels))


def test_detection_reports_how_much_it_threw_away(detection):
    assert detection.raw_segment_count >= len(detection.segments)


def test_a_facade_keeps_its_window_openings_as_holes(sim_hires, detection):
    """Dropping holes would project light straight through the glass."""
    from shapely.geometry import Polygon

    facade, iou = best_match(detection.regions, sim_hires.surface_mask("main_facade"))
    assert iou > 0.6
    assert facade.holes, "the facade region has no openings in it"

    solid = Polygon(facade.polygon).area
    assert facade.area < solid, "holes are not reducing the region's area"

    # The openings sit where the windows are, not somewhere arbitrary.
    glazing = sim_hires.glass_mask
    for hole in facade.holes:
        assert region_iou(hole, glazing) > 0.0

    # And the rasterised region excludes them.
    filled = facade.rasterize(sim_hires.white.shape)
    assert filled[glazing].mean() < 0.35


def test_rasterize_punches_holes_out():
    outer = np.array([[0.0, 0], [100, 0], [100, 100], [0, 100]])
    inner = np.array([[40.0, 40], [60, 40], [60, 60], [40, 60]])
    region = Region(polygon=outer, holes=[inner])
    mask = region.rasterize((120, 120))
    assert mask[10, 10] and not mask[50, 50]
    assert region.area == pytest.approx(100 * 100 - 20 * 20)


# --------------------------------------------------------------------------- #
# Resolution independence
# --------------------------------------------------------------------------- #
def test_thresholds_scale_with_image_size():
    small = DetectConfig().resolve((300, 480))
    large = DetectConfig().resolve((2000, 3000))
    assert large.min_segment_length_px > 4 * small.min_segment_length_px
    assert large.extend_px > 4 * small.extend_px
    assert large.min_region_area_px > 20 * small.min_region_area_px


def test_a_fraction_of_zero_pins_a_threshold_to_its_absolute_value():
    cfg = DetectConfig(min_segment_length_frac=0.0, min_segment_length_px=40.0)
    assert cfg.resolve((2000, 3000)).min_segment_length_px == 40.0


def test_the_same_house_is_detected_at_a_different_resolution():
    """A user's camera is not our camera. The pipeline must not need retuning."""
    small = build_sim(projector=(320, 200), camera=(480, 300))
    found = detect(small.white, DetectConfig())
    for name in ("garage_door", "front_door", "window_lower_left"):
        _, iou = best_match(found.regions, small.surface_mask(name))
        assert iou > 0.6, f"{name} lost at low resolution (IoU {iou:.2f})"


# --------------------------------------------------------------------------- #
# House mask and plumbing
# --------------------------------------------------------------------------- #
def test_house_mask_covers_illuminated_surface_only(sim_hires, decoded_hires, detection):
    mask = detection.house_mask
    assert mask is not None
    lit_house = sim_hires.cache.hit & sim_hires.cache.lit & ~sim_hires.glass_mask
    assert mask[lit_house].mean() > 0.97
    assert mask[~sim_hires.cache.hit].mean() < 0.01


def test_house_region_thresholds_illumination():
    illumination = np.array([[0.0, 0.05], [0.2, 0.9]], dtype=np.float32)
    assert house_region(illumination, 0.08).tolist() == [[False, False], [True, True]]


def test_detect_lines_is_the_whole_line_pipeline(sim_hires, detection):
    segments, vps = detect_lines(sim_hires.white, DetectConfig())
    assert np.allclose(segments, detection.segments)
    assert len(vps) == len(detection.vanishing_points)


# --------------------------------------------------------------------------- #
# Optional SAM
# --------------------------------------------------------------------------- #
def test_torch_is_not_a_hard_dependency():
    """Importing the package must not drag in torch."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, facade_scan, facade_scan.detect, facade_scan.cli;"
         " assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_sam_is_off_by_default():
    assert DetectConfig().use_sam is False


def test_asking_for_sam_without_it_installed_is_explained():
    if sam_available():
        pytest.skip("segment-anything is installed in this environment")
    with pytest.raises(SamUnavailable, match="optional"):
        segment_with_sam(np.zeros((32, 32), np.uint8), DetectConfig(use_sam=True))


def test_a_scan_still_produces_regions_when_sam_is_requested_but_missing(sim_hires):
    """An unavailable optional extra must never fail a scan."""
    if sam_available():
        pytest.skip("segment-anything is installed in this environment")
    result = detect(sim_hires.white, DetectConfig(use_sam=True))
    assert len(result.regions) >= 5
    assert any("Segment Anything skipped" in note for note in result.notes)


def test_sam_without_a_checkpoint_is_explained():
    if not sam_available():
        pytest.skip("segment-anything is not installed")
    with pytest.raises(SamUnavailable, match="checkpoint"):
        segment_with_sam(np.zeros((32, 32), np.uint8),
                         DetectConfig(use_sam=True, sam_checkpoint=None))


# --------------------------------------------------------------------------- #
# Geometric edges from the correspondence
# --------------------------------------------------------------------------- #
def test_depth_edges_are_found_where_the_image_has_no_contrast():
    """The failure this exists for.

    A square of the *same brightness* as its background is invisible to a line
    detector working on the photograph. It is perfectly visible as a step in the
    decoded correspondence, because it is nearer.
    """
    from facade_scan.detect import detect_depth_segments

    flat = np.full((400, 600), 160, np.uint8)          # no contrast anywhere
    assert len(detect_segments(flat, DetectConfig())) == 0

    edges = np.zeros((400, 600), bool)
    edges[120, 150:450] = True                          # the square's outline
    edges[280, 150:450] = True
    edges[120:280, 150] = True
    edges[120:280, 450] = True

    segments = detect_depth_segments(edges, DetectConfig().resolve(flat.shape))
    assert len(segments) >= 4, "should find the four sides"


def test_depth_edges_let_a_region_close_that_otherwise_would_not():
    flat = np.full((400, 600), 160, np.uint8)
    edges = np.zeros((400, 600), bool)
    edges[120, 150:450] = True
    edges[280, 150:450] = True
    edges[120:280, 150] = True
    edges[120:280, 450] = True

    without = detect(flat, DetectConfig())
    with_depth = detect(flat, DetectConfig(), depth_edges=edges)
    assert not without.regions
    assert with_depth.regions, "the depth step should close the square"

    best = max(with_depth.regions, key=lambda r: r.area)
    assert best.area == pytest.approx(300 * 160, rel=0.25)


def test_depth_edges_are_optional_and_an_empty_map_changes_nothing(sim_hires):
    plain = detect(sim_hires.white, DetectConfig())
    empty = detect(sim_hires.white, DetectConfig(),
                   depth_edges=np.zeros(sim_hires.white.shape, bool))
    assert len(plain.segments) == len(empty.segments)


def test_depth_edges_find_the_bump_out_silhouette_on_the_simulated_house(
    sim_hires, decoded_hires
):
    """On the synthetic house, where the answer is known."""
    from facade_scan.decode import discontinuity_mask
    from facade_scan.detect import detect_depth_segments

    edges = discontinuity_mask(decoded_hires.proj_map, decoded_hires.valid,
                               jump_px=3.0)
    assert edges.sum() > 100
    cfg = DetectConfig().resolve(sim_hires.white.shape)
    segments = detect_depth_segments(edges, cfg)
    assert len(segments) > 0

    # They should lie along real surface boundaries, not wander the facade.
    surface = sim_hires.cache.surface
    boundary = np.zeros_like(edges)
    for axis in (0, 1):
        differs = np.diff(surface, axis=axis) != 0
        pad = [(0, 0), (0, 0)]
        pad[axis] = (0, 1)
        boundary |= np.pad(differs, pad)
    boundary = cv2.dilate(boundary.astype(np.uint8),
                          np.ones((9, 9), np.uint8)).astype(bool)

    midpoints_xy = np.stack([(segments[:, 0] + segments[:, 2]) / 2,
                             (segments[:, 1] + segments[:, 3]) / 2], -1)
    on_boundary = boundary[np.clip(midpoints_xy[:, 1].astype(int), 0, edges.shape[0] - 1),
                           np.clip(midpoints_xy[:, 0].astype(int), 0, edges.shape[1] - 1)]
    assert on_boundary.mean() > 0.8


# --------------------------------------------------------------------------- #
# Endpoint snapping
# --------------------------------------------------------------------------- #
def _snap_cfg(**kwargs):
    base = {"snap_radius_px": 60.0, "snap_radius_frac": 0.0,
            "snap_min_angle_deg": 20.0}
    base.update(kwargs)
    return DetectConfig(**base)


def test_snapping_closes_a_corner_exactly():
    """Both ends land on the intersection, not merely near it."""
    from facade_scan.detect import snap_endpoints

    segments = np.array([
        [0.0, 100.0, 180.0, 100.0],      # horizontal, stops 20 short of x=200
        [200.0, 130.0, 200.0, 400.0],    # vertical, starts 30 below y=100
    ])
    snapped = snap_endpoints(segments, _snap_cfg())
    assert snapped[0, 2:4] == pytest.approx([200.0, 100.0], abs=0.01)
    assert snapped[1, 0:2] == pytest.approx([200.0, 100.0], abs=0.01)


def test_snapping_leaves_the_far_end_alone():
    from facade_scan.detect import snap_endpoints

    segments = np.array([[0.0, 100.0, 180.0, 100.0],
                         [200.0, 130.0, 200.0, 400.0]])
    snapped = snap_endpoints(segments, _snap_cfg())
    assert snapped[0, 0:2] == pytest.approx([0.0, 100.0])
    assert snapped[1, 2:4] == pytest.approx([200.0, 400.0])


def test_near_parallel_segments_are_never_snapped():
    """Their intersection swings wildly on a fraction of a degree of noise."""
    from facade_scan.detect import snap_endpoints

    segments = np.array([[0.0, 100.0, 180.0, 100.0],
                         [200.0, 103.0, 400.0, 106.0]])
    assert np.array_equal(snap_endpoints(segments, _snap_cfg()), segments)


def test_a_distant_intersection_is_not_snapped_to():
    """An edge must not be dragged to a line on the far side of the house."""
    from facade_scan.detect import snap_endpoints

    segments = np.array([[0.0, 100.0, 180.0, 100.0],
                         [900.0, 130.0, 900.0, 400.0]])
    assert np.array_equal(snap_endpoints(segments, _snap_cfg(snap_radius_px=60.0)),
                          segments)


def test_snapping_requires_the_corner_to_belong_to_both_segments():
    """The intersection is close to our end, but far along the other segment."""
    from facade_scan.detect import snap_endpoints

    segments = np.array([
        [0.0, 100.0, 180.0, 100.0],
        [200.0, 400.0, 200.0, 800.0],    # its line crosses y=100, its body does not
    ])
    assert np.array_equal(snap_endpoints(segments, _snap_cfg()), segments)


def test_snapping_can_be_disabled():
    from facade_scan.detect import snap_endpoints

    segments = np.array([[0.0, 100.0, 180.0, 100.0],
                         [200.0, 130.0, 200.0, 400.0]])
    assert np.array_equal(snap_endpoints(segments, _snap_cfg(snap_radius_px=0.0)),
                          segments)


def test_snapping_closes_a_square_that_extension_alone_would_not():
    """The point of the whole thing.

    Four sides, each stopping well short of its corners. Closing them by blanket
    extension needs enough growth to manufacture faces elsewhere; snapping puts
    the corners exactly where they belong and nothing else moves.
    """
    from facade_scan.detect import snap_endpoints

    gap = 40.0
    square = np.array([
        [100.0 + gap, 100.0, 400.0 - gap, 100.0],
        [400.0, 100.0 + gap, 400.0, 400.0 - gap],
        [400.0 - gap, 400.0, 100.0 + gap, 400.0],
        [100.0, 400.0 - gap, 100.0, 100.0 + gap],
    ])
    cfg = _snap_cfg(extend_px=4.0, extend_frac=0.0,
                    min_region_area_px=500.0, min_region_area_frac=0.0)

    assert polygonize_segments(square, (500, 500), cfg) == []
    faces = polygonize_segments(snap_endpoints(square, cfg), (500, 500), cfg)
    assert len(faces) == 1
    assert faces[0].area == pytest.approx(300 * 300, rel=0.05)


def test_snapping_does_not_disturb_the_simulated_house(sim_hires, decoded_hires):
    """No regression on a scene that already worked."""
    found = detect(sim_hires.white, DetectConfig(),
                   glass_mask=decoded_hires.likely_glass,
                   illumination=decoded_hires.illumination)
    missed = []
    for name in EXPECTED_FEATURES:
        _, iou = best_match(found.regions, sim_hires.surface_mask(name))
        if iou < 0.6:
            missed.append(f"{name} ({iou:.2f})")
    assert not missed, "features lost after snapping: " + ", ".join(missed)
