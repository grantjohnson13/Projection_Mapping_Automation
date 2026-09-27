"""Stage 6: camera space -> projector space.

Scored against the simulator throughout. The ground truth for projector space is
obtained by rendering the scene *from the projector's own viewpoint*, which
gives exactly the set of projector pixels that land on the house.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from facade_scan.config import DetectConfig, TransferConfig
from facade_scan.decode import discontinuity_mask
from facade_scan.detect import Region, detect
from facade_scan.sim import build_scene_cache
from facade_scan.transfer import (
    TransferResult,
    densify,
    fill_small_holes,
    house_mask,
    lookup_points,
    scatter_mask,
    transfer_mask,
    transfer_region,
    transfer_regions,
    transfer_ring,
)

from .conftest import GLASS_SURFACES


# --------------------------------------------------------------------------- #
# Ground truth in projector space
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def projector_view(sim_hires):
    """The scene as the projector itself sees it.

    ``hit`` is then precisely the set of projector pixels that land on the
    house, which is what a house mask is trying to reproduce.
    """
    return build_scene_cache(sim_hires.scene, sim_hires.projector,
                             sim_hires.projector, sim_hires.cfg)


@pytest.fixture(scope="module")
def truth_house_mask(sim_hires, projector_view):
    glass = [sim_hires.scene.index_of(n) for n in GLASS_SURFACES]
    return projector_view.hit & ~np.isin(projector_view.surface, glass)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


# --------------------------------------------------------------------------- #
# The house mask -- the headline one-command operation
# --------------------------------------------------------------------------- #
def test_house_mask_reproduces_what_the_projector_can_see(decoded_hires, truth_house_mask):
    mask = house_mask(decoded_hires)
    assert iou(mask, truth_house_mask) > 0.97
    assert (mask & truth_house_mask).sum() / truth_house_mask.sum() > 0.97   # recall
    assert (mask & truth_house_mask).sum() / mask.sum() > 0.97               # precision


def test_house_mask_does_not_spill_onto_the_sky(decoded_hires, projector_view):
    """The practical point: stop lighting the neighbours' hedge."""
    mask = house_mask(decoded_hires)
    assert mask[~projector_view.hit].mean() < 0.02


def test_house_mask_is_at_projector_resolution(decoded_hires):
    mask = house_mask(decoded_hires)
    assert mask.shape == (decoded_hires.projector_height, decoded_hires.projector_width)
    assert mask.dtype == np.bool_


def test_house_mask_leaves_the_windows_dark(decoded_hires, sim_hires, projector_view):
    """Glazing does not decode, so it never enters the mask -- which is what you
    want anyway, since projecting through glass lights up the living room."""
    glass = np.isin(projector_view.surface,
                    [sim_hires.scene.index_of(n) for n in GLASS_SURFACES])
    assert glass.sum() > 1000
    assert house_mask(decoded_hires)[glass].mean() < 0.15


def test_exclude_glass_removes_glazing_that_did_decode():
    """Glazing that returns *some* pattern is the case the option is for.

    A clean single pane returns so little that it never decodes and never
    reaches the mask anyway. A dirty, coated or double-glazed pane returns
    enough to decode, and then it is in the mask, and then you are projecting
    through someone's living room window.
    """
    from facade_scan.decode import array_loader, decode

    from .conftest import HIRES_CAMERA, HIRES_PROJECTOR, build_sim

    base = build_sim()
    for surface in base.scene.surfaces:
        if surface.name in GLASS_SURFACES:
            surface.transmission = 0.45      # a pane that half works
    dirty = build_sim(projector=HIRES_PROJECTOR, camera=HIRES_CAMERA, scene=base.scene)
    result = decode(array_loader(dirty.images), dirty.manifest)

    glass_camera = dirty.glass_mask
    assert result.valid[glass_camera].mean() > 0.5, "glazing should decode here"
    assert result.likely_glass[glass_camera].mean() > 0.5, "...and still be flagged"

    with_glass = house_mask(result)
    without = house_mask(result, exclude_glass=True)
    assert without.sum() < 0.95 * with_glass.sum()


# --------------------------------------------------------------------------- #
# The central claim: no single homography can do this
# --------------------------------------------------------------------------- #
def test_a_homography_fitted_to_the_wall_misses_the_bump_out(sim_hires, decoded_hires):
    """Quantify what the planarity assumption costs.

    A hand-aligned projection mapping is a homography fitted to whatever plane
    the operator clicked four corners on. It is exact on that plane and wrong
    everywhere else by the stereo disparity of the depth step. The decoded map
    is exact on both, because it never assumed a plane in the first place.
    """
    ys, xs = np.nonzero(decoded_hires.valid)
    camera_pts = np.stack([xs, ys], -1).astype(np.float32)
    decoded_pts = decoded_hires.proj_map[ys, xs].astype(np.float32)
    truth_pts = sim_hires.ground_truth_uv[ys, xs]
    surface = sim_hires.cache.surface[ys, xs]

    index = sim_hires.scene.index_of
    wall = np.isin(surface, [index("main_facade"), index("gable_main")])
    bump = np.isin(surface, [index("garage_face"), index("garage_door"),
                             index("gable_garage")])
    assert wall.sum() > 5000 and bump.sum() > 5000

    homography, _ = cv2.findHomography(camera_pts[wall], decoded_pts[wall],
                                       cv2.RANSAC, 2.0)
    predicted = cv2.perspectiveTransform(camera_pts.reshape(-1, 1, 2),
                                         homography).reshape(-1, 2)
    homography_error = np.linalg.norm(predicted - truth_pts, axis=1)
    structured_light_error = np.linalg.norm(decoded_pts - truth_pts, axis=1)

    # On the plane it was fitted to, the homography is as good as anything.
    assert np.median(homography_error[wall]) < 1.0

    # Off that plane it is wrong by about the stereo disparity of the step...
    baseline = float(np.linalg.norm(sim_hires.camera.center - sim_hires.projector.center))
    predicted_disparity = sim_hires.projector.fx * baseline * (1 / 13.0 - 1 / 14.0)
    assert np.median(homography_error[bump]) == pytest.approx(predicted_disparity, rel=0.3)

    # ...while structured light is unbothered by the depth step.
    assert np.median(structured_light_error[bump]) < 1.0
    assert np.median(homography_error[bump]) > 6 * np.median(structured_light_error[bump])


# --------------------------------------------------------------------------- #
# Densify
# --------------------------------------------------------------------------- #
def test_densify_keeps_vertices_and_bounds_edge_length():
    square = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    dense = densify(square, 10.0)
    for vertex in square:
        assert np.isclose(np.linalg.norm(dense - vertex, axis=1), 0).any()
    closed = np.vstack([dense, dense[:1]])
    assert np.linalg.norm(np.diff(closed, axis=0), axis=1).max() <= 10.0 + 1e-9
    assert len(dense) > len(square)


def test_densify_is_a_no_op_on_short_edges():
    triangle = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    assert len(densify(triangle, 100.0)) == 3


def test_densify_handles_degenerate_rings():
    assert len(densify(np.zeros((0, 2)), 5.0)) == 0
    assert len(densify(np.array([[1.0, 2.0]]), 5.0)) == 1


# --------------------------------------------------------------------------- #
# Point lookup
# --------------------------------------------------------------------------- #
def test_lookup_returns_the_decoded_coordinate(decoded_hires):
    ys, xs = np.nonzero(decoded_hires.valid)
    sample = np.stack([xs[::5000], ys[::5000]], -1).astype(float)
    mapped, found = lookup_points(decoded_hires, sample, radius=0)
    assert found.all()
    expected = decoded_hires.proj_map[sample[:, 1].astype(int), sample[:, 0].astype(int)]
    assert np.array_equal(mapped, expected)


def test_lookup_reaches_a_nearby_valid_pixel(decoded_hires):
    """A point landing on dark brick or a window gets the nearest good answer."""
    holed = replace_valid(decoded_hires)
    point = np.array([[holed.hole_x, holed.hole_y]], dtype=float)
    _, found_exact = lookup_points(holed.result, point, radius=0)
    mapped, found_near = lookup_points(holed.result, point, radius=4)
    assert not found_exact.any()
    assert found_near.all()
    assert np.isfinite(mapped).all()


def test_lookup_gives_up_beyond_the_radius(decoded_hires):
    empty = replace(decoded_hires)
    empty.valid = np.zeros_like(decoded_hires.valid)
    _, found = lookup_points(empty, np.array([[100.0, 100.0]]), radius=4)
    assert not found.any()


class _Holed:
    def __init__(self, result, hole_x, hole_y):
        self.result, self.hole_x, self.hole_y = result, hole_x, hole_y


def replace_valid(decoded):
    """A copy of the decode with a small invalid patch punched into it."""
    import copy

    out = copy.copy(decoded)
    out.valid = decoded.valid.copy()
    ys, xs = np.nonzero(decoded.valid)
    mid = len(ys) // 2
    y, x = int(ys[mid]), int(xs[mid])
    out.valid[y - 1:y + 2, x - 1:x + 2] = False
    return _Holed(out, x, y)


# --------------------------------------------------------------------------- #
# Vector transfer
# --------------------------------------------------------------------------- #
def test_transferred_regions_land_where_the_surfaces_are(sim_hires, decoded_hires,
                                                         projector_view):
    """Detect in camera space, transfer, and check against projector-space truth."""
    detection = detect(sim_hires.white, DetectConfig(),
                       glass_mask=decoded_hires.likely_glass,
                       illumination=decoded_hires.illumination)
    result = transfer_regions(decoded_hires, detection.regions)
    assert result.regions and not result.dropped

    shape = (decoded_hires.projector_height, decoded_hires.projector_width)
    checked = 0
    for name in ("garage_door", "front_door", "gable_main"):
        truth = projector_view.surface == sim_hires.scene.index_of(name)
        if truth.sum() < 200:
            continue
        best = max(iou(r.rasterize(shape), truth) for r in result.regions)
        assert best > 0.85, f"{name} transferred to the wrong place (IoU {best:.2f})"
        checked += 1
    assert checked >= 3


def test_a_straight_camera_line_does_not_stay_straight(sim_hires, decoded_hires):
    """Why edges are densified instead of being carried vertex to vertex.

    A horizontal line across the camera's view of this house runs over the main
    wall, up the garage's return wall -- which recedes a full metre over its
    width -- and onto the garage front. In projector space that is a curve. Join
    the transferred endpoints with a straight line instead and the middle of the
    line lands several projector pixels off the surface it was meant to follow.
    """
    surface = sim_hires.cache.surface
    wanted = [sim_hires.scene.index_of(n)
              for n in ("main_facade", "garage_return_left", "garage_face",
                        "garage_door")]

    rows = np.nonzero((surface == sim_hires.scene.index_of("garage_return_left")
                       ).sum(axis=1) > 5)[0]
    row = int(rows[len(rows) // 2])
    cols = np.nonzero(decoded_hires.valid[row] & np.isin(surface[row], wanted))[0]
    assert cols.size > 400

    points = np.stack([cols, np.full(cols.shape, row)], -1).astype(float)
    dense, found = lookup_points(decoded_hires, points, radius=2)
    points, dense = points[found], dense[found]
    assert np.ptp(points[:, 1]) == 0, "the camera-space line is straight"

    # What you get by transferring only the two endpoints and joining them.
    span = points[-1, 0] - points[0, 0]
    fraction = ((points[:, 0] - points[0, 0]) / span)[:, None]
    naive = dense[0] + (dense[-1] - dense[0]) * fraction

    error = np.linalg.norm(naive - dense, axis=1)
    assert error.max() > 4.0, f"endpoint-only transfer was only off by {error.max():.2f} px"
    assert np.median(error) > 1.0

    # Densifying is what removes that error: every sampled point is looked up
    # rather than interpolated, so by construction there is nothing left to be
    # wrong about.
    ring_error = np.linalg.norm(
        dense - decoded_hires.proj_map[points[:, 1].astype(int),
                                       points[:, 0].astype(int)], axis=1)
    assert ring_error.max() == 0.0


def test_densify_step_controls_how_faithfully_an_edge_is_carried(decoded_hires):
    """A huge step degenerates to endpoint-only transfer; a small one does not."""
    ring = np.array([[200.0, 430.0], [1100.0, 430.0], [1100.0, 470.0], [200.0, 470.0]])
    fine = transfer_ring(decoded_hires, ring, TransferConfig(densify_step_px=4.0))
    coarse = transfer_ring(decoded_hires, ring, TransferConfig(densify_step_px=1e9))
    assert fine is not None and coarse is not None
    assert len(fine) > 20 * len(coarse)
    assert len(coarse) == 4


def test_region_holes_survive_the_transfer(decoded_hires):
    outer = np.array([[300.0, 250.0], [900.0, 250.0], [900.0, 600.0], [300.0, 600.0]])
    inner = np.array([[450.0, 330.0], [600.0, 330.0], [600.0, 470.0], [450.0, 470.0]])
    region = Region(polygon=outer, label="wall", holes=[inner])

    moved = transfer_region(decoded_hires, region)
    assert moved is not None
    assert len(moved.holes) == 1
    assert moved.area < Region(polygon=moved.polygon).area
    assert moved.attributes["camera_area_px"] == pytest.approx(region.area)


def test_a_region_over_undecodable_ground_is_dropped_not_guessed(decoded_hires):
    """Off in the sky, where nothing decoded at all."""
    sky = Region(polygon=np.array([[2.0, 2.0], [60.0, 2.0], [60.0, 60.0], [2.0, 60.0]]),
                 label="sky")
    assert transfer_region(decoded_hires, sky) is None

    result = transfer_regions(decoded_hires, [sky])
    assert result.regions == []
    assert len(result.dropped) == 1 and "sky" in result.dropped[0]


def test_tiny_transferred_polygons_are_dropped(decoded_hires):
    ys, xs = np.nonzero(decoded_hires.valid)
    x, y = float(xs[len(xs) // 2]), float(ys[len(ys) // 2])
    speck = Region(polygon=np.array([[x, y], [x + 2, y], [x + 2, y + 2], [x, y + 2]]))
    assert transfer_region(decoded_hires, speck,
                           TransferConfig(min_polygon_area_px=500.0)) is None


def test_transfer_ring_needs_enough_decoded_points(decoded_hires):
    assert transfer_ring(decoded_hires, np.array([[10.0, 10.0], [20.0, 20.0]])) is None


def test_transfer_result_reports_its_projector_size(decoded_hires):
    result = transfer_regions(decoded_hires, [])
    assert result.shape == (decoded_hires.projector_height, decoded_hires.projector_width)
    assert isinstance(result, TransferResult)


# --------------------------------------------------------------------------- #
# Raster transfer
# --------------------------------------------------------------------------- #
def test_scatter_excludes_depth_discontinuities(decoded_hires):
    """Boundary pixels must not seed a fill across the gap between surfaces."""
    cfg = TransferConfig()
    lit = decoded_hires.illumination >= 0.08
    boundary = discontinuity_mask(decoded_hires.proj_map, decoded_hires.valid,
                                  cfg.discontinuity_jump_px)
    assert boundary.sum() > 100

    everything = scatter_mask(decoded_hires, lit, replace(cfg, discontinuity_jump_px=1e9))
    excluded = scatter_mask(decoded_hires, lit, cfg)
    assert excluded.sum() < everything.sum()


def test_closing_and_filling_only_add_pixels(decoded_hires):
    lit = decoded_hires.illumination >= 0.08
    raw = scatter_mask(decoded_hires, lit)
    tidied = transfer_mask(decoded_hires, lit)
    assert tidied.sum() >= raw.sum()
    assert (raw & ~tidied).sum() == 0


def test_fill_small_holes_fills_enclosed_gaps_only():
    mask = np.zeros((200, 200), bool)
    mask[40:160, 40:160] = True
    mask[90:96, 90:96] = False        # enclosed pinhole, 36 px
    mask[40:70, 40:50] = False        # a bite out of the edge of the square

    filled = fill_small_holes(mask, max_area=100.0)
    assert filled[92, 92], "enclosed hole not filled"
    assert not filled[50, 45], "a notch open to the outside must be left alone"


def test_fill_small_holes_leaves_large_holes_alone():
    mask = np.zeros((200, 200), bool)
    mask[20:180, 20:180] = True
    mask[60:140, 60:140] = False      # a window: 6400 px
    filled = fill_small_holes(mask, max_area=1000.0)
    assert not filled[100, 100]


def test_fill_small_holes_ignores_background_touching_the_border():
    mask = np.zeros((100, 100), bool)
    mask[10:90, 10:90] = True
    filled = fill_small_holes(mask, max_area=1e9)
    assert not filled[0, 0]


def test_transfer_mask_closes_pinholes_left_by_resolution_mismatch(decoded_hires):
    """Where the projector out-resolves the camera, scattering leaves gaps."""
    lit = decoded_hires.illumination >= 0.08
    raw = scatter_mask(decoded_hires, lit)
    interior = cv2.erode(raw.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    tidied = transfer_mask(decoded_hires, lit)
    # Anything well inside the mask must be solid after tidying.
    assert tidied[interior].all()


def test_closing_can_be_switched_off(decoded_hires):
    lit = decoded_hires.illumination >= 0.08
    cfg = TransferConfig(close_kernel_px=0, max_hole_area_px=0)
    assert np.array_equal(transfer_mask(decoded_hires, lit, cfg),
                          scatter_mask(decoded_hires, lit, cfg))


def test_masks_transfer_at_the_low_resolution_too(sim, decoded):
    """Nothing here should depend on the generous hi-res fixture."""
    mask = house_mask(decoded)
    assert mask.shape == (decoded.projector_height, decoded.projector_width)
    assert 0.2 < mask.mean() < 0.9


# --------------------------------------------------------------------------- #
# Ring spike rejection
# --------------------------------------------------------------------------- #
def test_an_isolated_spike_is_removed():
    """A bad lookup jumps away and straight back."""
    from facade_scan.transfer import drop_ring_spikes

    ring = np.array([[float(x), 100.0] for x in range(0, 120, 4)])
    ring[10] = [900.0, 900.0]                     # one wild vertex
    cleaned = drop_ring_spikes(ring, threshold=20.0)
    assert len(cleaned) == len(ring) - 1
    assert not (np.abs(cleaned - np.array([900.0, 900.0])).sum(axis=1) < 1).any()


def test_a_genuine_depth_step_is_kept():
    """The bend that lets an edge follow a real surface must survive.

    Everything after the step sits on the nearer surface, so each point is close
    to one neighbour and far from the other -- not a spike.
    """
    from facade_scan.transfer import drop_ring_spikes

    ring = np.array([[float(x), 100.0] for x in range(0, 120, 4)])
    ring[15:, 0] += 60.0                          # a sustained offset
    assert len(drop_ring_spikes(ring, threshold=20.0)) == len(ring)


def test_spike_rejection_leaves_a_clean_ring_alone():
    from facade_scan.transfer import drop_ring_spikes

    ring = np.array([[float(x), 100.0] for x in range(0, 120, 4)])
    assert np.array_equal(drop_ring_spikes(ring, threshold=20.0), ring)


def test_spike_rejection_handles_tiny_rings():
    from facade_scan.transfer import drop_ring_spikes

    for n in (0, 1, 2):
        ring = np.zeros((n, 2))
        assert len(drop_ring_spikes(ring, threshold=5.0)) == n


def test_transferred_regions_have_no_spikes_on_a_real_scan(decoded_hires, sim_hires):
    """End to end: nothing in a transferred outline flies off into space."""
    from facade_scan.detect import detect

    detection = detect(sim_hires.white, DetectConfig(),
                       glass_mask=decoded_hires.likely_glass,
                       illumination=decoded_hires.illumination)
    result = transfer_regions(decoded_hires, detection.regions)
    assert result.regions

    for region in result.regions:
        for ring in region.rings:
            if len(ring) < 3:
                continue
            steps = np.linalg.norm(np.diff(np.vstack([ring, ring[:1]]), axis=0), axis=1)
            # No single hop should dwarf the ring's own extent.
            extent = np.ptp(ring, axis=0).max()
            assert steps.max() <= extent + 1.0, f"{region.label} has a stray vertex"


# --------------------------------------------------------------------------- #
# Tangled rings
# --------------------------------------------------------------------------- #
def test_a_sane_outline_is_plausible():
    from facade_scan.transfer import ring_is_plausible

    square = np.array([[0.0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    assert ring_is_plausible(square, 8.0)
    dense = densify(square, 2.0)
    assert ring_is_plausible(dense, 8.0), "densifying must not make it implausible"


def test_a_zigzag_ring_is_rejected():
    """The failure spike rejection cannot see.

    Points alternating between two surfaces are each far from both neighbours
    *and* the neighbours are far from each other, so nothing local tells it
    apart from a real bend. Its total path length does.
    """
    from facade_scan.transfer import drop_ring_spikes, ring_is_plausible

    # Spaced so that each point's neighbours are far from *each other* too,
    # which is precisely the case spike rejection cannot see: it looks for a
    # point that departs and returns, and here nothing returns.
    #
    # Ratios measured on a real scan: sane outlines came in at 2.0-3.2, one
    # borderline at 6.8, and the ring that projected as a bundle of parallel
    # beams at 17.6. The default of 8 sits in that gap.
    n = 40
    zigzag = np.empty((n, 2))
    zigzag[:, 0] = np.arange(n) * 15.0
    zigzag[:, 1] = np.where(np.arange(n) % 2 == 0, 0.0, 300.0)

    assert len(drop_ring_spikes(zigzag, threshold=20.0)) == n, \
        "spike rejection should leave this one alone"
    assert not ring_is_plausible(zigzag, 8.0), "the perimeter test should catch it"


def test_spike_rejection_still_handles_a_tight_zigzag():
    """Where the neighbours *are* close together, it is a run of spikes."""
    from facade_scan.transfer import drop_ring_spikes

    n = 60
    zigzag = np.empty((n, 2))
    zigzag[:, 0] = np.linspace(0, 100, n)
    zigzag[:, 1] = np.where(np.arange(n) % 2 == 0, 0.0, 180.0)
    assert len(drop_ring_spikes(zigzag, threshold=20.0)) < 5


def test_a_genuinely_bent_outline_survives():
    """A real depth step bends an edge once; it does not make it zigzag."""
    from facade_scan.transfer import ring_is_plausible

    ring = np.array([[float(x), 0.0] for x in range(0, 100, 4)]
                    + [[float(x), 60.0] for x in range(100, 0, -4)])
    assert ring_is_plausible(ring, 8.0)


def test_a_tangled_region_is_dropped_rather_than_projected(decoded_hires):
    """Better to show nothing than to throw beams across the house."""
    cfg = TransferConfig(ring_max_perimeter_ratio=1.2)   # reject almost anything
    region = Region(polygon=np.array([[300.0, 250.0], [900.0, 250.0],
                                      [900.0, 600.0], [300.0, 600.0]]))
    assert transfer_region(decoded_hires, region, cfg) is None

    result = transfer_regions(decoded_hires, [region], cfg)
    assert result.regions == []
    assert "tangled" in result.dropped[0]


def test_the_check_can_be_disabled(decoded_hires):
    from facade_scan.transfer import ring_is_plausible

    assert ring_is_plausible(np.array([[0.0, 0], [1, 50], [2, 0], [3, 50]]), 0.0)


# --------------------------------------------------------------------------- #
# Foreground isolation by depth
# --------------------------------------------------------------------------- #
def test_the_residual_is_near_zero_on_a_single_plane(decoded_hires, sim_hires):
    from facade_scan.transfer import dominant_plane_residual

    residual = dominant_plane_residual(decoded_hires)
    facade = sim_hires.surface_mask("main_facade") & decoded_hires.valid
    assert facade.sum() > 5000
    assert np.median(residual[facade]) < 2.0


def test_the_bump_out_stands_out_from_the_dominant_plane(decoded_hires, sim_hires):
    """Geometry, not brightness.

    The garage front is a perfectly ordinary painted surface; nothing in a
    photograph separates it from the wall. It is a metre nearer, and the
    correspondence says so.
    """
    from facade_scan.transfer import dominant_plane_residual

    residual = dominant_plane_residual(decoded_hires)
    facade = sim_hires.surface_mask("main_facade") & decoded_hires.valid
    garage = sim_hires.surface_mask("garage_face", "garage_door") & decoded_hires.valid
    assert garage.sum() > 2000
    assert np.median(residual[garage]) > 2 * max(np.median(residual[facade]), 0.3)


def test_foreground_mask_picks_out_what_stands_in_front(decoded_hires_wide,
                                                        sim_hires_wide):
    """With enough baseline to resolve the depth step."""
    from facade_scan.transfer import foreground_mask

    mask = foreground_mask(decoded_hires_wide)
    garage = sim_hires_wide.surface_mask("garage_face", "garage_door", "gable_garage")
    facade = sim_hires_wide.surface_mask("main_facade", "gable_main")

    assert mask.sum() > 2000
    assert mask[garage & decoded_hires_wide.valid].mean() > 0.9, "covers the bump-out"
    assert mask[facade & decoded_hires_wide.valid].mean() < 0.1, "excludes the wall"


def test_foreground_separation_degrades_with_a_short_baseline(
    decoded_hires, sim_hires, decoded_hires_wide, sim_hires_wide
):
    """The trade-off, measured.

    Depth resolution is stereo disparity, so it scales with the very baseline
    you otherwise want small to avoid shadows. Worth knowing which you are
    optimising for.
    """
    from facade_scan.transfer import foreground_mask

    def covered(decoded, sim):
        mask = foreground_mask(decoded)
        garage = sim.surface_mask("garage_face", "garage_door", "gable_garage")
        return mask[garage & decoded.valid].mean()

    narrow = covered(decoded_hires, sim_hires)
    wide = covered(decoded_hires_wide, sim_hires_wide)
    assert wide > narrow + 0.2, "a longer baseline should separate depth better"
    assert narrow > 0.3, "...but a short one is not useless"


def test_foreground_is_smaller_than_the_whole_house_mask(decoded_hires):
    from facade_scan.transfer import foreground_projector_mask, house_mask

    everything = house_mask(decoded_hires)
    front = foreground_projector_mask(decoded_hires)
    assert 0 < front.sum() < everything.sum()
    assert front.shape == everything.shape


def test_an_explicit_residual_threshold_overrides_the_automatic_one(decoded_hires):
    from facade_scan.transfer import foreground_mask

    generous = foreground_mask(decoded_hires, TransferConfig(foreground_residual_px=0.5))
    strict = foreground_mask(decoded_hires, TransferConfig(foreground_residual_px=500.0))
    assert generous.sum() > strict.sum()
    assert strict.sum() == 0, "nothing is 500 projector pixels off the plane"


def test_foreground_on_an_undecodable_scan_is_empty(decoded_hires):
    import copy

    from facade_scan.transfer import foreground_mask

    blank = copy.copy(decoded_hires)
    blank.valid = np.zeros_like(decoded_hires.valid)
    assert not foreground_mask(blank).any()


def test_only_the_largest_piece_survives():
    """Scatter leaves a spray of stray pixels; the subject is one object."""
    from facade_scan.transfer import largest_blob

    mask = np.zeros((200, 200), bool)
    mask[50:150, 50:150] = True          # the subject
    mask[10, 10] = True                  # strays
    mask[180:184, 180:184] = True
    kept = largest_blob(mask)
    assert kept[100, 100]
    assert not kept[10, 10] and not kept[181, 181]
    assert kept.sum() == 100 * 100


def test_largest_blob_leaves_a_single_piece_alone():
    from facade_scan.transfer import largest_blob

    mask = np.zeros((100, 100), bool)
    mask[20:80, 20:80] = True
    assert np.array_equal(largest_blob(mask), mask)
    assert not largest_blob(np.zeros((10, 10), bool)).any()


def test_inset_pulls_the_boundary_inward():
    from facade_scan.transfer import inset_mask

    mask = np.zeros((100, 100), bool)
    mask[20:80, 20:80] = True
    pulled = inset_mask(mask, 3)
    assert pulled.sum() < mask.sum()
    assert pulled[50, 50], "the middle stays lit"
    assert not pulled[20, 50], "the original edge goes dark"
    assert (mask & ~pulled).any() and not (pulled & ~mask).any()


def test_inset_of_zero_changes_nothing():
    from facade_scan.transfer import inset_mask

    mask = np.zeros((60, 60), bool)
    mask[10:50, 10:50] = True
    assert np.array_equal(inset_mask(mask, 0), mask)


def test_the_projector_foreground_is_one_clean_inset_shape(decoded_hires_wide):
    from facade_scan.transfer import foreground_projector_mask, largest_blob

    plain = foreground_projector_mask(
        decoded_hires_wide,
        TransferConfig(foreground_single_blob=False, foreground_inset_px=0.0))
    tidy = foreground_projector_mask(decoded_hires_wide)

    assert tidy.any()
    assert tidy.sum() < plain.sum(), "single blob plus inset should only remove"
    assert np.array_equal(largest_blob(tidy), tidy), "must be one connected piece"
