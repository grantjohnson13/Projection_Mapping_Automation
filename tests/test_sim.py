"""Stage 2: the synthetic simulator, which is the harness everything else is tested against.

If these fail, no later test means anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from facade_scan.config import SimConfig
from facade_scan.sim import Pinhole, Scene, Surface, build_scene_cache, load_house, look_at
from facade_scan.sim.render import load_ground_truth, simulate

from .conftest import GLASS_SURFACES, build_sim


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_house_has_surfaces_at_several_distinct_depths():
    """The point of the whole project: a facade is not one plane."""
    scene = load_house()
    depths = {round(float(s.polygon[:, 2].mean()), 2) for s in scene.surfaces}
    assert len(depths) >= 4, f"expected several depth layers, got {sorted(depths)}"
    assert max(depths) - min(depths) > 1.0  # more than a metre of relief


def test_named_features_are_present():
    scene = load_house()
    names = {s.name for s in scene.surfaces}
    assert {"main_facade", "garage_face", "gable_main", "gable_garage",
            "eave_soffit", "garage_door", "front_door"} <= names
    assert set(GLASS_SURFACES) <= names


def test_newell_normal_is_unit_and_perpendicular_to_every_edge():
    scene = load_house()
    for s in scene.surfaces:
        assert np.isclose(np.linalg.norm(s.normal), 1.0)
        edges = np.roll(s.polygon, -1, axis=0) - s.polygon
        assert np.allclose(edges @ s.normal, 0.0, atol=1e-9)


def test_plane_basis_is_orthonormal():
    for s in load_house().surfaces:
        basis = np.stack([s.e1, s.e2, s.normal])
        assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-9)


def test_hole_makes_the_surface_transparent_to_rays_and_reveals_what_is_behind():
    """A ray aimed at a window hole must skip the facade and hit the recessed pane."""
    scene = load_house()
    facade = scene.surfaces[scene.index_of("main_facade")]
    pane = scene.surfaces[scene.index_of("window_lower_left")]

    through_hole = np.array([-3.35, 1.8, 0.0])   # centre of the lower-left opening
    through_wall = np.array([-4.3, 1.8, 0.0])    # solid wall just outside it
    origin = np.array([0.0, 1.8, 0.0])

    for target, expected in ((through_hole, pane), (through_wall, facade)):
        d = np.array([target[0] - origin[0], 0.0, 14.0])
        d /= np.linalg.norm(d)
        hit = scene.raycast(origin, d)
        assert bool(hit.hit), f"ray toward {target} hit nothing"
        assert scene.surfaces[int(hit.surface)].name == expected.name

    assert np.isclose(pane.polygon[0, 2], 14.2)  # genuinely recessed, not painted on


def test_raycast_keeps_the_nearest_surface():
    """The garage bump-out must occlude the facade behind it."""
    scene = load_house()
    origin = np.array([0.0, 1.2, 0.0])
    target = np.array([2.8, 1.2, 13.0])
    d = (target - origin) / np.linalg.norm(target - origin)
    hit = scene.raycast(origin, d)
    assert scene.surfaces[int(hit.surface)].name in {"garage_face", "garage_door"}
    assert float(hit.t) < 14.0


def test_raycast_misses_return_no_hit():
    scene = load_house()
    hit = scene.raycast(np.array([0.0, 1.2, 0.0]), np.array([0.0, -1.0, 0.0]))
    assert not bool(hit.hit)
    assert np.isinf(hit.t) and int(hit.surface) == -1


def test_concave_polygon_point_containment():
    """An L-shaped surface, to prove the containment test is not convex-only."""
    l_shape = Surface(
        name="L",
        polygon=np.array([[0, 0, 5], [4, 0, 5], [4, 1, 5], [1, 1, 5], [1, 4, 5], [0, 4, 5]],
                         dtype=float),
    )
    pts = np.array([[0.5, 0.5], [3.5, 0.5], [0.5, 3.5], [3.0, 3.0], [-1.0, -1.0]])
    got = l_shape.contains_plane_points(l_shape.to_plane(
        np.stack([pts[:, 0], pts[:, 1], np.full(len(pts), 5.0)], axis=-1)))
    assert list(got) == [True, True, True, False, False]


def test_degenerate_and_non_coplanar_geometry_is_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        Surface(name="flat", polygon=np.array([[0, 0, 1], [1, 0, 1], [2, 0, 1]], dtype=float))
    with pytest.raises(ValueError, match="at least 3"):
        Surface(name="line", polygon=np.array([[0, 0, 1], [1, 0, 1]], dtype=float))
    with pytest.raises(ValueError, match="coplanar"):
        Surface(name="wall",
                polygon=np.array([[0, 0, 1], [4, 0, 1], [4, 4, 1], [0, 4, 1]], dtype=float),
                holes=[np.array([[1, 1, 1], [2, 1, 1], [2, 2, 2], [1, 2, 1]], dtype=float)])


def test_unknown_surface_key_is_rejected():
    with pytest.raises(ValueError, match="unknown surface keys"):
        Scene.from_dict({"surface": [{"polygon": [[0, 0, 1], [1, 0, 1], [1, 1, 1]],
                                      "colour": "red"}]})


# --------------------------------------------------------------------------- #
# Optics
# --------------------------------------------------------------------------- #
def test_look_at_is_orthogonal_and_oriented_intuitively():
    R = look_at(np.array([0.0, 1.2, 0.0]), np.array([0.0, 1.2, 14.0]))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    right, down, forward = R
    assert np.allclose(right, [1, 0, 0])     # world +X is image right
    assert np.allclose(down, [0, -1, 0])     # world -Y is image down
    assert np.allclose(forward, [0, 0, 1])


def test_look_at_rejects_degenerate_aims():
    with pytest.raises(ValueError, match="coincide"):
        look_at(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="straight up or down"):
        look_at(np.zeros(3), np.array([0.0, 5.0, 0.0]))


def test_pixel_rays_and_project_are_exact_inverses(sim):
    """The simulator's correctness rests on this round trip."""
    dirs = sim.camera.pixel_rays()
    points = sim.camera.center + 11.0 * dirs
    uv, in_front = sim.camera.project(points)
    u, v = np.meshgrid(np.arange(sim.camera.width, dtype=float),
                       np.arange(sim.camera.height, dtype=float))
    assert in_front.all()
    assert np.abs(uv[..., 0] - u).max() < 1e-9
    assert np.abs(uv[..., 1] - v).max() < 1e-9


def test_pixel_rays_are_unit_length(sim):
    assert np.allclose(np.linalg.norm(sim.camera.pixel_rays(), axis=-1), 1.0)


def test_points_behind_the_device_are_flagged(sim):
    behind = sim.camera.center - np.array([0.0, 0.0, 3.0])
    _, in_front = sim.camera.project(behind)
    assert not bool(in_front)


def test_projector_covers_the_whole_house(sim):
    """If the projector cannot reach part of the house, the scan is untestable."""
    house = sim.cache.hit
    assert np.array_equal(house, sim.cache.lit | sim.cache.unlit)
    assert sim.cache.lit[house].mean() > 0.99
    # On the house scene the projector covers everything, so nothing is unlit
    # merely for being out of frame.
    assert sim.cache.outside_frustum.sum() == 0


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_the_camera_sees_a_house_not_a_void(sim):
    assert 0.4 < sim.cache.hit.mean() < 0.9
    visible = {sim.scene.surfaces[i].name for i in np.unique(sim.cache.surface) if i >= 0}
    assert {"main_facade", "garage_face", "garage_door", "gable_main",
            "gable_garage", "eave_soffit", "garage_return_left"} <= visible


def test_the_garage_is_nearer_than_the_facade(sim):
    garage = sim.surface_mask("garage_face")
    facade = sim.surface_mask("main_facade")
    assert sim.cache.depth[garage].mean() < sim.cache.depth[facade].mean() - 0.8


def test_white_frame_is_brighter_than_black_frame_across_the_house(sim):
    lit_house = sim.cache.hit & sim.cache.lit & ~sim.glass_mask
    assert (sim.white[lit_house] > sim.black[lit_house]).mean() > 0.999


def test_background_is_black(sim):
    background = ~sim.cache.hit
    assert sim.white[background].max() <= 12  # noise only


def test_shadowed_pixels_receive_ambient_only(sim_wide_baseline):
    """A projector shadow must be a shadow, not just a darker shade of lit."""
    cache = sim_wide_baseline.cache
    shadow = cache.shadowed & cache.hit
    assert shadow.sum() > 200, "wide-baseline scene should cast real shadows"
    illumination = (sim_wide_baseline.white.astype(float)
                    - sim_wide_baseline.black.astype(float)) / 255.0
    lit = cache.lit & ~sim_wide_baseline.glass_mask
    assert np.median(illumination[shadow]) < 0.02
    assert np.median(illumination[lit]) > 0.3


def test_albedo_cancels_in_the_pattern_versus_inverse_comparison(sim):
    """The reason inverses are not optional.

    A dark surface under a lit stripe is darker than a bright surface under an
    unlit stripe, so no global threshold works -- but the per-pixel comparison
    against the inverse gets both right.
    """
    m = sim.manifest
    bit = m.frames_for("x", False)[4]
    inv = m.frames_for("x", True)[4]
    pattern = sim.images[bit.index].astype(np.float32) / 255.0
    anti = sim.images[inv.index].astype(np.float32) / 255.0

    dark = sim.surface_mask("front_door") & sim.cache.lit      # albedo 0.38
    bright = sim.surface_mask("eave_soffit") & sim.cache.lit   # albedo 0.78
    assert dark.sum() > 100 and bright.sum() > 100

    # Ground truth for this bit, straight from the projector coordinate.
    from facade_scan.patterns import gray_plane
    plane = gray_plane(sim.projector.width, bit.bit, m.bits_x)
    u = np.clip(np.rint(sim.cache.proj_uv[..., 0]).astype(int), 0, sim.projector.width - 1)
    expected = plane[u].astype(bool)

    for name, region in (("dark", dark), ("bright", bright)):
        read = (pattern - anti)[region] > 0
        assert (read == expected[region]).mean() > 0.995, f"{name} surface misread"

    # And confirm a single global threshold really would fail here.
    lit_dark = pattern[dark & expected]          # dark surface, stripe ON
    unlit_bright = pattern[bright & ~expected]   # bright surface, stripe OFF
    assert lit_dark.mean() < unlit_bright.mean() + 0.35


def test_glass_returns_almost_no_pattern_modulation(sim):
    glass = sim.glass_mask & sim.cache.lit
    masonry = sim.surface_mask("main_facade") & sim.cache.lit
    contrast = (sim.white.astype(float) - sim.black.astype(float)) / 255.0
    assert np.median(contrast[glass]) < 0.08
    assert np.median(contrast[masonry]) > 0.4
    # ...but glass is still bright, from glare. That combination is its signature.
    assert np.median(sim.white[glass]) > 25


def test_noise_and_shading_respond_to_config():
    quiet = build_sim(noise_sigma=0.0, lambert=0.0, ambient=0.0)
    loud = build_sim(noise_sigma=0.06, lambert=0.0, ambient=0.0)
    bg = ~quiet.cache.hit
    assert quiet.white[bg].max() == 0
    assert loud.white[bg].std() > 5


def test_ground_truth_map_lands_inside_the_projector_panel(sim):
    valid = sim.ground_truth_valid
    uv = sim.ground_truth_uv[valid]
    assert (uv[:, 0] >= -0.5).all() and (uv[:, 0] <= sim.projector.width - 0.5).all()
    assert (uv[:, 1] >= -0.5).all() and (uv[:, 1] <= sim.projector.height - 0.5).all()


@pytest.mark.parametrize("fixture_name", ["sim", "sim_wide_baseline"])
def test_ground_truth_map_is_smooth_within_a_surface_and_jumps_between_them(
    fixture_name, request
):
    """Depth discontinuities show up as jumps in the correspondence.

    How big a jump is set by stereo disparity, ``fx * B * (1/z1 - 1/z2)``, so it
    grows with the camera-projector baseline ``B`` and with projector
    resolution. Keeping ``B`` small is exactly the physical advice this tool
    gives, which is why the jump at the 0.6 m default baseline is only a couple
    of pixels -- but it is never zero, and it is never something a homography
    could reproduce.
    """
    sim = request.getfixturevalue(fixture_name)
    uv = sim.ground_truth_uv
    valid = sim.ground_truth_valid
    surface = sim.cache.surface
    step = np.linalg.norm(np.diff(uv, axis=1), axis=-1)
    pair_valid = valid[:, :-1] & valid[:, 1:]
    left, right = surface[:, :-1], surface[:, 1:]

    within = step[pair_valid & (left == right)]
    assert np.median(within) < 1.0, "correspondence should be smooth within a surface"

    # The garage gable sits a full metre in front of the facade with no return
    # wall between them, so its silhouette is a true step rather than a ramp.
    facade = sim.scene.index_of("main_facade")
    gable = sim.scene.index_of("gable_garage")
    boundary = pair_valid & (((left == facade) & (right == gable))
                             | ((left == gable) & (right == facade)))
    assert boundary.sum() > 20, "garage gable silhouette not visible"

    baseline = float(np.linalg.norm(sim.camera.center - sim.projector.center))
    predicted = sim.projector.fx * baseline * (1.0 / 13.0 - 1.0 / 14.0)
    measured = float(np.median(step[boundary]))
    # The silhouette is oblique, so the measured step carries a little
    # within-surface gradient on top of the disparity.
    assert predicted <= measured <= predicted + 2.0 * np.median(within)
    assert measured > 2.5 * np.median(within)


def test_depth_discontinuity_disparity_matches_the_stereo_prediction(sim):
    """Check the jump against closed-form stereo disparity, not a magic number.

    Two world points on the same camera ray, one on the garage bump-out and one
    on the facade a metre behind it, must land ``fx * B * (1/z1 - 1/z2)``
    projector pixels apart. If the simulator's two pinholes disagree with this,
    every later stage is being tested against a fiction.
    """
    baseline = float(np.linalg.norm(sim.camera.center - sim.projector.center))
    z_near, z_far = 13.0, 14.0
    predicted = sim.projector.fx * baseline * (1.0 / z_near - 1.0 / z_far)

    # A camera ray aimed just left of the garage's left edge, so the same ray
    # direction reaches both depths within the house.
    direction = np.array([1.30, 1.50, 14.0]) - sim.camera.center
    direction /= np.linalg.norm(direction)
    near = sim.camera.center + direction * (z_near - sim.camera.center[2]) / direction[2]
    far = sim.camera.center + direction * (z_far - sim.camera.center[2]) / direction[2]

    uv_near, _ = sim.projector.project(near)
    uv_far, _ = sim.projector.project(far)
    measured = float(np.linalg.norm(uv_near - uv_far))
    assert measured == pytest.approx(predicted, rel=0.02)
    assert measured > 0.5


# --------------------------------------------------------------------------- #
# simulate() end to end
# --------------------------------------------------------------------------- #
def test_simulate_writes_a_capture_set_the_folder_backend_can_read(tmp_path):
    cfg = SimConfig()
    cfg.projector.width, cfg.projector.height = 160, 120
    cfg.camera.width, cfg.camera.height = 240, 160
    result = simulate(tmp_path, cfg)

    assert result.manifest.num_frames == len(list(result.capture_dir.glob("*.png")))
    assert (result.capture_dir / "manifest.json").exists()
    assert len(list(result.pattern_dir.glob("*.png"))) == result.manifest.num_frames

    # Capture filenames match pattern filenames, so `folder` capture works unmodified.
    assert sorted(p.name for p in result.capture_dir.glob("*.png")) == \
           sorted(p.name for p in result.pattern_dir.glob("*.png"))

    gt = load_ground_truth(result.ground_truth_path)
    assert gt["proj_uv"].shape == (160, 240, 2)
    assert gt["valid"].shape == (160, 240)
    assert gt["valid"].any()


def test_shadows_can_be_disabled_for_debugging():
    with_shadow = build_sim(camera_position=(2.2, 1.1, 0.05))
    without = build_sim(camera_position=(2.2, 1.1, 0.05), shadows=False)
    assert with_shadow.cache.shadowed.sum() > without.cache.shadowed.sum()
    assert without.cache.shadowed.sum() == 0


def test_shadow_and_out_of_frame_are_not_the_same_thing():
    """A backdrop wider than the projected image is not in shadow.

    The tabletop rig has exactly this shape, and conflating the two made a
    perfectly good scene report 25% of the frame as shadow.
    """
    from facade_scan.config import Config

    cfg = Config.from_toml("facade_scan/sim/tabletop.sim.toml")
    scene = load_house(cfg.sim.house_path)
    cache = build_scene_cache(scene, Pinhole.from_config(cfg.sim.camera),
                              Pinhole.from_config(cfg.sim.projector), cfg.sim)

    assert cache.outside_frustum.sum() > 100_000, "backdrop should overspill the image"
    assert not (cache.shadowed & cache.outside_frustum).any()
    assert not (cache.lit & cache.unlit).any()

    # There is a real shadow at the box edge, and it is *tiny* -- a hundred-odd
    # pixels out of two million -- because the camera sits 90 mm from the
    # projector and the box is a closed solid. That is the whole payoff of
    # mounting them together, and it would be invisible if out-of-frame
    # geometry were still being counted as shadow.
    assert 50 < cache.shadowed.sum() < 0.001 * cache.hit.sum()
