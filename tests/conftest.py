"""Shared simulator fixtures.

Every numerical stage in this project is tested against the synthetic scene
rather than against mocks, so almost every test needs a rendered capture set.
Rendering one is a second or two, so the default set is built once per session
and shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from facade_scan.config import Config, DecodeConfig, SimConfig, SimPoseConfig
from facade_scan.decode import DecodeResult, array_loader, decode
from facade_scan.patterns import Manifest, build_manifest, render_frame
from facade_scan.sim import (
    Pinhole,
    Scene,
    SceneCache,
    build_scene_cache,
    load_house,
    render_capture,
)

#: Small enough to keep the suite quick, large enough that the camera still
#: out-resolves the projector by ~1.5x as a real DSLR massively does.
TEST_PROJECTOR = (320, 200)
TEST_CAMERA = (480, 300)

#: Surface names in house.toml that are glazing. Used to score `likely_glass`
#: and to exclude glass from coverage assertions.
GLASS_SURFACES = (
    "window_lower_left", "window_lower_mid", "window_upper_left",
    "window_upper_mid", "window_upper_right",
)


@dataclass
class SimFixture:
    """A rendered synthetic capture set plus everything needed to score it."""

    cfg: SimConfig
    scene: Scene
    camera: Pinhole
    projector: Pinhole
    cache: SceneCache
    manifest: Manifest
    images: dict[int, np.ndarray]

    @property
    def ground_truth_uv(self) -> np.ndarray:
        return self.cache.proj_uv

    @property
    def ground_truth_valid(self) -> np.ndarray:
        return self.cache.ground_truth_valid

    def surface_mask(self, *names: str) -> np.ndarray:
        idx = [self.scene.index_of(n) for n in names]
        return np.isin(self.cache.surface, idx)

    @property
    def glass_mask(self) -> np.ndarray:
        return self.surface_mask(*GLASS_SURFACES)

    @property
    def white(self) -> np.ndarray:
        return self.images[self.manifest.frame_by_role("white").index]

    @property
    def black(self) -> np.ndarray:
        return self.images[self.manifest.frame_by_role("black").index]


def build_sim(projector: tuple[int, int] = TEST_PROJECTOR,
              camera: tuple[int, int] = TEST_CAMERA,
              camera_position: tuple[float, float, float] = (0.60, 1.10, 0.05),
              scene: Scene | None = None,
              **sim_overrides: object) -> SimFixture:
    """Render a synthetic capture set in memory."""
    cfg = SimConfig()
    cfg.projector = SimPoseConfig(projector[0], projector[1], 43.0,
                                  (0.0, 1.20, 0.0), (0.0, 2.85, 14.0))
    cfg.camera = SimPoseConfig(camera[0], camera[1], 41.5,
                               camera_position, (0.0, 2.85, 14.0))
    for key, value in sim_overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"SimConfig has no field {key!r}")
        setattr(cfg, key, value)

    scene = scene if scene is not None else load_house()
    cam = Pinhole.from_config(cfg.camera)
    proj = Pinhole.from_config(cfg.projector)
    cache = build_scene_cache(scene, cam, proj, cfg)

    manifest = build_manifest(cfg.projector.width, cfg.projector.height)
    rng = np.random.default_rng(cfg.random_seed)
    images = {
        frame.index: render_capture(
            cache,
            render_frame(frame, cfg.projector.width, cfg.projector.height,
                         manifest.bits_x, manifest.bits_y),
            cfg, rng,
        )
        for frame in manifest.frames
    }
    return SimFixture(cfg=cfg, scene=scene, camera=cam, projector=proj, cache=cache,
                      manifest=manifest, images=images)


@pytest.fixture(scope="session")
def sim() -> SimFixture:
    """The standard synthetic capture set, rendered once for the whole session."""
    return build_sim()


@pytest.fixture(scope="session")
def sim_wide_baseline() -> SimFixture:
    """Camera 2.2 m from the projector, which throws real projector shadows.

    The physical advice is the opposite -- keep the camera as close to the
    projector as possible, precisely so these shadows stay small -- but the
    shadow-handling code needs a scene where they are big enough to test.
    """
    return build_sim(camera_position=(2.20, 1.10, 0.05))


#: A realistically-sized scan: the camera out-resolves the projector, and the
#: house occupies enough pixels that architectural features are tens of pixels
#: across rather than a handful. Detection and transfer are judged here, because
#: this is what a real capture looks like.
HIRES_PROJECTOR = (960, 600)
HIRES_CAMERA = (1440, 900)


@pytest.fixture(scope="session")
def sim_hires() -> SimFixture:
    return build_sim(projector=HIRES_PROJECTOR, camera=HIRES_CAMERA)


@pytest.fixture(scope="session")
def sim_hires_wide() -> SimFixture:
    """Hi-res with a 2.2 m baseline, where depth is properly resolved."""
    return build_sim(projector=HIRES_PROJECTOR, camera=HIRES_CAMERA,
                     camera_position=(2.20, 1.10, 0.05))


@pytest.fixture(scope="session")
def decoded_hires_wide(sim_hires_wide) -> DecodeResult:
    return decode(array_loader(sim_hires_wide.images), sim_hires_wide.manifest,
                  DecodeConfig())


@pytest.fixture(scope="session")
def decoded(sim) -> DecodeResult:
    return decode(array_loader(sim.images), sim.manifest, DecodeConfig())


@pytest.fixture(scope="session")
def decoded_hires(sim_hires) -> DecodeResult:
    return decode(array_loader(sim_hires.images), sim_hires.manifest, DecodeConfig())


@pytest.fixture(scope="session")
def config() -> Config:
    return Config()


def region_iou(region_polygon: np.ndarray, truth_mask: np.ndarray) -> float:
    """Intersection over union between a camera-space polygon and a mask."""
    import cv2

    drawn = np.zeros(truth_mask.shape, np.uint8)
    cv2.fillPoly(drawn, [np.round(region_polygon).astype(np.int32)], 1)
    drawn = drawn.astype(bool)
    union = (drawn | truth_mask).sum()
    return float((drawn & truth_mask).sum() / union) if union else 0.0


def best_match(regions, truth_mask: np.ndarray):
    """The region overlapping ``truth_mask`` most, and that overlap."""
    best, score = None, 0.0
    for region in regions:
        iou = region_iou(region.polygon, truth_mask)
        if iou > score:
            best, score = region, iou
    return best, score
