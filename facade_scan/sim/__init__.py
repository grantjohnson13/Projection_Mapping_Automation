"""Synthetic scene simulator: the test harness the decoder is scored against."""

from .geometry import Scene, Surface, default_house_path, load_house
from .optics import Pinhole, look_at
from .render import (
    SceneCache,
    SimulationResult,
    build_scene_cache,
    load_ground_truth,
    render_capture,
    simulate,
)

__all__ = [
    "Pinhole",
    "Scene",
    "SceneCache",
    "SimulationResult",
    "Surface",
    "build_scene_cache",
    "default_house_path",
    "load_ground_truth",
    "load_house",
    "look_at",
    "render_capture",
    "simulate",
]
