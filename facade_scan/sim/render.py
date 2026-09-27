"""Synthetic optical path: project a pattern, render what the camera sees.

This is the test harness for the whole project, and it exists *before* any
capture hardware on purpose. The decoder cannot be trusted until it has been
scored against a scene whose camera-to-projector correspondence is known
exactly, and no real scan can ever provide that.

The path modelled, per camera pixel:

1. Cast a ray from the camera centre through the pixel into the scene and take
   the nearest surface hit. Misses are background (no house there).
2. Project that world point back into the projector to get the ground-truth
   projector coordinate.
3. Check whether the projector can actually see the point: in front of it,
   inside its frustum, and not occluded by another surface. Points failing the
   occlusion test are in **projector shadow** -- the garage bump-out throws one
   onto the facade beside it, and the eave throws one across the top of the
   wall. They receive ambient light only.
4. Sample the pattern at the projector pixel, attenuate by surface albedo, a
   Lambertian term and inverse-square falloff, add ambient, blur by the
   projector's defocus, and add gaussian sensor noise.

Everything in step 1-3 depends only on geometry, so it is computed once into a
:class:`SceneCache` and reused for all 40-odd frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import SimConfig
from ..patterns import Manifest, render_frame, write_patterns
from .geometry import Scene, load_house
from .optics import Pinhole


@dataclass
class SceneCache:
    """Geometry-only precomputation, shared by every rendered frame."""

    #: (H, W) camera ray struck a surface.
    hit: np.ndarray
    #: (H, W) distance from the camera to that surface, inf on a miss.
    depth: np.ndarray
    #: (H, W) index into ``Scene.surfaces``, -1 on a miss.
    surface: np.ndarray
    #: (H, W) diffuse reflectance of the surface seen.
    albedo: np.ndarray
    #: (H, W, 3) world point seen by each camera pixel.
    point: np.ndarray

    #: (H, W, 2) **ground truth** projector coordinate for each camera pixel.
    proj_uv: np.ndarray
    #: (H, W) the projector reaches this point: in front, in frustum, unshadowed.
    lit: np.ndarray
    #: (H, W) hit geometry that the projector is *blocked* from -- a true
    #: projector shadow, thrown by something else in the scene. This is
    #: geometry no decoder can recover, because there is no light on it.
    shadowed: np.ndarray
    #: (H, W) hit geometry that simply falls outside the projector's image.
    #: Not a shadow: aim or zoom differently and it would be lit. Kept separate
    #: because conflating the two makes a wide backdrop look like a failure.
    outside_frustum: np.ndarray
    #: (H, W) albedo * lambert * falloff * (1 - transmission) -- the gain
    #: applied to pattern intensity. Low on glass, which is the point.
    gain: np.ndarray
    #: (H, W) pattern-independent glare added to every frame.
    glare: np.ndarray

    @property
    def ground_truth_valid(self) -> np.ndarray:
        """Pixels a perfect decoder could decode: lit house surface."""
        return self.hit & self.lit

    @property
    def unlit(self) -> np.ndarray:
        """Hit geometry the projector does not light, for either reason."""
        return self.shadowed | self.outside_frustum

    def save(self, path: str | Path) -> Path:
        """Write ground truth for the decoder to be scored against."""
        path = Path(path)
        np.savez_compressed(
            path,
            proj_uv=self.proj_uv.astype(np.float32),
            valid=self.ground_truth_valid,
            hit=self.hit,
            shadowed=self.shadowed,
            outside_frustum=self.outside_frustum,
            depth=np.where(self.hit, self.depth, 0.0).astype(np.float32),
            surface=self.surface.astype(np.int16),
            albedo=self.albedo.astype(np.float32),
            gain=self.gain.astype(np.float32),
        )
        return path


def build_scene_cache(scene: Scene, camera: Pinhole, projector: Pinhole,
                      cfg: SimConfig) -> SceneCache:
    """Ray-cast the scene once and work out the full projector visibility."""
    dirs = camera.pixel_rays()                       # (H, W, 3)
    origin = camera.center[None, None, :]
    hit = scene.raycast(origin, dirs)
    point = hit.point(origin, dirs)

    albedos = np.concatenate([scene.albedos, [0.0]])  # index -1 -> 0 albedo
    albedo = albedos[hit.surface] * hit.hit
    transmission = np.concatenate([scene.transmissions, [0.0]])[hit.surface] * hit.hit
    specular = np.concatenate([scene.speculars, [0.0]])[hit.surface] * hit.hit
    normals = np.concatenate([scene.normals, [[0.0, 0.0, 1.0]]])
    normal = normals[hit.surface]

    proj_uv, in_front = projector.project(point)
    reachable = hit.hit & in_front & projector.in_frame(proj_uv)

    # --- projector shadow -------------------------------------------------- #
    # Cast from the projector back to each visible point and see whether
    # anything gets in the way. This is what makes the garage bump-out behave
    # like a real bump-out instead of a texture.
    to_point = point - projector.center[None, None, :]
    distance = np.linalg.norm(to_point, axis=-1)
    safe_distance = np.where(distance > 1e-9, distance, 1.0)
    proj_dirs = to_point / safe_distance[..., None]
    if cfg.shadows:
        blocked = np.zeros_like(reachable)
        idx = np.nonzero(reachable)
        if idx[0].size:
            blocked[idx] = scene.occluded(
                projector.center[None, :], proj_dirs[idx], distance[idx]
            )
    else:
        blocked = np.zeros_like(reachable)

    lit = reachable & ~blocked
    shadowed = reachable & blocked           # blocked by something in the way
    outside_frustum = hit.hit & ~reachable   # simply not in the projected image

    # --- radiometry -------------------------------------------------------- #
    # |n . d| rather than a signed dot so that surface winding in house.toml
    # does not silently black out a wall.
    cosine = np.abs(np.sum(normal * proj_dirs, axis=-1))
    lw = float(np.clip(cfg.lambert, 0.0, 1.0))
    shading = (1.0 - lw) + lw * cosine
    falloff = (cfg.falloff_ref_m / np.maximum(safe_distance, 1e-6)) ** 2
    gain = albedo * shading * falloff * (1.0 - transmission)

    return SceneCache(
        hit=hit.hit, depth=hit.t, surface=hit.surface, albedo=albedo, point=point,
        proj_uv=proj_uv, lit=lit, shadowed=shadowed,
        outside_frustum=outside_frustum, gain=gain,
        glare=specular * hit.hit,
    )


def render_capture(cache: SceneCache, pattern: np.ndarray, cfg: SimConfig,
                   rng: np.random.Generator) -> np.ndarray:
    """Render one camera image of ``pattern`` being projected onto the scene."""
    import cv2

    h, w = pattern.shape[:2]
    # A projector is a pixel grid, so sample it nearest-neighbour. Anything
    # smoother would quietly make the finest Gray planes easier to decode than
    # they are in reality.
    u = np.clip(np.rint(cache.proj_uv[..., 0]).astype(np.int32), 0, w - 1)
    v = np.clip(np.rint(cache.proj_uv[..., 1]).astype(np.int32), 0, h - 1)
    value = pattern[v, u].astype(np.float32) / 255.0
    value *= cache.lit

    # Projector defocus lives between the panel and the wall, so it blurs the
    # projected pattern but not the camera's view of surface albedo edges.
    if cfg.blur_px > 0:
        value = np.asarray(
            cv2.GaussianBlur(value, (0, 0), sigmaX=float(cfg.blur_px),
                             sigmaY=float(cfg.blur_px),
                             borderType=cv2.BORDER_REPLICATE),
            dtype=np.float32,
        )

    image = cache.albedo * cfg.ambient + cache.glare + cache.gain * value
    if cfg.noise_sigma > 0:
        image = image + rng.normal(0.0, cfg.noise_sigma, size=image.shape)
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


@dataclass
class SimulationResult:
    manifest: Manifest
    cache: SceneCache
    capture_dir: Path
    pattern_dir: Path
    ground_truth_path: Path
    camera: Pinhole
    projector: Pinhole


def simulate(out_dir: str | Path, cfg: SimConfig | None = None,
             scene: Scene | None = None,
             progress: object = None) -> SimulationResult:
    """Render a complete synthetic capture set plus its ground truth.

    Produces, under ``out_dir``:

    - ``patterns/``      the projected frames and their manifest
    - ``captures/``      one camera image per frame, named to match the manifest
    - ``ground_truth.npz``  the exact camera->projector map

    The capture filenames deliberately match the pattern filenames, so the
    ``folder`` capture backend can consume the result unmodified.
    """
    import cv2

    cfg = cfg or SimConfig()
    scene = scene or load_house(cfg.house_path)
    camera = Pinhole.from_config(cfg.camera)
    projector = Pinhole.from_config(cfg.projector)

    out = Path(out_dir)
    pattern_dir = out / "patterns"
    capture_dir = out / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)

    manifest = write_patterns(pattern_dir, cfg.projector.width, cfg.projector.height)
    cache = build_scene_cache(scene, camera, projector, cfg)
    rng = np.random.default_rng(cfg.random_seed)

    for frame in manifest.frames:
        pattern = render_frame(frame, cfg.projector.width, cfg.projector.height,
                               manifest.bits_x, manifest.bits_y)
        image = render_capture(cache, pattern, cfg, rng)
        if not cv2.imwrite(str(capture_dir / frame.filename), image):
            raise OSError(f"failed to write {capture_dir / frame.filename}")
        if callable(progress):
            progress(frame)

    manifest.write(capture_dir / "manifest.json")
    gt_path = cache.save(out / "ground_truth.npz")
    return SimulationResult(
        manifest=manifest, cache=cache, capture_dir=capture_dir,
        pattern_dir=pattern_dir, ground_truth_path=gt_path,
        camera=camera, projector=projector,
    )


def load_ground_truth(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
