"""Pinhole camera and projector models.

A projector is optically a camera run backwards, so both are the same class.
Device coordinates follow the OpenCV convention: ``+x`` right, ``+y`` down,
``+z`` forward along the optical axis, with ``u = fx*X/Z + cx``.

Handedness
----------
The world frame is ``+X`` right, ``+Y`` up, ``+Z`` away from the devices, which
is a *left*-handed frame (the Unity/DirectX convention). It is the natural one
for authoring a house -- ground at ``y = 0``, facade at ``z = 14`` -- so we keep
it, and :func:`look_at` accordingly returns an orthogonal matrix with
determinant -1 rather than a determinant +1 rotation.

That is deliberate and harmless. The matrix is only ever used as a change of
basis between world and device, and because it is orthogonal,
:meth:`Pinhole.pixel_rays` and :meth:`Pinhole.project` remain exact inverses of
one another, which is the only property the simulator relies on. Had we insisted
on determinant +1 here, world ``+X`` would render on the *left* of the image and
every edit to ``house.toml`` would come out mirrored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SimPoseConfig

WORLD_UP = np.array([0.0, 1.0, 0.0])


def look_at(position: np.ndarray, target: np.ndarray,
            world_up: np.ndarray = WORLD_UP) -> np.ndarray:
    """World-to-device basis whose rows are (right, down, forward).

    Orthogonal, with determinant -1; see the module docstring for why.
    """
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - position
    n = np.linalg.norm(forward)
    if n < 1e-9:
        raise ValueError("device position and target coincide")
    forward /= n
    right = np.cross(world_up, forward)
    rn = np.linalg.norm(right)
    if rn < 1e-9:
        raise ValueError("device is aimed straight up or down; pick another world_up")
    right /= rn
    down = np.cross(right, forward)
    return np.stack([right, down, forward])


@dataclass
class Pinhole:
    """An ideal pinhole device with square pixels and no distortion."""

    width: int
    height: int
    #: 3x3 intrinsics.
    K: np.ndarray
    #: 3x3 world-to-device rotation.
    R: np.ndarray
    #: Device position in world coordinates (the optical centre).
    center: np.ndarray

    @classmethod
    def from_config(cls, cfg: SimPoseConfig) -> Pinhole:
        fx = (cfg.width / 2.0) / np.tan(np.radians(cfg.hfov_deg) / 2.0)
        fy = fx  # square pixels
        K = np.array([
            [fx, 0.0, (cfg.width - 1) / 2.0],
            [0.0, fy, (cfg.height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ])
        center = np.asarray(cfg.position, dtype=np.float64)
        return cls(width=cfg.width, height=cfg.height, K=K,
                   R=look_at(center, np.asarray(cfg.target, dtype=np.float64)),
                   center=center)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def vfov_deg(self) -> float:
        return float(2 * np.degrees(np.arctan((self.height / 2.0) / self.K[1, 1])))

    # ------------------------------------------------------------------------
    def pixel_rays(self) -> np.ndarray:
        """Unit world-space direction for every pixel, as ``(height, width, 3)``."""
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float64),
                           np.arange(self.height, dtype=np.float64))
        x = (u - self.K[0, 2]) / self.K[0, 0]
        y = (v - self.K[1, 2]) / self.K[1, 1]
        d_dev = np.stack([x, y, np.ones_like(x)], axis=-1)
        d_world = d_dev @ self.R  # == (R.T @ d_dev[..., None]).squeeze()
        return d_world / np.linalg.norm(d_world, axis=-1, keepdims=True)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to pixel coordinates.

        Returns ``(uv, in_front)`` where ``uv`` is (..., 2) float and
        ``in_front`` marks points with positive depth. Points behind the device
        get meaningless ``uv``; always check ``in_front``.
        """
        rel = np.asarray(points, dtype=np.float64) - self.center
        cam = rel @ self.R.T
        z = cam[..., 2]
        in_front = z > 1e-9
        safe_z = np.where(in_front, z, 1.0)
        u = self.K[0, 0] * cam[..., 0] / safe_z + self.K[0, 2]
        v = self.K[1, 1] * cam[..., 1] / safe_z + self.K[1, 2]
        return np.stack([u, v], axis=-1), in_front

    def in_frame(self, uv: np.ndarray) -> np.ndarray:
        """True where a pixel coordinate falls inside the panel."""
        u, v = uv[..., 0], uv[..., 1]
        return (u >= -0.5) & (u <= self.width - 0.5) & (v >= -0.5) & (v <= self.height - 0.5)
