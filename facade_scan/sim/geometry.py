"""Planar scene geometry and a vectorised ray caster.

The synthetic house is a set of planar polygons in world coordinates. World
axes are ``+X`` right, ``+Y`` up (0 = ground), ``+Z`` away from the viewer, so
a facade 14 metres from the projector sits at ``z = 14``.

Polygons may carry coplanar **holes**. That is what makes a recessed window
possible without a full CSG modeller: the facade polygon has a rectangular hole
where the window is, and a second polygon sits 200mm further back behind it.
Rays passing through the hole reach the recessed pane, which gives a genuine
depth discontinuity at the window reveal -- exactly the thing a single
homography cannot model and the thing the decoder has to survive.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# A ray must travel at least this far to count as a hit, so a ray leaving a
# surface does not immediately re-hit it.
RAY_EPS = 1e-6


@dataclass
class Surface:
    """A planar polygon, optionally with coplanar holes cut out of it."""

    name: str
    #: (N, 3) outer boundary in world coordinates, N >= 3, any winding.
    polygon: np.ndarray
    #: Diffuse reflectance in [0, 1]. Dark brick ~0.25, white soffit ~0.8.
    albedo: float = 0.5
    #: List of (M, 3) coplanar hole boundaries.
    holes: list[np.ndarray] = field(default_factory=list)
    #: Fraction of the projected pattern that passes straight through and never
    #: comes back. Window glass is ~0.85; masonry is 0. This is what destroys
    #: pattern contrast on windows.
    transmission: float = 0.0
    #: View-independent glare added regardless of what is being projected --
    #: the projector's own body reflected in the pane, a streetlight, the moon.
    #: It raises brightness while carrying no pattern information, which is the
    #: other half of the glass signature.
    specular: float = 0.0

    # Derived plane frame, filled in __post_init__.
    origin: np.ndarray = field(init=False, repr=False)
    normal: np.ndarray = field(init=False, repr=False)
    e1: np.ndarray = field(init=False, repr=False)
    e2: np.ndarray = field(init=False, repr=False)
    poly2d: np.ndarray = field(init=False, repr=False)
    holes2d: list[np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.polygon = np.asarray(self.polygon, dtype=np.float64).reshape(-1, 3)
        if len(self.polygon) < 3:
            raise ValueError(f"surface {self.name!r} needs at least 3 vertices")
        self.holes = [np.asarray(h, dtype=np.float64).reshape(-1, 3) for h in self.holes]

        self.origin = self.polygon[0]
        # Newell's method: robust normal for any planar polygon, including
        # slivers where a simple cross product of two edges is ill-conditioned.
        p = self.polygon
        q = np.roll(p, -1, axis=0)
        n = np.array([
            np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])),
            np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])),
            np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])),
        ])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise ValueError(f"surface {self.name!r} is degenerate (zero area)")
        self.normal = n / norm

        e1 = self.polygon[1] - self.polygon[0]
        e1 = e1 - self.normal * float(e1 @ self.normal)
        e1_norm = np.linalg.norm(e1)
        if e1_norm < 1e-12:
            raise ValueError(f"surface {self.name!r} has a zero-length first edge")
        self.e1 = e1 / e1_norm
        self.e2 = np.cross(self.normal, self.e1)

        self.poly2d = self.to_plane(self.polygon)
        self.holes2d = [self.to_plane(h) for h in self.holes]

        for hole in self.holes:
            offsets = (hole - self.origin) @ self.normal
            if np.abs(offsets).max() > 1e-6:
                raise ValueError(
                    f"hole in surface {self.name!r} is not coplanar with it "
                    f"(max offset {np.abs(offsets).max():.4g} m)"
                )

    def to_plane(self, points: np.ndarray) -> np.ndarray:
        """Project world points onto this surface's 2-D plane basis."""
        rel = np.asarray(points, dtype=np.float64) - self.origin
        return np.stack([rel @ self.e1, rel @ self.e2], axis=-1)

    def contains_plane_points(self, pts2d: np.ndarray) -> np.ndarray:
        """Point-in-polygon for already-projected 2-D points, holes removed."""
        inside = _crossing_number(pts2d, self.poly2d)
        for hole in self.holes2d:
            inside &= ~_crossing_number(pts2d, hole)
        return inside


def _crossing_number(pts: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point-in-polygon test.

    ``pts`` is (..., 2), ``verts`` is (N, 2). Works for convex and concave
    polygons alike, which matters because a gable is a triangle and a facade
    with a bumped-out corner need not be convex.
    """
    px, py = pts[..., 0], pts[..., 1]
    inside = np.zeros(px.shape, dtype=bool)
    n = len(verts)
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        if ay == by:
            continue  # horizontal edge contributes no crossing
        straddles = (ay > py) != (by > py)
        x_at_py = ax + (py - ay) * (bx - ax) / (by - ay)
        inside ^= straddles & (px < x_at_py)
    return inside


@dataclass
class Hit:
    """Result of casting a batch of rays."""

    #: (...,) True where the ray struck a surface.
    hit: np.ndarray
    #: (...,) distance along the (unit) ray direction. inf where no hit.
    t: np.ndarray
    #: (...,) index into ``Scene.surfaces``. -1 where no hit.
    surface: np.ndarray

    def point(self, origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        t = np.where(self.hit, self.t, 0.0)
        return origins + t[..., None] * dirs


@dataclass
class Scene:
    """A collection of planar surfaces."""

    surfaces: list[Surface]

    @property
    def albedos(self) -> np.ndarray:
        return np.array([s.albedo for s in self.surfaces], dtype=np.float64)

    @property
    def transmissions(self) -> np.ndarray:
        return np.array([s.transmission for s in self.surfaces], dtype=np.float64)

    @property
    def speculars(self) -> np.ndarray:
        return np.array([s.specular for s in self.surfaces], dtype=np.float64)

    def index_of(self, name: str) -> int:
        for i, s in enumerate(self.surfaces):
            if s.name == name:
                return i
        raise KeyError(f"no surface named {name!r}")

    @property
    def normals(self) -> np.ndarray:
        return np.stack([s.normal for s in self.surfaces])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        pts = np.concatenate([s.polygon for s in self.surfaces])
        return pts.min(axis=0), pts.max(axis=0)

    # ------------------------------------------------------------- casting --
    def raycast(self, origins: np.ndarray, dirs: np.ndarray,
                t_min: float = RAY_EPS) -> Hit:
        """Cast rays against every surface and keep the nearest hit.

        ``origins`` and ``dirs`` broadcast against each other; ``dirs`` should
        be unit length so that ``t`` is a distance in metres.
        """
        origins = np.asarray(origins, dtype=np.float64)
        dirs = np.asarray(dirs, dtype=np.float64)
        shape = np.broadcast_shapes(origins.shape[:-1], dirs.shape[:-1])
        origins = np.broadcast_to(origins, (*shape, 3))
        dirs = np.broadcast_to(dirs, (*shape, 3))

        best_t = np.full(shape, np.inf)
        best_surface = np.full(shape, -1, dtype=np.int32)

        for idx, surf in enumerate(self.surfaces):
            denom = dirs @ surf.normal
            # Rays parallel to the plane never hit it (and would divide by ~0).
            usable = np.abs(denom) > 1e-12
            if not usable.any():
                continue
            num = (surf.origin - origins) @ surf.normal
            t = np.full(shape, np.inf)
            np.divide(num, denom, out=t, where=usable)

            # Only bother with the point-in-polygon test where this surface
            # could actually win: in front of the ray and nearer than the best
            # hit so far. On a frontal facade that culls very little, but on
            # side returns and gables it culls almost everything.
            candidate = usable & (t > t_min) & (t < best_t)
            if not candidate.any():
                continue

            pts = origins[candidate] + t[candidate][:, None] * dirs[candidate]
            inside = surf.contains_plane_points(surf.to_plane(pts))

            accept = np.zeros(shape, dtype=bool)
            accept[candidate] = inside
            best_t = np.where(accept, t, best_t)
            best_surface = np.where(accept, np.int32(idx), best_surface)

        return Hit(hit=best_surface >= 0, t=best_t, surface=best_surface)

    def occluded(self, origins: np.ndarray, dirs: np.ndarray, distance: np.ndarray,
                 eps: float = 2e-3) -> np.ndarray:
        """True where something blocks the ray before ``distance``.

        ``eps`` (2mm) keeps the target surface itself from counting as its own
        occluder given floating-point round-off along the ray.
        """
        hit = self.raycast(origins, dirs, t_min=RAY_EPS)
        return hit.hit & (hit.t < distance - eps)

    # ---------------------------------------------------------------- load --
    @classmethod
    def from_toml(cls, path: str | Path) -> Scene:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> Scene:
        entries = data.get("surface")
        if not entries:
            raise ValueError("house config has no [[surface]] entries")
        surfaces = []
        for i, entry in enumerate(entries):
            unknown = set(entry) - {"name", "polygon", "albedo", "holes",
                                    "transmission", "specular"}
            if unknown:
                raise ValueError(f"unknown surface keys: {sorted(unknown)}")
            surfaces.append(
                Surface(
                    name=entry.get("name", f"surface_{i}"),
                    polygon=np.array(entry["polygon"], dtype=np.float64),
                    albedo=float(entry.get("albedo", 0.5)),
                    holes=[np.array(h, dtype=np.float64) for h in entry.get("holes", [])],
                    transmission=float(entry.get("transmission", 0.0)),
                    specular=float(entry.get("specular", 0.0)),
                )
            )
        return cls(surfaces=surfaces)


def default_house_path() -> Path:
    return Path(__file__).with_name("house.toml")


def load_house(path: str | Path | None = None) -> Scene:
    """Load the house geometry, defaulting to the bundled one."""
    return Scene.from_toml(path or default_house_path())
