"""Geometric shapes with known topology for TDA point-cloud sampling.

Each :class:`Shape` knows three things:

* its *intrinsic dimension* ``d`` (1 for curves, 2 for surfaces),
* its *unit measure* (arc length for curves, surface area for surfaces, at
  ``size == 1``), and
* its ground-truth *Betti numbers* ``(b0, b1, b2)``.

Points are sampled uniformly with respect to the intrinsic measure, so the
expected number of points per unit length/area is constant. Because of that,
the point count is controlled purely by ``size``::

    N = density * unit_measure * size ** intrinsic_dim

Scaling a circle by 2 doubles its points; scaling a sphere by 2 quadruples
them. Point and field noise amplitudes are relative to object scale.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import pi

import numpy as np

from .noise import PointNoiseKind, apply_noise

Betti = tuple[int, int, int]
RngLike = int | np.random.Generator | None
StretchLike = float | tuple[float, ...] | np.ndarray | None

# Pilot sample size used to estimate the stretched measure of a surface/curve.
_PILOT = 4096


def _embed(pts: np.ndarray, embed_dim: int) -> np.ndarray:
    """Zero-pad ``pts`` (n, k) into an ``embed_dim``-dimensional ambient space."""
    n, k = pts.shape
    if embed_dim < k:
        raise ValueError(f"embed_dim={embed_dim} smaller than native dim {k}")
    if embed_dim == k:
        return pts
    out = np.zeros((n, embed_dim), dtype=pts.dtype)
    out[:, :k] = pts
    return out


class Shape(ABC):
    """Abstract shape that samples uniformly by its intrinsic measure."""

    name: str
    intrinsic_dim: int  # 1 (curve) or 2 (surface)
    native_dim: int  # ambient dim of the native embedding (2 or 3)
    betti: Betti  # ground-truth (b0, b1, b2)

    @abstractmethod
    def unit_measure(self) -> float:
        """Total intrinsic measure (length/area) at ``size == 1``."""

    @abstractmethod
    def _sample_unit(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample ``n`` points on the unit-size shape, returns ``(n, native_dim)``."""

    def _unit_normals(self, pts: np.ndarray) -> np.ndarray:
        """Unit normals at unit-shape points (codimension-1 shapes only).

        Needed only for anisotropic ``stretch``; solids never call this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support anisotropic stretching"
        )

    @property
    def is_solid(self) -> bool:
        """Whether the shape fills its ambient space (codimension 0)."""
        return self.intrinsic_dim == self.native_dim

    def expected_n(self, density: float, size: float = 1.0) -> int:
        """Number of points drawn for the given ``density`` and isotropic ``size``."""
        return int(round(density * self.unit_measure() * size**self.intrinsic_dim))

    def size_for_n(self, density: float, n: int) -> float:
        """Inverse of :meth:`expected_n`: the size yielding ~``n`` points."""
        return (n / (density * self.unit_measure())) ** (1.0 / self.intrinsic_dim)

    def sample(
        self,
        *,
        density: float,
        size: float = 1.0,
        point_noise: float = 0.0,
        point_noise_kind: PointNoiseKind = "gaussian",
        field_noise: float = 0.0,
        field_length_scale: float = 0.25,
        rng: RngLike = None,
        embed_dim: int | None = None,
        stretch: StretchLike = None,
    ) -> np.ndarray:
        """Sample a point cloud from this shape.

        Parameters
        ----------
        density:
            Points per unit intrinsic measure (per unit length for curves, per
            unit area for surfaces, per unit volume for solids). The
            constant-density knob.
        size:
            Linear scale of the shape; controls the point count.
        point_noise:
            Per-axis iid noise standard deviation, relative to object scale.
        point_noise_kind:
            ``"gaussian"`` or ``"uniform"`` iid point noise.
        field_noise:
            Smooth random vector-field displacement standard deviation,
            relative to object scale.
        field_length_scale:
            Smooth field correlation length, relative to object scale.
        rng:
            Seed or :class:`numpy.random.Generator`.
        embed_dim:
            If given, zero-pad the points into this many ambient dimensions
            (e.g. embed planar shapes into 3-D). Must be >= ``native_dim``.
        stretch:
            Per-axis scale factors of length ``native_dim`` (e.g. ``(1, 1, 2)``
            turns a sphere into an ellipsoid). The effective transform is
            ``size * diag(stretch)``. Density stays constant on the *stretched*
            object: for solids a linear map preserves uniformity exactly, and
            for curves/surfaces the points are reweighted by the local
            area/length distortion so they remain uniform by measure.
        """
        rng = np.random.default_rng(rng)
        if stretch is None:
            pts = self._sample_unit(self.expected_n(density, size), rng) * size
            noise_scale = size
        else:
            scale = size * np.broadcast_to(
                np.asarray(stretch, dtype=float), (self.native_dim,)
            )
            pts = self._sample_anisotropic(density, scale, rng)
            noise_scale = float(np.max(np.abs(scale)))
        if embed_dim is not None:
            pts = _embed(pts, embed_dim)
        return apply_noise(
            pts,
            scale=noise_scale,
            rng=rng,
            point_noise=point_noise,
            point_noise_kind=point_noise_kind,
            field_noise=field_noise,
            field_length_scale=field_length_scale,
        )

    def _sample_anisotropic(
        self, density: float, scale: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample uniformly by measure on the shape scaled by ``diag(scale)``."""
        if self.is_solid:
            # A linear map sends a uniform solid to a uniform solid; the measure
            # scales by the product of all axis factors.
            measure = self.unit_measure() * float(np.prod(scale))
            n = int(round(density * measure))
            return self._sample_unit(max(n, 0), rng) * scale

        # Codimension-1: the local area/length element of the unit shape with
        # unit normal n is stretched by  factor = |det T| * |T^{-T} n|. Points
        # are accepted in proportion to this factor (rejection) so the result is
        # uniform by measure on the stretched surface/curve.
        det = float(np.prod(scale))
        inv2 = 1.0 / scale**2

        def factor(normals: np.ndarray) -> np.ndarray:
            return det * np.sqrt((normals**2 * inv2).sum(axis=1))

        pilot = factor(self._unit_normals(self._sample_unit(_PILOT, rng)))
        mean_f = float(pilot.mean())
        n = int(round(density * self.unit_measure() * mean_f))
        if n <= 0:
            return np.empty((0, self.native_dim))

        # |n| = 1, so factor <= |det T| * max_i(1/scale_i) is a valid bound.
        fmax = det * float(np.sqrt(inv2.max()))
        accept_rate = max(mean_f / fmax, 1e-3)

        kept: list[np.ndarray] = []
        got = 0
        while got < n:
            batch_n = int((n - got) / accept_rate) + 16
            pts = self._sample_unit(batch_n, rng)
            accept = rng.uniform(0.0, 1.0, size=batch_n) < factor(
                self._unit_normals(pts)
            ) / fmax
            sel = pts[accept]
            kept.append(sel)
            got += len(sel)
        return np.vstack(kept)[:n] * scale


def _circle_points(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` points uniform by arc length on the unit circle in the xy-plane."""
    theta = rng.uniform(0.0, 2.0 * pi, size=n)
    return np.column_stack((np.cos(theta), np.sin(theta)))


@dataclass
class Circle(Shape):
    """Unit circle S^1 in the plane. Betti (1, 1, 0)."""

    name: str = "circle"
    intrinsic_dim: int = 1
    native_dim: int = 2
    betti: Betti = (1, 1, 0)

    def unit_measure(self) -> float:
        return 2.0 * pi

    def _sample_unit(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return _circle_points(n, rng)

    def _unit_normals(self, pts: np.ndarray) -> np.ndarray:
        return pts.copy()  # unit-circle points are their own outward normals


@dataclass
class TwoCircles(Shape):
    """Two disjoint unit circles. Betti (2, 2, 0).

    ``separation`` is the center-to-center distance (in unit-size coordinates);
    it must exceed 2 so the circles do not overlap.
    """

    separation: float = 3.0
    name: str = "two_circles"
    intrinsic_dim: int = 1
    native_dim: int = 2
    betti: Betti = (2, 2, 0)

    def __post_init__(self) -> None:
        if self.separation <= 2.0:
            raise ValueError("separation must exceed 2 to keep the circles disjoint")

    def unit_measure(self) -> float:
        return 4.0 * pi  # two unit circles

    def _sample_unit(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # Split points between the two equal-length circles.
        n_left = rng.binomial(n, 0.5)
        offset = self.separation / 2.0
        left = _circle_points(n_left, rng) - np.array([offset, 0.0])
        right = _circle_points(n - n_left, rng) + np.array([offset, 0.0])
        return np.vstack((left, right))

    def _unit_normals(self, pts: np.ndarray) -> np.ndarray:
        offset = self.separation / 2.0
        centers = np.zeros_like(pts)
        centers[:, 0] = np.sign(pts[:, 0]) * offset
        return pts - centers  # radius 1, so already unit length


@dataclass
class EntangledCircles(Shape):
    """Two linked unit circles in 3-D (a Hopf link). Betti (2, 2, 0)."""

    name: str = "entangled_circles"
    intrinsic_dim: int = 1
    native_dim: int = 3
    betti: Betti = (2, 2, 0)

    def unit_measure(self) -> float:
        return 4.0 * pi  # two unit circles

    def _sample_unit(self, n: int, rng: np.random.Generator) -> np.ndarray:
        n_xy = rng.binomial(n, 0.5)

        theta = rng.uniform(0.0, 2.0 * pi, size=n_xy)
        xy = np.column_stack(
            (np.cos(theta) - 0.5, np.sin(theta), np.zeros(n_xy))
        )

        phi = rng.uniform(0.0, 2.0 * pi, size=n - n_xy)
        xz = np.column_stack(
            (np.cos(phi) + 0.5, np.zeros(n - n_xy), np.sin(phi))
        )
        return np.vstack((xy, xz))

    def _sample_anisotropic(
        self, density: float, scale: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        sx, sy, sz = np.abs(scale)

        def speed(theta: np.ndarray, a: float, b: float) -> np.ndarray:
            return np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)

        def length(a: float, b: float) -> float:
            theta = rng.uniform(0.0, 2.0 * pi, size=_PILOT)
            return 2.0 * pi * float(speed(theta, a, b).mean())

        def angles(n: int, a: float, b: float) -> np.ndarray:
            if n <= 0:
                return np.empty(0)
            bound = max(a, b)
            if bound == 0.0:
                return rng.uniform(0.0, 2.0 * pi, size=n)
            kept: list[np.ndarray] = []
            got = 0
            while got < n:
                cand = rng.uniform(0.0, 2.0 * pi, size=n - got + 16)
                accept = (
                    rng.uniform(0.0, 1.0, size=len(cand))
                    < speed(cand, a, b) / bound
                )
                sel = cand[accept]
                kept.append(sel)
                got += len(sel)
            return np.concatenate(kept)[:n]

        xy_len = length(sx, sy)
        xz_len = length(sx, sz)
        n = int(round(density * (xy_len + xz_len)))
        n_xy = rng.binomial(n, xy_len / (xy_len + xz_len)) if n else 0

        theta = angles(n_xy, sx, sy)
        xy = np.column_stack(
            (np.cos(theta) - 0.5, np.sin(theta), np.zeros(n_xy))
        )
        phi = angles(n - n_xy, sx, sz)
        xz = np.column_stack(
            (np.cos(phi) + 0.5, np.zeros(n - n_xy), np.sin(phi))
        )
        return np.vstack((xy, xz)) * scale


@dataclass
class FigureEight(Shape):
    """Wedge of two unit circles meeting at the origin. Betti (1, 2, 0)."""

    name: str = "figure_eight"
    intrinsic_dim: int = 1
    native_dim: int = 2
    betti: Betti = (1, 2, 0)

    def unit_measure(self) -> float:
        return 4.0 * pi  # two unit circles sharing one point

    def _sample_unit(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # Two circles tangent at the origin, centered at (-1, 0) and (1, 0).
        n_left = rng.binomial(n, 0.5)
        left = _circle_points(n_left, rng) - np.array([1.0, 0.0])
        right = _circle_points(n - n_left, rng) + np.array([1.0, 0.0])
        return np.vstack((left, right))

    def _unit_normals(self, pts: np.ndarray) -> np.ndarray:
        centers = np.zeros_like(pts)
        centers[:, 0] = np.sign(pts[:, 0]) * 1.0  # circles centered at (-+1, 0)
        return pts - centers


@dataclass
class Disk(Shape):
    """Filled unit disk D^2. Betti (1, 0, 0)."""

    name: str = "disk"
    intrinsic_dim: int = 2
    native_dim: int = 2
    betti: Betti = (1, 0, 0)

    def unit_measure(self) -> float:
        return pi

    def _sample_unit(self, n: int, rng: np.random.Generator) -> np.ndarray:
        r = np.sqrt(rng.uniform(0.0, 1.0, size=n))  # uniform by area
        theta = rng.uniform(0.0, 2.0 * pi, size=n)
        return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


@dataclass
class Annulus(Shape):
    """Annulus with outer radius 1 and inner radius ``inner_ratio``. Betti (1, 1, 0)."""

    inner_ratio: float = 0.5
    name: str = "annulus"
    intrinsic_dim: int = 2
    native_dim: int = 2
    betti: Betti = (1, 1, 0)

    def __post_init__(self) -> None:
        if not 0.0 < self.inner_ratio < 1.0:
            raise ValueError("inner_ratio must be in (0, 1)")

    def unit_measure(self) -> float:
        return pi * (1.0 - self.inner_ratio**2)

    def _sample_unit(self, n: int, rng: np.random.Generator) -> np.ndarray:
        rho2 = self.inner_ratio**2
        r = np.sqrt(rng.uniform(rho2, 1.0, size=n))  # uniform by area in the ring
        theta = rng.uniform(0.0, 2.0 * pi, size=n)
        return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


@dataclass
class Sphere(Shape):
    """Unit sphere S^2 in 3-D. Betti (1, 0, 1)."""

    name: str = "sphere"
    intrinsic_dim: int = 2
    native_dim: int = 3
    betti: Betti = (1, 0, 1)

    def unit_measure(self) -> float:
        return 4.0 * pi

    def _sample_unit(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # Normalized Gaussian directions are uniform on the sphere by area.
        v = rng.standard_normal((n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v

    def _unit_normals(self, pts: np.ndarray) -> np.ndarray:
        return pts.copy()  # outward normal of the unit sphere is the point itself


@dataclass
class Torus(Shape):
    """Torus T^2 in 3-D with major radius 1 and tube radius ``tube_ratio``.

    Betti (1, 2, 1). Surface area is ``4 * pi**2 * R * r`` with ``R = 1``.
    """

    tube_ratio: float = 0.35
    name: str = "torus"
    intrinsic_dim: int = 2
    native_dim: int = 3
    betti: Betti = (1, 2, 1)

    def __post_init__(self) -> None:
        if not 0.0 < self.tube_ratio < 1.0:
            raise ValueError("tube_ratio must be in (0, 1)")

    def unit_measure(self) -> float:
        return 4.0 * pi**2 * self.tube_ratio  # R = 1, r = tube_ratio

    def _sample_unit(self, n: int, rng: np.random.Generator) -> np.ndarray:
        R, r = 1.0, self.tube_ratio
        # Rejection-sample theta with acceptance (R + r cos theta) / (R + r)
        # so the points are uniform by surface area, not by the (theta, phi)
        # parameter rectangle.
        theta = np.empty(n)
        filled = 0
        while filled < n:
            m = n - filled
            cand = rng.uniform(0.0, 2.0 * pi, size=m)
            accept = rng.uniform(0.0, 1.0, size=m) < (R + r * np.cos(cand)) / (R + r)
            k = int(np.count_nonzero(accept))
            theta[filled : filled + k] = cand[accept]
            filled += k
        phi = rng.uniform(0.0, 2.0 * pi, size=n)
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)
        return np.column_stack((x, y, z))

    def _unit_normals(self, pts: np.ndarray) -> np.ndarray:
        R, r = 1.0, self.tube_ratio
        rho = np.hypot(pts[:, 0], pts[:, 1])
        cos_phi, sin_phi = pts[:, 0] / rho, pts[:, 1] / rho
        cos_theta, sin_theta = (rho - R) / r, pts[:, 2] / r
        return np.column_stack(
            (cos_theta * cos_phi, cos_theta * sin_phi, sin_theta)
        )


@dataclass
class Ball(Shape):
    """Filled unit ball B^3 in 3-D (a solid sphere). Betti (1, 0, 0)."""

    name: str = "ball"
    intrinsic_dim: int = 3
    native_dim: int = 3
    betti: Betti = (1, 0, 0)

    def unit_measure(self) -> float:
        return 4.0 / 3.0 * pi  # volume of the unit ball

    def _sample_unit(self, n: int, rng: np.random.Generator) -> np.ndarray:
        v = rng.standard_normal((n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)  # uniform direction
        r = rng.uniform(0.0, 1.0, size=(n, 1)) ** (1.0 / 3.0)  # uniform by volume
        return r * v


def default_shapes() -> list[Shape]:
    """The curated topological set spanning H0/H1/H2."""
    return [
        Circle(),
        TwoCircles(),
        EntangledCircles(),
        FigureEight(),
        Disk(),
        Annulus(),
        Sphere(),
        Ball(),
        Torus(),
    ]


DEFAULT_SHAPES = default_shapes()
