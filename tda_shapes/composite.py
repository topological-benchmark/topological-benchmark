"""Composite point clouds containing ``k`` disjoint shapes.

A composite scene places ``k`` shapes (drawn with replacement from a pool) so
that their *clean* (noiseless) bounding balls do not overlap. Because the parts
are disjoint, the union's homology is the direct sum of the parts, so the
**combined Betti numbers are the element-wise sum** of the components'. Every
component is sampled at the same ``density``, so the whole scene stays uniform.

Packing uses each component's noiseless bounding radius; isotropic Gaussian
noise is added only *after* placement, so the spacing is not inflated to
accommodate the noise tail (a few boundary points may cross the gap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shapes import DEFAULT_SHAPES, RngLike, Shape


@dataclass
class Composite:
    """A single scene of ``k`` disjoint shapes.

    Attributes
    ----------
    points:
        The combined cloud, ``(N, embed_dim)``.
    betti:
        Combined Betti numbers ``(b0, b1, b2)`` = element-wise sum of components.
    counts:
        Number of each pool shape present, ``(n_types,)`` (indexed like
        ``shape_names``); ``counts.sum() == k``.
    component_labels:
        Pool-index of the shape each point came from, ``(N,)``. Kept for
        visualization/segmentation; not stored in :class:`CompositeDataset`.
    shape_names:
        Pool shape names, indexing the columns of ``counts``.
    """

    points: np.ndarray
    betti: np.ndarray
    counts: np.ndarray
    component_labels: np.ndarray
    shape_names: list[str]


def _random_rotation(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform rotation matrix via QR of a Gaussian matrix."""
    a = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))  # remove QR sign ambiguity -> uniform on O(dim)
    if np.linalg.det(q) < 0:  # make it a proper rotation (det +1)
        q[:, 0] = -q[:, 0]
    return q


def _pack_centers(
    radii: np.ndarray,
    clearance: float,
    dim: int,
    rng: np.random.Generator,
    max_tries: int = 2000,
    max_grow: int = 20,
) -> np.ndarray:
    """Place non-overlapping balls: ``‖c_i - c_j‖ >= r_i + r_j + clearance``.

    Random sequential insertion in a box that grows ×1.5 on repeated failure,
    with a deterministic axis-line fallback so packing always terminates.
    """
    n = len(radii)
    # Start from a compact box just large enough to plausibly hold the balls
    # (side ~ the d-th root of their summed footprints); grow only if needed so
    # the arrangement stays tight rather than sparse.
    span = float(np.sum((2.0 * radii + clearance) ** dim) ** (1.0 / dim)) + clearance

    for _ in range(max_grow):
        centers = np.zeros((n, dim))
        ok = True
        for i in range(n):
            for _ in range(max_tries):
                c = rng.uniform(-span, span, size=dim)
                gaps = np.linalg.norm(centers[:i] - c, axis=1) - (
                    radii[:i] + radii[i] + clearance
                )
                if i == 0 or gaps.min() >= 0.0:
                    centers[i] = c
                    break
            else:
                ok = False
                break
        if ok:
            return centers
        span *= 1.5  # not enough room — enlarge the box and start over

    return _pack_centers_line(radii, clearance, dim)


def _pack_centers_line(radii: np.ndarray, clearance: float, dim: int) -> np.ndarray:
    """Deterministic fallback: lay balls out along the x-axis, just touching+gap."""
    n = len(radii)
    centers = np.zeros((n, dim))
    x = 0.0
    for i in range(n):
        if i > 0:
            x += radii[i - 1] + radii[i] + clearance
        centers[i, 0] = x
    centers[:, 0] -= centers[:, 0].mean()  # recentre
    return centers


def sample_composite(
    k: int,
    pool: list[Shape] | None = None,
    *,
    density: float,
    size_range: tuple[float, float] = (1.0, 3.0),
    noise: float = 0.0,
    embed_dim: int = 3,
    clearance: float = 0.5,
    rotate: bool = True,
    rng: RngLike = None,
) -> Composite:
    """Sample one composite scene of ``k`` disjoint shapes.

    Parameters
    ----------
    k:
        Number of shapes in the scene.
    pool:
        Shapes to draw from with replacement (default: the curated set).
    density:
        Points per unit intrinsic measure, constant across all components.
    size_range:
        Per-component size is drawn uniformly from this range.
    noise:
        Std. dev. of isotropic Gaussian noise added after placement.
    embed_dim:
        Ambient dimension every component is embedded into (must be >= every
        pool shape's ``native_dim``).
    clearance:
        Minimum gap between component bounding balls (clean radii).
    rotate:
        Apply a random rotation to each component (an isometry, so density is
        unaffected). Gives planar shapes varied 3-D orientations.
    rng:
        Seed or generator.
    """
    if pool is None:
        pool = DEFAULT_SHAPES
    rng = np.random.default_rng(rng)
    shape_names = [s.name for s in pool]

    indices = rng.integers(0, len(pool), size=k)
    clouds: list[np.ndarray] = []
    radii = np.empty(k)
    betti = np.zeros(3, dtype=int)

    for j, idx in enumerate(indices):
        shape = pool[idx]
        size = rng.uniform(*size_range)
        pts = shape.sample(
            density=density, size=size, noise=0.0, embed_dim=embed_dim, rng=rng
        )
        if rotate:
            pts = pts @ _random_rotation(embed_dim, rng).T
        radii[j] = float(np.linalg.norm(pts, axis=1).max()) if len(pts) else 0.0
        clouds.append(pts)
        betti += np.asarray(shape.betti, dtype=int)

    centers = _pack_centers(radii, clearance, embed_dim, rng)

    points = np.vstack([c + centers[j] for j, c in enumerate(clouds)])
    component_labels = np.concatenate(
        [np.full(len(c), indices[j], dtype=int) for j, c in enumerate(clouds)]
    )
    if noise:
        points = points + noise * rng.standard_normal(points.shape)

    counts = np.bincount(indices, minlength=len(pool)).astype(int)
    return Composite(
        points=points,
        betti=betti,
        counts=counts,
        component_labels=component_labels,
        shape_names=shape_names,
    )


@dataclass
class CompositeDataset:
    """A collection of composite scenes labeled by combined Betti + shape counts.

    Attributes
    ----------
    clouds:
        ``M`` arrays, each ``(N_i, embed_dim)``.
    betti:
        Combined Betti per cloud, ``(M, 3)``.
    shape_counts:
        Per-cloud count of each pool shape, ``(M, n_types)``.
    shape_names:
        Names indexing the columns of ``shape_counts``.
    k:
        Number of shapes per cloud.
    """

    clouds: list[np.ndarray]
    betti: np.ndarray
    shape_counts: np.ndarray
    shape_names: list[str]
    k: int

    def __len__(self) -> int:
        return len(self.clouds)

    def save(self, path: str) -> None:
        """Save to a ``.npz`` file (clouds stored as a ragged object array)."""
        np.savez(
            path,
            clouds=np.array(self.clouds, dtype=object),
            betti=self.betti,
            shape_counts=self.shape_counts,
            shape_names=np.array(self.shape_names, dtype=object),
            k=self.k,
        )

    @classmethod
    def load(cls, path: str) -> "CompositeDataset":
        """Load a dataset previously written by :meth:`save`."""
        with np.load(path, allow_pickle=True) as data:
            return cls(
                clouds=list(data["clouds"]),
                betti=data["betti"],
                shape_counts=data["shape_counts"],
                shape_names=list(data["shape_names"]),
                k=int(data["k"]),
            )


def make_composite_dataset(
    n_samples: int,
    k: int,
    pool: list[Shape] | None = None,
    *,
    density: float = 20.0,
    size_range: tuple[float, float] = (1.0, 3.0),
    noise: float = 0.02,
    embed_dim: int = 3,
    clearance: float = 0.5,
    rotate: bool = True,
    rng: RngLike = None,
) -> CompositeDataset:
    """Generate ``n_samples`` composite scenes, each containing ``k`` shapes.

    See :func:`sample_composite` for the per-scene parameters. Each scene's label
    is its combined Betti vector and its shape-type count vector.
    """
    if pool is None:
        pool = DEFAULT_SHAPES
    rng = np.random.default_rng(rng)

    clouds: list[np.ndarray] = []
    betti: list[np.ndarray] = []
    counts: list[np.ndarray] = []

    for _ in range(n_samples):
        scene = sample_composite(
            k,
            pool,
            density=density,
            size_range=size_range,
            noise=noise,
            embed_dim=embed_dim,
            clearance=clearance,
            rotate=rotate,
            rng=rng,
        )
        clouds.append(scene.points)
        betti.append(scene.betti)
        counts.append(scene.counts)

    return CompositeDataset(
        clouds=clouds,
        betti=np.array(betti, dtype=int),
        shape_counts=np.array(counts, dtype=int),
        shape_names=[s.name for s in pool],
        k=k,
    )
