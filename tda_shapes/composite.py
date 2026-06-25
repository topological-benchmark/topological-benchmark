"""Composite point clouds containing ``k`` disjoint shapes.

A composite scene places ``k`` shapes (drawn with replacement from a pool) so
that their *clean* (noiseless) bounding balls do not overlap. Because the parts
are disjoint, the union's homology is the direct sum of the parts, so the
**combined Betti numbers are the element-wise sum** of the components'. Every
component is sampled at the same ``density``, so the whole scene stays uniform.
Packing uses each component's clean bounding radius; point/field noise is added
only *after* placement, so spacing is not inflated to accommodate the noise tail
(a few boundary points may cross the gap).

Optionally, uniform background clutter can be scattered across the scene's
bounding box at a fixed volumetric density (``background_density`` > 0, in points
per unit ambient volume — pick it well below the shapes' sampling ``density`` so
the shapes stay dominant). These distractor points belong to no shape and do not
change the Betti / shape-count labels; they only make the task harder by
polluting the cloud.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .noise import PointNoiseKind, apply_noise
from .shapes import DEFAULT_SHAPES, RngLike, Shape

#: ``component_labels`` value marking uniformly-sampled background clutter
#: (points that belong to no shape).
BACKGROUND_LABEL = -1


def _uniform_background(
    points: np.ndarray,
    density: float,
    rng: np.random.Generator,
    margin: float = 0.0,
) -> np.ndarray:
    """Sample uniform background clutter at a fixed *volumetric* density.

    Clutter fills the axis-aligned box spanning ``points`` (each axis' ``[min,
    max]`` extent), optionally padded on every side by ``margin`` times that
    axis' extent so it can spill a little beyond the shapes. The number of points
    is ``round(density * box_volume)``, where the volume is the product of the
    box's per-axis extents — so ``density`` is points per unit ambient volume and
    is invariant to how large the scene is. These points belong to no shape; they
    are pure distractors that leave the scene's Betti label unchanged.

    Returns an empty ``(0, dim)`` array if the density is non-positive or the box
    is degenerate (zero volume, e.g. a flat planar scene with no noise/rotation).
    """
    if density <= 0:
        return np.empty((0, points.shape[1]))
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    if margin:
        pad = margin * (hi - lo)
        lo = lo - pad
        hi = hi + pad
    volume = float(np.prod(hi - lo))
    count = int(round(density * volume))
    if volume <= 0 or count <= 0:
        return np.empty((0, points.shape[1]))
    return rng.uniform(lo, hi, size=(count, points.shape[1]))


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
        Pool-index of the shape each point came from, ``(N,)``; background
        clutter points are marked with :data:`BACKGROUND_LABEL` (``-1``). Kept
        for visualization/segmentation; not stored in :class:`CompositeDataset`.
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
    point_noise: float = 0.0,
    point_noise_kind: PointNoiseKind = "gaussian",
    field_noise: float = 0.0,
    field_length_scale: float = 0.25,
    background_density: float = 0.0,
    background_margin: float = 0.0,
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
    point_noise:
        Per-axis iid noise standard deviation, relative to each component size.
    point_noise_kind:
        ``"gaussian"`` or ``"uniform"`` iid point noise.
    field_noise:
        Smooth random vector-field displacement standard deviation, relative to
        each component size.
    field_length_scale:
        Smooth field correlation length, relative to each component size.
    background_density:
        Volumetric density of uniform background clutter, in points per unit
        ambient (``embed_dim``-dimensional) volume. The clutter fills the scene's
        bounding box, so the point count is ``round(background_density *
        box_volume)``. Pick it well below the shapes' sampling ``density`` so the
        shapes stay dominant. These points belong to no shape (labelled
        :data:`BACKGROUND_LABEL`) and do not change the Betti / shape-count
        labels. ``0`` disables it.
    background_margin:
        Fractional padding of the background box beyond the shapes' extent, per
        axis (e.g. ``0.1`` extends each side by 10% of that axis's span). Enlarges
        the filled volume, hence the clutter point count, accordingly.
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
    sizes = np.empty(k)
    betti = np.zeros(3, dtype=int)

    for j, idx in enumerate(indices):
        shape = pool[idx]
        size = rng.uniform(*size_range)
        sizes[j] = size
        pts = shape.sample(
            density=density, size=size, point_noise=0.0, embed_dim=embed_dim, rng=rng
        )
        if rotate:
            pts = pts @ _random_rotation(embed_dim, rng).T
        radii[j] = float(np.linalg.norm(pts, axis=1).max()) if len(pts) else 0.0
        clouds.append(pts)
        betti += np.asarray(shape.betti, dtype=int)

    centers = _pack_centers(radii, clearance, embed_dim, rng)
    perturbed = [
        apply_noise(
            c + centers[j],
            scale=sizes[j],
            rng=rng,
            point_noise=point_noise,
            point_noise_kind=point_noise_kind,
            field_noise=field_noise,
            field_length_scale=field_length_scale,
        )
        for j, c in enumerate(clouds)
    ]

    points = np.vstack(perturbed)
    component_labels = np.concatenate(
        [np.full(len(c), indices[j], dtype=int) for j, c in enumerate(clouds)]
    )
    if background_density > 0:
        bg = _uniform_background(points, background_density, rng, margin=background_margin)
        if len(bg):
            points = np.vstack([points, bg])
            component_labels = np.concatenate(
                [component_labels, np.full(len(bg), BACKGROUND_LABEL, dtype=int)]
            )

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
    point_noise: float = 0.02,
    point_noise_kind: PointNoiseKind = "gaussian",
    field_noise: float = 0.0,
    field_length_scale: float = 0.25,
    background_density: float = 0.0,
    background_margin: float = 0.0,
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
            point_noise=point_noise,
            point_noise_kind=point_noise_kind,
            field_noise=field_noise,
            field_length_scale=field_length_scale,
            background_density=background_density,
            background_margin=background_margin,
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
