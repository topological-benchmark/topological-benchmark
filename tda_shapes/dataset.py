"""Build labeled point-cloud datasets from :mod:`tda_shapes.shapes`.

A :class:`ShapeDataset` is a collection of point clouds, each labeled with the
shape it was sampled from and that shape's ground-truth Betti numbers. Point
counts vary across samples because the size is drawn at random per sample,
while the *density* (points per unit length/area) stays constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shapes import DEFAULT_SHAPES, RngLike, Shape


@dataclass
class ShapeDataset:
    """A labeled collection of sampled point clouds.

    Attributes
    ----------
    clouds:
        List of ``M`` arrays, each ``(N_i, embed_dim)``. Point counts differ
        between clouds (constant density, varying size).
    labels:
        Integer class id per cloud, ``(M,)``.
    label_names:
        Shape name for each class id (indexed by label).
    betti:
        Ground-truth Betti numbers per cloud, ``(M, 3)`` = ``(b0, b1, b2)``.
    """

    clouds: list[np.ndarray]
    labels: np.ndarray
    label_names: list[str]
    betti: np.ndarray

    def __len__(self) -> int:
        return len(self.clouds)

    def save(self, path: str) -> None:
        """Save to a ``.npz`` file (clouds stored as a ragged object array)."""
        np.savez(
            path,
            clouds=np.array(self.clouds, dtype=object),
            labels=self.labels,
            label_names=np.array(self.label_names, dtype=object),
            betti=self.betti,
        )

    @classmethod
    def load(cls, path: str) -> "ShapeDataset":
        """Load a dataset previously written by :meth:`save`."""
        with np.load(path, allow_pickle=True) as data:
            return cls(
                clouds=list(data["clouds"]),
                labels=data["labels"],
                label_names=list(data["label_names"]),
                betti=data["betti"],
            )


def make_dataset(
    shapes: list[Shape] | None = None,
    *,
    n_per_class: int = 50,
    density: float = 20.0,
    size_range: tuple[float, float] = (1.0, 3.0),
    noise: float = 0.02,
    embed_dim: int | None = 3,
    stretch_range: tuple[float, float] | None = None,
    rng: RngLike = None,
) -> ShapeDataset:
    """Generate a :class:`ShapeDataset`.

    Parameters
    ----------
    shapes:
        Shapes to sample from (defaults to the curated topological set).
    n_per_class:
        Number of point clouds to draw per shape.
    density:
        Points per unit intrinsic measure (per unit length for curves, per unit
        area for surfaces). Constant across the whole dataset. Note the unit
        differs between 1-D and 2-D shapes.
    size_range:
        Range ``(lo, hi)`` from which each cloud's size is drawn uniformly.
        Varying the size is what makes point counts vary at fixed density.
    noise:
        Standard deviation of ambient isotropic Gaussian noise.
    embed_dim:
        Common ambient dimension to embed every cloud into (default 3, so all
        clouds share R^3). Pass ``None`` to keep native dimensions.
    stretch_range:
        If given, draw per-axis stretch factors uniformly from this range for
        each cloud (anisotropy / data augmentation). Density stays constant on
        the stretched object. ``None`` keeps shapes isotropic.
    rng:
        Seed or generator for reproducibility.
    """
    if shapes is None:
        shapes = DEFAULT_SHAPES
    rng = np.random.default_rng(rng)
    lo, hi = size_range

    clouds: list[np.ndarray] = []
    labels: list[int] = []
    betti: list[tuple[int, int, int]] = []
    label_names = [s.name for s in shapes]

    for label, shape in enumerate(shapes):
        for _ in range(n_per_class):
            size = rng.uniform(lo, hi)
            stretch = (
                rng.uniform(*stretch_range, size=shape.native_dim)
                if stretch_range is not None
                else None
            )
            cloud = shape.sample(
                density=density,
                size=size,
                noise=noise,
                rng=rng,
                embed_dim=embed_dim,
                stretch=stretch,
            )
            clouds.append(cloud)
            labels.append(label)
            betti.append(shape.betti)

    return ShapeDataset(
        clouds=clouds,
        labels=np.array(labels, dtype=int),
        label_names=label_names,
        betti=np.array(betti, dtype=int),
    )
