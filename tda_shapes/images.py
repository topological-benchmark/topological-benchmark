"""Build labeled image datasets by rasterizing sampled shape points with KDE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .noise import PointNoiseKind
from .shapes import DEFAULT_SHAPES, RngLike, Shape


@dataclass
class ImageDataset:
    """A labeled collection of KDE-rasterized shape images."""

    images: np.ndarray
    labels: np.ndarray
    label_names: list[str]
    betti: np.ndarray
    clouds: list[np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.images)

    def save(self, path: str) -> None:
        """Save to a ``.npz`` file."""
        label_names = np.array(self.label_names, dtype=object)
        if self.clouds is None:
            np.savez(
                path,
                images=self.images,
                labels=self.labels,
                label_names=label_names,
                betti=self.betti,
            )
        else:
            clouds = np.empty(len(self.clouds), dtype=object)
            for i, cloud in enumerate(self.clouds):
                clouds[i] = cloud
            np.savez(
                path,
                images=self.images,
                labels=self.labels,
                label_names=label_names,
                betti=self.betti,
                clouds=clouds,
            )

    @classmethod
    def load(cls, path: str) -> "ImageDataset":
        """Load a dataset previously written by :meth:`save`."""
        with np.load(path, allow_pickle=True) as data:
            clouds = list(data["clouds"]) if "clouds" in data.files else None
            return cls(
                images=data["images"],
                labels=data["labels"],
                label_names=list(data["label_names"]),
                betti=data["betti"],
                clouds=clouds,
            )


def rasterize_kde(
    points: np.ndarray,
    *,
    resolution: int = 48,
    bandwidth: float = 0.15,
    padding: float = 0.2,
    min_pixels_per_bandwidth: float = 1.0,
    normalize: bool = True,
    filtration: bool = True,
) -> np.ndarray:
    """Rasterize ``points`` to a cubic Gaussian-KDE image.

    ``bandwidth`` is in the same physical units as ``points``. If ``filtration``
    is true, high-density cells get low values so sublevel cubical persistence
    sees the sampled shape before the background.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if len(points) == 0:
        raise ValueError("cannot rasterize an empty point cloud")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if padding < 0:
        raise ValueError("padding must be non-negative")

    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float(np.max(hi - lo))
    if radius == 0.0:
        radius = bandwidth
    radius = radius * (1.0 + padding) + bandwidth
    extent = 2.0 * radius
    pixel_size = extent / (resolution - 1)
    if bandwidth < min_pixels_per_bandwidth * pixel_size:
        raise ValueError(
            "bandwidth too small for resolution: "
            f"bandwidth={bandwidth:.6g}, pixel_size={pixel_size:.6g}"
        )

    axes = [np.linspace(c - radius, c + radius, resolution) for c in center]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    density = np.empty(len(grid), dtype=float)
    point_norms = np.sum(points**2, axis=1)
    h2 = bandwidth * bandwidth
    batch = max(1, min(8192, 2_000_000 // len(points)))
    for start in range(0, len(grid), batch):
        chunk = grid[start : start + batch]
        dist2 = (
            np.sum(chunk**2, axis=1)[:, None]
            + point_norms[None, :]
            - 2.0 * chunk @ points.T
        )
        np.maximum(dist2, 0.0, out=dist2)
        density[start : start + batch] = np.exp(-0.5 * dist2 / h2).sum(axis=1)

    image = density.reshape((resolution, resolution, resolution))
    if normalize and image.max() > 0:
        image = image / image.max()
    if filtration:
        image = image.max() - image
    return image.astype(np.float32, copy=False)


def make_image_dataset(
    shapes: list[Shape] | None = None,
    *,
    n_per_class: int = 50,
    density: float = 50.0,
    size_range: tuple[float, float] = (1.0, 3.0),
    resolution: int = 48,
    bandwidth: float = 0.15,
    padding: float = 0.2,
    min_pixels_per_bandwidth: float = 1.0,
    point_noise: float = 0.0,
    point_noise_kind: PointNoiseKind = "gaussian",
    field_noise: float = 0.0,
    field_length_scale: float = 0.25,
    stretch_range: tuple[float, float] | None = None,
    keep_clouds: bool = False,
    rng: RngLike = None,
) -> ImageDataset:
    """Generate an image dataset by sampling shapes then KDE-rasterizing them.

    ``bandwidth`` is relative to each sampled shape's linear ``size``. The
    effective physical KDE bandwidth is ``bandwidth * size``.
    """
    if shapes is None:
        shapes = DEFAULT_SHAPES
    rng = np.random.default_rng(rng)
    lo, hi = size_range

    images: list[np.ndarray] = []
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
                point_noise=point_noise,
                point_noise_kind=point_noise_kind,
                field_noise=field_noise,
                field_length_scale=field_length_scale,
                rng=rng,
                embed_dim=3,
                stretch=stretch,
            )
            image = rasterize_kde(
                cloud,
                resolution=resolution,
                bandwidth=bandwidth * size,
                padding=padding,
                min_pixels_per_bandwidth=min_pixels_per_bandwidth,
            )
            images.append(image)
            if keep_clouds:
                clouds.append(cloud)
            labels.append(label)
            betti.append(shape.betti)

    image_array = (
        np.stack(images)
        if images
        else np.empty((0, resolution, resolution, resolution), dtype=np.float32)
    )
    betti_array = np.array(betti, dtype=int).reshape((-1, 3))
    return ImageDataset(
        images=image_array,
        labels=np.array(labels, dtype=int),
        label_names=label_names,
        betti=betti_array,
        clouds=clouds if keep_clouds else None,
    )
