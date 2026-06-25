"""Shared data preparation for the Betti-number prediction benchmark."""

from __future__ import annotations

import numpy as np

RngLike = int | np.random.Generator | None


def normalize_cloud(pts: np.ndarray) -> np.ndarray:
    """Center to the centroid and scale to unit RMS radius.

    Applied to both methods so that absolute scale (which trivially encodes the
    point count, hence the shape) is *not* a usable shortcut feature.
    """
    pts = np.asarray(pts, dtype=np.float64)
    pts = pts - pts.mean(axis=0, keepdims=True)
    rms = np.sqrt((pts**2).sum(axis=1).mean())
    if rms > 0:
        pts = pts / rms
    return np.ascontiguousarray(pts, dtype=np.float64)


def to_fixed_size(pts: np.ndarray, p: int, rng: np.random.Generator) -> np.ndarray:
    """Resample ``pts`` to exactly ``p`` points (subsample or duplicate-pad)."""
    n = len(pts)
    if n == p:
        return pts
    replace = n < p
    idx = rng.choice(n, size=p, replace=replace)
    return pts[idx]


def pointnet_arrays(
    dataset,
    *,
    n_points: int = 512,
    rng: RngLike = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a dataset into dense PointNet tensors.

    Returns ``X (M, n_points, 3) float32`` of normalized, fixed-size clouds and
    ``y (M, 3) float32`` Betti targets.
    """
    rng = np.random.default_rng(rng)
    x = np.empty((len(dataset.clouds), n_points, 3), dtype=np.float32)
    for i, cloud in enumerate(dataset.clouds):
        pts = to_fixed_size(np.asarray(cloud, dtype=np.float64), n_points, rng)
        x[i] = normalize_cloud(pts).astype(np.float32)
    y = np.asarray(dataset.betti, dtype=np.float32)
    return x, y


def train_test_split_idx(
    m: int, *, frac: float = 0.3, rng: RngLike = None
) -> tuple[np.ndarray, np.ndarray]:
    """Shared train/test index split so every method sees the same partition."""
    rng = np.random.default_rng(rng)
    perm = rng.permutation(m)
    n_test = int(round(frac * m))
    return perm[n_test:], perm[:n_test]
