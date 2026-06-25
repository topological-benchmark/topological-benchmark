"""Point-cloud perturbations that keep labels unchanged when amplitudes are small."""

from __future__ import annotations

from typing import Literal

import numpy as np

RngLike = int | np.random.Generator | None
PointNoiseKind = Literal["gaussian", "uniform"]


def apply_noise(
    points: np.ndarray,
    *,
    scale: float,
    rng: np.random.Generator,
    point_noise: float = 0.0,
    point_noise_kind: PointNoiseKind = "gaussian",
    field_noise: float = 0.0,
    field_length_scale: float = 0.25,
    field_features: int = 32,
) -> np.ndarray:
    """Apply iid point noise plus a smooth random vector-field displacement.

    Noise amplitudes and field length scale are relative to ``scale``.
    """
    out = np.asarray(points, dtype=np.float64)
    if not point_noise and not field_noise:
        return out
    out = out.copy()
    if point_noise:
        sigma = float(point_noise) * scale
        if point_noise_kind == "gaussian":
            out += sigma * rng.standard_normal(out.shape)
        elif point_noise_kind == "uniform":
            out += rng.uniform(-np.sqrt(3.0) * sigma, np.sqrt(3.0) * sigma, size=out.shape)
        else:
            raise ValueError(f"unknown point_noise_kind: {point_noise_kind!r}")
    if field_noise:
        out += _smooth_vector_field(
            points,
            amplitude=float(field_noise) * scale,
            length_scale=max(float(field_length_scale) * scale, 1e-12),
            n_features=field_features,
            rng=rng,
        )
    return out


def _smooth_vector_field(
    points: np.ndarray,
    *,
    amplitude: float,
    length_scale: float,
    n_features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random Fourier approximation of a Gaussian-kernel vector field."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros_like(points)
    n_features = max(int(n_features), 1)
    w = rng.normal(scale=1.0 / length_scale, size=(n_features, points.shape[1]))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_features, points.shape[1]))
    weights = rng.standard_normal((n_features, points.shape[1]))
    projection = points @ w.T
    disp = np.empty_like(points)
    factor = np.sqrt(2.0 / n_features)
    for axis in range(points.shape[1]):
        shifted = np.cos(projection + phase[:, axis])
        vals = factor * (shifted @ weights[:, axis])
        std = float(vals.std())
        disp[:, axis] = vals / std if std > 1e-12 else vals
    return amplitude * disp
