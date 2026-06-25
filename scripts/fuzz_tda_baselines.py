"""Small randomized checks for TDA baseline invariants.

Run:
    uv run python scripts/fuzz_tda_baselines.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tda_shapes import Annulus, Circle, Disk, Sphere
from tda_shapes.ml.ph import (
    PointCloudPreprocessor,
    RipserFeatures,
    cech_betti_pipeline,
    direct_betti,
    gudhi_betti_pipeline,
    ripser_betti_pipeline,
)


SHAPES = [
    (Circle(), (1, 1, 0)),
    (Disk(), (1, 0, 0)),
    (Sphere(), (1, 0, 1)),
    (Annulus(), (1, 1, 0)),
]


def _clouds(seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    clouds = []
    labels = []
    for shape, betti in SHAPES:
        density = float(rng.integers(25, 45))
        size = float(rng.uniform(0.7, 1.4))
        cloud = shape.sample(density=density, size=size, embed_dim=3, rng=rng)
        clouds.append(cloud)
        labels.append(betti)
    return clouds, np.asarray(labels, dtype=int)


def _assert_preprocessed(clouds: list[np.ndarray], n_points: int, seed: int) -> list[np.ndarray]:
    pts = PointCloudPreprocessor(n_points=n_points, random_state=seed).fit_transform(clouds)
    assert len(pts) == len(clouds)
    for cloud in pts:
        assert cloud.shape == (n_points, 3)
        assert cloud.dtype == np.float64
        assert cloud.flags.c_contiguous
        assert np.all(np.isfinite(cloud))
        assert np.allclose(cloud.mean(axis=0), 0.0, atol=1e-12)
        rms = np.sqrt((cloud**2).sum(axis=1).mean())
        assert np.isclose(rms, 1.0)
    return pts


def _assert_ripser_features(clouds: list[np.ndarray], n_points: int, seed: int) -> None:
    features = RipserFeatures(n_points=n_points, grid=8, random_state=seed).fit_transform(clouds)
    assert features.shape == (len(clouds), 3 * (8 + 4))
    assert np.all(np.isfinite(features))


def _assert_pipeline(name: str, pipe, clouds: list[np.ndarray], y: np.ndarray) -> None:
    pred = pipe.fit(clouds, y).predict(clouds)
    assert pred.shape == y.shape, name
    assert pred.dtype.kind == "i", name
    assert np.all(pred >= 0), name


def run(seed: int, *, include_gudhi: bool) -> None:
    clouds, y = _clouds(seed)
    n_points = 32

    _assert_preprocessed(clouds, n_points, seed)
    _assert_ripser_features(clouds, n_points, seed)
    _assert_pipeline(
        "ripser",
        ripser_betti_pipeline(n_points=n_points, grid=8, n_estimators=5, random_state=seed),
        clouds,
        y,
    )

    features = RipserFeatures(n_points=n_points, grid=8, random_state=seed).fit_transform(clouds)
    assert np.all(features[:, :8] >= 0)

    if include_gudhi:
        _assert_pipeline(
            "gudhi_rips",
            gudhi_betti_pipeline(n_points=n_points, grid=8, n_estimators=5, random_state=seed),
            clouds,
            y,
        )
        _assert_pipeline(
            "gudhi_cech",
            cech_betti_pipeline(n_points=24, grid=8, n_estimators=5, random_state=seed),
            clouds,
            y,
        )

    # Direct Betti must always return a nonnegative integer triple.
    from tda_shapes.ml.ph import cloud_diagrams

    dgms = cloud_diagrams(clouds[0], n_points=n_points, rng=seed)
    betti = direct_betti(dgms, [0.2, 0.2, 0.2])
    assert betti.shape == (3,)
    assert betti.dtype.kind == "i"
    assert np.all(betti >= 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--skip-gudhi", action="store_true")
    args = parser.parse_args()

    include_gudhi = not args.skip_gudhi and importlib.util.find_spec("gudhi") is not None
    for seed in range(args.seeds):
        run(seed, include_gudhi=include_gudhi)
    print(f"ok: {args.seeds} seeds, gudhi={include_gudhi}")


if __name__ == "__main__":
    main()
