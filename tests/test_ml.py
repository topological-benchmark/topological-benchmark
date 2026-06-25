"""Tests for the Betti-prediction ML baselines (PointNet + persistent homology)."""

from __future__ import annotations

import numpy as np
import pytest

from tda_shapes import Annulus, Circle, Disk, Sphere, make_image_dataset
from tda_shapes.ml.data import normalize_cloud, pointnet_arrays, to_fixed_size
from tda_shapes.ml.metrics import betti_metrics

# --- data helpers ----------------------------------------------------------


def test_normalize_cloud_centers_and_scales():
    pts = Circle().sample(density=50, size=3.0, rng=0) + 5.0
    z = normalize_cloud(pts)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-9)
    assert np.isclose(np.sqrt((z**2).sum(axis=1).mean()), 1.0)


def test_to_fixed_size_up_and_down():
    rng = np.random.default_rng(0)
    pts = np.random.default_rng(1).standard_normal((10, 3))
    assert to_fixed_size(pts, 25, rng).shape == (25, 3)
    assert to_fixed_size(pts, 4, rng).shape == (4, 3)


# --- persistent homology ---------------------------------------------------


@pytest.mark.parametrize(
    "shape, expected",
    [
        (Circle(), (1, 1, 0)),
        (Sphere(), (1, 0, 1)),
        (Disk(), (1, 0, 0)),
        (Annulus(), (1, 1, 0)),
    ],
)
def test_direct_betti_on_clean_shapes(shape, expected):
    from tda_shapes.ml.ph import cloud_diagrams, direct_betti

    rng = np.random.default_rng(0)
    cloud = shape.sample(density=80, size=1.0, point_noise=0.0, embed_dim=3, rng=rng)
    dgms = cloud_diagrams(cloud, n_points=250, rng=rng)
    assert tuple(direct_betti(dgms, [0.25, 0.25, 0.25])) == expected


def test_diagram_features_shape_and_finiteness():
    from tda_shapes.ml.ph import cloud_diagrams, diagram_features

    rng = np.random.default_rng(0)
    cloud = Sphere().sample(density=60, size=1.0, rng=rng)
    feats = diagram_features(cloud_diagrams(cloud, n_points=150, rng=rng), grid=32)
    # 3 dims * (32 betti-curve points + 4 stats)
    assert feats.shape == (3 * (32 + 4),)
    assert np.isfinite(feats).all()


def test_ph_regressor_fits_and_predicts_integers():
    from tda_shapes.ml.ph import PHRegressor

    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 12))
    y = rng.integers(0, 3, size=(20, 3))
    model = PHRegressor(n_estimators=20).fit(x, y)
    pred = model.predict(x)
    assert pred.shape == (20, 3)
    assert pred.dtype.kind == "i"
    assert (pred >= 0).all()


def test_ripser_pipeline_fits_point_clouds():
    from tda_shapes.ml.ph import ripser_betti_pipeline

    rng = np.random.default_rng(0)
    clouds = [
        Circle().sample(density=30, size=1.0, embed_dim=3, rng=rng),
        Disk().sample(density=30, size=1.0, embed_dim=3, rng=rng),
        Sphere().sample(density=30, size=1.0, embed_dim=3, rng=rng),
        Annulus().sample(density=30, size=1.0, embed_dim=3, rng=rng),
    ]
    y = np.array([(1, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)])
    model = ripser_betti_pipeline(n_points=48, n_estimators=5, random_state=0)
    pred = model.fit(clouds, y).predict(clouds)
    assert pred.shape == y.shape
    assert pred.dtype.kind == "i"


def test_gudhi_pipeline_fits_point_clouds():
    pytest.importorskip("gudhi")
    from tda_shapes.ml.ph import gudhi_betti_pipeline

    rng = np.random.default_rng(0)
    clouds = [
        Circle().sample(density=30, size=1.0, embed_dim=3, rng=rng),
        Disk().sample(density=30, size=1.0, embed_dim=3, rng=rng),
        Sphere().sample(density=30, size=1.0, embed_dim=3, rng=rng),
        Annulus().sample(density=30, size=1.0, embed_dim=3, rng=rng),
    ]
    y = np.array([(1, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)])
    model = gudhi_betti_pipeline(n_points=48, grid=8, n_estimators=5, random_state=0)
    assert "clouds" in model.named_steps
    pred = model.fit(clouds, y).predict(clouds)
    assert pred.shape == y.shape
    assert pred.dtype.kind == "i"


@pytest.mark.parametrize(
    "representation",
    [
        "betti",
        "landscape",
        "image",
        "silhouette",
        "entropy",
        "topological_vector",
        "complex_polynomial",
        "atol",
        "persistence_lengths",
    ],
)
def test_gudhi_pipeline_representations_fit(representation):
    pytest.importorskip("gudhi")
    from tda_shapes.ml.ph import gudhi_betti_pipeline

    rng = np.random.default_rng(0)
    clouds = [
        Circle().sample(density=20, size=1.0, embed_dim=3, rng=rng),
        Annulus().sample(density=20, size=1.0, embed_dim=3, rng=rng),
        Disk().sample(density=20, size=1.0, embed_dim=3, rng=rng),
    ]
    y = np.array([(1, 1, 0), (1, 1, 0), (1, 0, 0)])
    model = gudhi_betti_pipeline(
        n_points=24,
        homology_dimensions=(0, 1),
        representation=representation,
        grid=2,
        n_estimators=2,
        random_state=0,
    )
    pred = model.fit(clouds, y).predict(clouds)
    assert pred.shape == y.shape


def test_cech_pipeline_fits_point_clouds():
    pytest.importorskip("gudhi")
    from tda_shapes.ml.ph import cech_betti_pipeline

    rng = np.random.default_rng(0)
    clouds = [
        Circle().sample(density=30, size=1.0, embed_dim=3, rng=rng),
        Disk().sample(density=30, size=1.0, embed_dim=3, rng=rng),
        Sphere().sample(density=30, size=1.0, embed_dim=3, rng=rng),
        Annulus().sample(density=30, size=1.0, embed_dim=3, rng=rng),
    ]
    y = np.array([(1, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)])
    model = cech_betti_pipeline(n_points=32, grid=8, n_estimators=5, random_state=0)
    pred = model.fit(clouds, y).predict(clouds)
    assert pred.shape == y.shape
    assert pred.dtype.kind == "i"


def test_cubical_pipeline_fits_images():
    pytest.importorskip("gudhi")
    from tda_shapes.ml.ph import cubical_betti_pipeline

    ds = make_image_dataset(
        [Circle(), Disk(), Sphere(), Annulus()],
        n_per_class=1,
        density=40.0,
        size_range=(1.0, 1.0),
        resolution=18,
        bandwidth=0.25,
        rng=0,
    )
    model = cubical_betti_pipeline(
        homology_dimensions=(0, 1), grid=8, n_estimators=5, random_state=0
    )
    assert "images" in model.named_steps
    pred = model.fit(ds.images, ds.betti).predict(ds.images)
    assert pred.shape == ds.betti.shape
    assert pred.dtype.kind == "i"


# --- PointNet --------------------------------------------------------------


def test_pointnet_forward_shape():
    import torch

    from tda_shapes.ml.pointnet import PointNetRegressor

    model = PointNetRegressor()
    out = model(torch.randn(4, 100, 3))
    assert out.shape == (4, 3)


def test_pointnet_overfits_tiny_set():
    from tda_shapes.ml.pointnet import predict_pointnet, train_pointnet

    rng = np.random.default_rng(0)
    # A few clean clouds with distinct Betti targets; the net should memorize them.
    clouds = [
        Circle().sample(density=60, size=1.0, embed_dim=3, rng=rng),
        Sphere().sample(density=60, size=1.0, embed_dim=3, rng=rng),
        Disk().sample(density=60, size=1.0, embed_dim=3, rng=rng),
        Annulus().sample(density=60, size=1.0, embed_dim=3, rng=rng),
    ] * 3

    class _DS:
        pass

    ds = _DS()
    ds.clouds = clouds
    ds.betti = np.array(
        [(1, 1, 0), (1, 0, 1), (1, 0, 0), (1, 1, 0)] * 3, dtype=float
    )
    x, y = pointnet_arrays(ds, n_points=128, rng=rng)
    model = train_pointnet(x, y, epochs=120, batch=6, rng=0)
    pred = predict_pointnet(model, x)
    assert np.abs(pred - y).mean() < 0.5  # learned the tiny training set


# --- metrics ---------------------------------------------------------------


def test_betti_metrics_values():
    y_true = np.array([[1, 1, 0], [2, 0, 1]])
    y_pred = np.array([[1, 1, 0], [2, 0, 0]])  # second row b2 wrong
    m = betti_metrics(y_true, y_pred)
    assert m["acc"]["b0"] == 1.0
    assert m["acc"]["b2"] == 0.5
    assert m["exact"] == 0.5
    assert m["mae"]["overall"] == pytest.approx(1 / 6)


# --- end-to-end benchmark smoke -------------------------------------------


def test_run_benchmark_smoke():
    from tda_shapes.ml.benchmark import run_benchmark

    results = run_benchmark(
        k=2,
        n_samples=20,
        density=12.0,
        epochs=2,
        n_points_ph=64,
        run_rips=True,
        seed=0,
        verbose=False,
    )
    assert set(results) == {
        "pointnet",
        "gudhi_rips",
        "gudhi_cech",
        "ripser_learned",
        "ripser_direct",
    }
    for m in results.values():
        assert 0.0 <= m["exact"] <= 1.0
        assert m["mae"]["overall"] >= 0.0
