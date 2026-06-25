"""Tests for KDE-rasterized image datasets."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from tda_shapes import Circle, ImageDataset, make_image_dataset, rasterize_kde


def test_make_image_dataset_shapes_labels_and_betti():
    ds = make_image_dataset([Circle()], n_per_class=3, density=80.0, rng=0)
    assert len(ds) == 3
    assert ds.images.shape == (3, 48, 48, 48)
    assert ds.images.dtype == np.float32
    assert np.isfinite(ds.images).all()
    assert ds.labels.tolist() == [0, 0, 0]
    assert ds.label_names == ["circle"]
    assert np.array_equal(ds.betti, np.array([[1, 1, 0]] * 3))
    assert ds.clouds is None


def test_image_dataset_save_load_round_trip(tmp_path):
    ds = make_image_dataset(
        [Circle()], n_per_class=2, density=80.0, keep_clouds=True, rng=1
    )
    path = str(tmp_path / "images.npz")
    ds.save(path)

    loaded = ImageDataset.load(path)
    assert np.array_equal(loaded.images, ds.images)
    assert np.array_equal(loaded.labels, ds.labels)
    assert loaded.label_names == ds.label_names
    assert np.array_equal(loaded.betti, ds.betti)
    assert loaded.clouds is not None
    assert ds.clouds is not None
    assert all(np.array_equal(a, b) for a, b in zip(loaded.clouds, ds.clouds))
    assert all(c.dtype == np.float64 for c in loaded.clouds)


def test_make_image_dataset_reproducible():
    a = make_image_dataset([Circle()], n_per_class=2, density=80.0, rng=7)
    b = make_image_dataset([Circle()], n_per_class=2, density=80.0, rng=7)
    assert np.array_equal(a.images, b.images)
    assert np.array_equal(a.labels, b.labels)


def test_make_image_dataset_allows_empty():
    ds = make_image_dataset([Circle()], n_per_class=0)
    assert len(ds) == 0
    assert ds.images.shape == (0, 48, 48, 48)
    assert ds.labels.shape == (0,)
    assert ds.betti.shape == (0, 3)


def test_kde_image_has_resolved_feature():
    ds = make_image_dataset([Circle()], n_per_class=1, density=120.0, rng=2)
    image = ds.images[0]
    assert image.min() == pytest.approx(0.0)
    assert image.max() > 0.5
    assert np.count_nonzero(image < 0.25) > 20


def test_bandwidth_pixel_compatibility_rejects_aliasing():
    points = Circle().sample(density=80.0, size=1.0, embed_dim=3, rng=3)
    with pytest.raises(ValueError, match="bandwidth too small"):
        rasterize_kde(points, resolution=16, bandwidth=0.01)


def test_rasterize_rejects_unknown_backend():
    points = Circle().sample(density=80.0, size=1.0, embed_dim=3, rng=3)
    with pytest.raises(ValueError, match="unknown image backend"):
        rasterize_kde(points, resolution=16, bandwidth=0.25, backend=cast(Any, "bogus"))


def test_rasterize_mps_matches_numpy_when_available():
    jax = pytest.importorskip("jax")
    try:
        jax.devices("mps")
    except RuntimeError:
        pytest.skip("JAX MPS backend unavailable")
    points = Circle().sample(density=40.0, size=1.0, embed_dim=3, rng=4)
    numpy_image = rasterize_kde(points, resolution=16, bandwidth=0.25, backend="numpy")
    mps_image = rasterize_kde(points, resolution=16, bandwidth=0.25, backend="mps")
    assert np.max(np.abs(numpy_image - mps_image)) < 1e-4


def test_default_relative_bandwidth_has_cubical_h1_signal():
    gudhi = pytest.importorskip("gudhi")
    ds = make_image_dataset(
        [Circle()],
        n_per_class=1,
        density=180.0,
        size_range=(1.0, 1.0),
        rng=4,
    )
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=ds.images[0])
    complex_.persistence(homology_coeff_field=2, min_persistence=0.0)
    h1 = complex_.persistence_intervals_in_dimension(1)

    assert len(h1) > 0
    assert np.max(h1[:, 1] - h1[:, 0]) > 0.1
