"""Tests for make_dataset and ShapeDataset save/load."""

from __future__ import annotations

import numpy as np

from tda_shapes import DEFAULT_SHAPES, ShapeDataset, make_dataset


def test_make_dataset_shapes_labels_and_betti():
    ds = make_dataset(n_per_class=5, density=50.0, rng=0)
    n_classes = len(DEFAULT_SHAPES)
    assert len(ds) == 5 * n_classes
    assert ds.labels.shape == (5 * n_classes,)
    assert ds.betti.shape == (5 * n_classes, 3)
    assert ds.label_names == [s.name for s in DEFAULT_SHAPES]
    # Each cloud's betti matches its label's shape.
    for label, betti in zip(ds.labels, ds.betti):
        assert tuple(betti) == DEFAULT_SHAPES[label].betti


def test_make_dataset_embed_dim_uniform():
    ds = make_dataset(n_per_class=3, density=50.0, embed_dim=3, rng=1)
    assert all(c.shape[1] == 3 for c in ds.clouds)


def test_make_dataset_counts_vary_with_size():
    ds = make_dataset(n_per_class=20, density=50.0, size_range=(1.0, 3.0), rng=2)
    counts = [c.shape[0] for c in ds.clouds]
    assert min(counts) < max(counts)  # size variation -> count variation


def test_make_dataset_reproducible():
    a = make_dataset(n_per_class=4, density=40.0, rng=7)
    b = make_dataset(n_per_class=4, density=40.0, rng=7)
    assert np.array_equal(a.labels, b.labels)
    assert all(np.array_equal(x, y) for x, y in zip(a.clouds, b.clouds))


def test_make_dataset_with_stretch_range():
    ds = make_dataset(
        n_per_class=3, density=40.0, stretch_range=(0.6, 1.6), embed_dim=3, rng=5
    )
    assert all(c.shape[1] == 3 for c in ds.clouds)
    assert len(ds) == 3 * len(DEFAULT_SHAPES)


def test_save_load_round_trip(tmp_path):
    ds = make_dataset(n_per_class=3, density=50.0, rng=3)
    path = str(tmp_path / "ds.npz")
    ds.save(path)
    loaded = ShapeDataset.load(path)
    assert loaded.label_names == ds.label_names
    assert np.array_equal(loaded.labels, ds.labels)
    assert np.array_equal(loaded.betti, ds.betti)
    assert all(np.array_equal(a, b) for a, b in zip(loaded.clouds, ds.clouds))
