"""Tests for composite (k-shape) scenes and datasets."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tda_shapes import (
    CompositeDataset,
    make_composite_dataset,
    sample_composite,
)
from tda_shapes.composite import _pack_centers, _pack_centers_line
from tda_shapes.shapes import Circle, Sphere

DENSITY = 80.0


def test_counts_sum_to_k():
    scene = sample_composite(5, density=DENSITY, rng=0)
    assert scene.counts.sum() == 5
    assert len(scene.shape_names) == scene.counts.shape[0]


def test_betti_is_sum_of_components():
    # betti must equal counts-weighted sum of pool shapes' betti.
    from tda_shapes import DEFAULT_SHAPES

    scene = sample_composite(6, density=DENSITY, rng=1)
    expected = sum(
        n * np.asarray(DEFAULT_SHAPES[i].betti)
        for i, n in enumerate(scene.counts)
    )
    assert np.array_equal(scene.betti, expected)


def test_three_circles_betti():
    scene = sample_composite(3, [Circle()], density=DENSITY, rng=2)
    assert tuple(scene.betti) == (3, 3, 0)  # 3 components, 3 loops
    assert scene.counts.tolist() == [3]


def test_two_spheres_betti():
    scene = sample_composite(2, [Sphere()], density=DENSITY, rng=3)
    assert tuple(scene.betti) == (2, 0, 2)  # 2 components, 2 voids


def test_constant_density_carries_through():
    # k=1 with a pinned size must reproduce the single-shape point count exactly
    # (the union neither adds nor drops points).
    circle = Circle()
    scene = sample_composite(
        1, [circle], density=DENSITY, size_range=(2.0, 2.0), noise=0.0, rng=4
    )
    assert len(scene.points) == circle.expected_n(DENSITY, 2.0)


def test_components_are_disjoint_when_clean():
    # With distinct shape types and no noise, points of different components keep
    # at least the clearance gap.
    from tda_shapes import Disk, Sphere, Torus, TwoCircles

    pool = [Disk(), Sphere(), Torus(), TwoCircles()]
    clearance = 0.5
    scene = sample_composite(
        4, pool, density=DENSITY, noise=0.0, clearance=clearance, rotate=False, rng=5
    )
    pts, lab = scene.points, scene.component_labels
    # component_labels are shape-type ids, so same-type components merge into one
    # group; any two distinct-type groups are still >= clearance apart because
    # every underlying component is clearance-separated from every other.
    ids = np.unique(lab)
    for a, b in itertools.combinations(ids, 2):
        d = np.linalg.norm(pts[lab == a][:, None] - pts[lab == b][None], axis=2).min()
        assert d >= clearance - 1e-9


def test_pack_centers_are_non_overlapping():
    rng = np.random.default_rng(6)
    radii = rng.uniform(0.5, 3.0, size=8)
    clearance = 0.4
    centers = _pack_centers(radii, clearance, dim=3, rng=rng)
    for i, j in itertools.combinations(range(len(radii)), 2):
        d = np.linalg.norm(centers[i] - centers[j])
        assert d >= radii[i] + radii[j] + clearance - 1e-9


def test_pack_centers_line_fallback_non_overlapping():
    radii = np.array([1.0, 2.0, 0.5, 1.5])
    clearance = 0.3
    centers = _pack_centers_line(radii, clearance, dim=3)
    for i, j in itertools.combinations(range(len(radii)), 2):
        d = np.linalg.norm(centers[i] - centers[j])
        assert d >= radii[i] + radii[j] + clearance - 1e-9


def test_large_k_packs():
    scene = sample_composite(15, density=30.0, noise=0.0, rng=7)
    assert scene.counts.sum() == 15
    assert scene.points.shape[1] == 3


def test_component_labels_cover_all_points():
    scene = sample_composite(4, density=DENSITY, rng=8)
    assert scene.component_labels.shape[0] == scene.points.shape[0]
    assert scene.component_labels.max() < len(scene.shape_names)


def test_reproducible():
    a = sample_composite(4, density=DENSITY, noise=0.03, rng=42)
    b = sample_composite(4, density=DENSITY, noise=0.03, rng=42)
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(a.betti, b.betti)


def test_make_composite_dataset_shapes():
    cds = make_composite_dataset(10, k=3, density=DENSITY, rng=9)
    assert len(cds) == 10
    assert cds.betti.shape == (10, 3)
    assert cds.shape_counts.shape[0] == 10
    assert (cds.shape_counts.sum(axis=1) == 3).all()
    assert cds.k == 3


def test_make_composite_dataset_save_load(tmp_path):
    cds = make_composite_dataset(6, k=2, density=DENSITY, rng=10)
    path = str(tmp_path / "comp.npz")
    cds.save(path)
    loaded = CompositeDataset.load(path)
    assert loaded.k == cds.k
    assert loaded.shape_names == cds.shape_names
    assert np.array_equal(loaded.betti, cds.betti)
    assert np.array_equal(loaded.shape_counts, cds.shape_counts)
    assert all(np.array_equal(a, b) for a, b in zip(loaded.clouds, cds.clouds))
