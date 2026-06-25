"""Headless smoke tests for the visualization helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display needed

import numpy as np

from tda_shapes import BACKGROUND_LABEL, Circle, Sphere
from tda_shapes.visualize import composite_gallery, gallery, plot_cloud


def test_plot_cloud_3d_returns_axis():
    pts = Sphere().sample(density=20.0, size=1.0, rng=0)
    ax = plot_cloud(pts, title="sphere")
    assert ax is not None


def test_plot_cloud_2d_returns_axis():
    pts = Circle().sample(density=20.0, size=1.0, rng=0)
    assert pts.shape[1] == 2
    ax = plot_cloud(pts)
    assert ax is not None


def test_gallery_renders_and_saves(tmp_path):
    fig = gallery(density=10.0, size=1.0, rng=0)
    out = tmp_path / "g.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0


def test_gallery_with_stretch():
    fig = gallery(density=10.0, size=1.0, stretch=(1, 1, 2), rng=0)
    # one axis per default shape
    assert len(fig.axes) >= 8


def test_plot_cloud_renders_background_labels():
    # A cloud whose labels include BACKGROUND_LABEL must render without error.
    shape = Sphere().sample(density=20.0, size=1.0, rng=0)
    clutter = np.random.default_rng(1).uniform(-1.5, 1.5, size=(40, 3))
    pts = np.vstack([shape, clutter])
    labels = np.concatenate(
        [np.zeros(len(shape), dtype=int), np.full(len(clutter), BACKGROUND_LABEL)]
    )
    ax = plot_cloud(pts, labels=labels)
    assert ax is not None


def test_composite_gallery_with_background_saves(tmp_path):
    fig = composite_gallery(k=3, n=2, density=20.0, background_density=2.0, rng=0)
    out = tmp_path / "clutter.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0
