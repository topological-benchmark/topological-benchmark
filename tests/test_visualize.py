"""Headless smoke tests for the visualization helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display needed

import numpy as np

from tda_shapes import Circle, Sphere
from tda_shapes.visualize import gallery, plot_cloud


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
