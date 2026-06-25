"""Tests for shape sampling: density, count scaling, on-shape, betti, uniformity."""

from __future__ import annotations

import numpy as np
import pytest

from tda_shapes import (
    DEFAULT_SHAPES,
    Annulus,
    Ball,
    Circle,
    Disk,
    Sphere,
    Torus,
    TwoCircles,
)

DENSITY = 200.0


def test_count_scales_with_size_to_intrinsic_power():
    circle = Circle()
    # 1-D: doubling size doubles the count.
    assert circle.expected_n(DENSITY, 2.0) == pytest.approx(
        2 * circle.expected_n(DENSITY, 1.0), rel=0.01
    )
    sphere = Sphere()
    # 2-D: doubling size quadruples the count.
    assert sphere.expected_n(DENSITY, 2.0) == pytest.approx(
        4 * sphere.expected_n(DENSITY, 1.0), rel=0.01
    )


def test_constant_density_recovered_from_counts():
    # Empirical N / (unit_measure * size^d) should match the requested density.
    for shape in DEFAULT_SHAPES:
        for size in (1.0, 2.5):
            n = shape.expected_n(DENSITY, size)
            est = n / (shape.unit_measure() * size**shape.intrinsic_dim)
            assert est == pytest.approx(DENSITY, rel=0.02)


def test_size_for_n_round_trips():
    for shape in DEFAULT_SHAPES:
        size = shape.size_for_n(DENSITY, 500)
        assert shape.expected_n(DENSITY, size) == pytest.approx(500, abs=2)


def test_actual_point_count_matches_expected():
    rng = np.random.default_rng(0)
    for shape in DEFAULT_SHAPES:
        pts = shape.sample(density=DENSITY, size=1.5, rng=rng)
        assert pts.shape[0] == shape.expected_n(DENSITY, 1.5)
        assert pts.shape[1] == shape.native_dim


@pytest.mark.parametrize("size", [0.7, 2.0])
def test_circle_points_on_circle(size):
    rng = np.random.default_rng(1)
    pts = Circle().sample(density=DENSITY, size=size, noise=0.0, rng=rng)
    radii = np.linalg.norm(pts, axis=1)
    assert np.allclose(radii, size, atol=1e-9)


def test_sphere_points_on_sphere():
    rng = np.random.default_rng(2)
    size = 1.3
    pts = Sphere().sample(density=DENSITY, size=size, noise=0.0, rng=rng)
    radii = np.linalg.norm(pts, axis=1)
    assert np.allclose(radii, size, atol=1e-9)


def test_annulus_radii_within_ring():
    rng = np.random.default_rng(3)
    shape = Annulus(inner_ratio=0.5)
    size = 2.0
    pts = shape.sample(density=DENSITY, size=size, noise=0.0, rng=rng)
    radii = np.linalg.norm(pts, axis=1)
    assert radii.min() >= 0.5 * size - 1e-9
    assert radii.max() <= size + 1e-9


def test_two_circles_split_into_two_components():
    rng = np.random.default_rng(4)
    shape = TwoCircles(separation=4.0)
    pts = shape.sample(density=DENSITY, size=1.0, noise=0.0, rng=rng)
    # Left/right halves should both be populated and well separated in x.
    left = pts[pts[:, 0] < 0]
    right = pts[pts[:, 0] > 0]
    assert len(left) > 0 and len(right) > 0
    assert right[:, 0].min() - left[:, 0].max() > 1.0


def test_betti_numbers_are_ground_truth():
    expected = {
        "circle": (1, 1, 0),
        "two_circles": (2, 2, 0),
        "figure_eight": (1, 2, 0),
        "disk": (1, 0, 0),
        "annulus": (1, 1, 0),
        "sphere": (1, 0, 1),
        "ball": (1, 0, 0),
        "torus": (1, 2, 1),
    }
    for shape in DEFAULT_SHAPES:
        assert shape.betti == expected[shape.name]


def test_disk_is_uniform_by_area():
    # Uniform-by-area means the fraction of points inside radius t*size is t^2.
    rng = np.random.default_rng(5)
    pts = Disk().sample(density=2000.0, size=1.0, noise=0.0, rng=rng)
    radii = np.linalg.norm(pts, axis=1)
    for t in (0.25, 0.5, 0.75):
        frac = np.mean(radii < t)
        assert frac == pytest.approx(t**2, abs=0.04)


def test_torus_theta_distribution_matches_area_density():
    # After rejection sampling, theta near the outer rim (cos~1) should be more
    # common than near the inner rim (cos~-1), matching (R + r cos theta).
    rng = np.random.default_rng(6)
    shape = Torus(tube_ratio=0.35)
    pts = shape.sample(density=4000.0, size=1.0, noise=0.0, rng=rng)
    # Recover theta from z = r sin(theta) and rho = R + r cos(theta).
    rho = np.hypot(pts[:, 0], pts[:, 1])
    cos_theta = (rho - 1.0) / 0.35
    outer = np.mean(cos_theta > 0)  # rim half
    assert outer > 0.5  # area-weighted toward the outer rim


def test_embed_dim_pads_planar_shapes_into_3d():
    rng = np.random.default_rng(7)
    pts = Circle().sample(density=DENSITY, size=1.0, noise=0.0, rng=rng, embed_dim=3)
    assert pts.shape[1] == 3
    assert np.allclose(pts[:, 2], 0.0)


def test_noise_perturbs_points_off_shape():
    rng = np.random.default_rng(8)
    clean = Circle().sample(density=DENSITY, size=1.0, noise=0.0, rng=np.random.default_rng(8))
    noisy = Circle().sample(density=DENSITY, size=1.0, noise=0.1, rng=rng)
    # Same count, but noisy points no longer lie exactly on the unit circle.
    assert clean.shape == noisy.shape
    assert not np.allclose(np.linalg.norm(noisy, axis=1), 1.0, atol=1e-3)


def test_sampling_is_reproducible():
    a = Sphere().sample(density=DENSITY, size=1.0, noise=0.05, rng=42)
    b = Sphere().sample(density=DENSITY, size=1.0, noise=0.05, rng=42)
    assert np.array_equal(a, b)


# --- filled ball -----------------------------------------------------------


def test_ball_points_inside_ball():
    rng = np.random.default_rng(10)
    size = 1.4
    pts = Ball().sample(density=300.0, size=size, noise=0.0, rng=rng)
    assert pts.shape[1] == 3
    assert np.linalg.norm(pts, axis=1).max() <= size + 1e-9


def test_ball_is_uniform_by_volume():
    # Fraction of points within radius t*size should be t^3 for a solid ball.
    rng = np.random.default_rng(11)
    pts = Ball().sample(density=3000.0, size=1.0, noise=0.0, rng=rng)
    radii = np.linalg.norm(pts, axis=1)
    for t in (0.3, 0.6, 0.9):
        assert np.mean(radii < t) == pytest.approx(t**3, abs=0.05)


def test_ball_count_scales_as_cube():
    ball = Ball()
    assert ball.expected_n(300.0, 2.0) == pytest.approx(
        8 * ball.expected_n(300.0, 1.0), rel=0.01
    )


# --- anisotropic stretch ---------------------------------------------------


def test_stretch_sphere_lies_on_ellipsoid():
    rng = np.random.default_rng(12)
    a, b, c = 1.0, 1.0, 2.0
    pts = Sphere().sample(density=300.0, size=1.0, noise=0.0, stretch=(a, b, c), rng=rng)
    resid = (pts[:, 0] / a) ** 2 + (pts[:, 1] / b) ** 2 + (pts[:, 2] / c) ** 2
    assert np.allclose(resid, 1.0, atol=1e-9)


def test_stretch_circle_lies_on_ellipse():
    rng = np.random.default_rng(13)
    a, b = 1.0, 3.0
    pts = Circle().sample(density=300.0, size=1.0, noise=0.0, stretch=(a, b), rng=rng)
    resid = (pts[:, 0] / a) ** 2 + (pts[:, 1] / b) ** 2
    assert np.allclose(resid, 1.0, atol=1e-9)


def test_isotropic_stretch_matches_size_count():
    # stretch=(k,k,k) must reproduce the count of size=k (constant density).
    sphere = Sphere()
    n = len(sphere.sample(density=300.0, size=1.0, stretch=(2.0, 2.0, 2.0), noise=0.0, rng=0))
    assert n == pytest.approx(sphere.expected_n(300.0, 2.0), rel=0.03)


def test_stretched_surface_density_is_uniform():
    # Density per unit area should be ~constant over a strongly stretched
    # ellipsoid: split into z<0 / z>0 halves and compare points-per-area. We
    # estimate each half's area from its own point count is circular, so instead
    # check that the two poles (small area) are not over-represented relative to
    # the equator the way a naive (non-reweighted) map would make them.
    rng = np.random.default_rng(14)
    pts = Sphere().sample(density=4000.0, size=1.0, noise=0.0, stretch=(1, 1, 3), rng=rng)
    z = pts[:, 2] / 3.0  # back to unit-sphere latitude cosine
    # Uniform-by-area on the ellipsoid puts more points near the equator (where
    # the stretched area is largest), so |z| small should dominate.
    assert np.mean(np.abs(z) < 0.5) > np.mean(np.abs(z) > 0.5)


def test_solid_stretch_count_scales_with_product():
    disk = Disk()
    n = len(disk.sample(density=300.0, size=1.0, stretch=(2.0, 3.0), noise=0.0, rng=0))
    # area of stretched disk = pi * 2 * 3 ; count = density * area
    assert n == pytest.approx(300.0 * np.pi * 2.0 * 3.0, rel=0.02)


def test_stretch_solid_disk_fills_ellipse():
    rng = np.random.default_rng(15)
    a, b = 2.0, 3.0
    pts = Disk().sample(density=300.0, size=1.0, noise=0.0, stretch=(a, b), rng=rng)
    resid = (pts[:, 0] / a) ** 2 + (pts[:, 1] / b) ** 2
    assert resid.max() <= 1.0 + 1e-9
