"""Matplotlib visualization of sampled point clouds (2-D and 3-D).

Run as a script to render a gallery of the default shapes::

    uv run python -m tda_shapes.visualize --out gallery.png
    uv run python -m tda_shapes.visualize --stretch 1,1,2 --show

``plot_cloud`` and ``gallery`` can also be imported and used directly.
"""

from __future__ import annotations

import argparse

import numpy as np

from .shapes import DEFAULT_SHAPES, RngLike, Shape, StretchLike


def _equalize_3d(ax, pts: np.ndarray) -> None:
    """Give a 3-D axis an equal aspect ratio around the data."""
    center = pts.mean(axis=0)
    radius = float(np.abs(pts - center).max())
    if radius == 0.0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_cloud(
    pts: np.ndarray,
    ax=None,
    title: str | None = None,
    labels: np.ndarray | None = None,
    **scatter_kw,
):
    """Scatter-plot a single point cloud.

    Works for 2-D and 3-D clouds. If ``ax`` is omitted a new figure/axis is
    created with the matching projection. If ``labels`` is given, points are
    colored by label (e.g. per-component in a composite scene). Returns the axis.
    """
    import matplotlib.pyplot as plt

    dim = pts.shape[1]
    if dim not in (2, 3):
        raise ValueError(f"can only plot 2-D or 3-D clouds, got dim={dim}")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)

    kw = {"s": 6, "alpha": 0.6, "edgecolors": "none"}
    kw.update(scatter_kw)
    if labels is not None:
        kw.setdefault("cmap", "tab10")
        kw["c"] = labels

    coords = [pts[:, i] for i in range(dim)]
    if dim == 3:
        ax.scatter(*coords, **kw)
        _equalize_3d(ax, pts)
    else:
        ax.scatter(*coords, **kw)
        ax.set_aspect("equal")

    if title:
        ax.set_title(title, fontsize=10)
    return ax


def gallery(
    shapes: list[Shape] | None = None,
    *,
    density: float = 30.0,
    size: float = 1.5,
    noise: float = 0.02,
    stretch: StretchLike = None,
    embed_dim: int | None = 3,
    ncols: int = 4,
    rng: RngLike = None,
):
    """Render one 3-D subplot per shape and return the matplotlib figure.

    All clouds are embedded into ``embed_dim`` (default 3) so they share a 3-D
    view; planar shapes simply lie in the ``z = 0`` plane. ``stretch`` is passed
    through to :meth:`Shape.sample`, so a single call can show every shape
    stretched (e.g. the sphere as an ellipsoid).
    """
    import matplotlib.pyplot as plt

    if shapes is None:
        shapes = DEFAULT_SHAPES
    rng = np.random.default_rng(rng)

    nrows = (len(shapes) + ncols - 1) // ncols
    fig = plt.figure(figsize=(3.2 * ncols, 3.2 * nrows))

    for i, shape in enumerate(shapes):
        # A 2-D shape only accepts a length-2 stretch; trim if needed.
        shp = stretch
        if stretch is not None:
            arr = np.broadcast_to(np.asarray(stretch, float), (3,))
            shp = arr[: shape.native_dim]
        pts = shape.sample(
            density=density,
            size=size,
            noise=noise,
            stretch=shp,
            embed_dim=embed_dim,
            rng=rng,
        )
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        plot_cloud(pts, ax=ax, title=f"{shape.name}  b={shape.betti}  n={len(pts)}")

    fig.tight_layout()
    return fig


def composite_gallery(
    k: int = 5,
    n: int = 6,
    pool: list[Shape] | None = None,
    *,
    density: float = 30.0,
    size_range: tuple[float, float] = (1.0, 2.0),
    noise: float = 0.02,
    clearance: float = 0.5,
    rotate: bool = True,
    ncols: int = 3,
    rng: RngLike = None,
):
    """Render ``n`` composite scenes (each with ``k`` shapes) colored by component."""
    import matplotlib.pyplot as plt

    from .composite import sample_composite

    rng = np.random.default_rng(rng)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(3.4 * ncols, 3.4 * nrows))

    for i in range(n):
        scene = sample_composite(
            k,
            pool,
            density=density,
            size_range=size_range,
            noise=noise,
            clearance=clearance,
            rotate=rotate,
            rng=rng,
        )
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        betti = tuple(int(x) for x in scene.betti)
        plot_cloud(
            scene.points,
            ax=ax,
            labels=scene.component_labels,
            s=10,
            title=f"k={k}  b={betti}  n={len(scene.points)}",
        )

    fig.tight_layout()
    return fig


def _parse_stretch(text: str | None):
    if text is None:
        return None
    return tuple(float(x) for x in text.split(","))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize TDA shape point clouds.")
    parser.add_argument("--density", type=float, default=30.0)
    parser.add_argument("--size", type=float, default=1.5)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument(
        "--stretch",
        type=str,
        default=None,
        help="comma-separated per-axis factors, e.g. '1,1,2' for an ellipsoid",
    )
    parser.add_argument(
        "--composite",
        type=int,
        default=None,
        metavar="K",
        help="render a gallery of composite scenes, each containing K shapes",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="shapes_gallery.png")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args(argv)

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")  # headless rendering
    import matplotlib.pyplot as plt

    if args.composite is not None:
        fig = composite_gallery(
            args.composite, density=args.density, noise=args.noise, rng=args.seed
        )
    else:
        fig = gallery(
            density=args.density,
            size=args.size,
            noise=args.noise,
            stretch=_parse_stretch(args.stretch),
            rng=args.seed,
        )
    if args.out:
        fig.savefig(args.out, dpi=120)
        print(f"Saved gallery to {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
