"""TDA shapes: sample labeled point clouds from shapes with known topology."""

from .composite import (
    Composite,
    CompositeDataset,
    make_composite_dataset,
    sample_composite,
)
from .dataset import ShapeDataset, make_dataset
from .shapes import (
    DEFAULT_SHAPES,
    Annulus,
    Ball,
    Circle,
    Disk,
    FigureEight,
    Shape,
    Sphere,
    Torus,
    TwoCircles,
    default_shapes,
)

# Visualization (``plot_cloud``, ``gallery``) lives in ``tda_shapes.visualize``;
# it is imported lazily so that ``import tda_shapes`` does not require matplotlib.

__all__ = [
    "Shape",
    "Circle",
    "TwoCircles",
    "FigureEight",
    "Disk",
    "Annulus",
    "Sphere",
    "Ball",
    "Torus",
    "default_shapes",
    "DEFAULT_SHAPES",
    "ShapeDataset",
    "make_dataset",
    "Composite",
    "CompositeDataset",
    "sample_composite",
    "make_composite_dataset",
]
