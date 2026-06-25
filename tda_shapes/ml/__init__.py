"""Machine-learning baselines for predicting Betti numbers from point clouds.

Two methods are compared (see :mod:`tda_shapes.ml.benchmark`):

* :mod:`tda_shapes.ml.pointnet` — a PointNet regressor that learns from raw coordinates.
* :mod:`tda_shapes.ml.ph` — single-parameter (Vietoris–Rips) persistent homology, both a
  learned feature model and a training-free "count long bars" baseline.

These submodules require the optional deps ``torch``, ``ripser`` and ``scikit-learn``;
they are imported lazily so ``import tda_shapes`` stays lightweight.
"""

from .data import (
    normalize_cloud,
    pointnet_arrays,
    to_fixed_size,
    train_test_split_idx,
)
from .metrics import betti_metrics, format_comparison

__all__ = [
    "normalize_cloud",
    "to_fixed_size",
    "pointnet_arrays",
    "train_test_split_idx",
    "betti_metrics",
    "format_comparison",
]
