"""Single-parameter (Vietoris-Rips) persistent homology for Betti prediction.

Each cloud is normalized and subsampled, then ``ripser`` computes its persistence
diagrams up to H2. From the diagrams we build:

* a training-free :func:`direct_betti` baseline that counts long-lived intervals, and
* a feature vector (:func:`diagram_features`) — per-dimension Betti curves plus
  persistence statistics — fed to a learned :class:`PHClassifier`.

GUDHI's scikit-learn API is exposed through :func:`gudhi_betti_pipeline`,
:func:`cech_betti_pipeline`, and :func:`cubical_betti_pipeline`.

Filtration values are normalized per cloud (divided by the largest finite death) so
features and thresholds are comparable across clouds of different scale.
"""

from __future__ import annotations

import numpy as np
from ripser import ripser
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from typing import Literal

from .data import normalize_cloud, to_fixed_size

RngLike = int | np.random.Generator | None
GudhiRepresentation = Literal[
    "betti",
    "landscape",
    "image",
    "silhouette",
    "entropy",
    "topological_vector",
    "complex_polynomial",
    "atol",
    "persistence_lengths",
]


def _prepare_cloud(
    cloud: np.ndarray, *, n_points: int | None, rng: np.random.Generator
) -> np.ndarray:
    pts = np.asarray(cloud, dtype=np.float64)
    if n_points is not None and len(pts) != n_points:
        pts = to_fixed_size(pts, n_points, rng)
    return normalize_cloud(pts)


class PointCloudPreprocessor(BaseEstimator, TransformerMixin):
    """Normalize point clouds and optionally subsample larger clouds."""

    def __init__(self, n_points: int | None = None, random_state: int | None = None):
        self.n_points = n_points
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rng = np.random.default_rng(self.random_state)
        return [_prepare_cloud(c, n_points=self.n_points, rng=rng) for c in X]


class ImagePreprocessor(BaseEstimator, TransformerMixin):
    """Convert raster images to float arrays for GUDHI CubicalPersistence."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [np.asarray(image, dtype=np.float64) for image in X]


class RipserFeatures(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer wrapping :func:`compute_ph`."""

    def __init__(
        self,
        n_points: int | None = None,
        maxdim: int = 2,
        thresh: float | None = None,
        grid: int = 32,
        random_state: int | None = None,
    ):
        self.n_points = n_points
        self.maxdim = maxdim
        self.thresh = thresh
        self.grid = grid
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats, _ = compute_ph(
            X,
            n_points=self.n_points,
            maxdim=self.maxdim,
            thresh=self.thresh,
            grid=self.grid,
            rng=self.random_state,
        )
        return feats


class ComplexToReal(BaseEstimator, TransformerMixin):
    """Convert complex-valued feature matrices to real-valued sklearn inputs."""

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        X = np.asarray(X)
        if np.iscomplexobj(X):
            return np.hstack([X.real, X.imag])
        return X


class FitMarker(BaseEstimator, TransformerMixin):
    """No-op transformer that marks stateless GUDHI branches as fitted."""

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        return X


def cloud_diagrams(
    cloud: np.ndarray,
    *,
    n_points: int | None = None,
    maxdim: int = 2,
    thresh: float | None = None,
    rng: RngLike = None,
) -> list[np.ndarray]:
    """Persistence diagrams (dims 0..maxdim) with filtration normalized to [0, 1].

    Essential/infinite deaths are capped at the cloud diameter, then all (birth,
    death) pairs are divided by it. By default the **whole cloud** is used; pass
    an integer ``n_points`` to resample clouds to a fixed size.
    """
    rng = np.random.default_rng(rng)
    pts = _prepare_cloud(cloud, n_points=n_points, rng=rng)

    # Cloud diameter: a stable geometric scale to normalize the filtration by.
    # (Coordinates are already unit-RMS, so this is comparable across clouds and,
    # unlike "max finite death", does not blow up for shapes with no large
    # persistent feature such as a filled disk.)
    diam = float(np.linalg.norm(pts[:, None] - pts[None], axis=2).max())
    diam = max(diam, 1e-9)

    kw = {"maxdim": maxdim, "thresh": thresh if thresh is not None else diam}
    dgms = ripser(pts, **kw)["dgms"]

    out = []
    for d in dgms:
        d = d.copy()
        d[~np.isfinite(d[:, 1]), 1] = diam  # cap essential / still-alive bars
        out.append(np.clip(d / diam, 0.0, 1.0))  # normalize filtration to [0, 1]
    return out


def _persistences(dgms: list[np.ndarray]) -> list[np.ndarray]:
    return [d[:, 1] - d[:, 0] if len(d) else np.zeros(0) for d in dgms]


def direct_betti(dgms: list[np.ndarray], taus) -> np.ndarray:
    """Training-free estimate: count intervals with persistence above ``tau_i``."""
    pers = _persistences(dgms)
    taus = np.atleast_1d(taus)
    return np.array(
        [int((pers[i] > taus[min(i, len(taus) - 1)]).sum()) for i in range(3)],
        dtype=int,
    )


def tune_taus(
    dgms_list: list[list[np.ndarray]],
    y: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Pick the persistence threshold per dimension that maximizes train accuracy."""
    if grid is None:
        grid = np.linspace(0.02, 0.6, 30)
    taus = np.zeros(3)
    for dim in range(3):
        pers = [_persistences(d)[dim] for d in dgms_list]
        best_tau, best_acc = grid[0], -1.0
        for tau in grid:
            pred = np.array([int((p > tau).sum()) for p in pers])
            acc = float((pred == y[:, dim]).mean())
            if acc > best_acc:
                best_acc, best_tau = acc, tau
        taus[dim] = best_tau
    return taus


def diagram_features(dgms: list[np.ndarray], *, grid: int = 32) -> np.ndarray:
    """Feature vector: per-dim Betti curves (counts alive on a grid) + statistics."""
    ts = np.linspace(0.0, 1.0, grid)
    feats: list[float] = []
    for d in dgms:
        # Betti curve: number of intervals alive at each filtration value.
        if len(d):
            alive = (d[:, 0][None, :] <= ts[:, None]) & (ts[:, None] < d[:, 1][None, :])
            curve = alive.sum(axis=1).astype(float)
            pers = d[:, 1] - d[:, 0]
            stats = [len(d), pers.sum(), pers.max(), pers.mean()]
        else:
            curve = np.zeros(grid)
            stats = [0.0, 0.0, 0.0, 0.0]
        feats.extend(curve.tolist())
        feats.extend(stats)
    return np.asarray(feats, dtype=np.float64)


def compute_ph(
    clouds: list[np.ndarray],
    *,
    n_points: int | None = None,
    maxdim: int = 2,
    thresh: float | None = None,
    grid: int = 32,
    rng: RngLike = None,
) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    """Compute the feature matrix and per-cloud diagrams for a list of clouds.

    ``n_points=None`` (default) uses every point of each cloud (no resampling).
    """
    rng = np.random.default_rng(rng)
    dgms_list = [
        cloud_diagrams(c, n_points=n_points, maxdim=maxdim, thresh=thresh, rng=rng)
        for c in clouds
    ]
    feats = np.vstack([diagram_features(d, grid=grid) for d in dgms_list])
    return feats, dgms_list


class PHClassifier(BaseEstimator, ClassifierMixin):
    """Random-forest multi-output classifier on persistence-diagram features.

    Each Betti dimension is a separate categorical target; scikit-learn's random
    forest handles the multi-output labels ``(N, D)`` natively.
    """

    def __init__(
        self, n_estimators: int = 300, random_state: int = 0, n_jobs: int | None = -1
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PHClassifier":
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(x, np.asarray(y).astype(int))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.predict(x)).astype(int)


def ripser_betti_pipeline(
    *,
    n_points: int | None = None,
    maxdim: int = 2,
    thresh: float | None = None,
    grid: int = 32,
    n_estimators: int = 300,
    random_state: int = 0,
) -> Pipeline:
    """Scikit-learn pipeline using the installed ``ripser`` package."""
    return Pipeline(
        [
            (
                "features",
                RipserFeatures(
                    n_points=n_points,
                    maxdim=maxdim,
                    thresh=thresh,
                    grid=grid,
                    random_state=random_state,
                ),
            ),
            ("model", PHClassifier(n_estimators=n_estimators, random_state=random_state)),
        ]
    )


def _gudhi_diagram_pipeline(
    persistence,
    *,
    preprocessor,
    preprocessor_name: str,
    homology_dimensions: tuple[int, ...],
    representation: GudhiRepresentation,
    grid: int,
    n_estimators: int,
    random_state: int,
    n_jobs: int | None,
) -> Pipeline:
    from gudhi.representations import (
        Atol,
        BettiCurve,
        ComplexPolynomial,
        DiagramSelector,
        DimensionSelector,
        Entropy,
        Landscape,
        PersistenceImage,
        PersistenceLengths,
        Silhouette,
        TopologicalVector,
    )

    def vectorizer():
        match representation:
            case "betti":
                return BettiCurve(resolution=grid)
            case "landscape":
                return Landscape(resolution=grid)
            case "image":
                return PersistenceImage(resolution=[grid, grid])
            case "silhouette":
                return Silhouette(resolution=grid)
            case "entropy":
                return Entropy(mode="vector", normalized=False, resolution=grid)
            case "topological_vector":
                return TopologicalVector(threshold=grid)
            case "complex_polynomial":
                return Pipeline(
                    [
                        ("polynomial", ComplexPolynomial(threshold=grid)),
                        ("real", ComplexToReal()),
                    ]
                )
            case "atol":
                return Atol(
                    quantiser=KMeans(n_clusters=grid, random_state=random_state, n_init=10)
                )
            case "persistence_lengths":
                return PersistenceLengths(num_lengths=grid)
        raise ValueError(f"Unknown GUDHI representation: {representation}")

    branches = []
    for index, dim in enumerate(homology_dimensions):
        branches.append(
            (
                f"H{dim}",
                Pipeline(
                    [
                        ("select", DimensionSelector(index=index)),
                        ("finite", DiagramSelector(use=True, point_type="finite")),
                        ("vector", vectorizer()),
                        ("fit_marker", FitMarker()),
                    ]
                ),
            )
        )

    return Pipeline(
        [
            (preprocessor_name, preprocessor),
            ("persistence", persistence),
            ("features", FeatureUnion(branches)),
            (
                "model",
                PHClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def gudhi_betti_pipeline(
    *,
    n_points: int | None = None,
    homology_dimensions: tuple[int, ...] = (0, 1, 2),
    threshold: float = float("inf"),
    representation: GudhiRepresentation = "betti",
    grid: int = 32,
    n_estimators: int = 300,
    random_state: int = 0,
    n_jobs: int | None = -1,
) -> Pipeline:
    """Point-cloud scikit-learn pipeline backed by GUDHI's RipsPersistence."""
    from gudhi.sklearn import RipsPersistence

    return _gudhi_diagram_pipeline(
        RipsPersistence(
            homology_dimensions=homology_dimensions,
            threshold=threshold,
            n_jobs=n_jobs,
        ),
        preprocessor=PointCloudPreprocessor(n_points=n_points, random_state=random_state),
        preprocessor_name="clouds",
        homology_dimensions=homology_dimensions,
        representation=representation,
        grid=grid,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def cech_betti_pipeline(
    *,
    n_points: int | None = None,
    homology_dimensions: tuple[int, ...] = (0, 1, 2),
    precision: Literal["fast", "safe", "exact"] = "safe",
    threshold: float = float("inf"),
    representation: GudhiRepresentation = "betti",
    grid: int = 32,
    n_estimators: int = 300,
    random_state: int = 0,
    n_jobs: int | None = -1,
) -> Pipeline:
    """Point-cloud scikit-learn pipeline backed by GUDHI's CechPersistence."""
    from gudhi.sklearn import CechPersistence

    return _gudhi_diagram_pipeline(
        CechPersistence(
            homology_dimensions=homology_dimensions,
            precision=precision,
            threshold=threshold,
            n_jobs=n_jobs,
        ),
        preprocessor=PointCloudPreprocessor(n_points=n_points, random_state=random_state),
        preprocessor_name="clouds",
        homology_dimensions=homology_dimensions,
        representation=representation,
        grid=grid,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def cubical_betti_pipeline(
    *,
    homology_dimensions: tuple[int, ...] = (0, 1, 2),
    input_type: Literal["top_dimensional_cells", "vertices"] = "top_dimensional_cells",
    min_persistence: float = 0.0,
    representation: GudhiRepresentation = "betti",
    grid: int = 32,
    n_estimators: int = 300,
    random_state: int = 0,
    n_jobs: int | None = -1,
) -> Pipeline:
    """Image scikit-learn pipeline backed by GUDHI's CubicalPersistence."""
    from gudhi.sklearn import CubicalPersistence

    return _gudhi_diagram_pipeline(
        CubicalPersistence(
            homology_dimensions=homology_dimensions,
            input_type=input_type,
            min_persistence=min_persistence,
            n_jobs=n_jobs,
        ),
        preprocessor=ImagePreprocessor(),
        preprocessor_name="images",
        homology_dimensions=homology_dimensions,
        representation=representation,
        grid=grid,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
