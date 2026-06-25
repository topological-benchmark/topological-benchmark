"""Single-parameter (Vietoris-Rips) persistent homology for Betti prediction.

Each cloud is normalized and subsampled, then ``ripser`` computes its persistence
diagrams up to H2. From the diagrams we build:

* a training-free :func:`direct_betti` baseline that counts long-lived intervals, and
* a feature vector (:func:`diagram_features`) — per-dimension Betti curves plus
  persistence statistics — fed to a learned :class:`PHRegressor`.

Filtration values are normalized per cloud (divided by the largest finite death) so
features and thresholds are comparable across clouds of different scale.
"""

from __future__ import annotations

import numpy as np
from ripser import ripser
from sklearn.ensemble import RandomForestRegressor

from .data import normalize_cloud, to_fixed_size

RngLike = int | np.random.Generator | None


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
    an integer ``n_points`` to subsample larger clouds (Vietoris-Rips up to H2 is
    O(N^3), so subsampling is the lever for very large clouds).
    """
    rng = np.random.default_rng(rng)
    pts = normalize_cloud(cloud)
    if n_points is not None and len(pts) > n_points:
        pts = to_fixed_size(pts, n_points, rng)

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

    ``n_points=None`` (default) uses every point of each cloud (no subsampling).
    """
    rng = np.random.default_rng(rng)
    dgms_list = [
        cloud_diagrams(c, n_points=n_points, maxdim=maxdim, thresh=thresh, rng=rng)
        for c in clouds
    ]
    feats = np.vstack([diagram_features(d, grid=grid) for d in dgms_list])
    return feats, dgms_list


class PHRegressor:
    """Random-forest multi-output regressor on persistence-diagram features."""

    def __init__(self, n_estimators: int = 300, random_state: int = 0):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PHRegressor":
        self.model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred = self.model.predict(x)
        return np.clip(np.rint(pred), 0, None).astype(int)
