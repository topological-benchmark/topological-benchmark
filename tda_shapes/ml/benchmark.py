"""Compare PointNet vs. TDA baselines on predicting Betti numbers.

Run directly::

    uv run python -m tda_shapes.ml.benchmark --k 3 --n 300 --epochs 80
    uv run python -m tda_shapes.ml.benchmark --k 3 --n 200 --plot bench.png
"""

from __future__ import annotations

import argparse

import numpy as np

from ..composite import make_composite_dataset
from .data import pointnet_arrays, train_test_split_idx
from .metrics import betti_metrics, format_comparison


def run_benchmark(
    *,
    k: int = 3,
    n_samples: int = 600,
    density: float = 6.0,
    size_range: tuple[float, float] = (0.8, 1.3),
    noise: float = 0.02,
    n_points_net: int = 256,
    n_points_ph: int | None = None,
    epochs: int = 80,
    test_frac: float = 0.3,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, dict]:
    """Build a composite dataset and evaluate all predictors on one split.

    Returns ``{method_name: betti_metrics(...)}`` for ``pointnet``,
    ``gudhi_rips``, ``gudhi_cech``, ``ripser_learned`` and ``ripser_direct``.

    ``n_points_ph=None`` runs persistent homology on the **full** cloud (no
    subsampling). Because Rips-H2 is O(N^3), the default ``density``/``size_range``
    are kept low so even the largest cloud stays tractable.
    """
    # Imported here so the heavy deps load only when the benchmark runs.
    from .ph import (
        PHRegressor,
        cech_betti_pipeline,
        compute_ph,
        direct_betti,
        gudhi_betti_pipeline,
        tune_taus,
    )
    from .pointnet import predict_pointnet, train_pointnet

    rng = np.random.default_rng(seed)

    if verbose:
        print(f"Building {n_samples} composite scenes (k={k}) ...")
    ds = make_composite_dataset(
        n_samples,
        k=k,
        density=density,
        size_range=size_range,
        noise=noise,
        embed_dim=3,
        rng=rng,
    )
    if verbose:
        sizes = np.array([len(c) for c in ds.clouds])
        print(
            f"  cloud points: median={int(np.median(sizes))}, max={int(sizes.max())}"
        )
    y = np.asarray(ds.betti, dtype=int)
    train_idx, test_idx = train_test_split_idx(len(ds), frac=test_frac, rng=rng)
    y_tr, y_te = y[train_idx], y[test_idx]
    clouds_tr = [ds.clouds[i] for i in train_idx]
    clouds_te = [ds.clouds[i] for i in test_idx]
    results: dict[str, dict] = {}

    # --- PointNet -----------------------------------------------------------
    if verbose:
        print("Training PointNet ...")
    x, _ = pointnet_arrays(ds, n_points=n_points_net, rng=rng)
    model = train_pointnet(
        x[train_idx], y_tr.astype(np.float32), epochs=epochs, rng=seed, verbose=verbose
    )
    pred_net = predict_pointnet(model, x[test_idx])
    results["pointnet"] = betti_metrics(y_te, pred_net)

    # --- GUDHI sklearn point-cloud pipelines --------------------------------
    if verbose:
        where = "full cloud" if n_points_ph is None else f"{n_points_ph} pts"
        print(f"Fitting GUDHI Rips/Cech pipelines ({where}, H0-H2) ...")
    gudhi = gudhi_betti_pipeline(n_points=n_points_ph, random_state=seed)
    pred_gudhi = gudhi.fit(clouds_tr, y_tr).predict(clouds_te)
    results["gudhi_rips"] = betti_metrics(y_te, pred_gudhi)

    cech = cech_betti_pipeline(n_points=n_points_ph, random_state=seed)
    pred_cech = cech.fit(clouds_tr, y_tr).predict(clouds_te)
    results["gudhi_cech"] = betti_metrics(y_te, pred_cech)

    # --- Ripser persistent homology (shared diagram computation) -------------
    if verbose:
        where = "full cloud" if n_points_ph is None else f"{n_points_ph} pts"
        print(f"Computing persistence diagrams ({where}, H0-H2) ...")
    feats, dgms = compute_ph(ds.clouds, n_points=n_points_ph, rng=rng)
    feats_tr, feats_te = feats[train_idx], feats[test_idx]
    dgms_tr = [dgms[i] for i in train_idx]
    dgms_te = [dgms[i] for i in test_idx]

    if verbose:
        print("Fitting Ripser feature model ...")
    ph = PHRegressor(random_state=seed).fit(feats_tr, y_tr)
    results["ripser_learned"] = betti_metrics(y_te, ph.predict(feats_te))

    taus = tune_taus(dgms_tr, y_tr)
    pred_direct = np.vstack([direct_betti(d, taus) for d in dgms_te])
    results["ripser_direct"] = betti_metrics(y_te, pred_direct)

    if verbose:
        print(f"\nBetti-number prediction on {len(test_idx)} held-out scenes (k={k}):\n")
        print(format_comparison(results))
        print(f"\n(PH direct tuned taus = {np.round(taus, 3).tolist()})")
    return results


def _plot(results: dict[str, dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(results)
    metrics = [("acc b0", lambda m: m["acc"]["b0"]),
               ("acc b1", lambda m: m["acc"]["b1"]),
               ("acc b2", lambda m: m["acc"]["b2"]),
               ("exact", lambda m: m["exact"])]
    x = np.arange(len(metrics))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, name in enumerate(methods):
        vals = [f(results[name]) for _, f in metrics]
        ax.bar(x + j * width, vals, width, label=name)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([n for n, _ in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("Betti-number prediction: PointNet vs persistent homology")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved chart to {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3, help="shapes per composite scene")
    parser.add_argument("--n", type=int, default=600, dest="n_samples")
    parser.add_argument("--density", type=float, default=6.0)
    parser.add_argument("--smin", type=float, default=0.8, help="min shape size")
    parser.add_argument("--smax", type=float, default=1.3, help="max shape size")
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--points-net", type=int, default=256, dest="n_points_net")
    parser.add_argument(
        "--points-ph",
        type=int,
        default=0,
        dest="n_points_ph",
        help="subsample size for persistent homology; 0 = use the full cloud",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", type=str, default=None, metavar="PATH")
    args = parser.parse_args(argv)

    results = run_benchmark(
        k=args.k,
        n_samples=args.n_samples,
        density=args.density,
        size_range=(args.smin, args.smax),
        noise=args.noise,
        n_points_net=args.n_points_net,
        n_points_ph=args.n_points_ph or None,  # 0 -> None (no subsampling)
        epochs=args.epochs,
        seed=args.seed,
    )
    if args.plot:
        _plot(results, args.plot)


if __name__ == "__main__":
    main()
