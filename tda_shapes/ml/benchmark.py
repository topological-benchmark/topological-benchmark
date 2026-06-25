"""Compare PointNet vs. TDA baselines on predicting Betti numbers.

Run directly::

    uv run python -m tda_shapes.ml.benchmark --k 3 --n 300 --epochs 80
    uv run python -m tda_shapes.ml.benchmark --k 3 --n 200 --plot bench.png
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import numpy as np

from ..composite import make_composite_dataset
from ..images import ImageBackend, rasterize_kde
from .data import pointnet_arrays, train_test_split_idx
from .metrics import betti_metrics, format_comparison

if TYPE_CHECKING:
    from .ph import GudhiRepresentation

def run_benchmark(
    *,
    k: int = 3,
    n_samples: int = 600,
    density: float = 6.0,
    size_range: tuple[float, float] = (0.8, 1.3),
    background_density: float = 0.0,
    background_margin: float = 0.0,
    point_noise: float = 0.02,
    field_noise: float = 0.0,
    field_length_scale: float = 0.25,
    n_points_net: int | None = 256,
    n_points_ph: int | None = 256,
    representation: GudhiRepresentation = "silhouette",
    run_pointnet2: bool = True,
    run_rips: bool = False,
    run_cubical: bool = False,
    image_resolution: int = 32,
    image_bandwidth: float = 0.15,
    image_backend: ImageBackend = "gpu",
    image_min_pixels_per_bandwidth: float = 1.0,
    epochs: int = 80,
    test_frac: float = 0.3,
    val_frac: float = 0.2,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, dict]:
    """Build a composite dataset and evaluate all predictors on one split.

    Returns ``{method_name: betti_metrics(...)}`` always for ``pointnet`` and
    ``gudhi_cech``; ``pointnet++`` (PointNet++ set abstraction) is included when
    ``run_pointnet2=True`` (on by default), and the Vietoris-Rips predictors
    ``gudhi_rips``, ``ripser_learned`` and ``ripser_direct`` only when
    ``run_rips=True`` (off by default).

    ``n_points_net`` resamples each cloud to a fixed size for PointNet (default
    256); pass ``None`` to use the **full** cloud (duplicate-padded to the
    dataset's largest cloud, which the global max-pool ignores).

    ``n_points_ph`` subsamples each cloud before persistent homology (default
    256); pass ``None`` to run on the **full** cloud. Because Rips-H2 is O(N^3),
    subsampling keeps even the largest cloud tractable.

    ``representation`` selects the GUDHI diagram vectorization fed to the random
    forest (default ``"silhouette"``); see
    :data:`tda_shapes.ml.ph.GudhiRepresentation` for the full list. It applies
    to Cech, Cubical, and (opt-in) Rips pipelines.

    ``val_frac`` carves a validation split out of the training portion (default
    0.2; set 0 to disable). All learned models fit on the reduced train set, and
    PointNet reports train/val accuracy each logged epoch so overfitting shows up
    during training. Final metrics are still computed on the untouched test set.
    """
    # Imported here so the heavy deps load only when the benchmark runs.
    from .ph import cech_betti_pipeline
    from .pointnet import predict_pointnet, train_pointnet

    rng = np.random.default_rng(seed)

    if verbose:
        print(f"Building {n_samples} composite scenes (k={k}) ...")
    ds = make_composite_dataset(
        n_samples,
        k=k,
        density=density,
        size_range=size_range,
        point_noise=point_noise,
        field_noise=field_noise,
        field_length_scale=field_length_scale,
        background_density=background_density,
        background_margin=background_margin,
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
    # Carve a validation split out of the training portion so the test set stays
    # a clean final holdout; every learned model fits on the reduced train set.
    if val_frac > 0:
        rel_fit, rel_val = train_test_split_idx(len(train_idx), frac=val_frac, rng=rng)
        val_idx = train_idx[rel_val]
        train_idx = train_idx[rel_fit]
    else:
        val_idx = np.empty(0, dtype=int)
    y_tr, y_te = y[train_idx], y[test_idx]
    clouds_tr = [ds.clouds[i] for i in train_idx]
    clouds_te = [ds.clouds[i] for i in test_idx]
    results: dict[str, dict] = {}

    # --- PointNet -----------------------------------------------------------
    if verbose:
        where = "full cloud" if n_points_net is None else f"{n_points_net} pts"
        val_note = f", val={len(val_idx)}" if len(val_idx) else ""
        print(f"Training PointNet ({where}, train={len(train_idx)}{val_note}) ...")
    x, _ = pointnet_arrays(ds, n_points=n_points_net, rng=rng)
    val = (x[val_idx], y[val_idx]) if len(val_idx) else None
    model = train_pointnet(
        x[train_idx], y_tr, epochs=epochs, rng=seed, val=val, verbose=verbose
    )
    pred_net = predict_pointnet(model, x[test_idx])
    results["pointnet"] = betti_metrics(y_te, pred_net)

    # --- PointNet++ (hierarchical set abstraction) --------------------------
    if run_pointnet2:
        if verbose:
            print(f"Training PointNet++ ({where}, train={len(train_idx)}{val_note}) ...")
        model2 = train_pointnet(
            x[train_idx], y_tr, epochs=epochs, rng=seed, val=val,
            arch="pointnet2", verbose=verbose,
        )
        pred_net2 = predict_pointnet(model2, x[test_idx])
        results["pointnet++"] = betti_metrics(y_te, pred_net2)

    # --- GUDHI Cech sklearn point-cloud pipeline ----------------------------
    if verbose:
        where = "full cloud" if n_points_ph is None else f"{n_points_ph} pts"
        print(
            f"Fitting GUDHI Cech pipeline ({where}, H0-H2, {representation}) ..."
        )
    cech = cech_betti_pipeline(
        n_points=n_points_ph, representation=representation, random_state=seed
    )
    pred_cech = cech.fit(clouds_tr, y_tr).predict(clouds_te)
    results["gudhi_cech"] = betti_metrics(y_te, pred_cech)

    # --- GUDHI Cubical sklearn image pipeline -------------------------------
    if run_cubical:
        from .ph import cubical_betti_pipeline

        if verbose:
            print(
                "Rasterizing KDE images "
                f"({image_resolution}^3, backend={image_backend}) ..."
            )
        images = _rasterize_clouds(
            ds.clouds,
            resolution=image_resolution,
            bandwidth=image_bandwidth,
            backend=image_backend,
            min_pixels_per_bandwidth=image_min_pixels_per_bandwidth,
        )
        if verbose:
            print(f"Fitting GUDHI Cubical pipeline ({image_resolution}^3, H0-H2) ...")
        cubical = cubical_betti_pipeline(
            representation=representation, random_state=seed
        )
        pred_cubical = cubical.fit(images[train_idx], y_tr).predict(images[test_idx])
        results["gudhi_cubical"] = betti_metrics(y_te, pred_cubical)

    # --- Vietoris-Rips predictors (opt-in via run_rips) ---------------------
    taus = None
    if run_rips:
        from .ph import (
            PHClassifier,
            compute_ph,
            direct_betti,
            gudhi_betti_pipeline,
            tune_taus,
        )

        if verbose:
            where = "full cloud" if n_points_ph is None else f"{n_points_ph} pts"
            print(
                f"Fitting GUDHI Rips pipeline ({where}, H0-H2, {representation}) ..."
            )
        gudhi = gudhi_betti_pipeline(
            n_points=n_points_ph, representation=representation, random_state=seed
        )
        pred_gudhi = gudhi.fit(clouds_tr, y_tr).predict(clouds_te)
        results["gudhi_rips"] = betti_metrics(y_te, pred_gudhi)

        if verbose:
            print(f"Computing Ripser persistence diagrams ({where}, H0-H2) ...")
        feats, dgms = compute_ph(ds.clouds, n_points=n_points_ph, rng=rng)
        feats_tr, feats_te = feats[train_idx], feats[test_idx]
        dgms_tr = [dgms[i] for i in train_idx]
        dgms_te = [dgms[i] for i in test_idx]

        if verbose:
            print("Fitting Ripser feature model ...")
        ph = PHClassifier(random_state=seed).fit(feats_tr, y_tr)
        results["ripser_learned"] = betti_metrics(y_te, ph.predict(feats_te))

        taus = tune_taus(dgms_tr, y_tr)
        pred_direct = np.vstack([direct_betti(d, taus) for d in dgms_te])
        results["ripser_direct"] = betti_metrics(y_te, pred_direct)

    if verbose:
        print(f"\nBetti-number prediction on {len(test_idx)} held-out scenes (k={k}):\n")
        print(format_comparison(results))
        if taus is not None:
            print(f"\n(PH direct tuned taus = {np.round(taus, 3).tolist()})")
    return results


def _rasterize_clouds(
    clouds: list[np.ndarray],
    *,
    resolution: int,
    bandwidth: float,
    backend: ImageBackend,
    min_pixels_per_bandwidth: float,
) -> np.ndarray:
    images = []
    for cloud in clouds:
        lo = cloud.min(axis=0)
        hi = cloud.max(axis=0)
        scale = max(0.5 * float(np.max(hi - lo)), 1e-12)
        images.append(
            rasterize_kde(
                cloud,
                resolution=resolution,
                bandwidth=bandwidth * scale,
                min_pixels_per_bandwidth=min_pixels_per_bandwidth,
                backend=backend,
            )
        )
    return np.stack(images)


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
    from typing import get_args

    from .ph import GudhiRepresentation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3, help="shapes per composite scene")
    parser.add_argument("--n", type=int, default=600, dest="n_samples")
    parser.add_argument("--density", type=float, default=6.0)
    parser.add_argument("--smin", type=float, default=0.8, help="min shape size")
    parser.add_argument("--smax", type=float, default=1.3, help="max shape size")
    parser.add_argument(
        "--background-density",
        type=float,
        default=0.0,
        dest="background_density",
        help="volumetric density of uniform background clutter; 0 = none",
    )
    parser.add_argument(
        "--background-margin",
        type=float,
        default=0.0,
        dest="background_margin",
        help="fractional padding of the background box beyond the shapes",
    )
    parser.add_argument("--point-noise", type=float, default=0.02)
    parser.add_argument("--field-noise", type=float, default=0.0)
    parser.add_argument("--field-length-scale", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--points-net",
        type=int,
        default=0,
        dest="n_points_net",
        help="fixed cloud size fed to PointNet; 0 = use the full cloud",
    )
    parser.add_argument(
        "--points-ph",
        type=int,
        default=0,
        dest="n_points_ph",
        help="subsample size for persistent homology; 0 = use the full cloud",
    )
    parser.add_argument(
        "--rips",
        action="store_true",
        dest="run_rips",
        help="also run the Vietoris-Rips predictors (gudhi_rips, ripser_*); "
        "off by default",
    )
    parser.add_argument(
        "--no-pointnet2",
        action="store_false",
        dest="run_pointnet2",
        help="skip the PointNet++ predictor (on by default)",
    )
    parser.add_argument(
        "--representation",
        "--tda-representation",
        choices=get_args(GudhiRepresentation),
        default="silhouette",
        dest="representation",
        help="GUDHI diagram vectorization for Cech/Rips/Cubical pipelines",
    )
    parser.add_argument("--cubical", action="store_true", dest="run_cubical")
    parser.add_argument("--image-resolution", type=int, default=32)
    parser.add_argument("--image-bandwidth", type=float, default=0.15)
    parser.add_argument(
        "--image-backend",
        choices=("numpy", "gpu", "jax", "mps", "cuda"),
        default="gpu",
    )
    parser.add_argument(
        "--image-min-pixels-per-bandwidth",
        type=float,
        default=1.0,
        dest="image_min_pixels_per_bandwidth",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        dest="val_frac",
        help="fraction of the training set held out for validation; "
        "0 = no validation monitoring",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", type=str, default=None, metavar="PATH")
    args = parser.parse_args(argv)

    results = run_benchmark(
        k=args.k,
        n_samples=args.n_samples,
        density=args.density,
        size_range=(args.smin, args.smax),
        background_density=args.background_density,
        background_margin=args.background_margin,
        run_rips=args.run_rips,
        run_cubical=args.run_cubical,
        point_noise=args.point_noise,
        field_noise=args.field_noise,
        field_length_scale=args.field_length_scale,
        n_points_net=args.n_points_net or None,  # 0 -> None (use full cloud)
        n_points_ph=args.n_points_ph or None,  # 0 -> None (no subsampling)
        representation=args.representation,
        run_pointnet2=args.run_pointnet2,
        image_resolution=args.image_resolution,
        image_bandwidth=args.image_bandwidth,
        image_backend=args.image_backend,
        image_min_pixels_per_bandwidth=args.image_min_pixels_per_bandwidth,
        epochs=args.epochs,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    if args.plot:
        _plot(results, args.plot)


if __name__ == "__main__":
    main()
