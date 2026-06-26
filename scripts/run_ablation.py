"""Run laptop-bounded ablations for topology benchmark baselines.

Examples:
    uv run python scripts/run_ablation.py --stage smoke
    uv run python scripts/run_ablation.py --stage smart --out outputs/smart.csv
    uv run python scripts/run_ablation.py --stage k --jobs 4 --out outputs/k.csv
    uv run python scripts/run_ablation.py --stage main --out outputs/main.csv
    uv run python scripts/run_ablation.py --stage representations --limit 3
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, get_args

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tda_shapes.ml.benchmark import run_benchmark
from tda_shapes.ml.ph import GudhiRepresentation


FIELDNAMES = [
    "run_id",
    "stage",
    "sweep",
    "seed",
    "k",
    "n_samples",
    "points_net",
    "points_ph",
    "point_noise",
    "field_noise",
    "field_length_scale",
    "image_resolution",
    "image_backend",
    "image_min_pixels_per_bandwidth",
    "representation",
    "run_rips",
    "run_cubical",
    "epochs",
    "val_frac",
    "method",
    "acc_b0",
    "acc_b1",
    "acc_b2",
    "exact",
    "mae",
    "wall_time_s",
    "status",
    "error",
]

REPRESENTATIONS = list(get_args(GudhiRepresentation))


def _noise_grid(
    point_noise: list[float],
    field_noise: list[float],
    field_length_scale: list[float],
) -> list[dict[str, float]]:
    configs = []
    for pn, fn in itertools.product(point_noise, field_noise):
        lengths = [0.25] if fn == 0.0 else field_length_scale
        for fls in lengths:
            configs.append(
                {
                    "point_noise": pn,
                    "field_noise": fn,
                    "field_length_scale": fls,
                }
            )
    return configs


def _product(stage: str, **grid: list[Any]):
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(zip(keys, values, strict=True))
        cfg["stage"] = stage
        cfg.setdefault("sweep", stage)
        yield cfg


def _baseline(**overrides) -> dict[str, Any]:
    cfg = {
        "stage": "smart",
        "sweep": "baseline",
        "k": 3,
        "n_samples": 120,
        "points_net": 256,
        "points_ph": 128,
        "point_noise": 0.02,
        "field_noise": 0.01,
        "field_length_scale": 0.25,
        "image_resolution": 24,
        "representation": "silhouette",
        "seed": 0,
    }
    cfg.update(overrides)
    return cfg


def _smart_configs():
    seen = set()

    def emit(cfg: dict[str, Any]):
        key = _run_id(cfg)
        if key not in seen:
            seen.add(key)
            yield cfg

    for seed in [0, 1]:
        yield from emit(_baseline(sweep="seed", seed=seed))

    for noise in _noise_grid([0.0, 0.02, 0.05], [0.0, 0.01], [0.25]):
        yield from emit(_baseline(sweep="noise", **noise))

    for k in [1, 3]:
        yield from emit(_baseline(sweep="k", k=k))

    for points_net in [128, 256]:
        yield from emit(_baseline(sweep="points_net", points_net=points_net))

    for points_ph in [128, 256]:
        yield from emit(_baseline(sweep="points_ph", points_ph=points_ph))

    for image_resolution in [24, 32]:
        yield from emit(_baseline(sweep="image_resolution", image_resolution=image_resolution))

    for representation in ["silhouette", "betti"]:
        yield from emit(_baseline(sweep="representation", representation=representation))


def _stage_configs(stage: str):
    if stage == "smoke":
        base = _product(
            "smoke",
            k=[1, 3],
            n_samples=[60],
            points_net=[256],
            points_ph=[128],
            image_resolution=[24],
            representation=["silhouette"],
            seed=[0],
        )
        noise = _noise_grid([0.0, 0.02], [0.0, 0.01], [0.25])
    elif stage == "main":
        base = _product(
            "main",
            k=[1, 3, 10],
            n_samples=[300],
            points_net=[256, 1000],
            points_ph=[128, 512, 1000],
            image_resolution=[32, 50],
            representation=["silhouette"],
            seed=[0, 1, 2],
        )
        noise = _noise_grid([0.0, 0.02, 0.05], [0.0, 0.01, 0.03], [0.15, 0.25, 0.5])
    elif stage == "representations":
        base = _product(
            "representations",
            k=[1, 3, 10],
            n_samples=[300],
            points_net=[1000],
            points_ph=[512],
            image_resolution=[32],
            representation=REPRESENTATIONS,
            seed=[0, 1, 2],
        )
        noise = _noise_grid([0.02], [0.01], [0.25])
    elif stage == "k":
        base = _product(
            "k",
            sweep=["k"],
            k=[1, 2, 3, 5, 10],
            n_samples=[120],
            points_net=[256],
            points_ph=[128],
            image_resolution=[24],
            representation=["silhouette"],
            seed=[0],
        )
        noise = _noise_grid([0.02], [0.01], [0.25])
    elif stage == "smart":
        yield from _smart_configs()
        return
    else:
        raise ValueError(f"unknown stage: {stage}")

    for cfg in base:
        for ncfg in noise:
            yield {**cfg, **ncfg}


def _configs(stage: str):
    stages = ["smoke", "smart", "k", "main", "representations"] if stage == "all" else [stage]
    for name in stages:
        yield from _stage_configs(name)


def _effective_cfg(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(cfg)
    out["n_samples"] = args.n_samples or cfg["n_samples"]
    out["image_resolution"] = args.image_resolution or cfg["image_resolution"]
    out["image_backend"] = args.image_backend
    out["image_min_pixels_per_bandwidth"] = args.image_min_pixels_per_bandwidth
    out["run_rips"] = args.run_rips
    out["run_cubical"] = not args.no_cubical
    out["epochs"] = args.epochs
    out["val_frac"] = args.val_frac
    return out


def _run_id(cfg: dict[str, Any]) -> str:
    parts = [f"{key}={cfg[key]}" for key in sorted(cfg)]
    return "|".join(parts)


def _expected_methods(cfg: dict[str, Any]) -> set[str]:
    methods = {"pointnet", "pointnet++", "gudhi_cech"}
    if cfg["run_cubical"]:
        methods.add("gudhi_cubical")
    if cfg["run_rips"]:
        methods.update({"gudhi_rips", "ripser_learned", "ripser_direct"})
    return methods


def _check_header(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(newline="") as f:
        header = next(csv.reader(f), [])
    if header != FIELDNAMES:
        raise ValueError(f"CSV header mismatch in {path}; use a new --out path")


def _completed(path: Path, configs: list[dict[str, Any]]) -> set[str]:
    if not path.exists():
        return set()
    expected = {_run_id(cfg): _expected_methods(cfg) for cfg in configs}
    seen: dict[str, set[str]] = {run_id: set() for run_id in expected}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            run_id = row.get("run_id", "")
            if row.get("status") == "ok" and run_id in seen:
                seen[run_id].add(row.get("method", ""))
    return {run_id for run_id, methods in seen.items() if methods >= expected[run_id]}


def _row(run_id: str, cfg: dict[str, Any], method: str, wall: float, status: str, error: str = ""):
    row = {name: "" for name in FIELDNAMES}
    row.update(cfg)
    row.update(
        {
            "run_id": run_id,
            "method": method,
            "wall_time_s": f"{wall:.3f}",
            "status": status,
            "error": error,
        }
    )
    return row


def _metric_rows(run_id: str, cfg: dict[str, Any], results: dict[str, dict], wall: float):
    rows = []
    for method, metrics in results.items():
        row = _row(run_id, cfg, method, wall, "ok")
        row.update(
            {
                "acc_b0": metrics["acc"]["b0"],
                "acc_b1": metrics["acc"]["b1"],
                "acc_b2": metrics["acc"]["b2"],
                "exact": metrics["exact"],
                "mae": metrics["mae"]["overall"],
            }
        )
        rows.append(row)
    return rows


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    _check_header(path)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _run_one(i: int, total: int, cfg: dict[str, Any], verbose: bool) -> tuple[int, str, list[dict[str, Any]], str]:
    run_id = _run_id(cfg)
    start = time.perf_counter()
    try:
        results = run_benchmark(
            k=cfg["k"],
            n_samples=cfg["n_samples"],
            point_noise=cfg["point_noise"],
            field_noise=cfg["field_noise"],
            field_length_scale=cfg["field_length_scale"],
            n_points_net=cfg["points_net"],
            n_points_ph=cfg["points_ph"],
            representation=cfg["representation"],
            run_rips=cfg["run_rips"],
            run_cubical=cfg["run_cubical"],
            image_resolution=cfg["image_resolution"],
            image_backend=cfg["image_backend"],
            image_min_pixels_per_bandwidth=cfg["image_min_pixels_per_bandwidth"],
            epochs=cfg["epochs"],
            val_frac=cfg["val_frac"],
            seed=cfg["seed"],
            verbose=verbose,
        )
        wall = time.perf_counter() - start
        return i, "ok", _metric_rows(run_id, cfg, results, wall), f"[{i}/{total}] ok {wall:.1f}s"
    except Exception as exc:  # noqa: BLE001 - long ablations should continue.
        wall = time.perf_counter() - start
        rows = [_row(run_id, cfg, "ERROR", wall, "error", repr(exc))]
        return i, "error", rows, f"[{i}/{total}] error {type(exc).__name__}: {exc}"


def run(args: argparse.Namespace) -> None:
    out = args.out or Path(f"outputs/ablation_{datetime.now():%Y%m%d_%H%M%S}.csv")
    configs = []
    seen = set()
    for cfg in (_effective_cfg(c, args) for c in _configs(args.stage)):
        run_id = _run_id(cfg)
        if run_id not in seen:
            seen.add(run_id)
            configs.append(cfg)
    if args.limit is not None:
        configs = configs[: args.limit]
    _check_header(out)
    done = _completed(out, configs)
    total = len(configs)
    print(f"writing {out}")
    print(f"configs {total}; already complete {sum(_run_id(c) in done for c in configs)}")
    if args.stop_on_error and args.jobs != 1:
        raise ValueError("--stop-on-error requires --jobs 1")

    pending = []
    for i, cfg in enumerate(configs, start=1):
        run_id = _run_id(cfg)
        if run_id in done and not args.rerun:
            print(f"[{i}/{total}] skip {run_id}")
            continue
        pending.append((i, cfg))

    def handle(result) -> None:
        _, status, rows, message = result
        _append_rows(out, rows)
        print(message)
        if status == "error" and args.stop_on_error:
            raise RuntimeError(rows[0]["error"])

    if args.jobs == 1:
        for i, cfg in pending:
            handle(_run_one(i, total, cfg, args.verbose))
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(_run_one)(i, total, cfg, args.verbose) for i, cfg in pending
        )
        for result in results:
            handle(result)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "smart", "k", "main", "representations", "all"),
        default="smoke",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--image-resolution", type=int, default=None)
    parser.add_argument("--image-min-pixels-per-bandwidth", type=float, default=1.0)
    parser.add_argument("--image-backend", choices=("numpy", "gpu", "jax", "mps", "cuda"), default="gpu")
    parser.add_argument("--rips", action="store_true", dest="run_rips")
    parser.add_argument("--no-rips", action="store_false", dest="run_rips")
    parser.add_argument("--no-cubical", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
