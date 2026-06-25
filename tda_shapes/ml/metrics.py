"""Metrics and reporting for Betti-number prediction."""

from __future__ import annotations

import numpy as np

_BETTI_NAMES = ("b0", "b1", "b2")


def betti_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-dimension MAE & exact-match accuracy plus exact-triple accuracy.

    Returns a dict with ``mae`` (per dim + ``overall``), ``acc`` (per dim) and
    ``exact`` (fraction of clouds with all three Betti numbers correct).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    abs_err = np.abs(y_pred - y_true)
    correct = y_pred == y_true
    return {
        "mae": {
            **{_BETTI_NAMES[i]: float(abs_err[:, i].mean()) for i in range(3)},
            "overall": float(abs_err.mean()),
        },
        "acc": {_BETTI_NAMES[i]: float(correct[:, i].mean()) for i in range(3)},
        "exact": float(correct.all(axis=1).mean()),
    }


def format_comparison(results: dict[str, dict]) -> str:
    """Render ``{method: betti_metrics(...)}`` as an aligned text table."""
    header = (
        f"{'method':<16}{'acc b0':>8}{'acc b1':>8}{'acc b2':>8}"
        f"{'exact':>8}{'MAE':>8}"
    )
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        lines.append(
            f"{name:<16}"
            f"{m['acc']['b0']:>8.2f}{m['acc']['b1']:>8.2f}{m['acc']['b2']:>8.2f}"
            f"{m['exact']:>8.2f}{m['mae']['overall']:>8.3f}"
        )
    return "\n".join(lines)
