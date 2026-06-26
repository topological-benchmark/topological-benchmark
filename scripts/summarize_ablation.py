"""Summarize ablation CSV results as Markdown tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


GROUPS = [
    "sweep",
    "method",
    "k",
    "points_net",
    "points_ph",
    "point_noise",
    "field_noise",
    "field_length_scale",
    "image_resolution",
    "image_backend",
    "representation",
]


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(_cell(h) for h in headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)
    return "\n".join(lines)


def _summary(rows: list[dict[str, str]], group: str) -> str:
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            method = "" if group == "method" else row.get("method", "")
            buckets[(row.get(group, ""), method)].append(row)
    out = []
    for (key, method), vals in sorted(buckets.items(), key=lambda item: str(item[0])):
        exact = _mean([float(v["exact"]) for v in vals])
        mae = _mean([float(v["mae"]) for v in vals])
        wall = _mean([float(v["wall_time_s"]) for v in vals])
        if group == "method":
            out.append([str(key), str(len(vals)), f"{exact:.3f}", f"{mae:.3f}", f"{wall:.1f}"])
        else:
            out.append([str(key), method, str(len(vals)), f"{exact:.3f}", f"{mae:.3f}", f"{wall:.1f}"])
    if group == "method":
        return _table([group, "rows", "mean exact", "mean mae", "mean wall s"], out)
    return _table([group, "method", "rows", "mean exact", "mean mae", "mean wall s"], out)


def summarize(csv_path: Path) -> str:
    rows = _rows(csv_path)
    ok = [r for r in rows if r["status"] == "ok"]
    errors = [r for r in rows if r["status"] != "ok"]
    configs = {r["run_id"] for r in rows}
    lines = [
        f"# Ablation Summary: `{csv_path}`",
        "",
        f"- rows: {len(rows)}",
        f"- ok rows: {len(ok)}",
        f"- error rows: {len(errors)}",
        f"- configs seen: {len(configs)}",
        "",
    ]
    for group in GROUPS:
        lines += [f"## By {group}", "", _summary(rows, group), ""]
    if errors:
        lines += ["## Errors", ""]
        lines.append(
            _table(
                ["run_id", "method", "error"],
                [[e["run_id"], e["method"], e["error"]] for e in errors[:50]],
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    text = summarize(args.csv)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
