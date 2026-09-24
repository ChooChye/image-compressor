#!/usr/bin/env python3
"""Compare encoder configurations: BD-rate of each run's fixed-level curve vs an anchor run.

Usage: python3 compare_runs.py ANCHOR_RUN OTHER_RUN [OTHER_RUN ...] [--format avif]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_bench import GATE, HELD_OUT, bd_rate, fixed_curve, load


def format_bd(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--format", default="avif")
    args = parser.parse_args()

    loaded = [(run, *load(run)) for run in args.runs]
    common = set.intersection(*({i for (i, f) in sweeps if f == args.format} for _, _, sweeps in loaded))
    images = sorted(common)
    print(f"{len(images)} common images, format {args.format}, anchor {args.runs[0].name}\n")

    metrics = [*HELD_OUT, GATE]
    print("| run | encoder | " + " | ".join(f"{m} mean" for m in metrics) + " | encode time |")
    print("|---|---|" + "---|" * (len(metrics) + 1))
    anchor_curve, anchor_ms = None, None
    for run, _, sweeps in loaded:
        curve = fixed_curve(images, sweeps, args.format)
        ms = np.mean([r["encode_ms"] for i in images for r in sweeps[(i, args.format)].rows.values()])
        config = json.loads((run / "run.json").read_text())
        encoder = next(c[2] for c in config["codecs"] if c[0] == args.format)
        if anchor_curve is None:
            anchor_curve, anchor_ms = curve, ms
            cells = ["anchor"] * len(metrics)
        else:
            cells = [format_bd(bd_rate(anchor_curve, curve, m)) for m in metrics]
        print(f"| {run.name} | {encoder.split(' ', 1)[1]} | " + " | ".join(cells) + f" | {ms / anchor_ms:.2f}x |")


if __name__ == "__main__":
    main()
