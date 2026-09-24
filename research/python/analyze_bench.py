#!/usr/bin/env python3
"""Analyze a smartimg-bench sweep.

Question: does adaptive per-image search beat fixed codec presets at equal perceptual
quality, and at what compute cost?

Strategies are evaluated on the same measured encodes (every level of every format):

- fixed(L)        one quality level for every image (what most pipelines do)
- engine(t)       exact replay of smartimg-core's search: binary search for the lowest level
                  passing the gate, ±sweep_radius local sweep, evaluation budget
- oracle(t)       smallest encode passing the gate among all levels (search upper bound)

The engine gates on SSIM. Results are judged on held-out metrics it never saw
(SSIMULACRA2, Butteraugli 3-norm), so they are not circular. Judging on SSIM is reported
separately to show how much a circular benchmark would flatter the optimizer.

Curves are compared with BD-rate: the average bitrate difference at equal quality over the
overlapping quality range (piecewise-linear interpolation of log-bitrate). Negative means
fewer bytes than the anchor.

Usage: python3 analyze_bench.py benchmarks/results/run1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GATE = "ssim-y-gauss11"
HELD_OUT = {"ssimulacra2": +1, "butteraugli-pnorm3": -1}  # +1 higher is better
BUDGET = 16
SWEEP_RADIUS = 2
FORMATS = ("avif", "webp")


@dataclass
class Sweep:
    """All measured levels of one format for one image."""
    levels: list[int]
    rows: dict[int, dict]

    @property
    def contiguous(self) -> bool:
        """True when every integer level between the first and last is present, which the
        engine replay needs (it probes arbitrary midpoints)."""
        return bool(self.levels) and len(self.levels) == self.levels[-1] - self.levels[0] + 1


def load(run: Path):
    images: dict[str, dict] = {}
    sweeps: dict[tuple[str, str], Sweep] = {}
    for line in (run / "results.jsonl").open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # partially written last line of an interrupted run
        images[row["image"]] = {k: row[k] for k in ("corpus", "split", "width", "height", "has_alpha")}
        sweep = sweeps.setdefault((row["image"], row["format"]), Sweep([], {}))
        sweep.rows[row["level"]] = row
    for sweep in sweeps.values():
        sweep.levels = sorted(sweep.rows)
    return images, sweeps


def usable_images(images, sweeps, formats=FORMATS) -> tuple[list[str], list[str]]:
    """Split images into (usable, excluded). An image is usable when every format has a
    contiguous sweep over that format's full level range (the widest range seen in the run).
    Anything else (runner --step != 1, a partially flushed last image, a missing format) is
    excluded as a whole so every strategy is evaluated on the same image set."""
    expected: dict[str, tuple[int, int]] = {}
    for (_, fmt), sweep in sweeps.items():
        lo, hi = expected.get(fmt, (sweep.levels[0], sweep.levels[-1]))
        expected[fmt] = (min(lo, sweep.levels[0]), max(hi, sweep.levels[-1]))

    def complete(image: str, fmt: str) -> bool:
        sweep = sweeps.get((image, fmt))
        return (sweep is not None and sweep.contiguous
                and (sweep.levels[0], sweep.levels[-1]) == expected[fmt])

    usable, excluded = [], []
    for image in sorted(images):
        (usable if all(complete(image, f) for f in formats) else excluded).append(image)
    return usable, excluded


# --- strategies ---------------------------------------------------------------------------

def passes(row: dict, threshold: float, gate: str = GATE, direction: int = +1) -> bool:
    return row["metrics"][gate] * direction >= threshold * direction


def engine_search(sweep: Sweep, threshold: float, gate: str = GATE, direction: int = +1):
    """Mirror of smartimg-core FormatSearch::run. Returns (row or None, evaluations)."""
    if not sweep.contiguous:
        raise ValueError("engine replay needs every integer level (run smartimg-bench with --step 1)")
    lo, hi = sweep.levels[0], sweep.levels[-1]
    samples: dict[int, dict] = {}

    def sample(level: int) -> dict:
        if level not in samples:
            samples[level] = sweep.rows[level]
        return samples[level]

    boundary = None
    while lo <= hi and len(samples) < BUDGET:
        mid = (lo + hi) // 2
        if passes(sample(mid), threshold, gate, direction):
            boundary, hi = mid, mid - 1
        else:
            lo = mid + 1
    if boundary is None:
        return None, len(samples)
    for level in range(max(boundary - SWEEP_RADIUS, sweep.levels[0]), min(boundary + SWEEP_RADIUS, sweep.levels[-1]) + 1):
        if len(samples) >= BUDGET:
            break
        sample(level)
    accepted = [r for r in samples.values() if passes(r, threshold, gate, direction)]
    return min(accepted, key=lambda r: (r["bytes"], -r["level"])), len(samples)


def oracle_search(sweep: Sweep, threshold: float, gate: str = GATE, direction: int = +1):
    accepted = [r for r in sweep.rows.values() if passes(r, threshold, gate, direction)]
    return (min(accepted, key=lambda r: (r["bytes"], -r["level"])) if accepted else None), len(sweep.rows)


def pick(image: str, sweeps, formats, search, threshold, **gate):
    """Best candidate across formats. Unreachable targets fall back to each format's
    highest level (what a caller would do after NoAcceptableCandidate) and are counted."""
    found, evaluations = [], 0
    for fmt in formats:
        sweep = sweeps.get((image, fmt))
        if sweep is None:
            continue
        row, n = search(sweep, threshold, **gate)
        evaluations += n
        if row is not None:
            found.append(row)
    if found:
        return min(found, key=lambda r: r["bytes"]), evaluations, False
    fallback = [sweeps[(image, f)].rows[sweeps[(image, f)].levels[-1]] for f in formats if (image, f) in sweeps]
    return min(fallback, key=lambda r: r["bytes"]), evaluations, True


# --- aggregation --------------------------------------------------------------------------

@dataclass
class Point:
    parameter: float
    bpp: float
    quality: dict[str, float]  # metric -> mean
    worst: dict[str, float]    # metric -> worst-decile value
    evaluations: float
    misses: int


def aggregate(parameter, chosen: list[tuple[dict, int, bool]]) -> Point:
    rows = [c[0] for c in chosen]
    bits = sum(r["bytes"] for r in rows) * 8
    pixels = sum(r["width"] * r["height"] for r in rows)
    quality, worst = {}, {}
    for metric in (GATE, *HELD_OUT):
        values = np.array([r["metrics"][metric] for r in rows])
        direction = HELD_OUT.get(metric, +1)
        quality[metric] = float(values.mean())
        worst[metric] = float(np.percentile(values, 10 if direction > 0 else 90))
    return Point(parameter, bits / pixels, quality, worst,
                 float(np.mean([c[1] for c in chosen])), sum(c[2] for c in chosen))


def fixed_curve(images, sweeps, fmt):
    if not images:
        return []
    levels = sorted(set.intersection(*(set(sweeps[(i, fmt)].levels) for i in images)))
    return [aggregate(level, [(sweeps[(i, fmt)].rows[level], 1, False) for i in images]) for level in levels]


def search_curve(images, sweeps, formats, search, thresholds, **gate):
    return [aggregate(t, [pick(i, sweeps, formats, search, t, **gate) for i in images]) for t in thresholds]


# --- BD-rate ------------------------------------------------------------------------------

def frontier(points, metric, stat):
    """(quality, log bpp) pairs on the lower-rate envelope, quality ascending. Quality is
    oriented so that larger is better."""
    direction = HELD_OUT.get(metric, +1)
    pairs = sorted((getattr(p, stat)[metric] * direction, math.log(p.bpp)) for p in points)
    envelope, best = [], math.inf
    for q, r in reversed(pairs):  # walk from best quality down, keep strictly cheaper points
        if r < best:
            envelope.append((q, r))
            best = r
    return sorted(envelope)


def bd_rate(anchor, test, metric, stat="quality"):
    a, b = frontier(anchor, metric, stat), frontier(test, metric, stat)
    if len(a) < 2 or len(b) < 2:
        return None
    lo, hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    if hi <= lo:
        return None
    grid = np.linspace(lo, hi, 200)
    ra = np.interp(grid, [q for q, _ in a], [r for _, r in a])
    rb = np.interp(grid, [q for q, _ in b], [r for _, r in b])
    return float(math.exp(np.mean(rb - ra)) - 1) * 100


def rate_at(points, metric, target, stat="quality"):
    """Interpolated bpp at which a curve reaches `target` quality."""
    direction = HELD_OUT.get(metric, +1)
    f = frontier(points, metric, stat)
    qs = [q for q, _ in f]
    t = target * direction
    if not f or t < qs[0] or t > qs[-1]:
        return None
    return math.exp(float(np.interp(t, qs, [r for _, r in f])))


# --- report -------------------------------------------------------------------------------

def svg_chart(curves: dict[str, list[Point]], metric: str, title: str) -> str:
    w, h, pad = 720, 440, 56
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    pts = [(p.bpp, p.quality[metric]) for c in curves.values() for p in c]
    xs = [math.log(x) for x, _ in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(y for _, y in pts), max(y for _, y in pts)
    sx = lambda v: pad + (math.log(v) - x0) / (x1 - x0 or 1) * (w - 2 * pad)
    sy = lambda v: h - pad - (v - y0) / (y1 - y0 or 1) * (h - 2 * pad)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif" font-size="12">',
           f'<rect width="{w}" height="{h}" fill="white"/>',
           f'<text x="{w/2}" y="20" text-anchor="middle" font-size="14">{title}</text>',
           f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#888"/>',
           f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#888"/>',
           f'<text x="{w/2}" y="{h-14}" text-anchor="middle">bits per pixel (log scale)</text>',
           f'<text x="16" y="{h/2}" transform="rotate(-90 16 {h/2})" text-anchor="middle">mean {metric}</text>']
    for i in range(5):
        xv = math.exp(x0 + (x1 - x0) * i / 4)
        yv = y0 + (y1 - y0) * i / 4
        out.append(f'<text x="{sx(xv):.1f}" y="{h-pad+16}" text-anchor="middle">{xv:.2f}</text>')
        out.append(f'<text x="{pad-6}" y="{sy(yv)+4:.1f}" text-anchor="end">{yv:.1f}</text>')
    for k, (name, curve) in enumerate(curves.items()):
        c = colors[k % len(colors)]
        path = " ".join(f"{sx(p.bpp):.1f},{sy(p.quality[metric]):.1f}" for p in sorted(curve, key=lambda p: p.bpp))
        out.append(f'<polyline points="{path}" fill="none" stroke="{c}" stroke-width="2"/>')
        out.append(f'<text x="{w-pad-150}" y="{pad+16*k}" fill="{c}">{name}</text>')
    out.append("</svg>")
    return "\n".join(out)


def analyze(images_meta, sweeps, subset: list[str], label: str, out: Path, report: list[str], summary: dict):
    complete = [i for i in subset if all((i, f) in sweeps for f in FORMATS)]
    if len(complete) < 3:
        report.append(f"\n### {label}\n\nToo few complete images ({len(complete)}).\n")
        return
    ssim_t = [round(t, 4) for t in np.arange(0.85, 0.9981, 0.002)]
    s2_t = list(range(30, 92, 2))
    ba_t = [round(t, 2) for t in np.arange(3.5, 0.45, -0.1)]
    s2 = dict(gate="ssimulacra2", direction=+1)
    ba = dict(gate="butteraugli-pnorm3", direction=-1)
    auto = list(FORMATS)

    # name -> (curve, gate metric or None). A cell is circular when judged on its own gate.
    curves = {
        "fixed avif": (fixed_curve(complete, sweeps, "avif"), None),
        "fixed webp": (fixed_curve(complete, sweeps, "webp"), None),
        "engine avif [ssim]": (search_curve(complete, sweeps, ["avif"], engine_search, ssim_t), GATE),
        "engine auto [ssim]": (search_curve(complete, sweeps, auto, engine_search, ssim_t), GATE),
        "engine avif [s2]": (search_curve(complete, sweeps, ["avif"], engine_search, s2_t, **s2), "ssimulacra2"),
        "engine auto [s2]": (search_curve(complete, sweeps, auto, engine_search, s2_t, **s2), "ssimulacra2"),
        "oracle auto [s2]": (search_curve(complete, sweeps, auto, oracle_search, s2_t, **s2), "ssimulacra2"),
        "engine avif [ba]": (search_curve(complete, sweeps, ["avif"], engine_search, ba_t, **ba), "butteraugli-pnorm3"),
        "engine auto [ba]": (search_curve(complete, sweeps, auto, engine_search, ba_t, **ba), "butteraugli-pnorm3"),
    }
    comparisons = [
        ("fixed avif", "fixed webp"),
        ("engine avif [ssim]", "fixed avif"),
        ("engine auto [ssim]", "fixed avif"),
        ("engine avif [s2]", "fixed avif"),
        ("engine auto [s2]", "fixed avif"),
        ("engine avif [ba]", "fixed avif"),
        ("engine auto [ba]", "fixed avif"),
        ("engine auto [s2]", "engine auto [ssim]"),
        ("oracle auto [s2]", "engine auto [s2]"),
    ]
    metrics = [*HELD_OUT, GATE]

    counts = defaultdict(int)
    for i in complete:
        counts[images_meta[i]["corpus"]] += 1
    report.append(f"\n### {label}\n")
    report.append(f"{len(complete)} images ({', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))}).\n")
    report.append("BD-rate: bitrate change of *test* vs *anchor* at equal quality. Negative = test needs fewer bytes. "
                  "`mean` matches corpus-average quality; `worst 10%` matches quality of the worst decile of images.\n")
    header = "| test vs anchor | " + " | ".join(
        f"{m} mean | {m} worst 10%" for m in metrics) + " |"
    report.append(header)
    report.append("|" + "---|" * (1 + 2 * len(metrics)))
    section = {}
    for test, anchor in comparisons:
        gates = {curves[test][1], curves[anchor][1]}
        cells = []
        for m in metrics:
            for stat in ("quality", "worst"):
                if m in gates:
                    cells.append("circular")
                    continue
                v = bd_rate(curves[anchor][0], curves[test][0], m, stat)
                section[f"{test} vs {anchor} / {m} / {stat}"] = v
                cells.append("n/a" if v is None else f"{v:+.1f}%")
        report.append(f"| {test} vs {anchor} | " + " | ".join(cells) + " |")
    report.append("\n`[gate]` names the metric the search must satisfy. A strategy is never judged on its own "
                  "gate metric (\"circular\"): that is exactly how a benchmark flatters an optimizer.\n")

    # Matched presets: bytes each engine needs to reach a preset's held-out quality.
    report.append("Matched to common fixed presets at equal mean SSIMULACRA2 (engines gated on SSIM or Butteraugli, "
                  "so S2 is held out for both):\n")
    report.append("| preset | bpp | mean S2 | worst-10% S2 | [ssim] bytes | [ssim] worst-10% | [ba] bytes | [ba] worst-10% |")
    report.append("|---|---|---|---|---|---|---|---|")
    for fmt, level in [("avif", 50), ("avif", 60), ("avif", 70), ("avif", 80), ("webp", 75), ("webp", 85)]:
        point = next((p for p in curves[f"fixed {fmt}"][0] if p.parameter == level), None)
        if point is None:
            continue
        target = point.quality["ssimulacra2"]
        cells = []
        for name in ("engine auto [ssim]", "engine auto [ba]"):
            curve = curves[name][0]
            bpp = rate_at(curve, "ssimulacra2", target)
            near = min(curve, key=lambda p: abs(p.quality["ssimulacra2"] - target))
            cells.append(f"{(bpp / point.bpp - 1) * 100:+.1f}%" if bpp else "out of range")
            cells.append(f"{near.worst['ssimulacra2']:.1f}")
        report.append(f"| {fmt} q{level} | {point.bpp:.3f} | {target:.1f} | {point.worst['ssimulacra2']:.1f} | "
                      + " | ".join(cells) + " |")

    # Search cost and failure rate at typical targets.
    for name, t in (("engine auto [ssim]", 0.99), ("engine auto [s2]", 80), ("oracle auto [s2]", 80),
                    ("engine auto [ba]", 1.0)):
        curve = curves[name][0]
        p = min(curve, key=lambda p: abs(p.parameter - t))
        report.append(f"\n{name} @ {p.parameter}: {p.bpp:.3f} bpp, {p.evaluations:.1f} encodes/image, "
                      f"{p.misses} images missed the target, mean S2 {p.quality['ssimulacra2']:.1f} "
                      f"(worst-10% {p.worst['ssimulacra2']:.1f}), mean BA {p.quality['butteraugli-pnorm3']:.2f} "
                      f"(worst-10% {p.worst['butteraugli-pnorm3']:.2f}).")

    slug = label.lower().replace(" ", "-").replace("/", "-")
    chart_curves = {k: v[0] for k, v in curves.items()
                    if k in ("fixed avif", "fixed webp", "engine auto [ssim]", "engine auto [ba]")}
    (out / f"chart-{slug}.svg").write_text(svg_chart(chart_curves, "ssimulacra2", f"{label}: rate vs SSIMULACRA2"))
    report.append(f"\n![{label}](chart-{slug}.svg)\n")
    summary[label] = section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()

    images, sweeps = load(args.run)
    run_config = json.loads((args.run / "run.json").read_text())
    report = ["# smartimg benchmark report", "",
              f"Run: `{args.run}`. Codecs: " + ", ".join(f"{c[1]['name']} {c[1]['version']}" for c in run_config["codecs"])
              + f". Max dimension {run_config['max_dimension']}px. Gate metric `{GATE}`; held-out metrics "
              + ", ".join(f"`{m}`" for m in HELD_OUT) + "."]

    usable, excluded = usable_images(images, sweeps)
    if excluded:
        if not any(s.contiguous for s in sweeps.values()) or run_config.get("step", 1) != 1:
            step = f" (run step {run_config['step']})" if "step" in run_config else ""
            reason = (f"no image has a contiguous level sweep{step}; "
                      "engine replay requires `smartimg-bench --step 1`")
        else:
            reason = "incomplete or non-contiguous level sweeps (e.g. an interrupted run)"
        warning = f"Excluded {len(excluded)} of {len(images)} images: {reason}."
        print(f"warning: {warning}", file=sys.stderr)
        report.append(f"\n**Warning:** {warning}")

    summary: dict = {}
    by_split = defaultdict(list)
    for image in usable:
        meta = images[image]
        by_split[meta["split"]].append(image)
        by_split[f"{meta['split']} / {meta['corpus']}"].append(image)
    for label in ["test", "dev", *sorted(k for k in by_split if "/" in k and k.startswith("test"))]:
        analyze(images, sweeps, sorted(by_split.get(label, [])), label, args.run, report, summary)

    (args.run / "report.md").write_text("\n".join(report) + "\n")
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
