#!/usr/bin/env python3
"""Small benchmark harness for the experimental optimizer.

This compares fixed-format baselines at common settings against our search-based
optimizer. It is intentionally simple; the real benchmark should use a larger held-out
corpus, psychovisual metrics, and matched-quality (BD-rate) comparison from plan.md.

Fixed baselines land at different quality levels, so compare bytes only between entries
whose `meets_target` is true.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

from smart_compressor import (
    EXTENSIONS,
    FORMAT_SPECS,
    METRIC_ID,
    analyze,
    load_normalized,
    optimize,
    save_reference,
    ssim,
    target_dimensions,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".tif", ".tiff"}

# Typical "fixed preset" settings people use without searching.
BASELINES = [("jpeg", 85), ("webp", 80), ("avif", 60)]


def collect(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="image file or directory of images")
    parser.add_argument("--target", type=float, default=0.990)
    parser.add_argument("--max-dimension", type=int, default=2400)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    rows = []
    with tempfile.TemporaryDirectory(prefix="sic_bench_") as td:
        td_path = Path(td)
        for input_path in collect(args.input):
            source = load_normalized(input_path)
            analysis = analyze(source.image)
            width, height = target_dimensions(source.image.width, source.image.height, args.max_dimension)
            ref_png = td_path / "ref.png"
            reference = save_reference(source, ref_png, width, height)

            entries = []
            for fmt, parameter in BASELINES:
                if analysis.has_alpha and not FORMAT_SPECS[fmt].supports_alpha:
                    continue
                out = td_path / f"baseline_{fmt}{EXTENSIONS[fmt]}"
                t0 = time.perf_counter()
                try:
                    FORMAT_SPECS[fmt].encode(ref_png, out, parameter, analysis)
                except (subprocess.CalledProcessError, OSError):
                    continue
                ms = (time.perf_counter() - t0) * 1000
                score = ssim(reference, out)
                entries.append({
                    "name": f"{fmt}-fixed-{parameter}",
                    "format": fmt,
                    "bytes": out.stat().st_size,
                    "ssim": round(score, 6),
                    "meets_target": score >= args.target,
                    "time_ms": round(ms, 2),
                    "parameter": parameter,
                })

            result = optimize(input_path, td_path / "optimized", target=args.target,
                              max_dimension=args.max_dimension, formats=sorted(FORMAT_SPECS))
            entries.append({
                "name": "smart-optimizer",
                "format": result["format"],
                "bytes": result["output_bytes"],
                "ssim": result["quality_score"],
                "meets_target": True,
                # Whole search, not just the winning encode, so cost is comparable.
                "time_ms": result["total_ms"],
                "encodes": result["search_encodes"],
                "parameter": result["parameter"],
                "kept_original": result["kept_original"],
            })

            rows.append({
                "file": str(input_path),
                "original_bytes": input_path.stat().st_size,
                "width": width,
                "height": height,
                "likely_type": analysis.likely_type,
                "results": entries,
            })

    summary = {"files": len(rows), "metric": METRIC_ID, "target": args.target, "results": rows}
    payload = json.dumps(summary, indent=2)
    print(payload)
    if args.json_out:
        args.json_out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
