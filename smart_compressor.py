#!/usr/bin/env python3
"""Experimental perceptual image optimizer.

This is a research prototype, not a production service. It searches for the
smallest JPEG/WebP/AVIF/PNG candidate that satisfies an SSIM target.

External tools expected on PATH or configurable via environment variables:
- ImageMagick (`magick`) for JPEG encoding
- libwebp (`cwebp`) for WebP encoding
- libavif (`avifenc`) for AVIF encoding

Decoding uses Pillow (built with WebP and AVIF support).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

MAGICK = os.environ.get("MAGICK", shutil.which("magick") or "magick")
CWEBP = os.environ.get("CWEBP", shutil.which("cwebp") or "cwebp")
AVIFENC = os.environ.get("AVIFENC", shutil.which("avifenc") or "avifenc")

AVIF_SPEED = int(os.environ.get("AVIF_SPEED", "6"))

# Identifies the metric implementation in reports. Bump when the algorithm changes so
# benchmark results from different implementations are never silently compared.
METRIC_ID = "ssim-y-gauss11-v1"

PILLOW_FORMATS = {"JPEG": "jpeg", "PNG": "png", "WEBP": "webp", "AVIF": "avif"}
EXTENSIONS = {"jpeg": ".jpg", "webp": ".webp", "avif": ".avif", "png": ".png"}


@dataclass(frozen=True)
class Candidate:
    format: str
    path: str
    bytes: int
    score: float
    parameter: int
    width: int
    height: int
    encode_ms: float


@dataclass(frozen=True)
class Analysis:
    width: int
    height: int
    pixels: int
    has_alpha: bool
    grayscale: bool
    entropy: float
    edge_density: float
    flat_ratio: float
    likely_type: str

    @property
    def ui_like(self) -> bool:
        return self.likely_type in ("line-art-or-ui", "screenshot-or-detailed-art")


@dataclass(frozen=True)
class Source:
    image: Image.Image
    icc_profile: bytes | None
    format: str | None


@dataclass(frozen=True)
class FormatSpec:
    # Search axis is always quality-like: a larger parameter means higher quality and
    # (usually) more bytes. Encoders with inverted native scales map onto this axis.
    lo: int
    hi: int
    supports_alpha: bool
    encode: Callable[[Path, Path, int, Analysis], None]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def load_normalized(path: Path, *, flatten_alpha: bool = False) -> Source:
    Image.MAX_IMAGE_PIXELS = 200_000_000
    with Image.open(path) as im:
        source_format = PILLOW_FORMATS.get(im.format or "")
        icc = im.info.get("icc_profile")
        if im.mode == "CMYK":
            # A CMYK profile is invalid once pixels are converted to RGB.
            icc = None
        image = ImageOps.exif_transpose(im)
        if image.mode in ("P", "LA", "PA"):
            image = image.convert("RGBA")
        elif image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        if image.mode == "RGBA":
            alpha_min = image.getchannel("A").getextrema()[0]
            if flatten_alpha and alpha_min < 255:
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image)
            if alpha_min == 255 or flatten_alpha:
                # An alpha channel that is fully opaque (or flattened) carries no information;
                # dropping it lets alpha-less formats like JPEG compete.
                image = image.convert("RGB")
        return Source(image.copy(), icc, source_format)


def entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def analyze(image: Image.Image) -> Analysis:
    has_alpha = "A" in image.getbands()
    preview = image.copy()
    preview.thumbnail((640, 640), Image.Resampling.BILINEAR)
    gray = np.asarray(preview.convert("L"), dtype=np.int16)
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return Analysis(image.width, image.height, image.width * image.height,
                        has_alpha, False, 0, 0, 1, "unknown")

    # Cheap gradient/flatness estimates; deliberately heuristic.
    dx = np.abs(np.diff(gray, axis=1))[:-1, :]
    dy = np.abs(np.diff(gray, axis=0))[:, :-1]
    g = dx + dy
    edge_density = float((g >= 40).mean())
    flat_ratio = float((g <= 4).mean())

    small = np.asarray(preview.convert("RGB").resize((min(64, preview.width), min(64, preview.height))),
                       dtype=np.int16)
    grayscale = bool((small.max(axis=2) - small.min(axis=2) <= 3).all())

    e = entropy(gray.astype(np.uint8))
    if grayscale and flat_ratio > 0.65:
        likely = "line-art-or-ui"
    elif edge_density > 0.22:
        likely = "screenshot-or-detailed-art"
    elif e > 7.2:
        likely = "photo-or-noisy-image"
    else:
        likely = "mixed"

    return Analysis(
        width=image.width,
        height=image.height,
        pixels=image.width * image.height,
        has_alpha=has_alpha,
        grayscale=grayscale,
        entropy=round(e, 4),
        edge_density=round(edge_density, 4),
        flat_ratio=round(flat_ratio, 4),
        likely_type=likely,
    )


def target_dimensions(width: int, height: int, max_dimension: int) -> tuple[int, int]:
    scale = min(1.0, max_dimension / max(width, height)) if max_dimension else 1.0
    return max(1, round(width * scale)), max(1, round(height * scale))


def make_reference(source: Source, width: int, height: int) -> Image.Image:
    ref = source.image
    if ref.width != width or ref.height != height:
        ref = ref.resize((width, height), Image.Resampling.LANCZOS)
    return ref


def save_reference(source: Source, path: Path, width: int, height: int) -> Image.Image:
    ref = make_reference(source, width, height)
    # Keep the ICC profile so encoders embed it; dropping it shifts wide-gamut colors.
    ref.save(path, format="PNG", compress_level=1, icc_profile=source.icc_profile)
    return ref


# --- metric -----------------------------------------------------------------------------

_GAUSS = np.exp(-0.5 * (np.arange(11) - 5) ** 2 / 1.5 ** 2)
_GAUSS /= _GAUSS.sum()


def _blur(a: np.ndarray) -> np.ndarray:
    # Separable 11-tap Gaussian (sigma 1.5), valid region only.
    h = sum(_GAUSS[i] * a[:, i:a.shape[1] - 10 + i] for i in range(11))
    return sum(_GAUSS[i] * h[i:h.shape[0] - 10 + i, :] for i in range(11))


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def ssim_luma(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM on BT.601 luma with the standard Wang et al. constants."""
    if a.shape[0] < 11 or a.shape[1] < 11:
        return 1.0 if np.array_equal(a, b) else 0.0
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = _blur(a), _blur(b)
    var_a = _blur(a * a) - mu_a * mu_a
    var_b = _blur(b * b) - mu_b * mu_b
    cov = _blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    return float((num / den).mean())


def _composites(image: Image.Image) -> list[np.ndarray]:
    """Luma planes to score. Alpha images are composited over black and white so both
    color and transparency errors are visible to the metric."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float64)
    if "A" not in image.getbands():
        return [_luma(rgba[..., :3])]
    alpha = rgba[..., 3:4] / 255.0
    color = rgba[..., :3] * alpha
    return [_luma(color), _luma(color + 255.0 * (1.0 - alpha))]


def ssim(reference: Image.Image, candidate_path: Path) -> float:
    with Image.open(candidate_path) as im:
        decoded = im.convert("RGBA" if "A" in reference.getbands() else "RGB")
    if decoded.size != reference.size:
        raise RuntimeError(f"decoded size {decoded.size} != reference size {reference.size}")
    return min(ssim_luma(a, b) for a, b in zip(_composites(reference), _composites(decoded)))


# --- encoders ---------------------------------------------------------------------------

def encode_jpeg(reference_png: Path, out: Path, quality: int, analysis: Analysis) -> None:
    # 4:4:4 keeps text and UI edges sharp; 4:2:0 is cheaper for photos.
    sampling = "4:4:4" if analysis.ui_like else "4:2:0"
    run([
        MAGICK, str(reference_png), "-interlace", "Plane",
        "-sampling-factor", sampling, "-quality", str(quality), str(out)
    ])


def encode_webp(reference_png: Path, out: Path, quality: int, analysis: Analysis) -> None:
    run([
        CWEBP, "-quiet", "-q", str(quality), "-m", "6", "-sharp_yuv",
        "-metadata", "icc", str(reference_png), "-o", str(out)
    ])


def encode_avif(reference_png: Path, out: Path, quality: int, analysis: Analysis) -> None:
    run([
        AVIFENC, "-q", str(quality), "-s", str(AVIF_SPEED),
        "-y", "444" if analysis.ui_like else "420",
        "--ignore-exif", "--ignore-xmp", str(reference_png), str(out)
    ])


def encode_png_palette(reference_png: Path, out: Path, colors: int, analysis: Analysis) -> None:
    with Image.open(reference_png) as im:
        icc = im.info.get("icc_profile")
        method = Image.Quantize.FASTOCTREE if "A" in im.getbands() else Image.Quantize.MEDIANCUT
        im.quantize(colors=colors, method=method).save(out, format="PNG", optimize=True, icc_profile=icc)


def encode_png_lossless(reference_png: Path, out: Path) -> None:
    with Image.open(reference_png) as im:
        im.save(out, format="PNG", optimize=True, icc_profile=im.info.get("icc_profile"))


FORMAT_SPECS: dict[str, FormatSpec] = {
    # Capped at 98: above that JPEG grows fast for no visible gain.
    "jpeg": FormatSpec(30, 98, False, encode_jpeg),
    "webp": FormatSpec(10, 95, True, encode_webp),
    "avif": FormatSpec(20, 95, True, encode_avif),
    # Palette PNG: the "quality" axis is the palette size.
    "png": FormatSpec(4, 256, True, encode_png_palette),
}


# --- search -----------------------------------------------------------------------------

@dataclass
class SearchStats:
    encodes: int = 0
    total_ms: float = 0.0


def evaluate(fmt: str, param: int, *, reference: Image.Image, reference_png: Path, out_dir: Path,
             analysis: Analysis, stats: SearchStats) -> Candidate:
    path = out_dir / f"{fmt}_{param}{EXTENSIONS[fmt]}"
    t0 = time.perf_counter()
    FORMAT_SPECS[fmt].encode(reference_png, path, param, analysis)
    ms = (time.perf_counter() - t0) * 1000
    score = ssim(reference, path)
    stats.encodes += 1
    stats.total_ms += (time.perf_counter() - t0) * 1000
    return Candidate(fmt, str(path), path.stat().st_size, score, param, reference.width, reference.height, ms)


def search_format(fmt: str, *, target: float, sweep: int = 3, **ctx) -> Candidate | None:
    spec = FORMAT_SPECS[fmt]
    samples: dict[int, Candidate] = {}

    def sample(param: int) -> Candidate:
        if param not in samples:
            samples[param] = evaluate(fmt, param, **ctx)
        return samples[param]

    # Binary search for the lowest quality parameter that still meets the target.
    lo, hi = spec.lo, spec.hi
    acceptable: Candidate | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        c = sample(mid)
        if c.score >= target:
            acceptable = c
            hi = mid - 1
        else:
            lo = mid + 1
    if acceptable is None:
        return None

    # Local sweep around the boundary. Quality curves are not guaranteed to be perfectly
    # monotonic for every encoder/settings combination.
    p = acceptable.parameter
    for param in range(max(spec.lo, p - sweep), min(spec.hi, p + sweep) + 1):
        sample(param)
    passing = [c for c in samples.values() if c.score >= target]
    return min(passing, key=lambda c: c.bytes)


def resolve_output(input_path: Path, output_path: Path | None, ext: str) -> Path:
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}.optimized{ext}")
    elif output_path.suffix.lower() != ext:
        output_path = output_path.with_suffix(ext)
    if output_path.resolve() == input_path.resolve():
        raise RuntimeError(f"Refusing to overwrite the input file: {input_path}")
    return output_path


def optimize(input_path: Path, output_path: Path | None, *, target: float, max_dimension: int,
             formats: list[str], flatten_alpha: bool = False) -> dict:
    t_start = time.perf_counter()
    source = load_normalized(input_path, flatten_alpha=flatten_alpha)
    analysis = analyze(source.image)
    original_bytes = input_path.stat().st_size

    work_dir = Path(tempfile.mkdtemp(prefix="smart_compressor_"))
    try:
        width, height = target_dimensions(source.image.width, source.image.height, max_dimension)
        resized = (width, height) != source.image.size
        reference_png = work_dir / "reference.png"
        reference = save_reference(source, reference_png, width, height)

        stats = SearchStats()
        ctx = dict(reference=reference, reference_png=reference_png, out_dir=work_dir,
                   analysis=analysis, stats=stats)
        candidates: list[Candidate] = []
        skipped: dict[str, str] = {}
        for fmt in formats:
            if analysis.has_alpha and not FORMAT_SPECS[fmt].supports_alpha:
                # Skip rather than silently flatten transparency.
                skipped[fmt] = "format does not support alpha"
                continue
            try:
                candidate = search_format(fmt, target=target, **ctx)
            except (subprocess.CalledProcessError, OSError) as exc:
                detail = getattr(exc, "stderr", None) or str(exc)
                skipped[fmt] = f"encoder failed: {detail.strip()}"
                continue
            if candidate:
                candidates.append(candidate)
            else:
                skipped[fmt] = "no parameter met the quality target"

        # Lossless PNG is exact, so it always meets the target; only worth the encode time
        # for non-photographic content or PNG input.
        if "png" in formats and (analysis.ui_like or source.format == "png"):
            path = work_dir / "png_lossless.png"
            t0 = time.perf_counter()
            encode_png_lossless(reference_png, path)
            ms = (time.perf_counter() - t0) * 1000
            stats.encodes += 1
            stats.total_ms += ms
            candidates.append(Candidate("png", str(path), path.stat().st_size, 1.0, 0, width, height, ms))

        if not candidates:
            raise RuntimeError(f"No candidate satisfied the quality target: {skipped}")

        winner = min(candidates, key=lambda c: c.bytes)

        # Never emit something larger than the input when the original is itself a valid
        # answer (same dimensions, allowed format).
        kept_original = (
            not resized
            and source.format in formats
            and winner.bytes >= original_bytes
        )
        if kept_original:
            ext = EXTENSIONS[source.format]
            output_path = resolve_output(input_path, output_path, ext)
            shutil.copy2(input_path, output_path)
            output_bytes = original_bytes
            out_format, out_score, out_param = source.format, 1.0, None
        else:
            output_path = resolve_output(input_path, output_path, EXTENSIONS[winner.format])
            shutil.copy2(winner.path, output_path)
            output_bytes = winner.bytes
            out_format, out_score, out_param = winner.format, winner.score, winner.parameter

        best_per_format: dict[str, Candidate] = {}
        for c in candidates:
            if c.format not in best_per_format or c.bytes < best_per_format[c.format].bytes:
                best_per_format[c.format] = c

        return {
            "input": str(input_path),
            "output": str(output_path),
            "analysis": asdict(analysis),
            "original_bytes": original_bytes,
            "output_bytes": output_bytes,
            "saved_bytes": original_bytes - output_bytes,
            "saved_percent": round((1 - output_bytes / original_bytes) * 100, 2) if original_bytes else 0,
            "kept_original": kept_original,
            "quality_metric": METRIC_ID,
            "quality_target": target,
            "quality_score": round(out_score, 6),
            "format": out_format,
            "parameter": out_param,
            "width": width,
            "height": height,
            "resized": resized,
            "icc_profile": source.icc_profile is not None,
            "search_encodes": stats.encodes,
            "search_ms": round(stats.total_ms, 2),
            "total_ms": round((time.perf_counter() - t_start) * 1000, 2),
            "skipped": skipped,
            # Best candidate per format, e.g. for <picture> fallbacks.
            "candidates": [
                {
                    "format": c.format,
                    "bytes": c.bytes,
                    "score": round(c.score, 6),
                    "parameter": c.parameter,
                    "width": c.width,
                    "height": c.height,
                    "encode_ms": round(c.encode_ms, 2),
                }
                for c in sorted(best_per_format.values(), key=lambda c: c.bytes)
            ],
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Perceptual smart image compressor")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--target", type=float, default=0.990, help="minimum SSIM, default 0.990")
    parser.add_argument("--max-dimension", type=int, default=2400, help="maximum width/height; 0 disables resizing")
    parser.add_argument(
        "--formats", nargs="+", choices=sorted(FORMAT_SPECS),
        default=["jpeg", "webp", "avif", "png"],
    )
    parser.add_argument("--flatten-alpha", action="store_true",
                        help="composite transparent images onto white (allows JPEG)")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON report")
    args = parser.parse_args()

    result = optimize(
        args.input,
        args.output,
        target=args.target,
        max_dimension=args.max_dimension,
        formats=args.formats,
        flatten_alpha=args.flatten_alpha,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Input:      {result['original_bytes']:,} bytes")
        print(f"Output:     {result['output_bytes']:,} bytes")
        print(f"Saved:      {result['saved_percent']}%")
        print(f"Format:     {result['format']}" + (" (original kept)" if result["kept_original"] else ""))
        print(f"Dimensions: {result['width']}×{result['height']}")
        print(f"SSIM:       {result['quality_score']} (target {result['quality_target']})")
        print(f"Parameter:  {result['parameter']}")
        print(f"Search:     {result['search_encodes']} encodes, {result['search_ms']} ms")
        print(f"Output:     {result['output']}")


if __name__ == "__main__":
    main()
