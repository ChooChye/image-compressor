# Smart Image Compressor

Research project for perceptual rate-distortion image optimization.

The key idea is **not** to build a new JPEG/AVIF codec from scratch. The project builds a proprietary optimizer that searches native codec parameters and selects the smallest candidate that satisfies explicit perceptual and image constraints.

## Usage: compress images for websites

`smartimg web` resizes images to web size and writes an AVIF plus a WebP fallback for each, every file under 1 MB. It works from any folder in Terminal.

### Main command

```bash
smartimg web "FOLDER" --out-dir "FOLDER/compressed" --avif-quality 70 --max-height 1600
```

Example:

```bash
smartimg web "/Users/choochye/Developer/jrd/1. Master Planning/Jade Hills MCP" \
  --out-dir "/Users/choochye/Developer/jrd/1. Master Planning/Jade Hills MCP/compressed" \
  --avif-quality 70 --max-height 1600
```

Result in `compressed/`:

- `<name>_web.avif`: main file (~95% of browsers)
- `<name>_web.webp`: fallback for older browsers
- `picture-snippets.html`: ready-to-paste `<picture>` markup (fill in the `alt` text)
- `web-report.json`: sizes, dimensions, and settings per image

Output names are URL-friendly: `AERIAL VIEW.png` becomes `aerial-view_web.avif`, because spaces break web links.

### Other ways to use it

```bash
# One image
smartimg web "photo.jpg" --out-dir compressed

# A few specific images
smartimg web a.jpg b.png c.tif --out-dir compressed

# Include subfolders
smartimg web "FOLDER" --out-dir "FOLDER/compressed" --recursive

# Extra phone/tablet sizes for srcset
smartimg web "FOLDER" --out-dir "FOLDER/compressed" --sizes 640,1280,1920

# Stricter size limit
smartimg web "FOLDER" --out-dir "FOLDER/compressed" --max-bytes 500KB

# WebP only (for places that accept a single file)
smartimg web "FOLDER" --out-dir "FOLDER/compressed" --formats webp
```

### All options

| Option | Default | What it does |
|---|---|---|
| `--out-dir PATH` | same folder as the image | Where output files go |
| `--max-width N` | 2560 | Largest width in pixels (never upscales) |
| `--max-height N` | 2560 | Largest height; **1600 recommended** so portrait photos aren't huge |
| `--max-bytes SIZE` | 1MB | Hard size cap per file, e.g. `500KB`; `0` for none |
| `--avif-quality N` | 60 | AVIF quality; **70 recommended** for renders and detailed photos |
| `--webp-quality N` | 80 | WebP quality |
| `--formats` | avif,webp | Formats to create |
| `--sizes` | none | Extra widths, e.g. `640,1280,1920` |
| `--recursive` | off | Include subfolders |
| `--keep-names` | off | Keep original file names (spaces and all) |
| `--no-score` | off | Skip the quality score (faster) |

Run `smartimg web --help` to see this list in Terminal.

### How the size cap works

Images are encoded at a fixed quality. If a file is over `--max-bytes`, quality is lowered in steps of 5 (down to 40), then the image is downscaled 15% at a time. Lines starting with `note:` in the output list any image that needed this, so check those.

Each output line also shows a SSIMULACRA2 quality score: ~90 is visually lossless, 70+ is high quality, and below ~65 looks noticeably soft.

### Shortcut

Add this to `~/.zshrc`, then open a new Terminal window:

```bash
webimg() { smartimg web "$1" --out-dir "${1%/}/compressed" --avif-quality 70 --max-height 1600 "${@:2}"; }
```

Then:

```bash
webimg "/Users/choochye/Developer/jrd/1. Master Planning/Jade Hills MCP"
```

### Using the output on a website

The browser downloads only the first format it supports, so the fallback costs visitors nothing:

```html
<picture>
  <source type="image/avif" srcset="compressed/aerial-view_web.avif">
  <img src="compressed/aerial-view_web.webp" alt="Aerial view" width="2560" height="1374" loading="lazy">
</picture>
```

| Situation | Use |
|---|---|
| `<picture>` allowed | AVIF + WebP fallback |
| Only one file allowed (CSS background, some CMS fields) | WebP |
| Social share image (`og:image`) or email | JPEG, via `python3 smart_compressor.py image.jpg --formats jpeg` |

### Tips

- Drag a folder from Finder into Terminal to paste its path.
- Re-running is safe: it skips its own outputs and overwrites the previous files in `compressed/`.
- Each run rewrites `web-report.json` and `picture-snippets.html`. Compress a batch in one command to keep them in one report.
- Colour profiles (e.g. Adobe RGB) are kept; GPS and other EXIF data are removed; camera rotation is applied.

### Install or update

```bash
brew install libavif webp pkg-config
cd ~/projects/image-compressor && cargo install --path apps/cli
```

Re-run the `cargo install` line after changing the code to update the installed `smartimg`.

## Architecture

```text
Client
  |
  v
Thin API -----> Queue -----> Worker pool
                               |
                               v
                         smartimg-core
                               |
                +--------------+--------------+
                |              |              |
                v              v              v
            libvips        Native codecs    Metrics
          input/resize       JPEG/WebP/AVIF   SSIM/etc.
                \              |              /
                 +-------------+-------------+
                               |
                               v
                         Object storage
                               |
                               v
                              CDN
```

### Important boundaries

- **API is not the compression engine.** It creates jobs and reports status.
- **Workers own CPU-heavy optimization.** They can scale independently.
- **The optimizer drives the encode → decode → metric loop.**
- **libvips handles image processing, not every encoder knob.** Native codec adapters provide direct access to codec-specific parameters.
- **`smartimg-core` is a Rust library crate.** CLI, workers, WASM, and future integrations wrap the same core.
- **Python stays in the research lab.** Proven algorithms are ported into Rust once validated.

## Current prototype

The existing Python prototype remains a research baseline. It searches JPEG/WebP/AVIF/PNG candidates and uses SSIM (`ssim-y-gauss11-v1`, identical to the Rust `Ssim` metric) as the first acceptance metric.

Requires `magick` (JPEG), `cwebp` (WebP), `avifenc` (AVIF), and Pillow with WebP/AVIF support:

```bash
brew install imagemagick webp libavif
pip install -r requirements.txt
```

Run it with:

```bash
python3 smart_compressor.py input.jpg                # writes input.optimized.<ext>
python3 smart_compressor.py input.jpg -o out --json
python3 benchmark.py path/to/image-or-directory
```

The input file is never overwritten, and the original is kept when no candidate is smaller.

## Rust engine

```bash
brew install libavif webp pkg-config          # system codecs, found via pkg-config
cargo test --workspace --all-features
cargo build --release -p smartimg-cli
./target/release/smartimg input.jpg            # writes input.optimized.<ext>
./target/release/smartimg input.png --formats avif,webp --target 0.99 --json --trace
```

- `smartimg-core` runs the optimizer loop against injected `ImagePipeline`, `Codec`, and `Metric` traits, and builds with no native dependencies (`cargo test -p smartimg-core`).
- `smartimg-codecs` features `native-avif` (libavif + libaom) and `native-webp` (libwebp + libwebpmux) bind the system libraries with bindgen. Production builds should vendor and statically link pinned versions.
- `smartimg-pipeline` feature `image-rs` is a pure-Rust stand-in for the libvips pipeline: gamma-space Lanczos resize, 8-bit only, no AVIF input.
- JPEG (jpegli/mozjpeg) and PNG adapters are not implemented yet, so the CLI currently searches AVIF and WebP only.

See [`plan.md`](./plan.md) for the complete research and production roadmap.

## Benchmarks

```bash
# Sweep every quality level of every format (resumable; outputs are gitignored).
./target/release/smartimg-bench --corpus kodak=benchmarks/data/kodak --corpus web=~/path/to/images@100 \
    --out benchmarks/results/run1 --jobs 10
# Compare fixed presets, the engine's search, and an oracle on held-out metrics.
python3 research/python/analyze_bench.py benchmarks/results/run1
# Compare encoder configurations (e.g. --avif-speed / --avif-tune runs).
python3 research/python/compare_runs.py benchmarks/results/avif-s6-default benchmarks/results/avif-s4-default
```

Kodak images: `http://r0k.us/graphics/kodak/`. Findings are summarized in `plan.md` §18.

