//! `smartimg web`: web-ready images.
//!
//! For each input: resize to fit a max box, encode AVIF and WebP at fixed quality, and
//! enforce a hard byte budget (lower quality first, downscale only as a last resort).
//! Optional responsive widths, a JSON report, and ready-to-paste `<picture>` snippets.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use clap::Args as ClapArgs;
use rayon::prelude::*;
use serde::Serialize;
use smartimg_codecs::{Codec, QualityAxis};
use smartimg_codecs::avif::{AvifCodec, AvifConfig};
use smartimg_codecs::webp::WebpCodec;
use smartimg_core::budget::{BudgetOptions, encode_within_budget};
use smartimg_metrics::{Metric, Ssimulacra2};
use smartimg_pipeline::ImagePipeline;
use smartimg_pipeline::image_rs::ImageRsPipeline;
use smartimg_types::{DimensionPolicy, Dimensions, Image, ImageFormat};

use crate::optimize::Format;

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "tif", "tiff"];
/// Outputs of this and earlier runs are never re-processed as inputs.
const OUTPUT_MARKERS: &[&str] = &["_compressed", ".optimized"];

#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Image files and/or folders.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// Where to write outputs. Defaults to each input's own folder.
    #[arg(long)]
    out_dir: Option<PathBuf>,
    /// Include images in subfolders.
    #[arg(long)]
    recursive: bool,
    /// Largest output width in pixels (never upscales).
    #[arg(long, default_value_t = 2560)]
    max_width: u32,
    /// Largest output height in pixels (never upscales).
    #[arg(long, default_value_t = 2560)]
    max_height: u32,
    /// Hard size cap per file, e.g. `1MB`, `500KB`, or `0` for none.
    #[arg(long, default_value = "1MB", value_parser = parse_bytes)]
    max_bytes: u64,
    #[arg(long, value_delimiter = ',', default_values = ["avif", "webp"])]
    formats: Vec<Format>,
    /// AVIF quality 0-100 (libavif); values outside the supported 20-95 are clamped. 60 is high quality for photos.
    #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u32).range(0..=100))]
    avif_quality: u32,
    /// WebP quality 0-100; values outside the supported 10-95 are clamped.
    #[arg(long, default_value_t = 80, value_parser = clap::value_parser!(u32).range(0..=100))]
    webp_quality: u32,
    /// Extra responsive widths, e.g. `640,1280,1920`. Widths at or above the main width are skipped.
    #[arg(long, value_delimiter = ',')]
    sizes: Vec<u32>,
    /// Appended to output names: `hero.jpg` -> `hero_web.avif`.
    #[arg(long, default_value = "_web")]
    suffix: String,
    /// Keep original file names instead of URL-friendly ones (`AERIAL VIEW` -> `aerial-view`).
    #[arg(long)]
    keep_names: bool,
    /// Skip the SSIMULACRA2 quality score in the summary (faster).
    #[arg(long)]
    no_score: bool,
    /// libavif speed, 0 (slowest, smallest) to 10.
    #[arg(long, default_value_t = 6)]
    avif_speed: i32,
    /// Images processed in parallel (at least 1). Defaults to half the CPU cores.
    #[arg(long, value_parser = parse_jobs)]
    jobs: Option<usize>,
}

fn parse_jobs(s: &str) -> Result<usize, String> {
    match s.trim().parse::<usize>() {
        Ok(0) => Err("must be at least 1".into()),
        Ok(n) => Ok(n),
        Err(_) => Err(format!("invalid number `{s}`")),
    }
}

fn parse_bytes(s: &str) -> Result<u64, String> {
    let t = s.trim().to_ascii_uppercase();
    let (number, unit) = match t.find(|c: char| c.is_ascii_alphabetic()) {
        Some(i) => t.split_at(i),
        None => (t.as_str(), ""),
    };
    let value: f64 = number.trim().parse().map_err(|_| format!("invalid size `{s}`"))?;
    let multiplier = match unit.trim() {
        "" | "B" => 1.0,
        "KB" | "K" => 1_000.0,
        "MB" | "M" => 1_000_000.0,
        "KIB" => 1_024.0,
        "MIB" => 1_048_576.0,
        other => return Err(format!("unknown unit `{other}`")),
    };
    Ok((value * multiplier).round() as u64)
}

/// `AERIAL VIEW` -> `aerial-view`, `1_11 - Photo` -> `1-11-photo`.
fn slug(stem: &str) -> String {
    let mut out = String::new();
    for c in stem.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    let out = out.trim_end_matches('-').to_string();
    if out.is_empty() { "image".into() } else { out }
}

fn is_image(path: &Path, suffix: &str) -> bool {
    let ext_ok = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()));
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    // An empty suffix would match every stem; the overwrite guard in `process_inner` covers that case.
    let own_output = !suffix.is_empty() && (stem.ends_with(suffix) || stem.contains(&format!("{suffix}-")));
    ext_ok && !own_output && !OUTPUT_MARKERS.iter().any(|m| stem.contains(m))
}

/// Clamps a requested 0-100 quality into the codec's supported range, with a note when it had to.
fn start_level(ext: &str, requested: u32, axis: QualityAxis) -> (u32, Option<String>) {
    let level = requested.clamp(axis.min, axis.max);
    let note = (level != requested)
        .then(|| format!("{ext} quality {requested} is outside the supported range {}-{}; using {level}", axis.min, axis.max));
    (level, note)
}

fn collect(args: &Args) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for input in &args.inputs {
        if input.is_file() {
            files.push(input.clone());
            continue;
        }
        let mut stack = vec![input.clone()];
        while let Some(dir) = stack.pop() {
            let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            for entry in entries {
                let path = entry.map_err(|e| e.to_string())?.path();
                if path.is_dir() {
                    if args.recursive {
                        stack.push(path);
                    }
                } else if is_image(&path, &args.suffix) {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

#[derive(Debug, Serialize)]
struct FileReport {
    path: PathBuf,
    format: ImageFormat,
    width: u32,
    height: u32,
    bytes: u64,
    quality: u32,
    within_budget: bool,
}

#[derive(Debug, Serialize)]
struct ImageReport {
    input: PathBuf,
    input_bytes: u64,
    source: Dimensions,
    /// Main (largest) outputs, one per format.
    outputs: Vec<FileReport>,
    /// Responsive variants.
    variants: Vec<FileReport>,
    /// SSIMULACRA2 of each main output vs the resized source: ~90 visually lossless, ~70+ high.
    ssimulacra2: Vec<(ImageFormat, f64)>,
    warnings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

struct Encoders {
    pipeline: ImageRsPipeline,
    codecs: Vec<(Box<dyn Codec>, u32)>,
}

fn scaled(source: Dimensions, width: u32) -> Dimensions {
    let height = (f64::from(source.height) * f64::from(width) / f64::from(source.width)).round() as u32;
    Dimensions::new(width, height.max(1))
}

pub fn run(args: Args) -> Result<(), String> {
    let files = collect(&args)?;
    if files.is_empty() {
        return Err("no images found".into());
    }
    if let Some(dir) = &args.out_dir {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }

    let cores = std::thread::available_parallelism().map_or(2, |n| n.get());
    let jobs = args.jobs.unwrap_or(cores / 2).min(files.len()).max(1);
    let encoder_threads = (cores / jobs).max(1) as i32;
    let encoders = Encoders {
        pipeline: ImageRsPipeline::default(),
        codecs: args
            .formats
            .iter()
            .map(|f| -> (Box<dyn Codec>, u32) {
                match f {
                    Format::Avif => (
                        Box::new(AvifCodec::new(AvifConfig { speed: args.avif_speed, threads: encoder_threads, ..Default::default() })),
                        args.avif_quality,
                    ),
                    Format::Webp => (Box::new(WebpCodec::default()), args.webp_quality),
                }
            })
            .collect(),
    };

    // Output names must be unique per folder even when two inputs slug to the same name.
    let names = unique_names(&files, &args);

    eprintln!("Processing {} images ({jobs} at a time)...", files.len());
    let pool = rayon::ThreadPoolBuilder::new().num_threads(jobs).build().map_err(|e| e.to_string())?;
    let reports: Vec<ImageReport> = pool.install(|| {
        files
            .par_iter()
            .zip(names.par_iter())
            .map(|(path, name)| {
                let report = process(path, name, &args, &encoders);
                print_line(&report);
                report
            })
            .collect()
    });

    let out_root = args.out_dir.clone().unwrap_or_else(|| files[0].parent().unwrap_or(Path::new(".")).to_path_buf());
    let report_path = out_root.join("web-report.json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&reports).unwrap()).map_err(|e| e.to_string())?;
    let snippets_path = out_root.join("picture-snippets.html");
    std::fs::write(&snippets_path, snippets(&reports)).map_err(|e| e.to_string())?;

    print_summary(&reports, &args);
    eprintln!("Report:   {}", report_path.display());
    eprintln!("Snippets: {}", snippets_path.display());
    if reports.iter().any(|r| r.error.is_some()) {
        return Err("some images failed; see above".into());
    }
    Ok(())
}

fn unique_names(files: &[PathBuf], args: &Args) -> Vec<String> {
    let mut used = std::collections::HashSet::new();
    files
        .iter()
        .map(|path| {
            let stem = path.file_stem().unwrap_or_default().to_string_lossy();
            let base = if args.keep_names { stem.to_string() } else { slug(&stem) };
            let dir = args.out_dir.clone().unwrap_or_else(|| path.parent().unwrap_or(Path::new(".")).to_path_buf());
            let mut name = base.clone();
            let mut n = 2;
            while !used.insert((dir.clone(), name.clone())) {
                name = format!("{base}-{n}");
                n += 1;
            }
            name
        })
        .collect()
}

fn process(path: &Path, name: &str, args: &Args, enc: &Encoders) -> ImageReport {
    let mut report = ImageReport {
        input: path.to_path_buf(),
        input_bytes: std::fs::metadata(path).map_or(0, |m| m.len()),
        source: Dimensions::new(0, 0),
        outputs: Vec::new(),
        variants: Vec::new(),
        ssimulacra2: Vec::new(),
        warnings: Vec::new(),
        error: None,
    };
    if let Err(e) = process_inner(path, name, args, enc, &mut report) {
        report.error = Some(e);
    }
    report
}

fn process_inner(path: &Path, name: &str, args: &Args, enc: &Encoders, report: &mut ImageReport) -> Result<(), String> {
    let input = std::fs::read(path).map_err(|e| e.to_string())?;
    let source = enc.pipeline.decode(&input).map_err(|e| e.to_string())?;
    report.source = source.image.dimensions();
    let dir = args.out_dir.clone().unwrap_or_else(|| path.parent().unwrap_or(Path::new(".")).to_path_buf());

    let main_dims = DimensionPolicy::MaxBox { width: args.max_width, height: args.max_height }.target(report.source);
    let resize = |dims: Dimensions| -> Result<Image, String> {
        if dims == report.source {
            Ok(source.image.clone())
        } else {
            enc.pipeline.resize(&source.image, dims).map_err(|e| e.to_string())
        }
    };
    let main = resize(main_dims)?;

    let mut widths: Vec<u32> = args.sizes.iter().copied().filter(|&w| w < main_dims.width).collect();
    widths.sort_unstable();
    widths.dedup();

    let budget = |level: u32| BudgetOptions {
        level,
        max_bytes: (args.max_bytes > 0).then_some(args.max_bytes),
        ..Default::default()
    };

    for (codec, requested) in &enc.codecs {
        let ext = codec.format().extension();
        let (quality, note) = start_level(ext, *requested, codec.quality_axis());
        report.warnings.extend(note);
        let result = encode_within_budget(&enc.pipeline, codec.as_ref(), &main, &budget(quality)).map_err(|e| e.to_string())?;
        let out = dir.join(format!("{name}{}.{ext}", args.suffix));
        if out.canonicalize().ok() == path.canonicalize().ok() {
            return Err(format!("refusing to overwrite the input {}", path.display()));
        }
        std::fs::write(&out, &result.encoded.bytes).map_err(|e| format!("{}: {e}", out.display()))?;

        if !result.within_budget {
            report.warnings.push(format!("{ext} is still over the size cap ({} bytes)", result.encoded.bytes.len()));
        } else if result.level < quality {
            report.warnings.push(format!("{ext} quality lowered {quality} -> {} to fit the cap", result.level));
        }
        if result.dimensions != main_dims {
            report.warnings.push(format!(
                "{ext} downscaled to {}x{} to fit the cap",
                result.dimensions.width, result.dimensions.height
            ));
        }
        if !args.no_score {
            // Score at the dimensions actually written.
            let reference = if result.dimensions == main_dims { main.clone() } else { resize(result.dimensions)? };
            let decoded = codec.decode(&result.encoded.bytes).map_err(|e| e.to_string())?;
            let score = Ssimulacra2.measure(&reference, &decoded).map_err(|e| e.to_string())?.score;
            report.ssimulacra2.push((codec.format(), score));
        }
        report.outputs.push(FileReport {
            path: out,
            format: codec.format(),
            width: result.dimensions.width,
            height: result.dimensions.height,
            bytes: result.encoded.bytes.len() as u64,
            quality: result.level,
            within_budget: result.within_budget,
        });

        for &width in &widths {
            let image = resize(scaled(report.source, width))?;
            let result = encode_within_budget(&enc.pipeline, codec.as_ref(), &image, &budget(quality)).map_err(|e| e.to_string())?;
            let out = dir.join(format!("{name}{}-{width}.{ext}", args.suffix));
            std::fs::write(&out, &result.encoded.bytes).map_err(|e| format!("{}: {e}", out.display()))?;
            report.variants.push(FileReport {
                path: out,
                format: codec.format(),
                width: result.dimensions.width,
                height: result.dimensions.height,
                bytes: result.encoded.bytes.len() as u64,
                quality: result.level,
                within_budget: result.within_budget,
            });
        }
    }
    Ok(())
}

fn kb(bytes: u64) -> String {
    format!("{:.0} KB", bytes as f64 / 1000.0)
}

fn print_line(r: &ImageReport) {
    let name = r.input.file_name().unwrap_or_default().to_string_lossy();
    if let Some(e) = &r.error {
        eprintln!("  FAILED {name}: {e}");
        return;
    }
    let outputs: Vec<String> = r
        .outputs
        .iter()
        .map(|o| {
            let score = r.ssimulacra2.iter().find(|(f, _)| *f == o.format).map(|(_, s)| format!(", S2 {s:.0}")).unwrap_or_default();
            format!("{} {}{score}", o.format, kb(o.bytes))
        })
        .collect();
    let dims = r.outputs.first().map(|o| format!("{}x{}", o.width, o.height)).unwrap_or_default();
    eprintln!("  {name}: {} -> {dims}: {}", kb(r.input_bytes), outputs.join(" | "));
    for w in &r.warnings {
        eprintln!("    note: {w}");
    }
}

fn print_summary(reports: &[ImageReport], args: &Args) {
    let ok: Vec<&ImageReport> = reports.iter().filter(|r| r.error.is_none()).collect();
    let input: u64 = ok.iter().map(|r| r.input_bytes).sum();
    eprintln!();
    for format in args.formats.iter().map(|&f| ImageFormat::from(f)) {
        let files: Vec<&FileReport> = ok.iter().flat_map(|r| r.outputs.iter()).filter(|o| o.format == format).collect();
        if files.is_empty() {
            continue;
        }
        let total: u64 = files.iter().map(|o| o.bytes).sum();
        let largest = files.iter().map(|o| o.bytes).max().unwrap_or(0);
        let over = files.iter().filter(|o| !o.within_budget).count();
        eprintln!(
            "{format:>5}: {} files, total {} (was {}), largest {}, over cap: {over}",
            files.len(),
            kb(total),
            kb(input),
            kb(largest)
        );
    }
}

fn snippets(reports: &[ImageReport]) -> String {
    let file_name = |p: &Path| p.file_name().unwrap_or_default().to_string_lossy().to_string();
    let mut html = String::from(
        "<!-- Generated by smartimg web. Adjust paths, `alt` text, and `sizes` to your layout. -->\n",
    );
    for r in reports.iter().filter(|r| r.error.is_none()) {
        let Some(fallback) = r.outputs.iter().find(|o| o.format != ImageFormat::Avif).or(r.outputs.first()) else {
            continue;
        };
        let stem = r.input.file_stem().unwrap_or_default().to_string_lossy();
        let _ = writeln!(html, "\n<!-- {} -->\n<picture>", file_name(&r.input));
        for main in &r.outputs {
            let mut entries: Vec<(u32, String)> =
                r.variants.iter().filter(|v| v.format == main.format).map(|v| (v.width, file_name(&v.path))).collect();
            entries.push((main.width, file_name(&main.path)));
            let srcset: Vec<String> = entries.iter().map(|(w, f)| format!("{f} {w}w")).collect();
            let sizes = if entries.len() > 1 { " sizes=\"100vw\"" } else { "" };
            let _ = writeln!(
                html,
                "  <source type=\"{}\" srcset=\"{}\"{sizes}>",
                main.format.mime_type(),
                srcset.join(", ")
            );
        }
        let _ = writeln!(
            html,
            "  <img src=\"{}\" width=\"{}\" height=\"{}\" alt=\"{}\" loading=\"lazy\" decoding=\"async\">\n</picture>",
            file_name(&fallback.path),
            fallback.width,
            fallback.height,
            stem.replace('"', "")
        );
    }
    html
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sizes() {
        assert_eq!(parse_bytes("1MB").unwrap(), 1_000_000);
        assert_eq!(parse_bytes("500kb").unwrap(), 500_000);
        assert_eq!(parse_bytes("1.5M").unwrap(), 1_500_000);
        assert_eq!(parse_bytes("0").unwrap(), 0);
        assert!(parse_bytes("12 parsecs").is_err());
    }

    #[test]
    fn slugs_are_url_friendly() {
        assert_eq!(slug("AERIAL VIEW"), "aerial-view");
        assert_eq!(slug("1_11 - Photo"), "1-11-photo");
        assert_eq!(slug("OVERALL Landscape X bridge"), "overall-landscape-x-bridge");
        assert_eq!(slug("   "), "image");
    }

    #[test]
    fn skips_previous_outputs() {
        assert!(is_image(Path::new("a/hero.jpg"), "_web"));
        assert!(!is_image(Path::new("a/hero_web.avif"), "_web"));
        assert!(!is_image(Path::new("a/hero_web-640.webp"), "_web"));
        assert!(!is_image(Path::new("a/hero_compressed.jpg"), "_web"));
        assert!(!is_image(Path::new("a/Thumbs.db"), "_web"));
    }

    #[test]
    fn empty_suffix_keeps_inputs() {
        assert!(is_image(Path::new("a/hero.jpg"), ""));
        assert!(is_image(Path::new("a/hero-640.png"), ""));
        assert!(!is_image(Path::new("a/hero_compressed.jpg"), ""));
    }

    #[test]
    fn clamps_quality_to_codec_axis() {
        let axis = QualityAxis { min: 20, max: 95 };
        assert_eq!(start_level("avif", 60, axis), (60, None));
        assert_eq!(start_level("avif", 20, axis), (20, None));
        let (level, note) = start_level("avif", 10, axis);
        assert_eq!(level, 20);
        assert!(note.unwrap().contains("avif quality 10 is outside the supported range 20-95; using 20"));
        assert_eq!(start_level("webp", 100, QualityAxis { min: 10, max: 95 }).0, 95);
    }

    #[test]
    fn rejects_zero_jobs() {
        assert!(parse_jobs("0").is_err());
        assert!(parse_jobs("x").is_err());
        assert_eq!(parse_jobs("4").unwrap(), 4);
    }

    #[test]
    fn scales_preserving_aspect() {
        assert_eq!(scaled(Dimensions::new(3840, 2160), 1280), Dimensions::new(1280, 720));
    }
}
