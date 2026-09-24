//! `smartimg <input>`: search for the smallest encoding that meets an SSIM target.

use std::path::{Path, PathBuf};

use clap::{Args as ClapArgs, ValueEnum};
use smartimg_codecs::avif::{AvifCodec, AvifConfig, Chroma};
use smartimg_codecs::webp::WebpCodec;
use smartimg_core::{Engine, EngineError};
use smartimg_metrics::{Metric, Ssim};
use smartimg_pipeline::image_rs::ImageRsPipeline;
use smartimg_types::{
    DimensionPolicy, FormatOutcome, ImageFormat, OptimizationRequest, OptimizationResult, QualityGate, Selection,
};

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
pub enum Format {
    Avif,
    Webp,
}

impl From<Format> for ImageFormat {
    fn from(f: Format) -> Self {
        match f {
            Format::Avif => ImageFormat::Avif,
            Format::Webp => ImageFormat::WebP,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ChromaArg {
    #[value(name = "420")]
    Yuv420,
    #[value(name = "444")]
    Yuv444,
}

/// Find the smallest encoding of an image that still meets a perceptual quality target.
#[derive(Debug, ClapArgs)]
pub struct Args {
    pub input: Option<PathBuf>,
    /// Output path. Defaults to `<input>.optimized.<ext>`; the extension follows the winner.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Minimum SSIM (ssim-y-gauss11).
    #[arg(long, default_value_t = 0.99)]
    target: f64,
    /// Maximum width/height; 0 disables resizing.
    #[arg(long, default_value_t = 2400)]
    max_dimension: u32,
    #[arg(long, value_delimiter = ',', default_values = ["avif", "webp"])]
    formats: Vec<Format>,
    /// libavif speed, 0 (slowest, smallest) to 10.
    #[arg(long, default_value_t = 6)]
    avif_speed: i32,
    #[arg(long, value_enum, default_value = "420")]
    avif_chroma: ChromaArg,
    /// libaom tune option, e.g. `iq` or `ssim`.
    #[arg(long)]
    avif_tune: Option<String>,
    /// Maximum encode+measure evaluations per format.
    #[arg(long, default_value_t = 16)]
    budget: u32,
    /// Print the machine-readable JSON report.
    #[arg(long)]
    json: bool,
    /// Include every evaluated candidate in the JSON report.
    #[arg(long)]
    trace: bool,
}

pub fn run(args: Args) -> Result<(), String> {
    let input_path = args.input.clone().ok_or("missing input file")?;
    let input = std::fs::read(&input_path).map_err(|e| format!("reading {}: {e}", input_path.display()))?;

    let avif = AvifConfig {
        speed: args.avif_speed,
        chroma: match args.avif_chroma {
            ChromaArg::Yuv420 => Chroma::Yuv420,
            ChromaArg::Yuv444 => Chroma::Yuv444,
        },
        tune: args.avif_tune.clone(),
        ..AvifConfig::default()
    };
    let engine = Engine::new(ImageRsPipeline::default())
        .with_codec(AvifCodec::new(avif))
        .with_codec(WebpCodec::default())
        .with_metric(Ssim);

    let request = OptimizationRequest {
        allowed_formats: args.formats.iter().map(|&f| f.into()).collect(),
        quality: vec![QualityGate { metric: Ssim.id().name, threshold: args.target }],
        dimensions: match args.max_dimension {
            0 => DimensionPolicy::Preserve,
            max => DimensionPolicy::MaxDimension { max },
        },
        alpha: Default::default(),
        metadata: Default::default(),
        budget: smartimg_types::SearchBudget { max_evaluations_per_format: args.budget, ..Default::default() },
        trace: args.trace,
    };

    let result = match engine.optimize(&input, &request) {
        Ok(result) => result,
        Err(EngineError::NoAcceptableCandidate { formats }) => {
            let detail: Vec<String> = formats.iter().map(|r| format!("{}: {:?}", r.format, r.outcome)).collect();
            return Err(format!("no candidate met the target\n  {}", detail.join("\n  ")));
        }
        Err(e) => return Err(e.to_string()),
    };

    let format = match &result.selection {
        Selection::Encoded { best } => best.format,
        Selection::KeptOriginal { format } => *format,
    };
    let output = resolve_output(&input_path, args.output.as_deref(), format)?;
    std::fs::write(&output, &result.output).map_err(|e| format!("writing {}: {e}", output.display()))?;

    if args.json {
        let mut report = serde_json::to_value(&result).map_err(|e| e.to_string())?;
        report["input"] = input_path.display().to_string().into();
        report["output"] = output.display().to_string().into();
        println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    } else {
        print_summary(&result, &output);
    }
    Ok(())
}

fn resolve_output(input: &Path, output: Option<&Path>, format: ImageFormat) -> Result<PathBuf, String> {
    let ext = format.extension();
    let path = match output {
        None => {
            let stem = input.file_stem().unwrap_or_default().to_string_lossy();
            input.with_file_name(format!("{stem}.optimized.{ext}"))
        }
        Some(p) if p.extension().is_some_and(|e| e.eq_ignore_ascii_case(ext)) => p.to_path_buf(),
        Some(p) => p.with_extension(ext),
    };
    let same = |a: &Path, b: &Path| matches!((a.canonicalize(), b.canonicalize()), (Ok(x), Ok(y)) if x == y);
    if same(input, &path) {
        return Err(format!("refusing to overwrite the input file {}", input.display()));
    }
    Ok(path)
}

fn print_summary(result: &OptimizationResult, output: &Path) {
    let saved = 100.0 * (1.0 - result.output_bytes as f64 / result.source_bytes as f64);
    println!("Input:      {} bytes", result.source_bytes);
    println!("Output:     {} bytes ({saved:.2}% saved)", result.output_bytes);
    match &result.selection {
        Selection::Encoded { best } => {
            let score = best.metrics.first().map_or(f64::NAN, |m| m.score);
            println!("Format:     {} (level {}, SSIM {score:.6})", best.format, best.level);
        }
        Selection::KeptOriginal { format } => println!("Format:     {format} (original kept)"),
    }
    let d = result.output_dimensions;
    println!("Dimensions: {}x{}", d.width, d.height);
    for report in &result.formats {
        let summary = match &report.outcome {
            FormatOutcome::Found { best } => format!("{} bytes at level {}", best.bytes, best.level),
            FormatOutcome::NoAcceptableCandidate => "no level met the target".into(),
            FormatOutcome::Skipped { reason } => format!("skipped: {reason}"),
            FormatOutcome::Failed { error } => format!("failed: {error}"),
        };
        println!("  {:<5} {:>2} evals  {summary}", report.format.to_string(), report.evaluations);
    }
    println!("Time:       {:.0} ms", result.elapsed_ms);
    println!("Output:     {}", output.display());
}
