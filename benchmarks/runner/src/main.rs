//! Benchmark sweep runner.
//!
//! For every image in the corpora, encodes every quality level of every format, decodes it,
//! and records bytes plus all metrics as one JSON line per candidate. Strategies (fixed
//! presets, the engine's search, oracles) are then evaluated offline from the same table by
//! `research/python/analyze_bench.py`, so every strategy is judged on identical encodes.
//!
//! Runs are resumable: rows already in `results.jsonl` are skipped. A run directory is
//! bound to one configuration (`run.json`) so results from different settings never mix.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use smartimg_codecs::avif::{AvifCodec, AvifConfig};
use smartimg_codecs::webp::WebpCodec;
use smartimg_codecs::{Codec, EncodeSettings};
use smartimg_metrics::{Butteraugli, ButteraugliNorm, Metric, Ssim, Ssimulacra2};
use smartimg_pipeline::ImagePipeline;
use smartimg_pipeline::image_rs::ImageRsPipeline;
use smartimg_types::{CodecId, DimensionPolicy, ImageFormat, MetadataPolicy, MetricId};

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum Format {
    Avif,
    Webp,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum MetricArg {
    Ssim,
    Ssimulacra2,
    Butteraugli,
    ButteraugliMax,
}

#[derive(Debug, Clone)]
struct CorpusArg {
    name: String,
    dir: PathBuf,
    sample: Option<usize>,
}

fn parse_corpus(s: &str) -> Result<CorpusArg, String> {
    let (name, rest) = s.split_once('=').ok_or("expected NAME=DIR[@SAMPLE]")?;
    let (dir, sample) = match rest.rsplit_once('@') {
        Some((dir, n)) => (dir, Some(n.parse::<usize>().map_err(|e| e.to_string())?)),
        None => (rest, None),
    };
    let dir = match dir.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").map_err(|e| e.to_string())?).join(rest),
        None => PathBuf::from(dir),
    };
    Ok(CorpusArg { name: name.to_string(), dir, sample })
}

/// Sweep every quality level of every format over one or more image corpora.
#[derive(Debug, Parser)]
struct Args {
    /// `NAME=DIR` or `NAME=DIR@N` to take a deterministic content-hash sample of N images.
    #[arg(long = "corpus", required = true, value_parser = parse_corpus)]
    corpora: Vec<CorpusArg>,
    /// Run directory (manifest.json, run.json, results.jsonl).
    #[arg(long)]
    out: PathBuf,
    /// Longest side after resizing; 0 keeps source dimensions.
    #[arg(long, default_value_t = 1600)]
    max_dimension: u32,
    #[arg(long, value_delimiter = ',', default_values = ["avif", "webp"])]
    formats: Vec<Format>,
    #[arg(long, value_delimiter = ',', default_values = ["ssim", "ssimulacra2", "butteraugli"])]
    metrics: Vec<MetricArg>,
    /// Level step. Use 1 so the engine's search can be replayed exactly.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    step: u32,
    #[arg(long, default_value_t = 6)]
    avif_speed: i32,
    /// libaom tune option, e.g. `iq` or `ssim`.
    #[arg(long)]
    avif_tune: Option<String>,
    /// Parallel images (each encode is single-threaded).
    #[arg(long)]
    jobs: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct RunConfig {
    max_dimension: u32,
    step: u32,
    codecs: Vec<(String, CodecId, String)>,
    metrics: Vec<MetricId>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ManifestEntry {
    id: String,
    corpus: String,
    path: PathBuf,
    sha256: String,
    /// Frozen from the content hash; never chosen after looking at results.
    split: String,
    source_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct Row {
    image: String,
    corpus: String,
    split: String,
    format: ImageFormat,
    level: u32,
    bytes: u64,
    width: u32,
    height: u32,
    source_width: u32,
    source_height: u32,
    has_alpha: bool,
    encode_ms: f64,
    metrics: std::collections::BTreeMap<String, f64>,
    metric_ms: std::collections::BTreeMap<String, f64>,
}

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "tif", "tiff"];

fn collect(corpus: &CorpusArg) -> Result<Vec<ManifestEntry>, String> {
    let mut files = Vec::new();
    let mut stack = vec![corpus.dir.clone()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
            {
                files.push(path);
            }
        }
    }

    let mut entries: Vec<ManifestEntry> = files
        .into_par_iter()
        .map(|path| {
            let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let sha256 = format!("{:x}", Sha256::digest(&bytes));
            // Split on a different hash byte than the sampler, so sampling does not bias it.
            let split = if u8::from_str_radix(&sha256[62..64], 16).unwrap() % 2 == 0 { "test" } else { "dev" };
            Ok(ManifestEntry {
                id: sha256[..16].to_string(),
                corpus: corpus.name.clone(),
                path,
                sha256,
                split: split.into(),
                source_bytes: bytes.len() as u64,
            })
        })
        .collect::<Result<_, String>>()?;

    // Deterministic, content-based order; duplicates (same content) are dropped.
    entries.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    entries.dedup_by(|a, b| a.sha256 == b.sha256);
    if let Some(n) = corpus.sample {
        entries.truncate(n);
    }
    Ok(entries)
}

fn build_codecs(args: &Args) -> Vec<Box<dyn Codec>> {
    args.formats
        .iter()
        .map(|f| -> Box<dyn Codec> {
            match f {
                Format::Avif => Box::new(AvifCodec::new(AvifConfig {
                    speed: args.avif_speed,
                    tune: args.avif_tune.clone(),
                    threads: 1,
                    ..Default::default()
                })),
                Format::Webp => Box::new(WebpCodec::default()),
            }
        })
        .collect()
}

fn build_metrics(args: &Args) -> Vec<Box<dyn Metric>> {
    args.metrics
        .iter()
        .map(|m| -> Box<dyn Metric> {
            match m {
                MetricArg::Ssim => Box::new(Ssim),
                MetricArg::Ssimulacra2 => Box::new(Ssimulacra2),
                MetricArg::Butteraugli => Box::new(Butteraugli { norm: ButteraugliNorm::PNorm3 }),
                MetricArg::ButteraugliMax => Box::new(Butteraugli { norm: ButteraugliNorm::Max }),
            }
        })
        .collect()
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    if let Err(e) = run(Args::parse()) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), String> {
    if let Some(jobs) = args.jobs {
        rayon::ThreadPoolBuilder::new().num_threads(jobs).build_global().map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(&args.out).map_err(|e| e.to_string())?;

    let codecs = build_codecs(&args);
    let metrics = build_metrics(&args);
    let config = RunConfig {
        max_dimension: args.max_dimension,
        step: args.step,
        codecs: codecs
            .iter()
            .map(|c| {
                let axis = c.quality_axis();
                (c.format().to_string(), c.id(), format!("{axis:?} avif_speed={} avif_tune={:?}", args.avif_speed, args.avif_tune))
            })
            .collect(),
        metrics: metrics.iter().map(|m| m.id()).collect(),
    };
    let config_path = args.out.join("run.json");
    if config_path.exists() {
        let existing: RunConfig = serde_json::from_str(&fs::read_to_string(&config_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if existing != config {
            return Err(format!("{} was created with different settings; use a new --out", args.out.display()));
        }
    } else {
        fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).map_err(|e| e.to_string())?;
    }

    let mut manifest = Vec::new();
    for corpus in &args.corpora {
        let entries = collect(corpus)?;
        eprintln!("corpus {}: {} images from {}", corpus.name, entries.len(), corpus.dir.display());
        manifest.extend(entries);
    }
    fs::write(args.out.join("manifest.json"), serde_json::to_string_pretty(&manifest).unwrap())
        .map_err(|e| e.to_string())?;

    let results_path = args.out.join("results.jsonl");
    let done = load_done(&results_path)?;
    let writer = Mutex::new(BufWriter::new(
        OpenOptions::new().create(true).append(true).open(&results_path).map_err(|e| e.to_string())?,
    ));
    let errors = Mutex::new(Vec::new());
    let completed = AtomicUsize::new(0);
    let started = Instant::now();
    let pipeline = ImageRsPipeline::default();
    let policy = match args.max_dimension {
        0 => DimensionPolicy::Preserve,
        max => DimensionPolicy::MaxDimension { max },
    };

    manifest.par_iter().for_each(|entry| {
        let t0 = Instant::now();
        let result = sweep_image(entry, &pipeline, policy, &codecs, &metrics, args.step, &done, &writer);
        let n = completed.fetch_add(1, Ordering::Relaxed) + 1;
        match result {
            Ok(rows) => eprintln!(
                "[{n}/{}] {} {}: {rows} rows in {:.1}s (elapsed {:.0}s)",
                manifest.len(),
                entry.corpus,
                entry.path.display(),
                t0.elapsed().as_secs_f64(),
                started.elapsed().as_secs_f64()
            ),
            Err(e) => {
                eprintln!("[{n}/{}] {} FAILED: {e}", manifest.len(), entry.path.display());
                errors.lock().unwrap().push(serde_json::json!({ "image": entry.id, "path": entry.path, "error": e }));
            }
        }
    });

    let errors = errors.into_inner().unwrap();
    fs::write(args.out.join("errors.json"), serde_json::to_string_pretty(&errors).unwrap()).map_err(|e| e.to_string())?;
    eprintln!("done in {:.0}s, {} failed images", started.elapsed().as_secs_f64(), errors.len());
    Ok(())
}

fn load_done(path: &Path) -> Result<HashSet<(String, ImageFormat, u32)>, String> {
    let mut done = HashSet::new();
    if let Ok(file) = File::open(path) {
        for line in BufReader::new(file).lines() {
            let line = line.map_err(|e| e.to_string())?;
            // A partially written last line from an interrupted run is ignored and redone.
            if let Ok(row) = serde_json::from_str::<Row>(&line) {
                done.insert((row.image, row.format, row.level));
            }
        }
    }
    Ok(done)
}

#[allow(clippy::too_many_arguments)]
fn sweep_image(
    entry: &ManifestEntry,
    pipeline: &ImageRsPipeline,
    policy: DimensionPolicy,
    codecs: &[Box<dyn Codec>],
    metrics: &[Box<dyn Metric>],
    step: u32,
    done: &HashSet<(String, ImageFormat, u32)>,
    writer: &Mutex<BufWriter<File>>,
) -> Result<usize, String> {
    let input = fs::read(&entry.path).map_err(|e| e.to_string())?;
    let source = pipeline.decode(&input).map_err(|e| e.to_string())?;
    let source_dims = source.image.dimensions();
    let dims = policy.target(source_dims);
    let reference = if dims != source_dims {
        pipeline.resize(&source.image, dims).map_err(|e| e.to_string())?
    } else {
        source.image
    };
    let has_alpha = reference.has_transparency();

    let mut rows = Vec::new();
    for codec in codecs {
        let axis = codec.quality_axis();
        for level in (axis.min..=axis.max).step_by(step as usize) {
            if done.contains(&(entry.id.clone(), codec.format(), level)) {
                continue;
            }
            let settings = EncodeSettings { level, metadata: MetadataPolicy::KeepIccOnly };
            let t = Instant::now();
            let encoded = codec.encode(&reference, &settings).map_err(|e| e.to_string())?;
            let encode_ms = ms(t);
            let decoded = codec.decode(&encoded.bytes).map_err(|e| e.to_string())?;

            let mut scores = std::collections::BTreeMap::new();
            let mut metric_ms = std::collections::BTreeMap::new();
            for metric in metrics {
                let t = Instant::now();
                let result = metric.measure(&reference, &decoded).map_err(|e| e.to_string())?;
                metric_ms.insert(result.metric.name.clone(), ms(t));
                scores.insert(result.metric.name, result.score);
            }
            rows.push(Row {
                image: entry.id.clone(),
                corpus: entry.corpus.clone(),
                split: entry.split.clone(),
                format: codec.format(),
                level,
                bytes: encoded.bytes.len() as u64,
                width: dims.width,
                height: dims.height,
                source_width: source_dims.width,
                source_height: source_dims.height,
                has_alpha,
                encode_ms,
                metrics: scores,
                metric_ms,
            });
        }
    }

    // One locked write per image keeps rows of an image contiguous and the file consistent.
    let mut w = writer.lock().unwrap();
    for row in &rows {
        serde_json::to_writer(&mut *w, row).map_err(|e| e.to_string())?;
        w.write_all(b"\n").map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(rows.len())
}
