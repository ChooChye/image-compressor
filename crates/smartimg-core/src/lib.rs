//! The smartimg optimization engine.
//!
//! The optimizer owns the loop: choose a level → encode → decode → measure → decide.
//! Pipeline, codecs, and metrics are injected, so this crate never depends on HTTP, queues,
//! storage, or a specific native library.

use std::collections::BTreeMap;
use std::time::Instant;

use smartimg_codecs::{Codec, CodecError, EncodeSettings};
use smartimg_metrics::Metric;
use smartimg_pipeline::{ImagePipeline, PipelineError};
use smartimg_types::{
    AlphaPolicy, Evaluation, FormatOutcome, FormatReport, Image, ImageFormat, OptimizationRequest,
    OptimizationResult, QualityGate, Selection,
};
use thiserror::Error;

pub mod budget;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("pipeline failed: {0}")]
    Pipeline(#[from] PipelineError),
    #[error("codec failed: {0}")]
    Codec(#[from] CodecError),
    #[error("no candidate met the quality target in any allowed format")]
    NoAcceptableCandidate { formats: Vec<FormatReport> },
}

pub struct Engine<P> {
    pub pipeline: P,
    pub codecs: Vec<Box<dyn Codec>>,
    pub metrics: Vec<Box<dyn Metric>>,
}

impl<P: ImagePipeline> Engine<P> {
    pub fn new(pipeline: P) -> Self {
        Self { pipeline, codecs: Vec::new(), metrics: Vec::new() }
    }

    pub fn with_codec(mut self, codec: impl Codec + 'static) -> Self {
        self.codecs.push(Box::new(codec));
        self
    }

    pub fn with_metric(mut self, metric: impl Metric + 'static) -> Self {
        self.metrics.push(Box::new(metric));
        self
    }

    pub fn optimize(&self, input: &[u8], request: &OptimizationRequest) -> Result<OptimizationResult, EngineError> {
        let started = Instant::now();
        let gates = self.resolve_gates(request)?;

        let source = self.pipeline.decode(input)?;
        let source_dimensions = source.image.dimensions();
        let output_dimensions = request.dimensions.target(source_dimensions);
        let resized = output_dimensions != source_dimensions;
        let reference = if resized {
            self.pipeline.resize(&source.image, output_dimensions)?
        } else {
            source.image
        };
        let transparent = reference.has_transparency();

        let mut reports = Vec::new();
        let mut trace = Vec::new();
        let mut winners: Vec<(Evaluation, Vec<u8>, bool)> = Vec::new();

        for format in dedup(&request.allowed_formats) {
            let Some(codec) = self.codecs.iter().find(|c| c.format() == format) else {
                reports.push(FormatReport {
                    format,
                    codec: None,
                    evaluations: 0,
                    outcome: FormatOutcome::Skipped { reason: "no codec registered".into() },
                });
                continue;
            };

            // Transparency must never disappear silently.
            let flattened;
            let (work_image, alpha_flattened) = match (transparent && !codec.capabilities().alpha, request.alpha) {
                (false, _) => (&reference, false),
                (true, AlphaPolicy::Preserve) => {
                    reports.push(FormatReport {
                        format,
                        codec: Some(codec.id()),
                        evaluations: 0,
                        outcome: FormatOutcome::Skipped { reason: "format cannot preserve transparency".into() },
                    });
                    continue;
                }
                (true, AlphaPolicy::FlattenOnto { background }) => {
                    flattened = reference.flatten(background);
                    (&flattened, true)
                }
            };

            let mut search = FormatSearch {
                codec: codec.as_ref(),
                image: work_image,
                gates: &gates,
                request,
                samples: BTreeMap::new(),
            };
            let outcome = search.run();
            let evaluations = search.samples.len() as u32;
            if request.trace {
                trace.extend(search.samples.values().map(|(e, _)| e.clone()));
            }

            let outcome = match outcome {
                Ok(Some(level)) => {
                    let (best, bytes) = search.samples.remove(&level).expect("winner was sampled");
                    winners.push((best.clone(), bytes, alpha_flattened));
                    FormatOutcome::Found { best }
                }
                Ok(None) => FormatOutcome::NoAcceptableCandidate,
                Err(error) => FormatOutcome::Failed { error },
            };
            reports.push(FormatReport { format, codec: Some(codec.id()), evaluations, outcome });
        }

        // Smallest acceptable candidate wins; ties go to the higher quality level.
        let Some((best, bytes, alpha_flattened)) = winners
            .into_iter()
            .min_by_key(|(e, _, _)| (e.bytes, std::cmp::Reverse(e.level)))
        else {
            return Err(EngineError::NoAcceptableCandidate { formats: reports });
        };

        // The original is itself a valid answer when nothing about it has to change.
        let keep_original = !resized
            && !source.has_strippable_metadata
            && best.bytes >= input.len() as u64
            && source.format.is_some_and(|f| request.allowed_formats.contains(&f));

        let (selection, output, alpha_flattened) = if keep_original {
            let format = source.format.expect("checked above");
            (Selection::KeptOriginal { format }, input.to_vec(), false)
        } else {
            (Selection::Encoded { best }, bytes, alpha_flattened)
        };

        Ok(OptimizationResult {
            selection,
            source_bytes: input.len() as u64,
            output_bytes: output.len() as u64,
            source_dimensions,
            output_dimensions,
            alpha_flattened,
            formats: reports,
            trace,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            output,
        })
    }

    fn resolve_gates<'a>(&'a self, request: &'a OptimizationRequest) -> Result<Vec<Gate<'a>>, EngineError> {
        if request.allowed_formats.is_empty() {
            return Err(EngineError::InvalidRequest("no output formats allowed".into()));
        }
        if request.quality.is_empty() {
            return Err(EngineError::InvalidRequest("at least one quality gate is required".into()));
        }
        request
            .quality
            .iter()
            .map(|gate| {
                self.metrics
                    .iter()
                    .find(|m| m.id().name == gate.metric)
                    .map(|metric| Gate { metric: metric.as_ref(), spec: gate })
                    .ok_or_else(|| EngineError::InvalidRequest(format!("unknown metric `{}`", gate.metric)))
            })
            .collect()
    }
}

struct Gate<'a> {
    metric: &'a dyn Metric,
    spec: &'a QualityGate,
}

/// Search state for one format. Every evaluated level is memoized, so the binary search
/// and the local sweep never encode the same level twice.
struct FormatSearch<'a> {
    codec: &'a dyn Codec,
    image: &'a Image,
    gates: &'a [Gate<'a>],
    request: &'a OptimizationRequest,
    samples: BTreeMap<u32, (Evaluation, Vec<u8>)>,
}

impl FormatSearch<'_> {
    /// Returns the level of the smallest accepted candidate.
    fn run(&mut self) -> Result<Option<u32>, String> {
        let axis = self.codec.quality_axis();
        let budget = self.request.budget;

        // Binary search for the lowest level that passes every gate.
        let (mut lo, mut hi) = (i64::from(axis.min), i64::from(axis.max));
        let mut boundary = None;
        while lo <= hi && self.has_budget() {
            let mid = (lo + hi) / 2;
            if self.sample(mid as u32)?.accepted {
                boundary = Some(mid as u32);
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        let Some(boundary) = boundary else {
            return Ok(None);
        };

        // Byte size is not strictly monotonic in the level for every encoder, so check the
        // neighborhood of the boundary for a smaller passing candidate.
        let from = boundary.saturating_sub(budget.sweep_radius).max(axis.min);
        let to = boundary.saturating_add(budget.sweep_radius).min(axis.max);
        for level in from..=to {
            if !self.has_budget() {
                break;
            }
            self.sample(level)?;
        }

        Ok(self
            .samples
            .values()
            .filter(|(e, _)| e.accepted)
            .min_by_key(|(e, _)| (e.bytes, std::cmp::Reverse(e.level)))
            .map(|(e, _)| e.level))
    }

    fn has_budget(&self) -> bool {
        (self.samples.len() as u32) < self.request.budget.max_evaluations_per_format
    }

    fn sample(&mut self, level: u32) -> Result<&Evaluation, String> {
        if !self.samples.contains_key(&level) {
            let evaluation = self.evaluate(level)?;
            self.samples.insert(level, evaluation);
        }
        Ok(&self.samples[&level].0)
    }

    fn evaluate(&self, level: u32) -> Result<(Evaluation, Vec<u8>), String> {
        let settings = EncodeSettings { level, metadata: self.request.metadata };
        let encoded = self.codec.encode(self.image, &settings).map_err(|e| e.to_string())?;
        let decoded = self.codec.decode(&encoded.bytes).map_err(|e| e.to_string())?;

        let mut metrics = Vec::with_capacity(self.gates.len());
        let mut accepted = true;
        for gate in self.gates {
            let result = gate.metric.measure(self.image, &decoded).map_err(|e| e.to_string())?;
            accepted &= result.direction.meets(result.score, gate.spec.threshold);
            metrics.push(result);
        }

        let evaluation = Evaluation {
            format: self.codec.format(),
            level,
            bytes: encoded.bytes.len() as u64,
            metrics,
            accepted,
            encoder_parameters: encoded.parameters,
        };
        Ok((evaluation, encoded.bytes))
    }
}

fn dedup(formats: &[ImageFormat]) -> Vec<ImageFormat> {
    let mut seen = Vec::with_capacity(formats.len());
    for &f in formats {
        if !seen.contains(&f) {
            seen.push(f);
        }
    }
    seen
}

#[cfg(test)]
mod tests;
