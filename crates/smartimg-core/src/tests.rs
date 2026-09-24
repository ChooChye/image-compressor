//! Regression tests for the optimizer loop using deterministic mocks.
//!
//! The mock codec encodes the level into its output; the mock metric reads it back and
//! maps it through a quality curve. That lets each test define exact size and quality
//! curves and assert precisely which candidate the search must pick.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use smartimg_codecs::{Capabilities, CodecError, Encoded, QualityAxis};
use smartimg_metrics::MetricError;
use smartimg_pipeline::DecodedSource;
use smartimg_types::{
    DimensionPolicy, Dimensions, Direction, MetricId, PixelFormat, SearchBudget,
};

use super::*;

struct MockPipeline {
    image: Image,
    format: Option<ImageFormat>,
    has_strippable_metadata: bool,
}

impl MockPipeline {
    fn opaque() -> Self {
        Self::with(Image::new(Dimensions::new(40, 30), PixelFormat::Rgb8, vec![128; 40 * 30 * 3]).unwrap())
    }

    fn transparent() -> Self {
        let mut data = vec![128; 40 * 30 * 4];
        data[3] = 0;
        Self::with(Image::new(Dimensions::new(40, 30), PixelFormat::Rgba8, data).unwrap())
    }

    fn with(image: Image) -> Self {
        Self { image, format: Some(ImageFormat::Png), has_strippable_metadata: false }
    }
}

impl ImagePipeline for MockPipeline {
    fn decode(&self, _input: &[u8]) -> Result<DecodedSource, PipelineError> {
        Ok(DecodedSource {
            image: self.image.clone(),
            format: self.format,
            has_strippable_metadata: self.has_strippable_metadata,
        })
    }

    fn resize(&self, image: &Image, dimensions: Dimensions) -> Result<Image, PipelineError> {
        let channels = image.format().channels();
        Ok(Image::new(dimensions, image.format(), vec![128; dimensions.pixels() as usize * channels]).unwrap())
    }
}

/// Output is `[level, width(4), height(4), padding...]`, `size(level)` bytes long.
struct MockCodec {
    format: ImageFormat,
    alpha: bool,
    size: fn(u32) -> usize,
    fail: bool,
    /// Decoded image is this much wider than the input, to simulate a broken codec.
    decode_width_error: u32,
    encodes: Arc<AtomicU32>,
}

impl MockCodec {
    fn new(format: ImageFormat, size: fn(u32) -> usize) -> Self {
        Self { format, alpha: true, size, fail: false, decode_width_error: 0, encodes: Arc::default() }
    }
}

impl Codec for MockCodec {
    fn id(&self) -> smartimg_types::CodecId {
        smartimg_types::CodecId::new(format!("mock-{}", self.format), "0")
    }
    fn format(&self) -> ImageFormat {
        self.format
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities { alpha: self.alpha }
    }
    fn quality_axis(&self) -> QualityAxis {
        QualityAxis { min: 0, max: 100 }
    }
    fn encode(&self, image: &Image, settings: &EncodeSettings) -> Result<Encoded, CodecError> {
        if self.fail {
            return Err(CodecError::Encode("mock failure".into()));
        }
        self.encodes.fetch_add(1, Ordering::Relaxed);
        let d = image.dimensions();
        let mut bytes = vec![settings.level as u8];
        bytes.extend((d.width + self.decode_width_error).to_le_bytes());
        bytes.extend(d.height.to_le_bytes());
        bytes.resize((self.size)(settings.level).max(bytes.len()), 0);
        let parameters = [("level".to_string(), settings.level.to_string())].into();
        Ok(Encoded { bytes, parameters })
    }
    fn decode(&self, encoded: &[u8]) -> Result<Image, CodecError> {
        let w = u32::from_le_bytes(encoded[1..5].try_into().unwrap());
        let h = u32::from_le_bytes(encoded[5..9].try_into().unwrap());
        let format = if self.alpha { PixelFormat::Rgba8 } else { PixelFormat::Rgb8 };
        let data = vec![encoded[0]; (w * h) as usize * format.channels()];
        Image::new(Dimensions::new(w, h), format, data).map_err(|e| CodecError::Decode(e.to_string()))
    }
}

/// Reads the level back from the decoded pixels and maps it through `curve`.
struct MockMetric {
    name: &'static str,
    direction: Direction,
    curve: fn(u32) -> f64,
}

impl Metric for MockMetric {
    fn id(&self) -> MetricId {
        MetricId::new(self.name, "0")
    }
    fn direction(&self) -> Direction {
        self.direction
    }
    fn score(&self, _reference: &Image, candidate: &Image) -> Result<f64, MetricError> {
        Ok((self.curve)(u32::from(candidate.data()[0])))
    }
}

fn quality() -> MockMetric {
    MockMetric { name: "quality", direction: Direction::HigherIsBetter, curve: |l| f64::from(l) / 100.0 }
}

fn request(formats: &[ImageFormat], threshold: f64) -> OptimizationRequest {
    OptimizationRequest {
        allowed_formats: formats.to_vec(),
        quality: vec![QualityGate { metric: "quality".into(), threshold }],
        dimensions: DimensionPolicy::Preserve,
        alpha: AlphaPolicy::Preserve,
        metadata: Default::default(),
        budget: SearchBudget { max_evaluations_per_format: 32, sweep_radius: 2 },
        trace: false,
    }
}

// Large enough that no candidate ever falls under it by accident.
const INPUT: &[u8] = &[0; 1 << 20];

fn encoded_best(result: &OptimizationResult) -> &Evaluation {
    match &result.selection {
        Selection::Encoded { best } => best,
        other => panic!("expected an encoded winner, got {other:?}"),
    }
}

fn outcome(result: &[FormatReport], format: ImageFormat) -> &FormatOutcome {
    &result.iter().find(|r| r.format == format).unwrap().outcome
}

#[test]
fn finds_lowest_passing_level() {
    let engine = Engine::new(MockPipeline::opaque())
        .with_codec(MockCodec::new(ImageFormat::WebP, |l| 100 + 10 * l as usize))
        .with_metric(quality());
    let result = engine.optimize(INPUT, &request(&[ImageFormat::WebP], 0.5)).unwrap();
    let best = encoded_best(&result);
    assert_eq!(best.level, 50);
    assert_eq!(best.bytes, 600);
    assert_eq!(result.output.len(), 600);
    assert_eq!(best.encoder_parameters["level"], "50");
}

#[test]
fn smallest_format_wins_and_every_format_is_reported() {
    let engine = Engine::new(MockPipeline::opaque())
        .with_codec(MockCodec::new(ImageFormat::Jpeg, |l| 200 + 10 * l as usize))
        .with_codec(MockCodec::new(ImageFormat::Avif, |l| 50 + 10 * l as usize))
        .with_metric(quality());
    let result = engine.optimize(INPUT, &request(&[ImageFormat::Jpeg, ImageFormat::Avif], 0.5)).unwrap();
    assert_eq!(encoded_best(&result).format, ImageFormat::Avif);
    assert!(matches!(outcome(&result.formats, ImageFormat::Jpeg), FormatOutcome::Found { best } if best.bytes == 700));
    assert!(matches!(outcome(&result.formats, ImageFormat::Avif), FormatOutcome::Found { best } if best.bytes == 550));
}

#[test]
fn sweep_catches_non_monotonic_size() {
    // Level 52 is smaller than the boundary level 50 while still passing.
    let engine = Engine::new(MockPipeline::opaque())
        .with_codec(MockCodec::new(ImageFormat::WebP, |l| if l == 52 { 300 } else { 100 + 10 * l as usize }))
        .with_metric(quality());
    let result = engine.optimize(INPUT, &request(&[ImageFormat::WebP], 0.5)).unwrap();
    assert_eq!(encoded_best(&result).level, 52);
    assert_eq!(encoded_best(&result).bytes, 300);
}

#[test]
fn every_gate_must_pass() {
    // Secondary lower-is-better metric: error = (100 - level) / 100 must be <= 0.3.
    let engine = Engine::new(MockPipeline::opaque())
        .with_codec(MockCodec::new(ImageFormat::WebP, |l| 100 + l as usize))
        .with_metric(quality())
        .with_metric(MockMetric {
            name: "error",
            direction: Direction::LowerIsBetter,
            curve: |l| f64::from(100 - l) / 100.0,
        });
    let mut req = request(&[ImageFormat::WebP], 0.5);
    req.quality.push(QualityGate { metric: "error".into(), threshold: 0.3 });
    let best = engine.optimize(INPUT, &req).map(|r| encoded_best(&r).clone()).unwrap();
    assert_eq!(best.level, 70);
    assert_eq!(best.metrics.len(), 2);
}

#[test]
fn transparency_is_never_silently_dropped() {
    let mut jpeg = MockCodec::new(ImageFormat::Jpeg, |l| 10 + l as usize);
    jpeg.alpha = false;
    let engine = Engine::new(MockPipeline::transparent())
        .with_codec(jpeg)
        .with_codec(MockCodec::new(ImageFormat::Png, |l| 1000 + l as usize))
        .with_metric(quality());

    let result = engine.optimize(INPUT, &request(&[ImageFormat::Jpeg, ImageFormat::Png], 0.5)).unwrap();
    assert_eq!(encoded_best(&result).format, ImageFormat::Png);
    assert!(!result.alpha_flattened);
    assert!(matches!(outcome(&result.formats, ImageFormat::Jpeg), FormatOutcome::Skipped { .. }));

    let mut req = request(&[ImageFormat::Jpeg, ImageFormat::Png], 0.5);
    req.alpha = AlphaPolicy::FlattenOnto { background: [255, 255, 255] };
    let result = engine.optimize(INPUT, &req).unwrap();
    assert_eq!(encoded_best(&result).format, ImageFormat::Jpeg);
    assert!(result.alpha_flattened);
}

#[test]
fn failing_codec_does_not_abort_other_formats() {
    let mut broken = MockCodec::new(ImageFormat::Avif, |l| l as usize);
    broken.fail = true;
    let mut wrong_size = MockCodec::new(ImageFormat::Jpeg, |l| l as usize);
    wrong_size.decode_width_error = 1;
    let engine = Engine::new(MockPipeline::opaque())
        .with_codec(broken)
        .with_codec(wrong_size)
        .with_codec(MockCodec::new(ImageFormat::WebP, |l| 100 + l as usize))
        .with_metric(quality());
    let result = engine
        .optimize(INPUT, &request(&[ImageFormat::Avif, ImageFormat::Jpeg, ImageFormat::WebP], 0.5))
        .unwrap();
    assert_eq!(encoded_best(&result).format, ImageFormat::WebP);
    assert!(matches!(outcome(&result.formats, ImageFormat::Avif), FormatOutcome::Failed { .. }));
    assert!(matches!(outcome(&result.formats, ImageFormat::Jpeg), FormatOutcome::Failed { .. }));
}

#[test]
fn unreachable_target_reports_every_format() {
    let engine = Engine::new(MockPipeline::opaque())
        .with_codec(MockCodec::new(ImageFormat::WebP, |l| l as usize))
        .with_metric(quality());
    match engine.optimize(INPUT, &request(&[ImageFormat::WebP, ImageFormat::Avif], 1.5)) {
        Err(EngineError::NoAcceptableCandidate { formats }) => {
            assert!(matches!(outcome(&formats, ImageFormat::WebP), FormatOutcome::NoAcceptableCandidate));
            assert!(matches!(outcome(&formats, ImageFormat::Avif), FormatOutcome::Skipped { .. }));
        }
        other => panic!("expected NoAcceptableCandidate, got {other:?}"),
    }
}

#[test]
fn keeps_original_only_when_it_is_a_valid_answer() {
    let small_input = [0u8; 100];
    let engine = |pipeline| {
        Engine::new(pipeline)
            .with_codec(MockCodec::new(ImageFormat::Png, |l| 500 + l as usize))
            .with_metric(quality())
    };

    let result = engine(MockPipeline::opaque()).optimize(&small_input, &request(&[ImageFormat::Png], 0.5)).unwrap();
    assert_eq!(result.selection, Selection::KeptOriginal { format: ImageFormat::Png });
    assert_eq!(result.output, small_input);

    // Original format not allowed → must encode.
    let mut pipeline = MockPipeline::opaque();
    pipeline.format = Some(ImageFormat::Jpeg);
    let result = engine(pipeline).optimize(&small_input, &request(&[ImageFormat::Png], 0.5)).unwrap();
    assert!(matches!(result.selection, Selection::Encoded { .. }));

    // Metadata must be stripped → must re-encode.
    let mut pipeline = MockPipeline::opaque();
    pipeline.has_strippable_metadata = true;
    let result = engine(pipeline).optimize(&small_input, &request(&[ImageFormat::Png], 0.5)).unwrap();
    assert!(matches!(result.selection, Selection::Encoded { .. }));

    // Resized → original dimensions are no longer valid.
    let mut req = request(&[ImageFormat::Png], 0.5);
    req.dimensions = DimensionPolicy::MaxDimension { max: 20 };
    let result = engine(MockPipeline::opaque()).optimize(&small_input, &req).unwrap();
    assert!(matches!(result.selection, Selection::Encoded { .. }));
    assert_eq!(result.output_dimensions, Dimensions::new(20, 15));
}

#[test]
fn budget_caps_evaluations() {
    let codec = MockCodec::new(ImageFormat::WebP, |l| 100 + l as usize);
    let engine = Engine::new(MockPipeline::opaque()).with_codec(codec).with_metric(quality());
    let mut req = request(&[ImageFormat::WebP], 0.5);
    req.budget.max_evaluations_per_format = 3;
    req.trace = true;
    let result = engine.optimize(INPUT, &req).unwrap();
    assert_eq!(result.formats[0].evaluations, 3);
    assert_eq!(result.trace.len(), 3);
}

#[test]
fn search_is_deterministic_and_memoized() {
    let codec = MockCodec::new(ImageFormat::WebP, |l| 100 + 3 * l as usize);
    let encodes = Arc::clone(&codec.encodes);
    let engine = Engine::new(MockPipeline::opaque()).with_codec(codec).with_metric(quality());
    let mut req = request(&[ImageFormat::WebP], 0.37);
    req.trace = true;

    let a = engine.optimize(INPUT, &req).unwrap();
    // The sweep revisits levels the binary search already tried; none may be re-encoded.
    assert_eq!(encodes.load(Ordering::Relaxed), a.formats[0].evaluations);

    let b = engine.optimize(INPUT, &req).unwrap();
    assert_eq!(a.trace, b.trace);
    assert_eq!(a.selection, b.selection);
}

#[test]
fn rejects_invalid_requests() {
    let engine = Engine::new(MockPipeline::opaque())
        .with_codec(MockCodec::new(ImageFormat::WebP, |l| l as usize))
        .with_metric(quality());
    let mut req = request(&[ImageFormat::WebP], 0.5);
    req.quality[0].metric = "butteraugli".into();
    assert!(matches!(engine.optimize(INPUT, &req), Err(EngineError::InvalidRequest(_))));
    assert!(matches!(engine.optimize(INPUT, &request(&[], 0.5)), Err(EngineError::InvalidRequest(_))));
}

mod budget {
    use super::*;
    use crate::budget::{BudgetOptions, encode_within_budget};

    fn image() -> Image {
        Image::new(Dimensions::new(100, 50), PixelFormat::Rgb8, vec![128; 100 * 50 * 3]).unwrap()
    }

    // Size grows with level and with pixel count: bytes = level * pixels / 100.
    fn codec() -> MockCodec {
        MockCodec::new(ImageFormat::Avif, |l| l as usize * 50)
    }

    #[test]
    fn no_budget_is_a_single_encode_at_the_requested_level() {
        let r = encode_within_budget(&MockPipeline::opaque(), &codec(), &image(), &BudgetOptions::default()).unwrap();
        assert_eq!((r.level, r.attempts, r.within_budget), (60, 1, true));
        assert_eq!(r.dimensions, Dimensions::new(100, 50));
    }

    #[test]
    fn lowers_level_until_it_fits() {
        let opts = BudgetOptions { max_bytes: Some(2_600), ..Default::default() };
        let r = encode_within_budget(&MockPipeline::opaque(), &codec(), &image(), &opts).unwrap();
        // 60 -> 3000, 55 -> 2750, 50 -> 2500 fits.
        assert_eq!((r.level, r.attempts, r.within_budget), (50, 3, true));
        assert_eq!(r.encoded.bytes.len(), 2_500);
    }

    #[test]
    fn downscales_only_after_reaching_the_floor() {
        // Mock size depends on level only, so the resized image never fits either:
        // the result is the smallest attempt, flagged as over budget.
        let opts = BudgetOptions { max_bytes: Some(100), max_downscales: 2, ..Default::default() };
        let r = encode_within_budget(&MockPipeline::opaque(), &codec(), &image(), &opts).unwrap();
        assert!(!r.within_budget);
        assert_eq!(r.level, 40);
        // 5 levels (60..=40) at each of 3 scales.
        assert_eq!(r.attempts, 15);
    }

    #[test]
    fn downscaling_rescues_an_oversized_image() {
        // Size proportional to pixels: 100x50 at level 40 is 2000; 85x43 is ~1462.
        let codec = MockCodec::new(ImageFormat::Avif, |l| l as usize * 50);
        struct PixelSized(MockCodec);
        impl Codec for PixelSized {
            fn id(&self) -> smartimg_types::CodecId { self.0.id() }
            fn format(&self) -> ImageFormat { self.0.format() }
            fn capabilities(&self) -> Capabilities { self.0.capabilities() }
            fn quality_axis(&self) -> QualityAxis { self.0.quality_axis() }
            fn encode(&self, image: &Image, s: &EncodeSettings) -> Result<Encoded, CodecError> {
                let mut e = self.0.encode(image, s)?;
                e.bytes.resize(s.level as usize * image.dimensions().pixels() as usize / 100, 0);
                Ok(e)
            }
            fn decode(&self, b: &[u8]) -> Result<Image, CodecError> { self.0.decode(b) }
        }
        let opts = BudgetOptions { max_bytes: Some(1_500), ..Default::default() };
        let r = encode_within_budget(&MockPipeline::opaque(), &PixelSized(codec), &image(), &opts).unwrap();
        assert!(r.within_budget);
        assert_eq!(r.dimensions, Dimensions::new(85, 43));
        assert!(r.encoded.bytes.len() <= 1_500);
    }
}
