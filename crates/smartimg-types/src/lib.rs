//! Shared domain types for the smartimg engine.
//!
//! Nothing in this crate performs I/O. Every other crate speaks in these types so pixel
//! layout, color, and request semantics are defined in exactly one place.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    Jpeg,
    WebP,
    Avif,
    Png,
}

impl ImageFormat {
    pub fn extension(self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "jpg",
            ImageFormat::WebP => "webp",
            ImageFormat::Avif => "avif",
            ImageFormat::Png => "png",
        }
    }

    pub fn mime_type(self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "image/jpeg",
            ImageFormat::WebP => "image/webp",
            ImageFormat::Avif => "image/avif",
            ImageFormat::Png => "image/png",
        }
    }
}

impl fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.extension())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Dimensions {
    pub width: u32,
    pub height: u32,
}

impl Dimensions {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn pixels(self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// Interleaved 8-bit sRGB-encoded samples. Wider formats (16-bit, HDR) are added when a
/// pipeline can produce them; until then they must be converted before entering the core.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb8,
    Rgba8,
}

impl PixelFormat {
    pub fn channels(self) -> usize {
        match self {
            PixelFormat::Rgb8 => 3,
            PixelFormat::Rgba8 => 4,
        }
    }

    pub fn has_alpha(self) -> bool {
        matches!(self, PixelFormat::Rgba8)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageError {
    ZeroDimensions,
    BufferSize { expected: usize, actual: usize },
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageError::ZeroDimensions => f.write_str("image dimensions must be non-zero"),
            ImageError::BufferSize { expected, actual } => {
                write!(f, "pixel buffer is {actual} bytes, expected {expected}")
            }
        }
    }
}

impl std::error::Error for ImageError {}

/// A decoded raster image. Rows are tightly packed (stride = width × channels).
///
/// The ICC profile travels with the pixels so encoders can embed it; dropping it silently
/// shifts colors of wide-gamut sources.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    dimensions: Dimensions,
    format: PixelFormat,
    data: Vec<u8>,
    icc_profile: Option<Vec<u8>>,
}

impl Image {
    pub fn new(dimensions: Dimensions, format: PixelFormat, data: Vec<u8>) -> Result<Self, ImageError> {
        if dimensions.width == 0 || dimensions.height == 0 {
            return Err(ImageError::ZeroDimensions);
        }
        let expected = dimensions.pixels() as usize * format.channels();
        if data.len() != expected {
            return Err(ImageError::BufferSize { expected, actual: data.len() });
        }
        Ok(Self { dimensions, format, data, icc_profile: None })
    }

    pub fn with_icc_profile(mut self, icc_profile: Option<Vec<u8>>) -> Self {
        self.icc_profile = icc_profile;
        self
    }

    pub fn dimensions(&self) -> Dimensions {
        self.dimensions
    }

    pub fn format(&self) -> PixelFormat {
        self.format
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.icc_profile.as_deref()
    }

    pub fn stride(&self) -> usize {
        self.dimensions.width as usize * self.format.channels()
    }

    /// True if the image has an alpha channel with at least one non-opaque pixel.
    pub fn has_transparency(&self) -> bool {
        self.format.has_alpha() && self.data.chunks_exact(4).any(|px| px[3] != 255)
    }

    /// Composite onto an opaque background, producing an RGB image.
    pub fn flatten(&self, background: [u8; 3]) -> Image {
        let data = match self.format {
            PixelFormat::Rgb8 => self.data.clone(),
            PixelFormat::Rgba8 => self
                .data
                .chunks_exact(4)
                .flat_map(|px| {
                    let a = u32::from(px[3]);
                    let blend = |c: u8, bg: u8| ((u32::from(c) * a + u32::from(bg) * (255 - a) + 127) / 255) as u8;
                    [blend(px[0], background[0]), blend(px[1], background[1]), blend(px[2], background[2])]
                })
                .collect(),
        };
        Image {
            dimensions: self.dimensions,
            format: PixelFormat::Rgb8,
            data,
            icc_profile: self.icc_profile.clone(),
        }
    }
}

/// How output dimensions are derived from the source. Resizing is always explicit and
/// never upscales.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub enum DimensionPolicy {
    #[default]
    Preserve,
    /// Longest side at most this many pixels.
    MaxDimension { max: u32 },
    /// Fit inside a box, preserving aspect ratio.
    MaxBox { width: u32, height: u32 },
    MaxMegapixels { megapixels: f64 },
}

impl DimensionPolicy {
    pub fn target(self, source: Dimensions) -> Dimensions {
        let (w, h) = (f64::from(source.width), f64::from(source.height));
        let scale = match self {
            DimensionPolicy::Preserve => 1.0,
            DimensionPolicy::MaxDimension { max } => f64::from(max) / w.max(h),
            DimensionPolicy::MaxBox { width, height } => (f64::from(width) / w).min(f64::from(height) / h),
            DimensionPolicy::MaxMegapixels { megapixels } => (megapixels * 1_000_000.0 / (w * h)).sqrt(),
        };
        if !(scale < 1.0) {
            return source;
        }
        let scaled = |v: f64| ((v * scale).round() as u32).max(1);
        Dimensions::new(scaled(w), scaled(h))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub enum AlphaPolicy {
    /// Transparency must survive. Formats without alpha are skipped for transparent images.
    #[default]
    Preserve,
    /// Transparent images may be composited onto this background for alpha-less formats.
    FlattenOnto { background: [u8; 3] },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum MetadataPolicy {
    /// Keep only the ICC profile; drop EXIF/XMP (removes GPS and other private data).
    #[default]
    KeepIccOnly,
    StripAll,
}

/// A candidate is acceptable only when every gate passes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QualityGate {
    /// Must match a registered metric's [`MetricId::name`].
    pub metric: String,
    pub threshold: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchBudget {
    /// Maximum encode+measure evaluations per format.
    pub max_evaluations_per_format: u32,
    /// Levels checked on each side of the binary-search boundary.
    pub sweep_radius: u32,
}

impl Default for SearchBudget {
    fn default() -> Self {
        Self { max_evaluations_per_format: 16, sweep_radius: 2 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OptimizationRequest {
    pub allowed_formats: Vec<ImageFormat>,
    pub quality: Vec<QualityGate>,
    #[serde(default)]
    pub dimensions: DimensionPolicy,
    #[serde(default)]
    pub alpha: AlphaPolicy,
    #[serde(default)]
    pub metadata: MetadataPolicy,
    #[serde(default)]
    pub budget: SearchBudget,
    /// Include every evaluated candidate in the result.
    #[serde(default)]
    pub trace: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Direction {
    HigherIsBetter,
    LowerIsBetter,
}

impl Direction {
    pub fn meets(self, score: f64, threshold: f64) -> bool {
        match self {
            Direction::HigherIsBetter => score >= threshold,
            Direction::LowerIsBetter => score <= threshold,
        }
    }
}

/// Identifies an implementation precisely enough that benchmark results are only ever
/// compared across identical versions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ComponentId {
    pub name: String,
    pub version: String,
}

impl ComponentId {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self { name: name.into(), version: version.into() }
    }
}

pub type MetricId = ComponentId;
pub type CodecId = ComponentId;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricResult {
    pub metric: MetricId,
    pub direction: Direction,
    pub score: f64,
}

/// Native encoder settings recorded for reports, e.g. `{"quality": "72", "chroma": "420"}`.
pub type EncoderParameters = BTreeMap<String, String>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Evaluation {
    pub format: ImageFormat,
    pub level: u32,
    pub bytes: u64,
    pub metrics: Vec<MetricResult>,
    pub accepted: bool,
    pub encoder_parameters: EncoderParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum FormatOutcome {
    Found { best: Evaluation },
    NoAcceptableCandidate,
    Skipped { reason: String },
    Failed { error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FormatReport {
    pub format: ImageFormat,
    pub codec: Option<CodecId>,
    pub evaluations: u32,
    pub outcome: FormatOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Selection {
    /// A newly encoded candidate won.
    Encoded { best: Evaluation },
    /// The source was already the smallest valid answer (same dimensions, allowed format).
    KeptOriginal { format: ImageFormat },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OptimizationResult {
    pub selection: Selection,
    pub source_bytes: u64,
    pub output_bytes: u64,
    pub source_dimensions: Dimensions,
    pub output_dimensions: Dimensions,
    pub alpha_flattened: bool,
    /// Per-format winners, usable as `<picture>` fallbacks.
    pub formats: Vec<FormatReport>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub trace: Vec<Evaluation>,
    pub elapsed_ms: f64,
    /// Encoded output. Not serialized into reports.
    #[serde(skip)]
    pub output: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_rejects_wrong_buffer_size() {
        let err = Image::new(Dimensions::new(2, 2), PixelFormat::Rgb8, vec![0; 11]).unwrap_err();
        assert_eq!(err, ImageError::BufferSize { expected: 12, actual: 11 });
        assert_eq!(
            Image::new(Dimensions::new(0, 2), PixelFormat::Rgb8, vec![]).unwrap_err(),
            ImageError::ZeroDimensions
        );
    }

    #[test]
    fn flatten_blends_alpha() {
        let img = Image::new(Dimensions::new(2, 1), PixelFormat::Rgba8, vec![255, 0, 0, 255, 0, 0, 0, 0]).unwrap();
        let flat = img.flatten([255, 255, 255]);
        assert_eq!(flat.format(), PixelFormat::Rgb8);
        assert_eq!(flat.data(), &[255, 0, 0, 255, 255, 255]);
    }

    #[test]
    fn dimension_policy_never_upscales() {
        let src = Dimensions::new(4000, 3000);
        assert_eq!(DimensionPolicy::Preserve.target(src), src);
        assert_eq!(DimensionPolicy::MaxDimension { max: 2000 }.target(src), Dimensions::new(2000, 1500));
        assert_eq!(DimensionPolicy::MaxDimension { max: 8000 }.target(src), src);
        assert_eq!(DimensionPolicy::MaxBox { width: 1000, height: 1000 }.target(src), Dimensions::new(1000, 750));
        assert_eq!(DimensionPolicy::MaxMegapixels { megapixels: 3.0 }.target(src), Dimensions::new(2000, 1500));
    }
}
