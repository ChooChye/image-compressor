//! Codec boundary. Native adapters (jpegli, libwebp, libaom, PNG optimizers) implement
//! [`Codec`] and own their native parameter models; the optimizer only sees a normalized
//! quality axis.

use smartimg_types::{CodecId, EncoderParameters, Image, ImageFormat, MetadataPolicy};
use thiserror::Error;

#[cfg(feature = "native-avif")]
pub mod avif;
#[cfg(feature = "native-webp")]
pub mod webp;

#[cfg(all(test, any(feature = "native-avif", feature = "native-webp")))]
mod test_support;

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("codec backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("encoding failed: {0}")]
    Encode(String),
    #[error("decoding failed: {0}")]
    Decode(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    pub alpha: bool,
}

/// The searchable parameter range, normalized so that a larger level always means higher
/// quality and (usually) more bytes. Adapters map levels onto native scales, e.g. AVIF
/// quantizers where lower means better.
///
/// Multi-dimensional spaces (chroma mode, effort, tuning) are fixed per adapter for now;
/// they become additional axes once the benchmark shows they are worth searching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualityAxis {
    pub min: u32,
    pub max: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeSettings {
    pub level: u32,
    pub metadata: MetadataPolicy,
}

#[derive(Debug, Clone)]
pub struct Encoded {
    pub bytes: Vec<u8>,
    /// The native settings actually used, recorded in reports for reproducibility.
    pub parameters: EncoderParameters,
}

pub trait Codec: Send + Sync {
    /// Encoder name and exact version; output changes between encoder versions.
    fn id(&self) -> CodecId;
    fn format(&self) -> ImageFormat;
    fn capabilities(&self) -> Capabilities;
    fn quality_axis(&self) -> QualityAxis;
    fn encode(&self, image: &Image, settings: &EncodeSettings) -> Result<Encoded, CodecError>;
    /// Decode with the same decoder behavior a browser would use for this format.
    fn decode(&self, encoded: &[u8]) -> Result<Image, CodecError>;
}
