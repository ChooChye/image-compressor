//! Image-processing boundary: decode, orientation, color, resize. The production
//! implementation is backed by libvips. Encoding is deliberately not part of this trait;
//! the optimizer calls codec adapters directly.

use smartimg_types::{Dimensions, Image, ImageFormat};
use thiserror::Error;

#[cfg(feature = "image-rs")]
pub mod image_rs;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("pipeline backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("unsupported input: {0}")]
    Unsupported(String),
    #[error("processing failed: {0}")]
    Processing(String),
}

#[derive(Debug, Clone)]
pub struct DecodedSource {
    /// Orientation-normalized pixels, with the source ICC profile attached.
    pub image: Image,
    /// Container format of the input, if it is one the engine can output.
    pub format: Option<ImageFormat>,
    /// True if the input carries EXIF/XMP or other non-ICC metadata. The original file
    /// cannot be returned unchanged when the metadata policy requires stripping it.
    pub has_strippable_metadata: bool,
}

pub trait ImagePipeline: Send + Sync {
    fn decode(&self, input: &[u8]) -> Result<DecodedSource, PipelineError>;
    /// High-quality (linear-light) downscale to exactly `dimensions`.
    fn resize(&self, image: &Image, dimensions: Dimensions) -> Result<Image, PipelineError>;
}
