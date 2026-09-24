//! Fixed-quality encoding under a hard byte budget.
//!
//! The run-1 benchmark (plan.md §18) found that a well-chosen fixed quality matches or beats
//! per-image quality search on held-out metrics. Web delivery also needs a hard ceiling on
//! file size, so this encodes at a fixed level and only degrades when the budget demands it:
//! first by lowering the level, then, as a last resort, by downscaling.

use smartimg_codecs::{Codec, EncodeSettings, Encoded};
use smartimg_pipeline::ImagePipeline;
use smartimg_types::{Dimensions, Image, MetadataPolicy};

use crate::EngineError;

#[derive(Debug, Clone, Copy)]
pub struct BudgetOptions {
    /// Level to use when the budget allows it.
    pub level: u32,
    /// Lowest level the budget may push down to before downscaling.
    pub min_level: u32,
    pub level_step: u32,
    /// `None` disables the budget: exactly one encode at `level`.
    pub max_bytes: Option<u64>,
    /// Each downscale multiplies both dimensions by this factor.
    pub downscale_factor: f64,
    pub max_downscales: u32,
    pub metadata: MetadataPolicy,
}

impl Default for BudgetOptions {
    fn default() -> Self {
        Self {
            level: 60,
            min_level: 40,
            level_step: 5,
            max_bytes: None,
            downscale_factor: 0.85,
            max_downscales: 4,
            metadata: MetadataPolicy::KeepIccOnly,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BudgetedEncode {
    pub encoded: Encoded,
    pub level: u32,
    pub dimensions: Dimensions,
    pub attempts: u32,
    /// False only when even the smallest attempt exceeded the budget.
    pub within_budget: bool,
}

/// Encode `image` at `options.level`, degrading only as far as needed to fit `max_bytes`.
/// Downscaled attempts always resize from the full-quality `image`.
pub fn encode_within_budget(
    pipeline: &dyn ImagePipeline,
    codec: &dyn Codec,
    image: &Image,
    options: &BudgetOptions,
) -> Result<BudgetedEncode, EngineError> {
    let axis = codec.quality_axis();
    let clamp = |level: u32| level.clamp(axis.min, axis.max);
    let (start, floor) = (clamp(options.level), clamp(options.min_level.min(options.level)));
    let fits = |bytes: usize| options.max_bytes.is_none_or(|max| bytes as u64 <= max);

    let mut attempts = 0;
    let mut smallest: Option<BudgetedEncode> = None;
    let mut scaled: Option<Image> = None;

    for downscale in 0..=options.max_downscales {
        let current = scaled.as_ref().unwrap_or(image);
        let mut level = start;
        loop {
            attempts += 1;
            let settings = EncodeSettings { level, metadata: options.metadata };
            let encoded = codec.encode(current, &settings)?;
            let within_budget = fits(encoded.bytes.len());
            let result = BudgetedEncode { encoded, level, dimensions: current.dimensions(), attempts, within_budget };
            if within_budget {
                return Ok(result);
            }
            if smallest.as_ref().is_none_or(|s| result.encoded.bytes.len() < s.encoded.bytes.len()) {
                smallest = Some(result);
            }
            if level == floor {
                break;
            }
            level = level.saturating_sub(options.level_step).max(floor);
        }

        if downscale == options.max_downscales {
            break;
        }
        let d = image.dimensions();
        let factor = options.downscale_factor.powi(downscale as i32 + 1);
        let dims = Dimensions::new(
            ((f64::from(d.width) * factor).round() as u32).max(1),
            ((f64::from(d.height) * factor).round() as u32).max(1),
        );
        scaled = Some(pipeline.resize(image, dims)?);
    }

    let mut best = smallest.expect("at least one attempt");
    best.attempts = attempts;
    Ok(best)
}
