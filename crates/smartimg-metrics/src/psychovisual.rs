//! Psychovisual metrics: SSIMULACRA2 and Butteraugli.
//!
//! Both take opaque sRGB input. Images with alpha are composited over black and over
//! white and the worse result is reported, matching the SSIM implementation.

use smartimg_types::{Direction, Image, MetricId};

use crate::{Metric, MetricError};

fn backgrounds(reference: &Image, candidate: &Image) -> &'static [[u8; 3]] {
    if reference.format().has_alpha() || candidate.format().has_alpha() {
        &[[0, 0, 0], [255, 255, 255]]
    } else {
        &[[0, 0, 0]]
    }
}

fn worst(scores: impl Iterator<Item = Result<f64, MetricError>>, direction: Direction) -> Result<f64, MetricError> {
    let mut worst: Option<f64> = None;
    for score in scores {
        let score = score?;
        worst = Some(match (worst, direction) {
            (None, _) => score,
            (Some(w), Direction::HigherIsBetter) => w.min(score),
            (Some(w), Direction::LowerIsBetter) => w.max(score),
        });
    }
    worst.ok_or_else(|| MetricError::Evaluation("no composites".into()))
}

/// SSIMULACRA2 (rust-av implementation). Scale: 100 = identical, ~90 visually lossless,
/// ~70 high quality, ~50 medium, below 30 low.
#[cfg(feature = "ssimulacra2")]
#[derive(Debug, Default, Clone, Copy)]
pub struct Ssimulacra2;

#[cfg(feature = "ssimulacra2")]
impl Metric for Ssimulacra2 {
    fn id(&self) -> MetricId {
        MetricId::new("ssimulacra2", "rust-av-0.5")
    }

    fn direction(&self) -> Direction {
        Direction::HigherIsBetter
    }

    fn score(&self, reference: &Image, candidate: &Image) -> Result<f64, MetricError> {
        use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2};

        let to_rgb = |image: &Image, bg: [u8; 3]| {
            let d = image.dimensions();
            let data = image.flatten(bg).data().chunks_exact(3).map(|p| [0, 1, 2].map(|i| f32::from(p[i]) / 255.0)).collect();
            Rgb::new(data, d.width as usize, d.height as usize, TransferCharacteristic::SRGB, ColorPrimaries::BT709)
                .map_err(|e| MetricError::Evaluation(format!("{e:?}")))
        };
        worst(
            backgrounds(reference, candidate).iter().map(|&bg| {
                compute_frame_ssimulacra2(to_rgb(reference, bg)?, to_rgb(candidate, bg)?)
                    .map_err(|e| MetricError::Evaluation(e.to_string()))
            }),
            self.direction(),
        )
    }
}

#[cfg(feature = "butteraugli")]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ButteraugliNorm {
    /// Worst local difference: the classic Butteraugli distance (~1.0 = barely visible).
    #[default]
    Max,
    /// 3-norm over the difference map; far less sensitive to a single bad pixel.
    PNorm3,
}

/// Butteraugli (imazen pure-Rust port of libjxl's implementation). Lower is better.
#[cfg(feature = "butteraugli")]
#[derive(Debug, Default, Clone, Copy)]
pub struct Butteraugli {
    pub norm: ButteraugliNorm,
}

#[cfg(feature = "butteraugli")]
impl Metric for Butteraugli {
    fn id(&self) -> MetricId {
        let name = match self.norm {
            ButteraugliNorm::Max => "butteraugli",
            ButteraugliNorm::PNorm3 => "butteraugli-pnorm3",
        };
        MetricId::new(name, "imazen-0.9")
    }

    fn direction(&self) -> Direction {
        Direction::LowerIsBetter
    }

    fn score(&self, reference: &Image, candidate: &Image) -> Result<f64, MetricError> {
        use butteraugli::{ButteraugliParams, ImgRef, RGB8, butteraugli};

        let pixels = |image: &Image, bg: [u8; 3]| -> Vec<RGB8> {
            image.flatten(bg).data().chunks_exact(3).map(|p| RGB8::new(p[0], p[1], p[2])).collect()
        };
        let (w, h) = (reference.dimensions().width as usize, reference.dimensions().height as usize);
        let params = ButteraugliParams::default();
        worst(
            backgrounds(reference, candidate).iter().map(|&bg| {
                let (a, b) = (pixels(reference, bg), pixels(candidate, bg));
                let result = butteraugli(ImgRef::new(&a, w, h), ImgRef::new(&b, w, h), &params)
                    .map_err(|e| MetricError::Evaluation(e.to_string()))?;
                Ok(match self.norm {
                    ButteraugliNorm::Max => result.score,
                    ButteraugliNorm::PNorm3 => result.pnorm_3,
                })
            }),
            self.direction(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smartimg_types::{Dimensions, PixelFormat};

    fn image(noise: u32) -> Image {
        let (w, h) = (48u32, 40u32);
        let data = (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                let n = if noise > 0 { (i * 7919) % noise } else { 0 };
                let v = ((x * 5 + y * 3) % 200 + n) as u8;
                [v, 255 - v, v / 2]
            })
            .collect();
        Image::new(Dimensions::new(w, h), PixelFormat::Rgb8, data).unwrap()
    }

    #[cfg(feature = "ssimulacra2")]
    #[test]
    fn ssimulacra2_orders_quality() {
        let a = image(0);
        let same = Ssimulacra2.score(&a, &a).unwrap();
        let slight = Ssimulacra2.score(&a, &image(6)).unwrap();
        let heavy = Ssimulacra2.score(&a, &image(50)).unwrap();
        assert!(same > 99.0, "{same}");
        assert!(same > slight && slight > heavy, "{same} {slight} {heavy}");
    }

    #[cfg(feature = "butteraugli")]
    #[test]
    fn butteraugli_orders_quality() {
        let a = image(0);
        for norm in [ButteraugliNorm::Max, ButteraugliNorm::PNorm3] {
            let m = Butteraugli { norm };
            let same = m.score(&a, &a).unwrap();
            let slight = m.score(&a, &image(6)).unwrap();
            let heavy = m.score(&a, &image(50)).unwrap();
            assert!(same < 1e-6, "{same}");
            assert!(same < slight && slight < heavy, "{norm:?}: {same} {slight} {heavy}");
        }
    }
}
