//! Perceptual metric boundary plus built-in implementations.

use smartimg_types::{Direction, Image, MetricId, MetricResult, PixelFormat};
use thiserror::Error;

#[cfg(any(feature = "ssimulacra2", feature = "butteraugli"))]
mod psychovisual;
#[cfg(feature = "butteraugli")]
pub use psychovisual::{Butteraugli, ButteraugliNorm};
#[cfg(feature = "ssimulacra2")]
pub use psychovisual::Ssimulacra2;

#[derive(Debug, Error)]
pub enum MetricError {
    #[error("metric backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("reference is {reference:?} but candidate is {candidate:?}")]
    DimensionMismatch {
        reference: smartimg_types::Dimensions,
        candidate: smartimg_types::Dimensions,
    },
    #[error("metric evaluation failed: {0}")]
    Evaluation(String),
}

pub trait Metric: Send + Sync {
    /// Name and implementation version. Scores from different versions are not comparable.
    fn id(&self) -> MetricId;
    fn direction(&self) -> Direction;
    /// Compare a candidate against a reference of identical dimensions.
    fn score(&self, reference: &Image, candidate: &Image) -> Result<f64, MetricError>;

    fn measure(&self, reference: &Image, candidate: &Image) -> Result<MetricResult, MetricError> {
        if reference.dimensions() != candidate.dimensions() {
            return Err(MetricError::DimensionMismatch {
                reference: reference.dimensions(),
                candidate: candidate.dimensions(),
            });
        }
        Ok(MetricResult { metric: self.id(), direction: self.direction(), score: self.score(reference, candidate)? })
    }
}

/// Mean SSIM (Wang et al. 2004) on BT.601 luma with an 11-tap Gaussian window (σ = 1.5),
/// valid region only. Images with alpha are composited over black and over white and the
/// worse score is used, so transparency errors are penalized.
///
/// Matches `ssim-y-gauss11-v1` in the Python research prototype.
#[derive(Debug, Default, Clone, Copy)]
pub struct Ssim;

const WINDOW: usize = 11;

impl Metric for Ssim {
    fn id(&self) -> MetricId {
        MetricId::new("ssim-y-gauss11", "1")
    }

    fn direction(&self) -> Direction {
        Direction::HigherIsBetter
    }

    fn score(&self, reference: &Image, candidate: &Image) -> Result<f64, MetricError> {
        let backgrounds: &[[u8; 3]] =
            if reference.format().has_alpha() || candidate.format().has_alpha() {
                &[[0, 0, 0], [255, 255, 255]]
            } else {
                &[[0, 0, 0]]
            };
        Ok(backgrounds
            .iter()
            .map(|&bg| {
                let (w, h) = (reference.dimensions().width as usize, reference.dimensions().height as usize);
                ssim_plane(&luma(reference, bg), &luma(candidate, bg), w, h)
            })
            .fold(f64::INFINITY, f64::min))
    }
}

fn luma(image: &Image, background: [u8; 3]) -> Vec<f64> {
    let y = |r: f64, g: f64, b: f64| 0.299 * r + 0.587 * g + 0.114 * b;
    match image.format() {
        PixelFormat::Rgb8 => image
            .data()
            .chunks_exact(3)
            .map(|px| y(f64::from(px[0]), f64::from(px[1]), f64::from(px[2])))
            .collect(),
        // Unrounded compositing, so the metric does not inherit 8-bit rounding.
        PixelFormat::Rgba8 => image
            .data()
            .chunks_exact(4)
            .map(|px| {
                let a = f64::from(px[3]) / 255.0;
                let c = |i: usize| f64::from(px[i]) * a + f64::from(background[i]) * (1.0 - a);
                y(c(0), c(1), c(2))
            })
            .collect(),
    }
}

fn gaussian_window() -> [f64; WINDOW] {
    let mut w = [0.0; WINDOW];
    for (i, v) in w.iter_mut().enumerate() {
        let x = i as f64 - 5.0;
        *v = (-0.5 * x * x / (1.5 * 1.5)).exp();
    }
    let sum: f64 = w.iter().sum();
    w.map(|v| v / sum)
}

/// Separable Gaussian blur, valid region: output is (w-10) × (h-10).
fn blur(src: &[f64], w: usize, h: usize, g: &[f64; WINDOW]) -> Vec<f64> {
    let ow = w - (WINDOW - 1);
    let oh = h - (WINDOW - 1);
    let mut horiz = vec![0.0; ow * h];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        for x in 0..ow {
            horiz[y * ow + x] = (0..WINDOW).map(|i| g[i] * row[x + i]).sum();
        }
    }
    let mut out = vec![0.0; ow * oh];
    for y in 0..oh {
        for x in 0..ow {
            out[y * ow + x] = (0..WINDOW).map(|i| g[i] * horiz[(y + i) * ow + x]).sum();
        }
    }
    out
}

fn ssim_plane(a: &[f64], b: &[f64], w: usize, h: usize) -> f64 {
    if w < WINDOW || h < WINDOW {
        return if a == b { 1.0 } else { 0.0 };
    }
    let g = gaussian_window();
    let c1 = (0.01 * 255.0f64).powi(2);
    let c2 = (0.03 * 255.0f64).powi(2);
    let product = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| p * q).collect::<Vec<_>>();

    let mu_a = blur(a, w, h, &g);
    let mu_b = blur(b, w, h, &g);
    let aa = blur(&product(a, a), w, h, &g);
    let bb = blur(&product(b, b), w, h, &g);
    let ab = blur(&product(a, b), w, h, &g);

    let n = mu_a.len();
    let total: f64 = (0..n)
        .map(|i| {
            let (ma, mb) = (mu_a[i], mu_b[i]);
            let var_a = aa[i] - ma * ma;
            let var_b = bb[i] - mb * mb;
            let cov = ab[i] - ma * mb;
            ((2.0 * ma * mb + c1) * (2.0 * cov + c2)) / ((ma * ma + mb * mb + c1) * (var_a + var_b + c2))
        })
        .sum();
    total / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use smartimg_types::Dimensions;

    fn gradient(w: u32, h: u32, noise: u8) -> Image {
        let data = (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                let n = if noise > 0 { ((i * 7919) % u32::from(noise)) as u8 } else { 0 };
                let v = ((x * 4 + y * 2) % 200) as u8 + n;
                [v, v / 2, 255 - v]
            })
            .collect();
        Image::new(Dimensions::new(w, h), PixelFormat::Rgb8, data).unwrap()
    }

    #[test]
    fn identical_images_score_one() {
        let img = gradient(64, 48, 0);
        let s = Ssim.score(&img, &img).unwrap();
        assert!((s - 1.0).abs() < 1e-12, "{s}");
    }

    #[test]
    fn distortion_lowers_score() {
        let a = gradient(64, 48, 0);
        let b = gradient(64, 48, 40);
        let s = Ssim.score(&a, &b).unwrap();
        assert!(s < 0.99 && s > 0.0, "{s}");
    }

    #[test]
    fn alpha_errors_are_penalized() {
        let opaque = Image::new(Dimensions::new(16, 16), PixelFormat::Rgba8, [120, 80, 40, 255].repeat(256)).unwrap();
        let mut half = [120, 80, 40, 255].repeat(256);
        for px in half.chunks_exact_mut(4).skip(128) {
            px[3] = 0;
        }
        let half = Image::new(Dimensions::new(16, 16), PixelFormat::Rgba8, half).unwrap();
        assert!(Ssim.score(&opaque, &half).unwrap() < 0.9);
    }

    #[test]
    fn measure_rejects_dimension_mismatch() {
        let err = Ssim.measure(&gradient(20, 20, 0), &gradient(20, 21, 0)).unwrap_err();
        assert!(matches!(err, MetricError::DimensionMismatch { .. }));
    }
}
