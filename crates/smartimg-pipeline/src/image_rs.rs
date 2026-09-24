//! Pure-Rust pipeline backed by the `image` crate.
//!
//! Stand-in until the libvips pipeline lands. Known gaps: resizing happens in gamma space
//! (not linear light), 16-bit sources are truncated to 8 bits, and AVIF input is not
//! decoded.

use std::io::Cursor;

use image::{DynamicImage, ImageDecoder, ImageReader, imageops::FilterType};
use smartimg_types::{Dimensions, Image, ImageFormat, PixelFormat};

use crate::{DecodedSource, ImagePipeline, PipelineError};

#[derive(Debug, Clone)]
pub struct ImageRsPipeline {
    /// Inputs above this pixel count are rejected before decoding (decompression bombs).
    pub max_pixels: u64,
}

impl Default for ImageRsPipeline {
    fn default() -> Self {
        Self { max_pixels: 100_000_000 }
    }
}

fn processing(e: impl std::fmt::Display) -> PipelineError {
    PipelineError::Processing(e.to_string())
}

/// Only an RGB profile is valid once pixels are RGB; CMYK and gray sources are converted
/// during decode, so their profiles no longer describe the pixels.
fn rgb_profile(icc: Option<Vec<u8>>) -> Option<Vec<u8>> {
    icc.filter(|p| p.get(16..20) == Some(b"RGB "))
}

impl ImagePipeline for ImageRsPipeline {
    fn decode(&self, input: &[u8]) -> Result<DecodedSource, PipelineError> {
        let reader = ImageReader::new(Cursor::new(input)).with_guessed_format().map_err(processing)?;
        let format = match reader.format() {
            Some(image::ImageFormat::Jpeg) => Some(ImageFormat::Jpeg),
            Some(image::ImageFormat::Png) => Some(ImageFormat::Png),
            Some(image::ImageFormat::WebP) => Some(ImageFormat::WebP),
            Some(image::ImageFormat::Tiff) => None,
            other => return Err(PipelineError::Unsupported(format!("input format {other:?}"))),
        };

        let mut decoder = reader.into_decoder().map_err(processing)?;
        let (w, h) = decoder.dimensions();
        if u64::from(w) * u64::from(h) > self.max_pixels {
            return Err(PipelineError::Unsupported(format!("{w}x{h} exceeds {} pixels", self.max_pixels)));
        }

        let icc = rgb_profile(decoder.icc_profile().map_err(processing)?);
        let has_strippable_metadata = decoder.exif_metadata().map_err(processing)?.is_some()
            || decoder.xmp_metadata().map_err(processing)?.is_some();
        let orientation = decoder.orientation().map_err(processing)?;

        let mut decoded = DynamicImage::from_decoder(decoder).map_err(processing)?;
        decoded.apply_orientation(orientation);

        Ok(DecodedSource { image: to_image(decoded)?.with_icc_profile(icc), format, has_strippable_metadata })
    }

    fn resize(&self, image: &Image, dimensions: Dimensions) -> Result<Image, PipelineError> {
        let (w, h) = (image.dimensions().width, image.dimensions().height);
        let data = image.data().to_vec();
        let source = match image.format() {
            PixelFormat::Rgb8 => DynamicImage::ImageRgb8(image::RgbImage::from_raw(w, h, data).ok_or_else(|| processing("buffer"))?),
            PixelFormat::Rgba8 => {
                DynamicImage::ImageRgba8(image::RgbaImage::from_raw(w, h, data).ok_or_else(|| processing("buffer"))?)
            }
        };
        let resized = source.resize_exact(dimensions.width, dimensions.height, FilterType::Lanczos3);
        Ok(to_image(resized)?.with_icc_profile(image.icc_profile().map(<[u8]>::to_vec)))
    }
}

fn to_image(decoded: DynamicImage) -> Result<Image, PipelineError> {
    let dimensions = Dimensions::new(decoded.width(), decoded.height());
    if decoded.color().has_alpha() {
        Image::new(dimensions, PixelFormat::Rgba8, decoded.into_rgba8().into_raw()).map_err(processing)
    } else {
        Image::new(dimensions, PixelFormat::Rgb8, decoded.into_rgb8().into_raw()).map_err(processing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_png(img: DynamicImage) -> Vec<u8> {
        let mut out = Cursor::new(Vec::new());
        img.write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    #[test]
    fn decodes_png_with_alpha() {
        let png = encode_png(DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(3, 2, image::Rgba([1, 2, 3, 4]))));
        let source = ImageRsPipeline::default().decode(&png).unwrap();
        assert_eq!(source.format, Some(ImageFormat::Png));
        assert_eq!(source.image.format(), PixelFormat::Rgba8);
        assert_eq!(source.image.dimensions(), Dimensions::new(3, 2));
        assert!(!source.has_strippable_metadata);
    }

    #[test]
    fn rejects_oversized_input_before_decoding() {
        let png = encode_png(DynamicImage::ImageRgb8(image::RgbImage::new(100, 100)));
        let pipeline = ImageRsPipeline { max_pixels: 9_999 };
        assert!(matches!(pipeline.decode(&png), Err(PipelineError::Unsupported(_))));
    }

    #[test]
    fn keeps_only_rgb_profiles() {
        let mut profile = vec![0u8; 128];
        profile[16..20].copy_from_slice(b"RGB ");
        assert!(rgb_profile(Some(profile.clone())).is_some());
        profile[16..20].copy_from_slice(b"CMYK");
        assert!(rgb_profile(Some(profile)).is_none());
    }

    #[test]
    fn resize_hits_exact_dimensions() {
        let img = Image::new(Dimensions::new(8, 4), PixelFormat::Rgb8, vec![200; 96]).unwrap();
        let out = ImageRsPipeline::default().resize(&img, Dimensions::new(4, 2)).unwrap();
        assert_eq!(out.dimensions(), Dimensions::new(4, 2));
        assert!(out.data().iter().all(|&v| v.abs_diff(200) <= 1));
    }
}
