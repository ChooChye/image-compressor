//! AVIF adapter over libavif (libaom encoder, dav1d/aom decoder).

use std::ffi::{CStr, CString};

use smartimg_types::{CodecId, Dimensions, EncoderParameters, Image, ImageFormat, MetadataPolicy, PixelFormat};

use crate::{Capabilities, Codec, CodecError, EncodeSettings, Encoded, QualityAxis};

#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, clippy::all)]
mod sys {
    include!(concat!(env!("OUT_DIR"), "/avif.rs"));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chroma {
    Yuv420,
    Yuv444,
}

#[derive(Debug, Clone)]
pub struct AvifConfig {
    /// libavif speed 0 (slowest, smallest) ..= 10 (fastest).
    pub speed: i32,
    pub chroma: Chroma,
    /// libaom `tune` option, e.g. `"iq"` or `"ssim"`. `None` keeps the encoder default.
    pub tune: Option<String>,
    pub threads: i32,
}

impl Default for AvifConfig {
    fn default() -> Self {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()) as i32;
        Self { speed: 6, chroma: Chroma::Yuv420, tune: None, threads }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AvifCodec {
    pub config: AvifConfig,
}

impl AvifCodec {
    pub fn new(config: AvifConfig) -> Self {
        Self { config }
    }
}

// RAII owners for libavif allocations, so every early return frees native memory.
struct ImagePtr(*mut sys::avifImage);
impl Drop for ImagePtr {
    fn drop(&mut self) {
        unsafe { sys::avifImageDestroy(self.0) }
    }
}

struct EncoderPtr(*mut sys::avifEncoder);
impl Drop for EncoderPtr {
    fn drop(&mut self) {
        unsafe { sys::avifEncoderDestroy(self.0) }
    }
}

struct DecoderPtr(*mut sys::avifDecoder);
impl Drop for DecoderPtr {
    fn drop(&mut self) {
        unsafe { sys::avifDecoderDestroy(self.0) }
    }
}

struct RwData(sys::avifRWData);
impl Drop for RwData {
    fn drop(&mut self) {
        unsafe { sys::avifRWDataFree(&mut self.0) }
    }
}

struct RgbPixels(sys::avifRGBImage);
impl Drop for RgbPixels {
    fn drop(&mut self) {
        unsafe { sys::avifRGBImageFreePixels(&mut self.0) }
    }
}

fn check(result: sys::avifResult, what: &str) -> Result<(), String> {
    if result == sys::avifResult_AVIF_RESULT_OK {
        Ok(())
    } else {
        let message = unsafe { CStr::from_ptr(sys::avifResultToString(result)) };
        Err(format!("{what}: {}", message.to_string_lossy()))
    }
}

fn rgb_format(format: PixelFormat) -> sys::avifRGBFormat {
    match format {
        PixelFormat::Rgb8 => sys::avifRGBFormat_AVIF_RGB_FORMAT_RGB,
        PixelFormat::Rgba8 => sys::avifRGBFormat_AVIF_RGB_FORMAT_RGBA,
    }
}

impl Codec for AvifCodec {
    fn id(&self) -> CodecId {
        let mut codecs = [0 as std::ffi::c_char; 256];
        let (version, codecs) = unsafe {
            sys::avifCodecVersions(codecs.as_mut_ptr());
            (CStr::from_ptr(sys::avifVersion()), CStr::from_ptr(codecs.as_ptr()))
        };
        CodecId::new("libavif", format!("{} ({})", version.to_string_lossy(), codecs.to_string_lossy()))
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Avif
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { alpha: true }
    }

    fn quality_axis(&self) -> QualityAxis {
        // libavif quality: 100 is lossless. Matches the Python prototype's search range.
        QualityAxis { min: 20, max: 95 }
    }

    fn encode(&self, image: &Image, settings: &EncodeSettings) -> Result<Encoded, CodecError> {
        self.encode_inner(image, settings).map_err(CodecError::Encode)
    }

    fn decode(&self, encoded: &[u8]) -> Result<Image, CodecError> {
        decode_inner(encoded).map_err(CodecError::Decode)
    }
}

impl AvifCodec {
    fn encode_inner(&self, image: &Image, settings: &EncodeSettings) -> Result<Encoded, String> {
        let config = &self.config;
        let d = image.dimensions();
        let yuv = match config.chroma {
            Chroma::Yuv420 => sys::avifPixelFormat_AVIF_PIXEL_FORMAT_YUV420,
            Chroma::Yuv444 => sys::avifPixelFormat_AVIF_PIXEL_FORMAT_YUV444,
        };

        unsafe {
            let avif = ImagePtr(sys::avifImageCreate(d.width, d.height, 8, yuv));
            if avif.0.is_null() {
                return Err("avifImageCreate failed".into());
            }
            (*avif.0).yuvRange = sys::avifRange_AVIF_RANGE_FULL;

            match image.icc_profile().filter(|_| settings.metadata == MetadataPolicy::KeepIccOnly) {
                Some(icc) => check(sys::avifImageSetProfileICC(avif.0, icc.as_ptr(), icc.len()), "set ICC")?,
                None => {
                    // Untagged input is treated as sRGB; say so explicitly instead of leaving
                    // decoders to guess.
                    (*avif.0).colorPrimaries = sys::AVIF_COLOR_PRIMARIES_BT709 as _;
                    (*avif.0).transferCharacteristics = sys::AVIF_TRANSFER_CHARACTERISTICS_SRGB as _;
                    (*avif.0).matrixCoefficients = sys::AVIF_MATRIX_COEFFICIENTS_BT601 as _;
                }
            }

            let mut rgb = sys::avifRGBImage::default();
            sys::avifRGBImageSetDefaults(&mut rgb, avif.0);
            rgb.format = rgb_format(image.format());
            rgb.depth = 8;
            // Opaque RGBA input would otherwise spend bytes on a constant alpha plane.
            rgb.ignoreAlpha = i32::from(!image.has_transparency());
            rgb.pixels = image.data().as_ptr().cast_mut();
            rgb.rowBytes = image.stride() as u32;
            check(sys::avifImageRGBToYUV(avif.0, &rgb), "RGB to YUV")?;

            let encoder = EncoderPtr(sys::avifEncoderCreate());
            if encoder.0.is_null() {
                return Err("avifEncoderCreate failed".into());
            }
            (*encoder.0).quality = settings.level as i32;
            (*encoder.0).speed = config.speed;
            (*encoder.0).maxThreads = config.threads;
            if let Some(tune) = &config.tune {
                let value = CString::new(tune.as_str()).map_err(|e| e.to_string())?;
                check(
                    sys::avifEncoderSetCodecSpecificOption(encoder.0, c"tune".as_ptr(), value.as_ptr()),
                    "set tune",
                )?;
            }

            let mut output = RwData(sys::avifRWData::default());
            check(sys::avifEncoderWrite(encoder.0, avif.0, &mut output.0), "encode")?;
            let bytes = std::slice::from_raw_parts(output.0.data, output.0.size).to_vec();

            let mut parameters = EncoderParameters::new();
            parameters.insert("quality".into(), settings.level.to_string());
            parameters.insert("speed".into(), config.speed.to_string());
            parameters.insert("chroma".into(), if config.chroma == Chroma::Yuv444 { "444" } else { "420" }.into());
            parameters.insert("tune".into(), config.tune.clone().unwrap_or_else(|| "default".into()));
            Ok(Encoded { bytes, parameters })
        }
    }
}

fn decode_inner(encoded: &[u8]) -> Result<Image, String> {
    unsafe {
        let decoder = DecoderPtr(sys::avifDecoderCreate());
        let avif = ImagePtr(sys::avifImageCreateEmpty());
        if decoder.0.is_null() || avif.0.is_null() {
            return Err("allocation failed".into());
        }
        check(sys::avifDecoderReadMemory(decoder.0, avif.0, encoded.as_ptr(), encoded.len()), "read")?;

        let has_alpha = !(*avif.0).alphaPlane.is_null();
        let format = if has_alpha { PixelFormat::Rgba8 } else { PixelFormat::Rgb8 };

        let mut rgb = RgbPixels(sys::avifRGBImage::default());
        sys::avifRGBImageSetDefaults(&mut rgb.0, avif.0);
        rgb.0.format = rgb_format(format);
        rgb.0.depth = 8;
        check(sys::avifRGBImageAllocatePixels(&mut rgb.0), "allocate RGB")?;
        check(sys::avifImageYUVToRGB(avif.0, &mut rgb.0), "YUV to RGB")?;

        let (width, height) = ((*avif.0).width, (*avif.0).height);
        let row = width as usize * format.channels();
        let mut data = Vec::with_capacity(row * height as usize);
        for y in 0..height as usize {
            let start = rgb.0.pixels.add(y * rgb.0.rowBytes as usize);
            data.extend_from_slice(std::slice::from_raw_parts(start, row));
        }

        let icc = (*avif.0).icc;
        let icc = (icc.size > 0).then(|| std::slice::from_raw_parts(icc.data, icc.size).to_vec());
        Image::new(Dimensions::new(width, height), format, data)
            .map(|img| img.with_icc_profile(icc))
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;

    #[test]
    fn round_trip() {
        assert_round_trip(&AvifCodec::default());
    }

    #[test]
    fn icc_profile_is_embedded_unless_stripped() {
        let img = gradient(PixelFormat::Rgb8).with_icc_profile(Some(fake_icc()));
        let codec = AvifCodec::default();
        let kept = codec.decode(&codec.encode(&img, &settings(60)).unwrap().bytes).unwrap();
        assert_eq!(kept.icc_profile(), Some(fake_icc().as_slice()));

        let strip = EncodeSettings { level: 60, metadata: MetadataPolicy::StripAll };
        let stripped = codec.decode(&codec.encode(&img, &strip).unwrap().bytes).unwrap();
        assert_eq!(stripped.icc_profile(), None);
    }

    #[test]
    fn id_reports_versions() {
        let id = AvifCodec::default().id();
        assert_eq!(id.name, "libavif");
        assert!(id.version.contains("aom"), "{}", id.version);
    }
}
