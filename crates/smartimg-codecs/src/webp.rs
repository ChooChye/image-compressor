//! WebP adapter over libwebp (lossy encoding, ICC embedded via libwebpmux).

use std::ffi::c_void;

use smartimg_types::{CodecId, Dimensions, EncoderParameters, Image, ImageFormat, MetadataPolicy, PixelFormat};

use crate::{Capabilities, Codec, CodecError, EncodeSettings, Encoded, QualityAxis};

#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, clippy::all)]
mod sys {
    include!(concat!(env!("OUT_DIR"), "/webp.rs"));
}

#[derive(Debug, Clone)]
pub struct WebpConfig {
    /// Compression effort 0 (fastest) ..= 6 (smallest).
    pub method: i32,
    /// Sharper RGB→YUV conversion; reduces chroma bleeding at a small speed cost.
    pub sharp_yuv: bool,
}

impl Default for WebpConfig {
    fn default() -> Self {
        Self { method: 6, sharp_yuv: true }
    }
}

#[derive(Debug, Clone, Default)]
pub struct WebpCodec {
    pub config: WebpConfig,
}

impl WebpCodec {
    pub fn new(config: WebpConfig) -> Self {
        Self { config }
    }
}

struct Picture(sys::WebPPicture);
impl Drop for Picture {
    fn drop(&mut self) {
        unsafe { sys::WebPPictureFree(&mut self.0) }
    }
}

struct MemoryWriter(sys::WebPMemoryWriter);
impl Drop for MemoryWriter {
    fn drop(&mut self) {
        unsafe { sys::WebPMemoryWriterClear(&mut self.0) }
    }
}

struct Mux(*mut sys::WebPMux);
impl Drop for Mux {
    fn drop(&mut self) {
        unsafe { sys::WebPMuxDelete(self.0) }
    }
}

/// Memory owned by libwebp, released with `WebPFree`.
struct WebpBuffer(*mut u8);
impl Drop for WebpBuffer {
    fn drop(&mut self) {
        unsafe { sys::WebPFree(self.0.cast()) }
    }
}

fn version_string(v: i32) -> String {
    format!("{}.{}.{}", (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff)
}

impl Codec for WebpCodec {
    fn id(&self) -> CodecId {
        CodecId::new("libwebp", version_string(unsafe { sys::WebPGetEncoderVersion() }))
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::WebP
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { alpha: true }
    }

    fn quality_axis(&self) -> QualityAxis {
        // Matches the Python prototype's search range.
        QualityAxis { min: 10, max: 95 }
    }

    fn encode(&self, image: &Image, settings: &EncodeSettings) -> Result<Encoded, CodecError> {
        self.encode_inner(image, settings).map_err(CodecError::Encode)
    }

    fn decode(&self, encoded: &[u8]) -> Result<Image, CodecError> {
        decode_inner(encoded).map_err(CodecError::Decode)
    }
}

impl WebpCodec {
    fn encode_inner(&self, image: &Image, settings: &EncodeSettings) -> Result<Encoded, String> {
        let d = image.dimensions();
        unsafe {
            let mut config = sys::WebPConfig::default();
            if sys::WebPConfigInitInternal(
                &mut config,
                sys::WebPPreset_WEBP_PRESET_DEFAULT,
                settings.level as f32,
                sys::WEBP_ENCODER_ABI_VERSION as i32,
            ) == 0
            {
                return Err("WebPConfigInit failed (ABI mismatch?)".into());
            }
            config.method = self.config.method;
            config.use_sharp_yuv = i32::from(self.config.sharp_yuv);
            if sys::WebPValidateConfig(&config) == 0 {
                return Err("invalid WebP config".into());
            }

            let mut picture = Picture(sys::WebPPicture::default());
            if sys::WebPPictureInitInternal(&mut picture.0, sys::WEBP_ENCODER_ABI_VERSION as i32) == 0 {
                return Err("WebPPictureInit failed".into());
            }
            picture.0.width = d.width as i32;
            picture.0.height = d.height as i32;
            // ARGB input lets the encoder apply sharp YUV conversion itself.
            picture.0.use_argb = 1;
            let stride = image.stride() as i32;
            let imported = match image.format() {
                PixelFormat::Rgb8 => sys::WebPPictureImportRGB(&mut picture.0, image.data().as_ptr(), stride),
                PixelFormat::Rgba8 => sys::WebPPictureImportRGBA(&mut picture.0, image.data().as_ptr(), stride),
            };
            if imported == 0 {
                return Err("WebPPictureImport failed".into());
            }

            let mut writer = MemoryWriter(sys::WebPMemoryWriter::default());
            sys::WebPMemoryWriterInit(&mut writer.0);
            picture.0.writer = Some(sys::WebPMemoryWrite);
            picture.0.custom_ptr = (&mut writer.0 as *mut sys::WebPMemoryWriter).cast::<c_void>();
            if sys::WebPEncode(&config, &mut picture.0) == 0 {
                return Err(format!("WebPEncode failed (error code {})", picture.0.error_code));
            }
            let mut bytes = std::slice::from_raw_parts(writer.0.mem, writer.0.size).to_vec();

            if let Some(icc) = image.icc_profile().filter(|_| settings.metadata == MetadataPolicy::KeepIccOnly) {
                bytes = embed_icc(&bytes, icc)?;
            }

            let mut parameters = EncoderParameters::new();
            parameters.insert("quality".into(), settings.level.to_string());
            parameters.insert("method".into(), self.config.method.to_string());
            parameters.insert("sharp_yuv".into(), self.config.sharp_yuv.to_string());
            Ok(Encoded { bytes, parameters })
        }
    }
}

unsafe fn embed_icc(bitstream: &[u8], icc: &[u8]) -> Result<Vec<u8>, String> {
    unsafe {
        let mux = Mux(sys::WebPNewInternal(sys::WEBP_MUX_ABI_VERSION as i32));
        if mux.0.is_null() {
            return Err("WebPMuxNew failed".into());
        }
        let image = sys::WebPData { bytes: bitstream.as_ptr(), size: bitstream.len() };
        if sys::WebPMuxSetImage(mux.0, &image, 1) != sys::WebPMuxError_WEBP_MUX_OK {
            return Err("WebPMuxSetImage failed".into());
        }
        let profile = sys::WebPData { bytes: icc.as_ptr(), size: icc.len() };
        if sys::WebPMuxSetChunk(mux.0, c"ICCP".as_ptr(), &profile, 1) != sys::WebPMuxError_WEBP_MUX_OK {
            return Err("WebPMuxSetChunk(ICCP) failed".into());
        }
        let mut assembled = sys::WebPData::default();
        if sys::WebPMuxAssemble(mux.0, &mut assembled) != sys::WebPMuxError_WEBP_MUX_OK {
            return Err("WebPMuxAssemble failed".into());
        }
        let owned = WebpBuffer(assembled.bytes.cast_mut());
        Ok(std::slice::from_raw_parts(owned.0, assembled.size).to_vec())
    }
}

fn decode_inner(encoded: &[u8]) -> Result<Image, String> {
    unsafe {
        let mut features = sys::WebPBitstreamFeatures::default();
        let status = sys::WebPGetFeaturesInternal(
            encoded.as_ptr(),
            encoded.len(),
            &mut features,
            sys::WEBP_DECODER_ABI_VERSION as i32,
        );
        if status != sys::VP8StatusCode_VP8_STATUS_OK {
            return Err(format!("invalid WebP bitstream (status {status})"));
        }
        if features.has_animation != 0 {
            return Err("animated WebP is not supported".into());
        }

        let format = if features.has_alpha != 0 { PixelFormat::Rgba8 } else { PixelFormat::Rgb8 };
        let (mut width, mut height) = (0, 0);
        let pixels = WebpBuffer(match format {
            PixelFormat::Rgb8 => sys::WebPDecodeRGB(encoded.as_ptr(), encoded.len(), &mut width, &mut height),
            PixelFormat::Rgba8 => sys::WebPDecodeRGBA(encoded.as_ptr(), encoded.len(), &mut width, &mut height),
        });
        if pixels.0.is_null() {
            return Err("WebPDecode failed".into());
        }
        let len = width as usize * height as usize * format.channels();
        let data = std::slice::from_raw_parts(pixels.0, len).to_vec();
        Image::new(Dimensions::new(width as u32, height as u32), format, data).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;

    #[test]
    fn round_trip() {
        assert_round_trip(&WebpCodec::default());
    }

    #[test]
    fn icc_profile_is_embedded_unless_stripped() {
        let img = gradient(PixelFormat::Rgb8).with_icc_profile(Some(fake_icc()));
        let codec = WebpCodec::default();
        let has_iccp = |bytes: &[u8]| bytes.windows(4).any(|w| w == b"ICCP");

        let kept = codec.encode(&img, &settings(60)).unwrap().bytes;
        assert!(has_iccp(&kept));
        assert_eq!(codec.decode(&kept).unwrap().dimensions(), img.dimensions());

        let strip = EncodeSettings { level: 60, metadata: MetadataPolicy::StripAll };
        assert!(!has_iccp(&codec.encode(&img, &strip).unwrap().bytes));
    }

    #[test]
    fn rejects_garbage() {
        assert!(WebpCodec::default().decode(b"not a webp").is_err());
    }
}
