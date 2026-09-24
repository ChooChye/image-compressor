//! Shared fixtures for native codec round-trip tests.

use smartimg_types::{Dimensions, Image, MetadataPolicy, PixelFormat};

use crate::{Codec, EncodeSettings};

pub fn gradient(format: PixelFormat) -> Image {
    let (w, h) = (96u32, 64u32);
    let data = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let px = [(x * 2) as u8, (y * 3) as u8, ((x + y) * 2) as u8];
            match format {
                PixelFormat::Rgb8 => px.to_vec(),
                // Left half opaque, right half transparent.
                PixelFormat::Rgba8 => vec![px[0], px[1], px[2], if x < w / 2 { 255 } else { 0 }],
            }
        })
        .collect();
    Image::new(Dimensions::new(w, h), format, data).unwrap()
}

pub fn settings(level: u32) -> EncodeSettings {
    EncodeSettings { level, metadata: MetadataPolicy::KeepIccOnly }
}

fn mean_abs_error(a: &Image, b: &Image) -> f64 {
    let sum: u64 = a.data().iter().zip(b.data()).map(|(x, y)| u64::from(x.abs_diff(*y))).sum();
    sum as f64 / a.data().len() as f64
}

/// Round-trip invariants every adapter must satisfy.
pub fn assert_round_trip(codec: &dyn Codec) {
    let axis = codec.quality_axis();
    let img = gradient(PixelFormat::Rgb8);

    let high = codec.encode(&img, &settings(axis.max)).unwrap();
    let low = codec.encode(&img, &settings(axis.min)).unwrap();
    assert!(low.bytes.len() < high.bytes.len(), "lower level must produce fewer bytes");

    let decoded = codec.decode(&high.bytes).unwrap();
    assert_eq!(decoded.dimensions(), img.dimensions());
    assert_eq!(decoded.format(), PixelFormat::Rgb8, "opaque input must not gain an alpha channel");
    let err = mean_abs_error(&img, &decoded);
    assert!(err < 3.0, "high-quality round trip error {err}");
    assert!(!high.parameters.is_empty());

    // Transparency survives.
    let rgba = gradient(PixelFormat::Rgba8);
    let decoded = codec.decode(&codec.encode(&rgba, &settings(axis.max)).unwrap().bytes).unwrap();
    assert_eq!(decoded.format(), PixelFormat::Rgba8);
    let alpha = |img: &Image, x: usize| img.data()[x * 4 + 3];
    assert_eq!((alpha(&decoded, 0), alpha(&decoded, 95)), (255, 0));
}

pub fn fake_icc() -> Vec<u8> {
    let mut icc = vec![0u8; 132];
    icc[16..20].copy_from_slice(b"RGB ");
    icc
}
