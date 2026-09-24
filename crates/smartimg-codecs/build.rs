//! Generates FFI bindings for the native codec adapters against system libraries.
//!
//! Production builds will vendor and statically link pinned codec versions; linking the
//! system libraries is the M2 development path.

fn main() {
    #[cfg(feature = "native-avif")]
    bind(&["libavif"], "#include <avif/avif.h>", "avif.*|AVIF_.*", "avif.rs");

    #[cfg(feature = "native-webp")]
    bind(
        &["libwebp", "libwebpmux"],
        "#include <webp/encode.h>\n#include <webp/decode.h>\n#include <webp/mux.h>",
        "WebP.*|WEBP_.*|VP8.*",
        "webp.rs",
    );
}

#[cfg(any(feature = "native-avif", feature = "native-webp"))]
fn bind(libraries: &[&str], header: &str, allowlist: &str, output: &str) {
    let mut include_paths = Vec::new();
    for library in libraries {
        let lib = pkg_config::Config::new()
            .probe(library)
            .unwrap_or_else(|e| panic!("{library} not found via pkg-config: {e}"));
        include_paths.extend(lib.include_paths);
    }

    let bindings = bindgen::Builder::default()
        .header_contents("wrapper.h", header)
        .clang_args(include_paths.iter().map(|p| format!("-I{}", p.display())))
        .allowlist_function(allowlist)
        .allowlist_type(allowlist)
        .allowlist_var(allowlist)
        .derive_default(true)
        .layout_tests(false)
        .generate()
        .unwrap_or_else(|e| panic!("bindgen failed for {output}: {e}"));

    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join(output);
    bindings.write_to_file(out).expect("write bindings");
}
