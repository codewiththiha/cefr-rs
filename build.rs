//! Only active with the `embed-model` feature: downloads the English
//! LanguageTool data and compiles it into OUT_DIR so main.rs can include it
//! with include_bytes!. A no-op otherwise — plain builds never touch the
//! network, and the build dependency is not even compiled.

#[cfg(feature = "embed-model")]
fn main() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is always set for build scripts");
    nlprule_build::BinaryBuilder::new(&["en"], std::path::Path::new(&out_dir))
        .build()
        .expect("failed to build the English model (needs network)");
}

#[cfg(not(feature = "embed-model"))]
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
