//! Only active with the `embed-model` feature: downloads the English
//! LanguageTool data and compiles it into OUT_DIR so main.rs can include it
//! with include_bytes!. A no-op otherwise — plain `--features nlp` builds
//! never touch the network.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var("CARGO_FEATURE_EMBED_MODEL").is_err() {
        return;
    }

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is always set for build scripts");
    nlprule_build::BinaryBuilder::new(&["en"], std::path::Path::new(&out_dir))
        .build()
        .expect("failed to build the English model (needs network)");
}
