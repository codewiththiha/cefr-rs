//! Regenerate the English tokenizer model that `analyze` loads at runtime.
//!
//! Downloads the LanguageTool data once (network required) and writes
//! models/en_tokenizer.bin. For local regeneration only; the prebuilt
//! models/en_tokenizer.bin.zst is the distributed artifact.
//!
//!     cargo run --example make_model

use nlprule_build::BinaryBuilder;
use std::path::Path;

fn main() {
    let out = Path::new("models");
    std::fs::create_dir_all(out).expect("create models/");
    BinaryBuilder::new(&["en"], out)
        .build()
        .expect("failed to build the English model (needs network)");
    println!(
        "wrote {} — en_tokenizer.bin is the one the CLI loads",
        out.join("en_tokenizer.bin").display()
    );
}
