//! CEFR word-level lookups backed by the Words-CEFR-Dataset, as a library.
//!
//! The dataset ships as a small single-table parquet (1.4 / 2.8 MB); rebuild a
//! local sqlite db from it with [`db::build_db`], then query it with
//! [`db::CefrDb`] — or skip sqlite entirely by reading the parquet into memory
//! with [`dataset::read_parquet`].
//!
//! Column handling is name-based: any export from tools/csv_to_parquet.py works
//! as long as `word` and `level` exist; the sqlite schema follows the input.
//! The CLI in `main.rs` is a thin wrapper over this crate.
//!
//! [`level`] and [`tags`] carry no feature gate and no dependency: with
//! `default-features = false` this crate is two pure modules, so a consumer
//! that cannot carry parquet, sqlite or a tagger still reads the same bands
//! and word classes.

pub mod level;
pub mod tags;

#[cfg(feature = "parquet")]
pub mod dataset;
#[cfg(feature = "sqlite")]
pub mod db;
#[cfg(feature = "nlp")]
pub mod pos;
