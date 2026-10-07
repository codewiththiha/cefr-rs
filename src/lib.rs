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

pub mod level;

#[cfg(feature = "parquet")]
pub mod dataset;
#[cfg(feature = "sqlite")]
pub mod db;
