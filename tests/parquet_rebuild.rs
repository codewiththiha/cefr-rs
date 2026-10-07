//! The full pipeline against the repo's own dataset: parquet -> sqlite ->
//! lookups. Skips quietly when the built parquets are absent.
#![cfg(feature = "parquet")]
#![cfg(feature = "sqlite")]

use std::collections::HashMap;
use std::path::PathBuf;

use cefr::dataset::read_parquet;
use cefr::db::{CefrDb, build_db};
use cefr::level::level_band;

fn core_parquet() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/cefr.core.parquet");
    p.is_file().then_some(p)
}

#[test]
fn the_core_parquet_rebuilds_and_answers() {
    let Some(parquet) = core_parquet() else {
        eprintln!("skipping: data/cefr.core.parquet not built");
        return;
    };

    let (rows, cols) = read_parquet(&parquet).unwrap();
    assert!(
        rows.len() > 100_000,
        "the core export carries the whole word list"
    );
    assert!(!cols.pos_tag, "the core export has no pos_tag column");
    // words and levels only, in the canonical order
    assert_eq!(cols.names(), vec!["word", "level"]);
    // the export is mean-level per word: 1.0..=6.0
    let bad = rows
        .iter()
        .filter(|r| !(1.0..=6.0).contains(&r.level))
        .count();
    assert_eq!(bad, 0, "every level stays inside the documented range");

    // rebuild a db in a scratch dir
    let dir = std::env::temp_dir().join(format!("cefr_rebuild_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let db_path = dir.join("core.db");
    let stats = build_db(&parquet, &db_path).unwrap();
    assert_eq!(stats.rows, rows.len());

    let db = CefrDb::open(&db_path).unwrap();
    assert!(!db.has_pos(), "the rebuilt schema follows the core parquet");

    // common word in, absent word None, band math sane
    let hello = db.lookup("hello", "NN").unwrap();
    assert!(
        hello.is_some(),
        "'hello' is a CEFR-J headword the merge re-adds"
    );
    assert!((1.0..=6.0).contains(&hello.unwrap()));

    let pairs: Vec<(String, String)> = ["hello", "palimpsest", "zzz_no_such_word"]
        .iter()
        .map(|w| (w.to_string(), "NN".to_string()))
        .collect();
    let got: HashMap<(String, String), f64> = db.lookup_batch(&pairs).unwrap();
    assert_eq!(got.get(&pairs[0]), hello.as_ref());
    assert!(
        !got.contains_key(&pairs[2]),
        "absent words are absent from the batch map"
    );
    for level in got.values() {
        let band = level_band(*level);
        assert!((1..=6).contains(&band));
    }

    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn the_full_parquet_reads_with_every_column() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/cefr.zstd.parquet");
    if !p.is_file() {
        eprintln!("skipping: data/cefr.zstd.parquet not built");
        return;
    }
    let (rows, cols) = read_parquet(&p).unwrap();
    assert!(
        cols.pos_tag && cols.lemma && cols.categories,
        "the full export keeps all columns"
    );
    assert!(rows.iter().any(|r| r.pos_tag.is_some()));
    assert!(
        rows.iter()
            .any(|r| r.categories.as_deref().is_some_and(|c| !c.is_empty()))
    );
}
