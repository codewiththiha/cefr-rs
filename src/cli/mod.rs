//! The CLI subcommands that are presentation, not reusable behaviour:
//! `memory`, `analyze`, `bench`.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use rusqlite::types::Value;

/// One text run's stored value, printed.
pub fn fmt_value(v: &Value) -> String {
    match v {
        Value::Null => "-".into(),
        Value::Integer(i) => i.to_string(),
        Value::Real(f) => format!("{f:.2}"),
        Value::Text(s) => s.clone(),
        _ => "?".into(),
    }
}

// ---------------------------------------------------------------------------
// memory: skip SQLite entirely — leanest possible client
// ---------------------------------------------------------------------------
pub fn memory(parquet_path: &Path, word: &str, pos_tag: &str) -> Result<()> {
    use cefr::dataset::read_parquet;
    use cefr::level::level_to_cefr;

    let t0 = Instant::now();
    let (rows, cols) = read_parquet(parquet_path)?;
    // word -> list of (pos_tag, level); pos_tag is None when the file has no
    // such column
    let mut map: HashMap<String, Vec<(Option<String>, f32)>> = HashMap::new();
    for r in rows {
        map.entry(r.word).or_default().push((r.pos_tag, r.level));
    }
    println!(
        "loaded {} words in {:?} (one-time startup cost{})",
        map.len(),
        t0.elapsed(),
        if cols.pos_tag {
            ""
        } else {
            ", no pos_tag column -> word-level averages"
        }
    );

    let t = Instant::now();
    let level = map.get(word).map(|entries| {
        let matched: Vec<f32> = entries
            .iter()
            .filter(|(p, _)| p.as_deref() == Some(pos_tag))
            .map(|(_, l)| *l)
            .collect();
        if matched.is_empty() {
            entries.iter().map(|(_, l)| *l as f64).sum::<f64>() / entries.len() as f64
        } else {
            matched.iter().map(|l| *l as f64).sum::<f64>() / matched.len() as f64
        }
    });
    match level {
        Some(l) => println!(
            "{word} ({pos_tag}): {l:.2} -> {}   [{:?}]",
            level_to_cefr(l),
            t.elapsed()
        ),
        None => println!("{word}: not found"),
    }
    Ok(())
}

#[cfg(feature = "nlp")]
mod analyze_cmd;
#[cfg(feature = "nlp")]
pub use analyze_cmd::analyze;

#[cfg(not(feature = "nlp"))]
pub fn analyze(_db_path: &Path, _text_path: &str, _model: Option<&String>) -> Result<()> {
    anyhow::bail!(
        "the `analyze` command needs the NLP feature: cargo build --release --features nlp"
    )
}

mod bench_cmd;
pub use bench_cmd::bench;
