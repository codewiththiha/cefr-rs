//! `bench`: parquet read / sqlite rebuild / lookup latency, per file.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use rusqlite::params;

use cefr::dataset::read_parquet;
use cefr::db::{CefrDb, build_db};

struct BenchReport {
    file: String,
    parquet_mb: f64,
    rows: usize,
    words: usize,
    cols: String,
    read_ms: f64,
    build_ms: f64,
    db_mb: f64,
    single_us: f64,
    miss_us: f64,
    batch_us: f64,
    mem_load_ms: f64,
    mem_us: f64,
}

fn bench_one(path: &Path) -> Result<BenchReport> {
    const N: usize = 1_000;
    let parquet_mb = std::fs::metadata(path)?.len() as f64 / 1_048_576.0;

    // --- phase 1: read the parquet ---
    let t = Instant::now();
    let (rows, cols) = read_parquet(path)?;
    let read_ms = t.elapsed().as_secs_f64() * 1e3;

    // deterministic sample: every k-th word (rows stay frequency-ranked)
    let mut unique: Vec<&str> = Vec::new();
    let mut prev = "";
    for r in &rows {
        if r.word != prev {
            prev = r.word.as_str();
            unique.push(&r.word);
        }
    }
    let step = (unique.len() / N).max(1);
    let sample: Vec<&str> = unique.into_iter().step_by(step).take(N).collect();

    // --- phase 2: rebuild sqlite in a temp file ---
    let db_path = std::env::temp_dir().join(format!("cefr_bench_{}.db", std::process::id()));
    let _ = std::fs::remove_file(&db_path);
    let t = Instant::now();
    let stats = build_db(path, &db_path)?;
    let build_ms = t.elapsed().as_secs_f64() * 1e3;
    let db_mb = std::fs::metadata(&db_path)?.len() as f64 / 1_048_576.0;

    let db = CefrDb::open(&db_path)?;
    let words: i64 = db
        .conn()
        .query_row("SELECT COUNT(DISTINCT word) FROM cefr", [], |r| r.get(0))?;
    let words = words as usize;

    // --- phase 3: hot-path single lookups (statement prepared once) ---
    // rusqlite accepts trailing unused params, so (pos, word) binds fine
    // even for the no-pos variant's "WHERE word = ?2"
    let sql = if cols.pos_tag {
        "SELECT COALESCE(AVG(CASE WHEN pos_tag = ?1 THEN level END), AVG(level)) FROM cefr WHERE word = ?2"
    } else {
        "SELECT AVG(level) FROM cefr WHERE word = ?2"
    };
    let mut stmt = db.conn().prepare(sql)?;

    let t = Instant::now();
    let mut found = 0usize;
    for w in &sample {
        let lvl: Option<f64> = stmt.query_row(params!["NN", w], |r| r.get(0))?;
        std::hint::black_box(&lvl);
        found += lvl.is_some() as usize;
    }
    let single_us = t.elapsed().as_micros() as f64 / sample.len() as f64;
    std::hint::black_box(found);

    // --- phase 4: index misses (word absent) ---
    let t = Instant::now();
    for i in 0..sample.len() {
        let miss = format!("zzz_no_such_word_{i}");
        let lvl: Option<f64> = stmt.query_row(params!["NN", &miss], |r| r.get(0))?;
        std::hint::black_box(&lvl);
    }
    let miss_us = t.elapsed().as_micros() as f64 / sample.len() as f64;
    drop(stmt);

    // --- phase 5: batch lookup (one VALUES-CTE round trip for all pairs) ---
    let pairs: Vec<(String, String)> = sample
        .iter()
        .map(|w| (w.to_string(), "NN".into()))
        .collect();
    let t = Instant::now();
    let m = db.lookup_batch(&pairs)?;
    std::hint::black_box(m.len());
    let batch_us = t.elapsed().as_micros() as f64 / pairs.len() as f64;
    drop(db);

    // --- phase 6: no-sqlite in-memory map ---
    let t = Instant::now();
    let mut map: HashMap<&str, Vec<(Option<&str>, f32)>> = HashMap::with_capacity(words);
    for r in &rows {
        map.entry(r.word.as_str())
            .or_default()
            .push((r.pos_tag.as_deref(), r.level));
    }
    let mem_load_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let mut acc = 0f64;
    for w in &sample {
        if let Some(entries) = map.get(w) {
            acc += entries.iter().map(|e| e.1 as f64).sum::<f64>() / entries.len() as f64;
        }
    }
    std::hint::black_box(acc);
    let mem_us = t.elapsed().as_micros() as f64 / sample.len() as f64;

    let _ = std::fs::remove_file(&db_path);
    Ok(BenchReport {
        file: path.display().to_string(),
        parquet_mb,
        rows: stats.rows,
        words,
        cols: cols.names().join(","),
        read_ms,
        build_ms,
        db_mb,
        single_us,
        miss_us,
        batch_us,
        mem_load_ms,
        mem_us,
    })
}

pub fn bench(files: &[PathBuf]) -> Result<()> {
    let mut reports = Vec::new();
    for f in files {
        reports.push(bench_one(f)?);
    }
    println!();
    println!(
        "{:<28} {:>8} {:>10} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "file", "pq MB", "read ms", "build ms", "db MB", "1-lookup µs", "miss µs", "batch µs/pair"
    );
    println!("{}", "-".repeat(118));
    for r in &reports {
        println!(
            "{:<28} {:>8.2} {:>10.1} {:>10.1} {:>10.2} {:>10.2} {:>12.2} {:>12.3}",
            r.file,
            r.parquet_mb,
            r.read_ms,
            r.build_ms,
            r.db_mb,
            r.single_us,
            r.miss_us,
            r.batch_us
        );
    }
    println!();
    for r in &reports {
        println!("{}", r.file);
        println!(
            "  rows {:>7} | distinct words {:>7} | columns [{}]",
            r.rows, r.words, r.cols
        );
        println!(
            "  in-memory map: {:.1} ms load, {:.3} µs/lookup (no sqlite at all)",
            r.mem_load_ms, r.mem_us
        );
    }
    if reports.len() > 1 {
        let (a, b) = (&reports[0], &reports[1]);
        println!();
        println!(
            "'{}' is {:.0}% smaller to download than '{}' and rebuilds {:.0}% faster.",
            b.file,
            (a.parquet_mb - b.parquet_mb) / a.parquet_mb * 100.0,
            a.file,
            (a.build_ms - b.build_ms) / a.build_ms * 100.0
        );
    }
    Ok(())
}
