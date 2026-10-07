//! CEFR word-level lookups backed by the Words-CEFR-Dataset.
//!
//! Ship a small single-table parquet (1.4 / 2.8 MB), rebuild a local sqlite db
//! on the client with `builddb`, then query it — or skip sqlite via `memory`.
//!
//! The reusable half of this crate lives in the library (`cefr::`); this
//! binary is the CLI over it.

mod cli;

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use rusqlite::{params, types::Value};

use cefr::db::CefrDb;
use cefr::level::level_to_cefr;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let rest = args.get(1..).unwrap_or(&[]);
    match (cmd, rest) {
        ("builddb", [parquet, db]) => {
            let stats = cefr::db::build_db(Path::new(parquet), Path::new(db))?;
            println!(
                "builddb: {} rows, columns [{}] | read parquet {:?} | total {:?} -> {db}",
                stats.rows,
                stats.cols.names().join(", "),
                stats.read,
                stats.total
            );
            Ok(())
        }
        ("lookup", [db, word]) => {
            let conn = CefrDb::open(Path::new(db))?;
            let cols = cefr::db::table_columns(conn.conn(), "cefr")?;
            // show the informative columns that exist in this db
            let shown: Vec<&str> = ["pos_tag", "lemma", "level", "frequency_count"]
                .into_iter()
                .filter(|c| cols.iter().any(|x| x == c))
                .collect();
            let sql = format!(
                "SELECT {} FROM cefr WHERE word = ?1 ORDER BY level",
                shown.join(", ")
            );
            let mut stmt = conn.conn().prepare(&sql)?;
            let rows = stmt.query_map(params![word], |row| {
                let mut v = Vec::with_capacity(shown.len());
                for i in 0..shown.len() {
                    v.push(row.get::<_, Value>(i)?);
                }
                Ok(v)
            })?;
            let mut found = false;
            for r in rows {
                let vals = r?;
                found = true;
                let mut parts: Vec<String> = vec![word.clone()];
                for (col, v) in shown.iter().zip(&vals) {
                    parts.push(cli::fmt_value(v));
                    if *col == "level" {
                        if let Value::Real(l) = v {
                            parts.push(level_to_cefr(*l).to_string());
                        }
                    }
                }
                println!("{}", parts.join("\t"));
            }
            if !found {
                println!("{word}: not found");
            }
            Ok(())
        }
        ("lookup", [db, word, pos]) => {
            let conn = CefrDb::open(Path::new(db))?;
            match conn.lookup(word, pos)? {
                Some(l) => {
                    println!("{word} ({pos}): {l:.2} -> {}", level_to_cefr(l));
                    Ok(())
                }
                None => bail!("{word}: not found in dataset"),
            }
        }
        ("batch", [db, pairs @ ..]) if !pairs.is_empty() => {
            let conn = CefrDb::open(Path::new(db))?;
            let parsed: Vec<(String, String)> = pairs
                .iter()
                .map(|s| {
                    let (w, p) = s.rsplit_once(':').unwrap_or((s.as_str(), ""));
                    (w.to_string(), p.to_string())
                })
                .collect();
            let levels = conn.lookup_batch(&parsed)?;
            for (w, p) in &parsed {
                match levels.get(&(w.clone(), p.clone())) {
                    Some(l) => println!("{w}\t{p}\t{l:.2}\t{}", level_to_cefr(*l)),
                    None => println!("{w}\t{p}\tnot found"),
                }
            }
            eprintln!("-- {} pairs", parsed.len());
            Ok(())
        }
        ("memory", [parquet, word, pos]) => cli::memory(Path::new(parquet), word, pos),
        ("analyze", [db, text]) => cli::analyze(Path::new(db), text, None),
        ("analyze", [db, text, model]) => cli::analyze(Path::new(db), text, Some(model)),
        ("bench", _) if args.len() >= 2 => {
            let paths: Vec<PathBuf> = args[1..].iter().map(PathBuf::from).collect();
            cli::bench(&paths)
        }
        _ => {
            eprintln!(
                "usage:
  cefr builddb <in.parquet> <out.db>     rebuild sqlite db client-side (schema follows parquet)
  cefr lookup  <db> <word> [pos_tag]     look up one word (optionally with POS tag)
  cefr batch   <db> <word:pos> [...]     batched lookup, mirrors the python notebook
  cefr memory  <parquet> <word> <pos>    no-sqlite in-memory lookup demo
  cefr analyze <db> <text> [model]       full-text CEFR analysis (needs --features nlp;
                                         model: path/CEFR_MODEL, defaults to ./models/en_tokenizer.bin.zst)
  cefr bench   <parquet> [parquet...]    benchmark read/rebuild/lookup for one or more files"
            );
            std::process::exit(2);
        }
    }
}
