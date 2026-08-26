//! CEFR word-level lookups backed by the Words-CEFR-Dataset.
//!
//! Ship a small single-table parquet (1.4 / 2.8 MB), rebuild a local sqlite db
//! on the client with `builddb`, then query it — or skip sqlite via `memory`.
//!
//! Column handling is name-based: any export from tools/csv_to_parquet.py works
//! as long as `word` and `level` exist; the sqlite schema follows the input.

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use arrow::array::{Array, Float32Array, ListArray, StringArray, UInt64Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rusqlite::{Connection, params, params_from_iter, types::Value};

// ---------------------------------------------------------------------------
// Schema-flexible row: everything optional except word + level
// ---------------------------------------------------------------------------
#[derive(Debug, Default)]
struct CefrRow {
    word: String,
    pos_tag: Option<String>,
    lemma: Option<String>,
    stem: Option<String>,
    frequency_count: Option<u64>,
    level: f32,
    categories: Option<String>, // '|' separated titles
}

/// Which optional columns the parquet file actually has.
#[derive(Debug, Default, Clone, Copy)]
struct Columns {
    pos_tag: bool,
    lemma: bool,
    stem: bool,
    frequency_count: bool,
    categories: bool,
}

impl Columns {
    /// Canonical column order used for the SQLite table (matches the parquet builder).
    fn names(&self) -> Vec<&'static str> {
        let mut v = vec!["word"];
        if self.pos_tag {
            v.push("pos_tag");
        }
        if self.lemma {
            v.push("lemma");
        }
        if self.stem {
            v.push("stem");
        }
        if self.frequency_count {
            v.push("frequency_count");
        }
        v.push("level");
        if self.categories {
            v.push("categories");
        }
        v
    }
}

/// Column names of a table in the .db (PRAGMA table_info).
fn table_columns(conn: &Connection, table: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let names = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(names)
}

// ---------------------------------------------------------------------------
// Parquet reading (arrow-rs), columns resolved by NAME so any subset works.
// Handles both Utf8 and Utf8View producers.
// ---------------------------------------------------------------------------

/// Extract a StringArray out of a column no matter how the producer encoded it.
fn as_string_array(batch: &RecordBatch, col: usize) -> Result<StringArray> {
    let array = batch.column(col);
    if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(a.clone());
    }
    let casted = cast(array, &DataType::Utf8).context("cast to Utf8")?;
    Ok(casted.as_any().downcast_ref::<StringArray>().unwrap().clone())
}

/// Flatten a list<string> column into '|'-joined strings (empty for empty/null lists).
fn as_joined_categories(batch: &RecordBatch, col: usize) -> Result<Vec<String>> {
    let array = batch.column(col);
    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let owned;
    let lists: &ListArray = match array.as_any().downcast_ref::<ListArray>() {
        Some(l) => l,
        None => {
            owned = cast(array, &list_type).context("cast categories to List<Utf8>")?;
            owned.as_any().downcast_ref::<ListArray>().unwrap()
        }
    };
    let mut out = Vec::with_capacity(lists.len());
    for i in 0..lists.len() {
        if lists.is_null(i) {
            out.push(String::new());
            continue;
        }
        let vals = lists.value(i);
        let strs = match vals.as_any().downcast_ref::<StringArray>() {
            Some(s) => s.clone(),
            None => cast(&vals, &DataType::Utf8)?
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .clone(),
        };
        let joined = (0..strs.len()).map(|j| strs.value(j)).collect::<Vec<_>>().join("|");
        out.push(joined);
    }
    Ok(out)
}

fn read_parquet(path: &str) -> Result<(Vec<CefrRow>, Columns)> {
    let file = File::open(path).with_context(|| format!("open {path}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();

    // map column name -> parquet index for every available column
    let idx = |name: &str| schema.index_of(name).ok();
    let i_word = idx("word").context("parquet is missing required column 'word'")?;
    let i_level = idx("level").context("parquet is missing required column 'level'")?;
    let i_pos = idx("pos_tag");
    let i_lemma = idx("lemma");
    let i_stem = idx("stem");
    let i_freq = idx("frequency_count");
    let i_cats = idx("categories");

    let cols = Columns {
        pos_tag: i_pos.is_some(),
        lemma: i_lemma.is_some(),
        stem: i_stem.is_some(),
        frequency_count: i_freq.is_some(),
        categories: i_cats.is_some(),
    };

    let reader = builder.with_batch_size(65_536).build()?;
    let mut rows = Vec::with_capacity(250_000);
    for batch in reader {
        let batch = batch?;
        let word = as_string_array(&batch, i_word)?;
        let level = batch
            .column(i_level)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("level is not Float32")?;
        // optional columns: only materialize when present in this file
        let pos = i_pos.map(|i| as_string_array(&batch, i)).transpose()?;
        let lemma = i_lemma.map(|i| as_string_array(&batch, i)).transpose()?;
        let stem = i_stem.map(|i| as_string_array(&batch, i)).transpose()?;
        let freq = i_freq
            .map(|i| {
                batch
                    .column(i)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .cloned()
                    .context("frequency_count is not UInt64")
            })
            .transpose()?;
        let cats = i_cats.map(|i| as_joined_categories(&batch, i)).transpose()?;

        for r in 0..batch.num_rows() {
            rows.push(CefrRow {
                word: word.value(r).to_string(),
                pos_tag: pos.as_ref().filter(|a| !a.is_null(r)).map(|a| a.value(r).to_string()),
                lemma: lemma.as_ref().filter(|a| !a.is_null(r)).map(|a| a.value(r).to_string()),
                stem: stem.as_ref().filter(|a| !a.is_null(r)).map(|a| a.value(r).to_string()),
                frequency_count: freq.as_ref().and_then(|a| (!a.is_null(r)).then(|| a.value(r))),
                level: level.value(r),
                categories: cats.as_ref().map(|c| c[r].clone()),
            });
        }
    }
    Ok((rows, cols))
}

// ---------------------------------------------------------------------------
// 1) builddb: parquet -> SQLite db on the client machine (schema follows input)
// ---------------------------------------------------------------------------
fn row_values(r: &CefrRow, names: &[&str]) -> Vec<Value> {
    names
        .iter()
        .map(|n| match *n {
            "word" => Value::Text(r.word.clone()),
            "pos_tag" => r.pos_tag.clone().map_or(Value::Null, Value::Text),
            "lemma" => r.lemma.clone().map_or(Value::Null, Value::Text),
            "stem" => r.stem.clone().map_or(Value::Null, Value::Text),
            "frequency_count" => r
                .frequency_count
                .map_or(Value::Null, |f| Value::Integer(f as i64)), // SQLite INTEGER is signed
            "level" => Value::Real(r.level as f64),
            "categories" => r.categories.clone().map_or(Value::Null, Value::Text),
            _ => Value::Null,
        })
        .collect()
}

struct BuildStats {
    rows: usize,
    cols: Columns,
    read: Duration,
    total: Duration,
}

fn build_db_core(parquet_path: &str, db_path: &str) -> Result<BuildStats> {
    let t0 = Instant::now();
    let (rows, cols) = read_parquet(parquet_path)?;
    let t_read = t0.elapsed();

    let names = cols.names();
    let ddl = format!(
        "DROP TABLE IF EXISTS cefr;\nCREATE TABLE cefr (\n    {}\n);\nCREATE INDEX idx_cefr_word ON cefr(word);",
        names
            .iter()
            .map(|n| match *n {
                "word" => "word TEXT NOT NULL",
                "level" => "level REAL NOT NULL",
                "frequency_count" => "frequency_count INTEGER",
                other => match other {
                    "pos_tag" => "pos_tag TEXT",
                    "lemma" => "lemma TEXT",
                    "stem" => "stem TEXT",
                    "categories" => "categories TEXT",
                    _ => unreachable!(),
                },
            })
            .collect::<Vec<_>>()
            .join(",\n    ")
    );

    let mut conn = Connection::open(db_path).with_context(|| format!("create {db_path}"))?;
    conn.execute_batch("PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF;")?;
    let tx = conn.transaction()?;
    tx.execute_batch(&ddl)?;
    {
        let placeholders = names
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!("INSERT INTO cefr ({}) VALUES ({placeholders})", names.join(", "));
        let mut stmt = tx.prepare_cached(&sql)?;
        for r in &rows {
            stmt.execute(params_from_iter(row_values(r, &names)))?;
        }
    }
    tx.commit()?;
    // user_version lets the app skip the rebuild when the dataset hasn't changed
    conn.execute_batch("ANALYZE; PRAGMA optimize; PRAGMA user_version = 1;")?;

    Ok(BuildStats { rows: rows.len(), cols, read: t_read, total: t0.elapsed() })
}

fn build_db(parquet_path: &str, db_path: &str) -> Result<()> {
    let stats = build_db_core(parquet_path, db_path)?;
    println!(
        "builddb: {} rows, columns [{}] | read parquet {:?} | total {:?} -> {db_path}",
        stats.rows,
        stats.cols.names().join(", "),
        stats.read,
        stats.total
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 2) Lookups — direct port of the Python notebook's logic
// ---------------------------------------------------------------------------

/// Python equivalent:
///   SELECT COALESCE(AVG(CASE WHEN pt.tag = ? THEN wp.level END), AVG(wp.level)) ...
/// Falls back to the average over all senses when the exact (word, pos) combo is
/// unknown — or when the db has no pos_tag column at all.
fn lookup_level(conn: &Connection, word: &str, pos_tag: &str) -> Result<Option<f64>> {
    let has_pos = table_columns(conn, "cefr")?.iter().any(|c| c == "pos_tag");
    let sql = if has_pos {
        "SELECT COALESCE(
             AVG(CASE WHEN pos_tag = ?1 THEN level END),
             AVG(level)
         )
         FROM cefr
         WHERE word = ?2"
    } else {
        "SELECT AVG(level) FROM cefr WHERE word = ?2"
    };
    let level: Option<f64> = conn.query_row(sql, params![pos_tag, word], |row| row.get(0))?;
    Ok(level)
}

/// Batch version mirroring fetch_word_pos_level_tokens() from Text-Analizer.ipynb:
/// one round trip for many (word, pos) pairs, using a VALUES CTE.
fn fetch_levels_batch(
    conn: &Connection,
    pairs: &[(String, String)],
) -> Result<HashMap<(String, String), f64>> {
    let has_pos = table_columns(conn, "cefr")?.iter().any(|c| c == "pos_tag");
    let mut out = HashMap::with_capacity(pairs.len());
    for chunk in pairs.chunks(500) {
        // 2 bind params per pair; keep far below SQLITE_MAX_VARIABLE_NUMBER
        let placeholders = vec!["(?, ?)"; chunk.len()].join(", ");
        let level_expr = if has_pos {
            "COALESCE(AVG(CASE WHEN c.pos_tag = wanted.pos_tag THEN c.level END), AVG(c.level))"
        } else {
            "AVG(c.level)"
        };
        let sql = format!(
            "WITH wanted(word, pos_tag) AS (VALUES {placeholders})
             SELECT wanted.word, wanted.pos_tag, {level_expr} AS avg_level
             FROM wanted
             JOIN cefr c ON c.word = wanted.word
             GROUP BY wanted.word, wanted.pos_tag"
        );
        let mut stmt = conn.prepare(&sql)?;
        let flat = chunk.iter().flat_map(|(w, p)| [w.as_str(), p.as_str()]);
        let rows = stmt.query_map(params_from_iter(flat), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, f64>(2)?))
        })?;
        for r in rows {
            let (w, p, lvl) = r?;
            out.insert((w, p), lvl);
        }
    }
    Ok(out)
}

fn level_to_cefr(level: f64) -> &'static str {
    match level.round() as i64 {
        1 => "A1",
        2 => "A2",
        3 => "B1",
        4 => "B2",
        5 => "C1",
        6 => "C2",
        _ => "?",
    }
}

fn fmt_value(v: &Value) -> String {
    match v {
        Value::Null => "-".into(),
        Value::Integer(i) => i.to_string(),
        Value::Real(f) => format!("{f:.2}"),
        Value::Text(s) => s.clone(),
        _ => "?".into(),
    }
}

// ---------------------------------------------------------------------------
// 3) memory: skip SQLite entirely — leanest possible client
// ---------------------------------------------------------------------------
fn memory_demo(parquet_path: &str, word: &str, pos_tag: &str) -> Result<()> {
    let t0 = Instant::now();
    let (rows, cols) = read_parquet(parquet_path)?;
    // word -> list of (pos_tag, level); pos_tag is None when the file has no such column
    let mut map: HashMap<String, Vec<(Option<String>, f32)>> = HashMap::new();
    for r in rows {
        map.entry(r.word).or_default().push((r.pos_tag, r.level));
    }
    println!(
        "loaded {} words in {:?} (one-time startup cost{})",
        map.len(),
        t0.elapsed(),
        if cols.pos_tag { "" } else { ", no pos_tag column -> word-level averages" }
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
        Some(l) => println!("{word} ({pos_tag}): {:.2} -> {}   [{:?}]", l, level_to_cefr(l), t.elapsed()),
        None => println!("{word}: not found"),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 4) analyze: full Text-Analizer.ipynb port — tokenize+tag with nlprule
//    (replaces spaCy + LemmInflect), then look up every (word, pos) in cefr.db
// ---------------------------------------------------------------------------

// Same contraction map as the Python notebook
#[cfg(feature = "nlp")]
const ABBREVIATION_MAPPING: [(&str, &str); 7] = [
    ("'m", "am"),
    ("'s", "is"),
    ("'re", "are"),
    ("'ve", "have"),
    ("'d", "had"),
    ("n't", "not"),
    ("'ll", "will"),
];

/// LanguageTool tags are Penn-style but with extra suffixes (`NN:U`, `IN/that`);
/// strip them so they match this dataset's Penn Treebank tags.
#[cfg(feature = "nlp")]
fn normalize_pos(tag: &str) -> &str {
    match tag.find(|c| c == ':' || c == '/') {
        Some(end) => &tag[..end],
        None => tag,
    }
}

#[cfg(feature = "nlp")]
fn load_tokenizer(path: &std::path::Path) -> Result<nlprule::Tokenizer> {
    use std::io::{Cursor, Read};
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let bytes = match extension {
        "zst" => zstd::stream::decode_all(File::open(path).with_context(|| format!("open {}", path.display()))?)
            .context("decompress model")?,
        "gz" => {
            let mut out = Vec::new();
            flate2::read::GzDecoder::new(
                File::open(path).with_context(|| format!("open {}", path.display()))?,
            )
            .read_to_end(&mut out)
            .context("decompress model")?;
            out
        }
        _ => std::fs::read(path).with_context(|| format!("read model {}", path.display()))?,
    };
    nlprule::Tokenizer::from_reader(Cursor::new(bytes))
        .with_context(|| format!("load tokenizer model {}", path.display()))
}

/// Model path: explicit arg > $CEFR_MODEL > ./models/ (or ./data/) defaults.
/// Returns None when no model file is found (embedded build may still apply).
#[cfg(feature = "nlp")]
fn resolve_model_path(arg: Option<&String>) -> Option<std::path::PathBuf> {
    if let Some(p) = arg {
        return Some(p.into());
    }
    if let Ok(p) = std::env::var("CEFR_MODEL") {
        return Some(p.into());
    }
    for cand in [
        "models/en_tokenizer.bin.zst",
        "models/en_tokenizer.bin",
        "models/en_tokenizer.bin.gz",
        "data/en_tokenizer.bin.zst",
        "data/en_tokenizer.bin",
        "data/en_tokenizer.bin.gz",
    ] {
        let p = std::path::PathBuf::from(cand);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Model bytes compiled into the binary (`embed-model` feature, dev convenience).
#[cfg(all(feature = "nlp", feature = "embed-model"))]
fn embedded_model() -> Option<&'static [u8]> {
    Some(include_bytes!(concat!(env!("OUT_DIR"), "/en_tokenizer.bin")))
}

#[cfg(all(feature = "nlp", not(feature = "embed-model")))]
fn embedded_model() -> Option<&'static [u8]> {
    None
}

#[cfg(feature = "nlp")]
fn analyze(db_path: &str, text_path: &str, model: Option<&String>) -> Result<()> {
    let text = std::fs::read_to_string(text_path)
        .with_context(|| format!("read {text_path}"))?;

    // --- NLP phase (like `spacy.load("en_core_web_sm")` + lemminflect) ------
    let t0 = Instant::now();
    let tokenizer = match resolve_model_path(model) {
        Some(path) => load_tokenizer(&path)?,
        None => match embedded_model() {
            Some(bytes) => {
                use std::io::Cursor;
                nlprule::Tokenizer::from_reader(Cursor::new(bytes))
                    .context("load embedded tokenizer model")?
            }
            None => bail!(
                "tokenizer model not found: pass it as 3rd arg, set CEFR_MODEL, put \
                 en_tokenizer.bin[.zst|.gz] into ./models, or build with --features embed-model"
            ),
        },
    };

    // (word, lemma, pos_tag) per token — mirrors custom_tokenize_text()
    let mut tokens: Vec<(String, String, String)> = Vec::new();
    for sentence in tokenizer.pipe(&text) {
        for token in sentence {
            let word_ref = token.word();
            let mut word = word_ref.text().as_str().to_lowercase().trim().to_string();
            // first (best) disambiguated tag
            let (mut lemma, pos) = match word_ref.tags().first() {
                Some(wd) => (
                    wd.lemma().as_str().to_lowercase(),
                    normalize_pos(wd.pos().as_str()).to_string(),
                ),
                None => (word.clone(), String::new()),
            };
            // contractions -> base form, exactly like the python ABBREVIATION_MAPPING
            if let Some((_, expansion)) = ABBREVIATION_MAPPING.iter().find(|(a, _)| *a == word) {
                word = expansion.to_string();
                lemma = expansion.to_string();
            }
            tokens.push((word, lemma, pos));
        }
    }
    let nlp_ms = t0.elapsed().as_millis();

    // --- CEFR phase (like get_levels_tokens() in the notebook) ---------------
    let t1 = Instant::now();
    let conn = Connection::open(db_path)?;
    let has_pos = table_columns(&conn, "cefr")?.iter().any(|c| c == "pos_tag");

    // unique (word, pos) pairs, skipping punctuation (python: is_punctuation)
    let mut pairs: Vec<(String, String)> = tokens
        .iter()
        .filter(|(w, _, _)| w.chars().any(char::is_alphabetic))
        .map(|(w, _, p)| (w.clone(), p.clone()))
        .collect();
    pairs.sort();
    pairs.dedup();

    let levels = fetch_levels_batch(&conn, &pairs)?;
    let got = |w: &str, p: &str| levels.get(&(w.to_string(), p.to_string())).copied();

    // per-token level list, like level_tokens in the notebook
    let level_tokens: Vec<(String, String, String, Option<f64>)> = tokens
        .iter()
        .filter(|(w, _, _)| w.chars().any(char::is_alphabetic))
        .map(|(w, l, p)| (w.clone(), l.clone(), p.clone(), got(w, p)))
        .collect();
    let cefr_ms = t1.elapsed().as_millis();

    println!("NLP: {nlp_ms} ms");
    println!("CEFR levels: {cefr_ms} ms");
    println!("{}", "-".repeat(30));
    println!("Text length: {}", text.len());
    println!("Total tokens: {}", tokens.len());
    if !has_pos {
        println!("note: db has no pos_tag column -> word-average levels; POS column is the tokenizer's guess");
    }
    println!();

    // --- per-token table (python cell 10) ------------------------------------
    println!("{:<26}\t{:<26}\tPOS\tLEVEL\tCEFR", "WORD", "LEMMA");
    println!("{}", "-".repeat(85));
    for (w, l, p, lvl) in &level_tokens {
        match lvl {
            Some(lvl) => println!("{w:<26}\t{l:<26}\t{p}\t{lvl:.2}\t{}", level_to_cefr(*lvl)),
            None => println!("{w:<26}\t{l:<26}\t{p}\t--\tN/A"),
        }
    }
    println!();

    // --- statistics blocks (python cells 11-14) ------------------------------
    let mut total = [0usize; 6];
    for (_, _, _, lvl) in &level_tokens {
        if let Some(l) = lvl {
            let i = (l.round() as usize).clamp(1, 6);
            total[i - 1] += 1;
        }
    }
    println!("CEFR statistic (total words):");
    for (i, c) in total.iter().enumerate() {
        println!("{}: {}", level_to_cefr((i + 1) as f64), c);
    }
    println!();

    // dedup key: (word, pos) with a pos-aware db; word alone for word-averaged dbs,
    // otherwise the tagger tagging one word two ways would count it twice
    let dedup_key = |w: &str, p: &str| (w.to_string(), if has_pos { p.to_string() } else { String::new() });

    let mut unique_seen = std::collections::HashSet::new();
    let mut unique = [0usize; 6];
    for (w, _, p, lvl) in &level_tokens {
        if let Some(l) = lvl {
            if unique_seen.insert(dedup_key(w, p)) {
                let i = (l.round() as usize).clamp(1, 6);
                unique[i - 1] += 1;
            }
        }
    }
    println!("CEFR statistic (unique words):");
    for (i, c) in unique.iter().enumerate() {
        println!("{}: {}", level_to_cefr((i + 1) as f64), c);
    }
    println!();

    let mut not_found: Vec<String> = level_tokens
        .iter()
        .filter(|(_, _, _, lvl)| lvl.is_none())
        .map(|(w, _, _, _)| w.clone())
        .collect();
    not_found.sort();
    not_found.dedup();
    println!("Not found words: {}", not_found.len());
    if !not_found.is_empty() {
        for w in &not_found {
            println!("{w}");
        }
    }
    println!();

    // words with level B2 and higher (round(level) >= 4), like filter_for_desired_level(_, 4)
    let mut hard: Vec<(String, String, f64)> = Vec::new();
    let mut hard_seen = std::collections::HashSet::new();
    for (w, _, p, lvl) in &level_tokens {
        if let Some(l) = lvl {
            if l.round() >= 4.0 && hard_seen.insert(dedup_key(w, p)) {
                hard.push((w.clone(), p.clone(), *l));
            }
        }
    }
    hard.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    println!("\tWords with level B2 and higher: {}", hard.len());
    for (w, p, l) in &hard {
        println!("{w:<26} {p:<6} {l:.2}   {}", level_to_cefr(*l));
    }
    Ok(())
}

#[cfg(not(feature = "nlp"))]
fn analyze(_db_path: &str, _text_path: &str, _model: Option<&String>) -> Result<()> {
    anyhow::bail!("the `analyze` command needs the NLP feature: cargo build --release --features nlp")
}

// ---------------------------------------------------------------------------
// 5) bench: parquet read / sqlite rebuild / lookup latency, per file
// ---------------------------------------------------------------------------
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

fn bench_one(path: &str) -> Result<BenchReport> {
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
    let stats = build_db_core(path, db_path.to_str().context("temp path utf8")?)?;
    let build_ms = t.elapsed().as_secs_f64() * 1e3;
    let db_mb = std::fs::metadata(&db_path)?.len() as f64 / 1_048_576.0;

    let conn = Connection::open(&db_path)?;
    let words: i64 = conn.query_row("SELECT COUNT(DISTINCT word) FROM cefr", [], |r| r.get(0))?;
    let words = words as usize;

    // --- phase 3: hot-path single lookups (statement prepared once) ---
    // rusqlite accepts trailing unused params, so (pos, word) binds fine
    // even for the no-pos variant's "WHERE word = ?2"
    let sql = if cols.pos_tag {
        "SELECT COALESCE(AVG(CASE WHEN pos_tag = ?1 THEN level END), AVG(level)) FROM cefr WHERE word = ?2"
    } else {
        "SELECT AVG(level) FROM cefr WHERE word = ?2"
    };
    let mut stmt = conn.prepare(sql)?;

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
    let pairs: Vec<(String, String)> = sample.iter().map(|w| (w.to_string(), "NN".into())).collect();
    let t = Instant::now();
    let m = fetch_levels_batch(&conn, &pairs)?;
    std::hint::black_box(m.len());
    let batch_us = t.elapsed().as_micros() as f64 / pairs.len() as f64;
    drop(conn);

    // --- phase 6: no-sqlite in-memory map ---
    let t = Instant::now();
    let mut map: HashMap<&str, Vec<(Option<&str>, f32)>> = HashMap::with_capacity(words);
    for r in &rows {
        map.entry(r.word.as_str()).or_default().push((r.pos_tag.as_deref(), r.level));
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
        file: path.to_string(),
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

fn bench(files: &[String]) -> Result<()> {
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
            r.file, r.parquet_mb, r.read_ms, r.build_ms, r.db_mb, r.single_us, r.miss_us, r.batch_us
        );
    }
    println!();
    for r in &reports {
        println!("{}", r.file);
        println!("  rows {:>7} | distinct words {:>7} | columns [{}]", r.rows, r.words, r.cols);
        println!("  in-memory map: {:.1} ms load, {:.3} µs/lookup (no sqlite at all)", r.mem_load_ms, r.mem_us);
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

// ---------------------------------------------------------------------------
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let rest = args.get(1..).unwrap_or(&[]);
    match (cmd, rest) {
        ("builddb", [parquet, db]) => build_db(parquet, db),
        ("lookup", [db, word]) => {
            let conn = Connection::open(db)?;
            let cols = table_columns(&conn, "cefr")?;
            // show the informative columns that exist in this db
            let shown: Vec<&str> = ["pos_tag", "lemma", "level", "frequency_count"]
                .into_iter()
                .filter(|c| cols.iter().any(|x| x == c))
                .collect();
            let sql = format!(
                "SELECT {} FROM cefr WHERE word = ?1 ORDER BY level",
                shown.join(", ")
            );
            let mut stmt = conn.prepare(&sql)?;
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
                    parts.push(fmt_value(v));
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
            let conn = Connection::open(db)?;
            match lookup_level(&conn, word, pos)? {
                Some(l) => {
                    println!("{word} ({pos}): {l:.2} -> {}", level_to_cefr(l));
                    Ok(())
                }
                None => bail!("{word}: not found in dataset"),
            }
        }
        ("batch", [db, pairs @ ..]) if !pairs.is_empty() => {
            let conn = Connection::open(db)?;
            let parsed: Vec<(String, String)> = pairs
                .iter()
                .map(|s| {
                    let (w, p) = s.rsplit_once(':').unwrap_or((s.as_str(), ""));
                    (w.to_string(), p.to_string())
                })
                .collect();
            let t = Instant::now();
            let levels = fetch_levels_batch(&conn, &parsed)?;
            for (w, p) in &parsed {
                match levels.get(&(w.clone(), p.clone())) {
                    Some(l) => println!("{w}\t{p}\t{l:.2}\t{}", level_to_cefr(*l)),
                    None => println!("{w}\t{p}\tnot found"),
                }
            }
            eprintln!("-- {} pairs in {:?}", parsed.len(), t.elapsed());
            Ok(())
        }
        ("memory", [parquet, word, pos]) => memory_demo(parquet, word, pos),
        ("analyze", [db, text]) => analyze(db, text, None),
        ("analyze", [db, text, model]) => analyze(db, text, Some(model)),
        ("bench", _) if args.len() >= 2 => bench(&args[1..]),
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
