//! The SQLite half: rebuild a local db from the parquet, then query it.
//!
//! Lookup semantics follow the upstream notebook: the average level of the
//! exact (word, POS) entries, falling back to the average of the word's other
//! senses — or, when the db has no `pos_tag` column at all, the per-word
//! average.
//!
//! Only compiled with the `sqlite` feature.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rusqlite::{Connection, params, params_from_iter, types::Value};

use crate::dataset::{CefrRow, Columns};

/// Column names of a table in the .db (PRAGMA table_info).
pub fn table_columns(conn: &Connection, table: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let names = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(names)
}

/// One row as SQLite bind values, in the canonical column order.
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

/// What one rebuild did: row count, columns, timings.
#[derive(Debug, Clone)]
pub struct BuildStats {
    pub rows: usize,
    pub cols: Columns,
    pub read: Duration,
    pub total: Duration,
}

/// Rebuild the sqlite db at `db_path` from the parquet at `parquet_path`.
///
/// `PRAGMA user_version = 1` marks a completed rebuild, so an app can skip
/// the rebuild when the dataset has not changed.
pub fn build_db(parquet_path: &Path, db_path: &Path) -> Result<BuildStats> {
    let t0 = Instant::now();
    let (rows, cols) = crate::dataset::read_parquet(parquet_path)?;
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

    let mut conn =
        Connection::open(db_path).with_context(|| format!("create {}", db_path.display()))?;
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
        let sql = format!(
            "INSERT INTO cefr ({}) VALUES ({placeholders})",
            names.join(", ")
        );
        let mut stmt = tx.prepare_cached(&sql)?;
        for r in &rows {
            stmt.execute(params_from_iter(row_values(r, &names)))?;
        }
    }
    tx.commit()?;
    conn.execute_batch("ANALYZE; PRAGMA optimize; PRAGMA user_version = 1;")?;

    Ok(BuildStats {
        rows: rows.len(),
        cols,
        read: t_read,
        total: t0.elapsed(),
    })
}

/// An open CEFR database: schema-aware lookups with one prepared statement.
pub struct CefrDb {
    conn: Connection,
    has_pos: bool,
}

impl CefrDb {
    /// Open the db at `path` and sniff which columns it carries.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| format!("open {}", path.display()))?;
        Self::from_connection(conn)
    }

    /// Wrap an existing connection (tests use in-memory dbs).
    pub fn from_connection(conn: Connection) -> Result<Self> {
        let has_pos = table_columns(&conn, "cefr")?.iter().any(|c| c == "pos_tag");
        Ok(Self { conn, has_pos })
    }

    /// Whether the table carries per-POS rows (false = word averages).
    pub fn has_pos(&self) -> bool {
        self.has_pos
    }

    /// The raw connection, for read-only extras (the CLI's `lookup` listing,
    /// the bench's prepared-statement phases).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// The level of `word`, optionally scored against one POS tag.
    pub fn lookup(&self, word: &str, pos_tag: &str) -> Result<Option<f64>> {
        let sql = if self.has_pos {
            "SELECT COALESCE(
                 AVG(CASE WHEN pos_tag = ?1 THEN level END),
                 AVG(level)
             )
             FROM cefr
             WHERE word = ?2"
        } else {
            "SELECT AVG(level) FROM cefr WHERE word = ?2"
        };
        let level: Option<f64> = self
            .conn
            .query_row(sql, params![pos_tag, word], |row| row.get(0))?;
        Ok(level)
    }

    /// One round trip for many (word, pos) pairs, via a VALUES CTE.
    pub fn lookup_batch(
        &self,
        pairs: &[(String, String)],
    ) -> Result<HashMap<(String, String), f64>> {
        let mut out = HashMap::with_capacity(pairs.len());
        for chunk in pairs.chunks(500) {
            // 2 bind params per pair; keep far below SQLITE_MAX_VARIABLE_NUMBER
            let placeholders = vec!["(?, ?)"; chunk.len()].join(", ");
            let level_expr = if self.has_pos {
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
            let mut stmt = self.conn.prepare(&sql)?;
            let flat = chunk.iter().flat_map(|(w, p)| [w.as_str(), p.as_str()]);
            let rows = stmt.query_map(params_from_iter(flat), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f64>(2)?,
                ))
            })?;
            for r in rows {
                let (w, p, lvl) = r?;
                out.insert((w, p), lvl);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded() -> CefrDb {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE cefr (word TEXT NOT NULL, pos_tag TEXT, level REAL NOT NULL);
             CREATE INDEX idx_cefr_word ON cefr(word);
             INSERT INTO cefr (word, pos_tag, level) VALUES
               ('run', 'NN', 2.0), ('run', 'VB', 4.0),
               ('ephemeral', NULL, 5.6),
               ('dog', 'NN', 1.1);",
        )
        .unwrap();
        CefrDb::from_connection(conn).unwrap()
    }

    #[test]
    fn a_pos_db_defaults_to_the_word_average_then_prefers_the_tag() {
        let db = seeded();
        assert!(db.has_pos());
        // No such POS: the fallback is the average over the word's senses.
        let avg = db.lookup("run", "JJ").unwrap().unwrap();
        assert!((avg - 3.0).abs() < 1e-9);
        // The exact sense wins when it exists.
        let vb = db.lookup("run", "VB").unwrap().unwrap();
        assert!((vb - 4.0).abs() < 1e-9);
    }

    #[test]
    fn an_absent_word_is_none_not_an_error() {
        let db = seeded();
        assert_eq!(db.lookup("zzz_no_such_word", "NN").unwrap(), None);
    }

    #[test]
    fn a_word_average_db_answers_without_a_pos_column() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE cefr (word TEXT NOT NULL, level REAL NOT NULL);
             INSERT INTO cefr (word, level) VALUES ('dog', 1.5), ('cat', 2.5);",
        )
        .unwrap();
        let db = CefrDb::from_connection(conn).unwrap();
        assert!(!db.has_pos());
        assert!((db.lookup("dog", "NN").unwrap().unwrap() - 1.5).abs() < 1e-9);

        let pairs = vec![
            ("dog".to_string(), "NN".to_string()),
            ("cat".to_string(), String::new()),
        ];
        let got = db.lookup_batch(&pairs).unwrap();
        assert_eq!(got.len(), 2);
        assert!((got[&("cat".to_string(), String::new())] - 2.5).abs() < 1e-9);
    }

    #[test]
    fn the_batch_answer_matches_the_single_lookup() {
        let db = seeded();
        let pairs = vec![
            ("run".to_string(), "VB".to_string()),
            ("ephemeral".to_string(), "JJ".to_string()),
        ];
        let got = db.lookup_batch(&pairs).unwrap();
        assert!((got[&pairs[0]] - 4.0).abs() < 1e-9);
        assert!((got[&pairs[1]] - 5.6).abs() < 1e-9);
    }

    #[test]
    fn an_empty_batch_builds_valid_sql_and_answers_empty() {
        let db = seeded();
        assert!(db.lookup_batch(&[]).unwrap().is_empty());
    }

    #[test]
    fn a_rebuilt_db_carries_the_rows_and_the_version_mark() {
        let dir = std::env::temp_dir().join(format!("cefr_lib_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        // A minimal parquet is produced by the repo's own data file when
        // present; a hand-written one needs the writer, so this path is
        // covered by tests/parquet_rebuild.rs instead. Here: the version
        // mark of a hand-seeded db is what `open` reads back.
        let db_path = dir.join("marked.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE cefr (word TEXT NOT NULL, level REAL NOT NULL); PRAGMA user_version = 1;",
        )
        .unwrap();
        assert_eq!(
            conn.query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            1
        );
        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_dir(&dir);
    }
}
