//! Reading the denormalized parquet: rows resolved by column NAME, so any
//! export from tools/csv_to_parquet.py works as long as `word` and `level`
//! exist.
//!
//! Only compiled with the `parquet` feature; the reader handles both Utf8 and
//! Utf8View producers, and `level` as Float32 or Float64.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{Array, Float32Array, ListArray, StringArray, UInt64Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// One denormalized row: everything optional except word + level.
#[derive(Debug, Default)]
pub struct CefrRow {
    pub word: String,
    pub pos_tag: Option<String>,
    pub lemma: Option<String>,
    pub stem: Option<String>,
    pub frequency_count: Option<u64>,
    pub level: f32,
    /// '|'-separated category titles, empty when the column is absent.
    pub categories: Option<String>,
}

/// Which optional columns the parquet file actually has.
#[derive(Debug, Default, Clone, Copy)]
pub struct Columns {
    pub pos_tag: bool,
    pub lemma: bool,
    pub stem: bool,
    pub frequency_count: bool,
    pub categories: bool,
}

impl Columns {
    /// Canonical column order used for the SQLite table (matches the parquet
    /// builder).
    pub fn names(&self) -> Vec<&'static str> {
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

/// Extract a StringArray out of a column no matter how the producer encoded
/// it.
fn as_string_array(batch: &RecordBatch, col: usize) -> Result<StringArray> {
    let array = batch.column(col);
    if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(a.clone());
    }
    let casted = cast(array, &DataType::Utf8).context("cast to Utf8")?;
    Ok(casted
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("cast to Utf8 yields a StringArray")
        .clone())
}

/// Flatten a list<string> column into '|'-joined strings (empty for empty/null
/// lists).
fn as_joined_categories(batch: &RecordBatch, col: usize) -> Result<Vec<String>> {
    let array = batch.column(col);
    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let owned;
    let lists: &ListArray = match array.as_any().downcast_ref::<ListArray>() {
        Some(l) => l,
        None => {
            owned = cast(array, &list_type).context("cast categories to List<Utf8>")?;
            owned
                .as_any()
                .downcast_ref::<ListArray>()
                .expect("cast to List<Utf8> yields a ListArray")
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
                .expect("cast to Utf8 yields a StringArray")
                .clone(),
        };
        let joined = (0..strs.len())
            .map(|j| strs.value(j))
            .collect::<Vec<_>>()
            .join("|");
        out.push(joined);
    }
    Ok(out)
}

/// The level column as Float32, casting Float64 producers on the way.
fn as_f32_levels(batch: &RecordBatch, col: usize) -> Result<Float32Array> {
    let array = batch.column(col);
    if let Some(a) = array.as_any().downcast_ref::<Float32Array>() {
        return Ok(a.clone());
    }
    let casted = cast(array, &DataType::Float32).context("cast level to Float32")?;
    Ok(casted
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("cast to Float32 yields a Float32Array")
        .clone())
}

/// Read every row of the parquet at `path`.
pub fn read_parquet(path: &Path) -> Result<(Vec<CefrRow>, Columns)> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    read_parquet_from(file).with_context(|| format!("read parquet {}", path.display()))
}

/// Read every row of an already-open parquet source.
pub fn read_parquet_from(reader: File) -> Result<(Vec<CefrRow>, Columns)> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(reader)?;
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
        let level = as_f32_levels(&batch, i_level)?;
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
        let cats = i_cats
            .map(|i| as_joined_categories(&batch, i))
            .transpose()?;

        for r in 0..batch.num_rows() {
            rows.push(CefrRow {
                word: word.value(r).to_string(),
                pos_tag: pos
                    .as_ref()
                    .filter(|a| !a.is_null(r))
                    .map(|a| a.value(r).to_string()),
                lemma: lemma
                    .as_ref()
                    .filter(|a| !a.is_null(r))
                    .map(|a| a.value(r).to_string()),
                stem: stem
                    .as_ref()
                    .filter(|a| !a.is_null(r))
                    .map(|a| a.value(r).to_string()),
                frequency_count: freq
                    .as_ref()
                    .and_then(|a| (!a.is_null(r)).then(|| a.value(r))),
                level: level.value(r),
                categories: cats.as_ref().map(|c| c[r].clone()),
            });
        }
    }
    Ok((rows, cols))
}
