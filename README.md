# cefr-rust

CEFR word-level lookup CLI in Rust, built on the
[Words-CEFR-Dataset](https://github.com/Maximax67/Words-CEFR-Dataset).

The dataset ships as a 19.6 MB SQLite database. This repo stores it as a single
denormalized parquet (1.4–2.8 MB), rebuilds a local `.db` in ~0.3–0.6 s on first
run, then serves lookups through SQLite — or straight from memory when SQLite is
not wanted.

## Layout

```
src/lib.rs               library root: the reusable half, feature-gated
src/level.rs             level bands: 1.0-6.0 floats <-> A1-C2 labels
src/dataset.rs           parquet reader (feature `parquet`)
src/db.rs                sqlite rebuild + CefrDb lookups (feature `sqlite`)
src/main.rs              CLI: builddb, lookup, batch, memory, analyze, bench
src/cli/                 the subcommands that are presentation, not library
build.rs                 model compilation, active only with --features embed-model
examples/make_model.rs   dev helper: regenerate the language model (needs network)
models/                  en_tokenizer.bin.zst — prebuilt runtime language model
tools/csv_to_parquet.py  builds the parquet from the CSV source tables
datasets/                source data: csv/*.csv, word_list_cefr.csv, database_model.png
data/                    built parquets + sample texts
demo/                    captured analyze/bench outputs
```

## As a library

Depend on it with the features you need; the CLI consumes the same API.

```toml
cefr = { git = "https://github.com/codewiththiha/cefr-rs.git" }
```

| feature | default | pulls in | gives you |
|---|---|---|---|
| `parquet` | yes | arrow, parquet | `dataset::{read_parquet, CefrRow, Columns}` |
| `sqlite` | yes | rusqlite (bundled) | `db::{build_db, CefrDb}` — rebuild + batch lookups |
| `nlp` | no | nlprule | the `analyze` pipeline (tokenizer model from a file) |
| `embed-model` | no | nlprule-build | `analyze` with the model compiled into the binary |

`CefrDb::lookup(word, pos)` and `CefrDb::lookup_batch(&[(word, pos)])` follow
the upstream notebook's semantics: the exact (word, POS) average, falling back
to the word's other senses — or, without a `pos_tag` column, the word average.


## Parquet files

| file | columns | size | rows |
|---|---|---|---|
| `data/cefr.zstd.parquet` | word, pos_tag, lemma, stem, frequency_count, level, categories | 2.80 MB | 248,447 |
| `data/cefr.core.parquet` | word, lemma, level | 1.36 MB | 173,033 |

Both are produced by `tools/csv_to_parquet.py`, which flattens the five
relational tables (`words` 1─N `word_pos` N─1 `pos_tags`, `word_pos` 1─N
`word_categories` N─1 `categories`, plus two self-FKs on `words` for stem/lemma).

## Regenerating the parquets

```bash
python3 tools/csv_to_parquet.py --columns word,lemma,level --out data/cefr.core.parquet
python3 tools/csv_to_parquet.py --out data/cefr.zstd.parquet          # all columns
python3 tools/csv_to_parquet.py --columns word,cefr --compression brotli --out x.parquet
```

- `--columns`: any subset of `word pos_tag lemma stem frequency_count level
  categories`, plus the derived `cefr` label (A1–C2 from `round(level)`).
  `word` and `level` are always kept. Without `pos_tag` the export collapses to
  one row per word (mean level, first non-null lemma, category union).
- `--merge-cefrj` / `--no-merge-cefrj` (default on): adds CEFR-J levels for the
  263 headwords missing from the base tables (e.g. `hello`). Existing rows are
  never overridden.
- `--compression zstd|snappy|brotli|gzip|lz4`, `--compression-level`,
  `--row-group-size`, `--repo` (defaults to `datasets/` in this repo).

## Build

```bash
cargo build --release                        # lookup CLI (12.6 MB)
cargo build --release --features nlp         # + analyze; model loaded from file (15.2 MB)
cargo build --release --features embed-model # + analyze; model compiled into the
                                             # binary (26.9 MB, no model file needed)
```

`embed-model` runs `build.rs`, which downloads LanguageTool data and compiles
`en_tokenizer.bin` into the binary on first build. The other modes build fully
offline.

## Usage

```bash
cefr builddb <in.parquet> <out.db>       # parquet -> sqlite (schema follows the input)
cefr lookup  <db> <word> [pos]           # single word, optionally per POS tag
cefr batch   <db> <word:pos> [...]       # one round trip per chunk (VALUES CTE)
cefr memory  <parquet> <word> <pos>      # no sqlite, lookups from a HashMap
cefr analyze <db> <text> [model]         # per-token CEFR stats for a text file
cefr bench   <parquet> [...]             # read/rebuild/lookup timings
```

The parquet reader resolves columns by name, so any export from the builder
works as long as `word` and `level` exist; the rebuilt sqlite schema follows the
input. When the db has no `pos_tag` column, queries fall back to per-word
averages automatically.

Lookup semantics follow the upstream notebook: average level of the exact
(word, POS) entries, falling back to the average of the word's other senses:

```sql
SELECT COALESCE(AVG(CASE WHEN pos_tag = ?1 THEN level END), AVG(level))
FROM cefr
WHERE word = ?2;
```

## Language model

`analyze` tags text with an nlprule LanguageTool model. Model bytes come from
the first match of:

1. 3rd CLI arg
2. `CEFR_MODEL` env var
3. `./models/en_tokenizer.bin[.zst|.gz]` (or `data/...`)
4. compiled-in bytes (`embed-model` builds only)

`.zst`, `.gz` and raw `.bin` are all accepted. A prebuilt
`models/en_tokenizer.bin.zst` (6.70 MB) is included in this repo. Equivalent
upstream files:

- `en_tokenizer.bin.gz` (7.17 MB, byte-identical after decompression):
  `https://github.com/bminixhofer/nlprule/releases/download/0.6.4/en_tokenizer.bin.gz`
- `en_rules.bin.gz` (grammar rules, not used here):
  `https://github.com/bminixhofer/nlprule/releases/download/0.6.4/en_rules.bin.gz`
- releases page: `https://github.com/bminixhofer/nlprule/releases`
- LanguageTool source zip (for building from scratch):
  `https://f000.backblazeb2.com/file/nlprule/en.zip`

Regenerate locally: `cargo run --example make_model` (dev-only, needs network).

## Benchmarks

`cefr bench data/cefr.zstd.parquet data/cefr.core.parquet` (median of 3 runs,
sandbox hardware, 1,000-lookup samples):

| metric | full (2.80 MB) | core (1.36 MB) |
|---|---|---|
| parquet read | 84 ms | 24 ms |
| sqlite rebuild | 540 ms → 12.2 MB db | 290 ms → 6.7 MB db |
| single lookup (prepared stmt) | ~11 µs | ~10 µs |
| miss (absent word) | ~8 µs | ~7 µs |
| batch (VALUES CTE) | ~5 µs/pair | ~4.6 µs/pair |
| in-memory map | 27 ms load, 0.14 µs/lookup | 24 ms load, 0.13 µs/lookup |

Raw output: `demo/bench.txt`; `analyze` captures for both texts and both db
variants: `demo/`.

## Data notes

- Levels are `REAL` in [1.0, 6.0] and map to A1–C2 by rounding (1=A1 … 6=C2),
  same as the upstream notebook's `DIFFICULTY_MAPPING_REVERSE`. No CEFR label is
  stored unless the derived `cefr` column is exported.
- `level` is an estimated CEFR level, not raw frequency: the upstream pipeline
  computed it from CEFR-J levels, per-level average frequencies, lemma/stem
  levels and cross-POS averaging. The raw signal lives in `frequency_count`.
- ~77 % of rows have NULL `lemma`: the dataset stores lemmas for inflected
  forms only.
- The upstream word list lost 263 CEFR-J headwords (`hello` is absent from
  `words.csv` entirely). The builder's CEFR-J merge re-adds them with their gold
  level; CEFR-J POS names map to Penn tags; `frequency_count` stays NULL.
- The POS column in `analyze` output always comes from the runtime tagger, never
  from the db. Against a core (word/lemma/level) db the levels are word
  averages, marked by a `note:` line; core dbs dedup unique/hard-word stats per
  word, full dbs per (word, pos).
- Runtime tagging (nlprule) is not byte-identical to spaCy used by the upstream
  notebook; the COALESCE fallback absorbs most mismatches. See the distribution
  drift in `demo/forest_full.txt` vs the upstream README numbers.

## Credits

- [Words-CEFR-Dataset](https://github.com/Maximax67/Words-CEFR-Dataset) (MIT) —
  source tables in `datasets/`, license: `datasets/CEFR-DATASET-LICENSE`.
- [CEFR-J](https://cefr-j.org/) wordlist (`word_list_cefr.csv`).
- [LanguageTool](https://languagetool.org/) data, via
  [nlprule](https://github.com/bminixhofer/nlprule) for `analyze`.
- Upstream pipeline references: spaCy, LemmInflect, Google Books 1-grams,
  Penn Treebank tag set.

License: MIT.
