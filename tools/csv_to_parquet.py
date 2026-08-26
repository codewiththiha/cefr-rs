#!/usr/bin/env python3
r"""Flatten the CEFR dataset CSVs into one parquet table.

Reads datasets/csv/*.csv from this repo (FK layout: datasets/database_model.png),
resolves the foreign keys into readable values and writes a single denormalized
parquet to --out. Any column subset works; word and level are always kept.

    python3 tools/csv_to_parquet.py --columns word,lemma,level --out data/cefr.core.parquet
    python3 tools/csv_to_parquet.py --out data/cefr.zstd.parquet
    python3 tools/csv_to_parquet.py --columns word,cefr --compression brotli --out /tmp/x.parquet
"""

from __future__ import annotations
import argparse
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ALL_COLUMNS = ["word", "pos_tag", "lemma", "stem", "frequency_count", "level", "categories"]
DERIVABLE = ["cefr"]            # computed from level; only exported on request
REQUIRED = ["word", "level"]

CEFR_FROM_LEVEL = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C1", 6: "C2"}
LEVEL_FROM_CEFR = {v: k for k, v in CEFR_FROM_LEVEL.items()}

# CEFR-J wordlist uses POS names, the dataset uses Penn Treebank tags
CEFRJ_POS_TO_PENN = {
    "noun": "NN",
    "adjective": "JJ",
    "verb": "VB",
    "be-verb": "VB",
    "do-verb": "VB",
    "have-verb": "VB",
    "adverb": "RB",
    "pronoun": "PRP",
    "preposition": "IN",
    "determiner": "DT",
    "conjunction": "CC",
    "number": "CD",
    "modal auxiliary": "MD",
    "interjection": "UH",
    "infinitive-to": "TO",
}

ARROW_SCHEMA = {
    "word": pa.string(),
    "pos_tag": pa.string(),
    "lemma": pa.string(),
    "stem": pa.string(),
    "frequency_count": pa.uint64(),
    "level": pa.float32(),
    "cefr": pa.string(),
    "categories": pa.list_(pa.string()),
}


def load_tables(repo: str) -> dict[str, pd.DataFrame]:
    csv_dir = os.path.join(repo, "csv")
    return {
        name: pd.read_csv(os.path.join(csv_dir, f"{name}.csv"))
        for name in ("words", "pos_tags", "word_pos", "word_categories", "categories")
    }


def denormalize(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    words, wp, tags = t["words"].copy(), t["word_pos"].copy(), t["pos_tags"]

    id2word = words.set_index("word_id")["word"]
    id2stem = words.set_index("word_id")["stem_word_id"]

    df = wp.copy().sort_values("word_id", kind="stable")  # keep frequency rank order
    df["word"] = df["word_id"].map(id2word)
    df["pos_tag"] = df["pos_tag_id"].map(tags.set_index("tag_id")["tag"])
    df["lemma"] = df["lemma_word_id"].map(id2word)              # FK into words
    df["stem"] = df["word_id"].map(id2stem).map(id2word)        # two hops: word_pos -> words -> words

    # M:N through the junction table
    title = t["categories"].set_index("category_id")["category_title"]
    cat_lists = (
        t["word_categories"]
        .assign(title=t["word_categories"]["category_id"].map(title))
        .groupby("word_pos_id")["title"]
        .agg(list)
    )
    df["categories"] = df["word_pos_id"].map(cat_lists)
    df["categories"] = df["categories"].apply(lambda c: c if isinstance(c, list) else [])

    df["_origin"] = "dataset"
    return df[["word", "pos_tag", "lemma", "stem", "frequency_count", "level",
               "categories", "word_id", "_origin"]]


def load_cefrj_missing(repo: str, existing_words: set[str]) -> pd.DataFrame:
    """CEFR-J gold rows for headwords the upstream dataset lost (e.g. hello)."""
    for cand in (os.path.join(repo, "word_list_cefr.csv"),                    # this repo
                 os.path.join(repo, "datasets", "word_list_cefr.csv")):       # upstream checkout
        if os.path.exists(cand):
            path = cand
            break
    else:
        raise FileNotFoundError(f"word_list_cefr.csv not found under {repo}")

    raw = pd.read_csv(path, sep=";")
    raw = raw.rename(columns=str.lower).rename(columns={"coreinventory 1": "core1",
                                                        "coreinventory 2": "core2",
                                                        "threshold": "core3"})
    raw["headword"] = raw["headword"].str.strip().str.lower()
    missing = raw[~raw["headword"].isin(existing_words)].copy()

    missing["word"] = missing["headword"]
    missing["pos_tag"] = missing["pos"].map(CEFRJ_POS_TO_PENN)
    missing["lemma"] = missing["headword"]                 # headwords are base forms
    missing["stem"] = None
    missing["frequency_count"] = pd.NA                     # genuinely unknown
    missing["level"] = missing["cefr"].str.strip().str.upper().map(LEVEL_FROM_CEFR)
    missing["categories"] = missing.apply(
        lambda r: [c.strip() for c in (r["core1"], r["core2"], r["core3"]) if isinstance(c, str) and c.strip()],
        axis=1,
    )
    missing["word_id"] = pd.NA
    missing["_origin"] = "cefrj-gold"

    unmapped = missing[missing["pos_tag"].isna()]["pos"].unique()
    if len(unmapped):
        print(f"warning: unmapped CEFR-J pos dropped: {list(unmapped)}", file=sys.stderr)
    return missing.dropna(subset=["pos_tag", "level"])[
        ["word", "pos_tag", "lemma", "stem", "frequency_count", "level", "categories", "word_id", "_origin"]
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    ap.add_argument("--repo", default=default_repo,
                    help=f"folder with csv/*.csv and word_list_cefr.csv (default: {default_repo}); "
                         "point at a Words-CEFR-Dataset checkout to use upstream instead")
    ap.add_argument("--out", required=True, help="output .parquet path")
    ap.add_argument("--columns", default=",".join(ALL_COLUMNS),
                    help=f"comma list, any subset of {ALL_COLUMNS + DERIVABLE}; "
                         "word and level are auto-added (default: all non-derived)")
    ap.add_argument("--merge-cefrj", dest="merge_cefrj", action="store_true", default=True,
                    help="add CEFR-J rows for headwords missing from the dataset (default: on)")
    ap.add_argument("--no-merge-cefrj", dest="merge_cefrj", action="store_false")
    ap.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "brotli", "gzip", "lz4"])
    ap.add_argument("--compression-level", type=int, default=None,
                    help="codec level (zstd 1-22 default 19, brotli 0-11 def 11, gzip 0-9)")
    ap.add_argument("--row-group-size", type=int, default=131072)
    args = ap.parse_args()

    want = [c.strip() for c in args.columns.split(",") if c.strip()]
    for c in REQUIRED:
        if c not in want:
            want.insert(0, c)
            print(f"note: required column '{c}' auto-added")
    bad = [c for c in want if c not in ALL_COLUMNS + DERIVABLE]
    if bad:
        ap.error(f"unknown columns {bad}; choose from {ALL_COLUMNS + DERIVABLE}")
    order = [c for c in ALL_COLUMNS + DERIVABLE if c in want]

    t = load_tables(args.repo)
    df = denormalize(t)
    n_base = len(df)

    if args.merge_cefrj:
        extra = load_cefrj_missing(args.repo, set(t["words"]["word"].str.lower()))
        df = pd.concat([df, extra], ignore_index=True)
        print(f"merged {len(extra)} CEFR-J headwords missing from the dataset; base rows: {n_base}")

    if "cefr" in order:
        df["cefr"] = df["level"].round().astype("Int64").map(CEFR_FROM_LEVEL)
    if "categories" in order:
        df["categories"] = df["categories"].apply(lambda c: c if isinstance(c, list) else [])

    if "pos_tag" not in order:
        # one row per word: mean level, first non-null for the rest
        def first_valid(s):
            s = s.dropna()
            return s.iloc[0] if len(s) else None
        agg = {c: first_valid for c in order if c not in ("word", "level", "categories")}
        df = (df.groupby("word", as_index=False, sort=False)
                .agg(level=("level", "mean"),
                     **{c: pd.NamedAgg(column=c, aggfunc=a) for c, a in agg.items() if c != "word"},
                     **({"categories": pd.NamedAgg(column="categories",
                                                  aggfunc=lambda s: sorted({x for lst in s for x in lst}))}
                        if "categories" in order else {})))
        df["level"] = df["level"].astype("float32")

    schema = pa.schema([(c, ARROW_SCHEMA[c]) for c in order])
    table = pa.Table.from_pandas(df[order], schema=schema, preserve_index=False)

    lvl = args.compression_level
    if lvl is None:
        lvl = {"zstd": 19, "brotli": 11, "gzip": 9}.get(args.compression)
    pq.write_table(table, args.out, compression=args.compression, compression_level=lvl,
                   use_dictionary=True, write_statistics=True, row_group_size=args.row_group_size)

    mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"wrote {args.out}: {table.num_rows:,} rows, {mb:.2f} MB, codec={args.compression}"
          f"{f' lvl {lvl}' if lvl is not None else ''}")
    print(table.schema.to_string(show_field_metadata=False))

    chk = table.filter(pa.compute.equal(table["word"], "hello")).to_pandas()
    print(f"hello rows: {len(chk)}")
    if len(chk):
        print(chk.to_string())


if __name__ == "__main__":
    main()
