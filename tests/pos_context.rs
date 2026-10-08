//! The context tagger against the real model and the real dataset:
//! a word's role in the sentence decides which sense answers.

#![cfg(all(feature = "nlp", feature = "sqlite", feature = "parquet"))]

use std::path::PathBuf;

use cefr::db::{CefrDb, build_db};
use cefr::pos::Tagger;

fn full_parquet() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/cefr.zstd.parquet")
}

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/en_tokenizer.bin.zst")
}

fn tagged_db(dir: &std::path::Path) -> CefrDb {
    let db_path = dir.join("pos_context.db");
    build_db(&full_parquet(), &db_path).expect("rebuild the full parquet");
    CefrDb::open(&db_path).expect("open the rebuilt db")
}

#[test]
fn record_reads_as_verb_and_noun_by_position() {
    let dir = std::env::temp_dir().join(format!("cefr_pos_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let db = tagged_db(&dir);
    let tagger = Tagger::from_model_path(&model_path()).expect("load the repo model");

    let as_verb = tagger
        .pos_in_context("record", "They record a new song every winter.")
        .expect("the verb reading is tagged");
    let as_noun = tagger
        .pos_in_context("record", "She kept a record of every promise.")
        .expect("the noun reading is tagged");

    assert!(as_verb.pos.starts_with('V'), "got {}", as_verb.pos);
    assert!(as_noun.pos.starts_with('N'), "got {}", as_noun.pos);
    assert_ne!(as_verb.pos, as_noun.pos);

    // Each sentence's sense answers with a level, same tag family.
    let verb_sense = db
        .sense_level("record", &as_verb.pos)
        .unwrap()
        .expect("a verb-family sense exists");
    assert!(verb_sense.pos.starts_with('V'), "got {}", verb_sense.pos);
    assert!(
        db.sense_level("record", &as_noun.pos)
            .unwrap()
            .is_some_and(|s| s.pos.starts_with('N'))
    );
    let senses = db.pos_senses("record").unwrap();
    assert!(senses.len() >= 2, "record should list several senses");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_absent_word_or_sentence_tags_to_none() {
    let dir = std::env::temp_dir().join(format!("cefr_pos_none_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let _db = tagged_db(&dir);
    let tagger = Tagger::from_model_path(&model_path()).expect("load the repo model");

    assert!(tagger.pos_in_context("", "Some words here.").is_none());
    assert!(
        tagger
            .pos_in_context("record", "Not a trace of the target in this one.")
            .is_none()
    );

    std::fs::remove_dir_all(&dir).ok();
}
