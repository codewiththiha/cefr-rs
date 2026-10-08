//! `analyze`: full Text-Analizer.ipynb port — tokenize+tag with nlprule
//! (replaces spaCy + LemmInflect), then look up every (word, pos) in cefr.db.

use std::collections::HashSet;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};

use cefr::db::CefrDb;
use cefr::level::level_to_cefr;
use cefr::pos::{ABBREVIATION_MAPPING, normalize_pos, tokenizer_from_model_path};

fn load_tokenizer(path: &Path) -> Result<nlprule::Tokenizer> {
    tokenizer_from_model_path(path)
}

/// Model path: explicit arg > $CEFR_MODEL > ./models/ (or ./data/) defaults.
/// Returns None when no model file is found (embedded build may still apply).
fn resolve_model_path(arg: Option<&String>) -> Option<PathBuf> {
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
        let p = PathBuf::from(cand);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Model bytes compiled into the binary (`embed-model` feature, dev convenience).
#[cfg(feature = "embed-model")]
fn embedded_model() -> Option<&'static [u8]> {
    Some(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/en_tokenizer.bin"
    )))
}

#[cfg(not(feature = "embed-model"))]
fn embedded_model() -> Option<&'static [u8]> {
    None
}

pub fn analyze(db_path: &Path, text_path: &str, model: Option<&String>) -> Result<()> {
    let text = std::fs::read_to_string(text_path).with_context(|| format!("read {text_path}"))?;

    // --- NLP phase (like `spacy.load("en_core_web_sm")` + lemminflect) ------
    let t0 = Instant::now();
    let tokenizer = match resolve_model_path(model) {
        Some(path) => load_tokenizer(&path)?,
        None => match embedded_model() {
            Some(bytes) => nlprule::Tokenizer::from_reader(Cursor::new(bytes))
                .context("load embedded tokenizer model")?,
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
    let conn = CefrDb::open(db_path)?;
    let has_pos = conn.has_pos();

    // unique (word, pos) pairs, skipping punctuation (python: is_punctuation)
    let mut pairs: Vec<(String, String)> = tokens
        .iter()
        .filter(|(w, _, _)| w.chars().any(char::is_alphabetic))
        .map(|(w, _, p)| (w.clone(), p.clone()))
        .collect();
    pairs.sort();
    pairs.dedup();

    let levels = conn.lookup_batch(&pairs)?;
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
        println!(
            "note: db has no pos_tag column -> word-average levels; POS column is the tokenizer's guess"
        );
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
            let i = cefr::level::level_band(*l) as usize;
            if i >= 1 {
                total[i - 1] += 1;
            }
        }
    }
    println!("CEFR statistic (total words):");
    for (i, c) in total.iter().enumerate() {
        println!("{}: {}", level_to_cefr((i + 1) as f64), c);
    }
    println!();

    // dedup key: (word, pos) with a pos-aware db; word alone for word-averaged
    // dbs, otherwise the tagger tagging one word two ways would count it twice
    let dedup_key = |w: &str, p: &str| {
        (
            w.to_string(),
            if has_pos {
                p.to_string()
            } else {
                String::new()
            },
        )
    };

    let mut unique_seen = HashSet::new();
    let mut unique = [0usize; 6];
    for (w, _, p, lvl) in &level_tokens {
        if let Some(l) = lvl {
            if unique_seen.insert(dedup_key(w, p)) {
                let i = cefr::level::level_band(*l) as usize;
                if i >= 1 {
                    unique[i - 1] += 1;
                }
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
    let mut hard_seen = HashSet::new();
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
