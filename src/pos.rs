//! Sentence-context POS for one word: the runtime tagger's verdict on the
//! word's role where it actually appears.
//!
//! The tag vocabulary and the surface rules this module reasons with live in
//! [`crate::tags`], feature-free; only the model does not.

use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

use anyhow::{Context, Result};

use crate::tags::{ABBREVIATION_MAPPING, contraction_stem, hyphen_head, normalize_pos};

/// The answer's shape, owned by [`crate::tags`]; re-exported because it is
/// this module's public return type.
pub use crate::tags::ContextPos;

/// Decode `.zst`, `.gz` or raw model bytes and build the tokenizer.
pub fn tokenizer_from_model_path(path: &Path) -> Result<nlprule::Tokenizer> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let bytes = match extension {
        "zst" => zstd::stream::decode_all(
            File::open(path).with_context(|| format!("open {}", path.display()))?,
        )
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

/// A loaded tokenizer model; reuse it across asks, loading is the cost.
pub struct Tagger {
    tokenizer: nlprule::Tokenizer,
}

impl Tagger {
    /// Load the model from `path` (`.zst`, `.gz` or raw `.bin`).
    pub fn from_model_path(path: &Path) -> Result<Self> {
        Ok(Self {
            tokenizer: tokenizer_from_model_path(path)?,
        })
    }

    /// The tag of `word` where it appears in `sentence`. The surface match
    /// answers a plain word; a contraction answers through its stem and a
    /// compound through its head, both of which the tokenizer splits off.
    /// The tagger's lemma is the last resort; `None` when the word is
    /// absent.
    pub fn pos_in_context(&self, word: &str, sentence: &str) -> Option<ContextPos> {
        let target = word.trim().to_lowercase();
        if target.is_empty() {
            return None;
        }
        let stem = contraction_stem(&target);
        let head = hyphen_head(&target);
        let mut by_lemma = None;
        let mut by_stem = None;
        let mut by_head = None;
        for token in self.tokenizer.pipe(sentence).flatten() {
            let entry = token.word();
            let surface = entry.text().as_str().trim().to_lowercase();
            let surface = match ABBREVIATION_MAPPING.iter().find(|(a, _)| *a == surface) {
                Some((_, expansion)) => expansion.to_string(),
                None => surface,
            };
            let Some(tag) = entry.tags().first() else {
                continue;
            };
            let lemma = tag.lemma().as_str().trim().to_lowercase();
            let pos = normalize_pos(tag.pos().as_str()).to_string();
            // The surface match answers at once; the three fallbacks keep
            // the first token that resolved each way, so the later tokens
            // cannot overwrite an earlier, better answer.
            if surface == target {
                return Some(ContextPos {
                    pos,
                    lemma: Some(lemma),
                });
            }
            if by_stem.is_none() && stem == Some(surface.as_str()) {
                by_stem = Some(ContextPos {
                    pos: pos.clone(),
                    lemma: Some(lemma.clone()),
                });
            }
            if by_head.is_none() && head == Some(surface.as_str()) {
                by_head = Some(ContextPos {
                    pos: pos.clone(),
                    lemma: Some(lemma.clone()),
                });
            }
            if by_lemma.is_none() && lemma == target {
                by_lemma = Some(ContextPos {
                    pos,
                    lemma: Some(lemma),
                });
            }
        }
        // The stem is the contraction's own verb; a compound's head is its
        // noun or participle; a lemma only ever guesses.
        by_stem.or(by_head).or(by_lemma)
    }
}
