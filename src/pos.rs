//! Sentence-context POS for one word: the runtime tagger's verdict on
//! the word's role where it actually appears.

use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

use anyhow::{Context, Result};

/// LanguageTool tags are Penn-style but with extra suffixes (`NN:U`,
/// `IN/that`); strip them so they match this dataset's Penn Treebank
/// tags.
pub fn normalize_pos(tag: &str) -> &str {
    match tag.find([':', '/']) {
        Some(end) => &tag[..end],
        None => tag,
    }
}

/// Same contraction map as the Python notebook.
pub const ABBREVIATION_MAPPING: [(&str, &str); 7] = [
    ("'m", "am"),
    ("'s", "is"),
    ("'re", "are"),
    ("'ve", "have"),
    ("'d", "had"),
    ("n't", "not"),
    ("'ll", "will"),
];

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

/// The tagger's answer for one word in one sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextPos {
    /// The Penn Treebank tag at this word's position (`NN`, `VB`, ...).
    pub pos: String,
    /// The tagger's lemma for the token, lowercased.
    pub lemma: Option<String>,
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

    /// The tag of `word` where it appears in `sentence`: surface match
    /// first, the tagger's lemma second. `None` when the word is absent.
    pub fn pos_in_context(&self, word: &str, sentence: &str) -> Option<ContextPos> {
        let target = word.trim().to_lowercase();
        if target.is_empty() {
            return None;
        }
        let mut by_lemma = None;
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
            if surface == target {
                return Some(ContextPos {
                    pos,
                    lemma: Some(lemma),
                });
            }
            if by_lemma.is_none() && lemma == target {
                by_lemma = Some(ContextPos {
                    pos,
                    lemma: Some(lemma),
                });
            }
        }
        by_lemma
    }
}
