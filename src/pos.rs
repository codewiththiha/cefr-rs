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

/// The readable word class a Penn Treebank tag stands for: `NNS` -> noun.
/// One home for the mapping, so a UI never re-derives it from the tags.
pub fn kind_of(tag: &str) -> &'static str {
    if tag.starts_with("VB") {
        "verb"
    } else if tag.starts_with("NN") || tag.starts_with("NP") {
        "noun"
    } else if tag.starts_with("JJ") {
        "adjective"
    } else if tag.starts_with("RB") {
        "adverb"
    } else if tag == "PRP" || tag.starts_with("WP") {
        "pronoun"
    } else if tag == "IN" || tag == "TO" {
        "preposition"
    } else if tag == "CC" {
        "conjunction"
    } else if tag == "CD" {
        "number"
    } else if tag == "MD" {
        "modal verb"
    } else if tag == "DT" || tag == "PDT" || tag == "WDT" {
        "determiner"
    } else {
        "other"
    }
}

/// The stem a contraction's tag belongs to: `don't` tags as `do`, because
/// the tokenizer splits the `n't` off and the head is the verb.
pub fn contraction_stem(word: &str) -> Option<&str> {
    ABBREVIATION_MAPPING
        .iter()
        .find_map(|(suffix, _)| word.strip_suffix(suffix))
        .filter(|stem| stem.chars().any(|c| c.is_ascii_alphanumeric()))
}

/// The head of a hyphenated compound. English compounds are head-final:
/// `well-known` tags as `known`, `mother-in-law` as `law`.
pub fn hyphen_head(word: &str) -> Option<&str> {
    word.rsplit('-')
        .find(|part| part.chars().any(|c| c.is_ascii_alphabetic()))
        .filter(|head| *head != word)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_map_to_their_readable_class() {
        assert_eq!(kind_of("NN"), "noun");
        assert_eq!(kind_of("NNS"), "noun");
        assert_eq!(kind_of("VBG"), "verb");
        assert_eq!(kind_of("JJ"), "adjective");
        assert_eq!(kind_of("RBR"), "adverb");
        assert_eq!(kind_of("PRP"), "pronoun");
        assert_eq!(kind_of("IN"), "preposition");
        assert_eq!(kind_of("MD"), "modal verb");
        assert_eq!(kind_of("DT"), "determiner");
        // LanguageTool's suffixes are stripped before this is asked.
        assert_eq!(kind_of(normalize_pos("NN:U")), "noun");
        assert_eq!(kind_of("XX"), "other");
        assert_eq!(kind_of(""), "other");
    }

    #[test]
    fn a_contraction_resolves_to_its_head() {
        assert_eq!(contraction_stem("don't"), Some("do"));
        assert_eq!(contraction_stem("it's"), Some("it"));
        assert_eq!(contraction_stem("we've"), Some("we"));
        assert_eq!(contraction_stem("can't"), Some("ca"));
        assert_eq!(contraction_stem("record"), None);
        // A bare suffix is not a stem.
        assert_eq!(contraction_stem("'s"), None);
    }

    #[test]
    fn a_compound_resolves_to_its_last_part() {
        assert_eq!(hyphen_head("well-known"), Some("known"));
        assert_eq!(hyphen_head("mother-in-law"), Some("law"));
        assert_eq!(hyphen_head("state-of-the-art"), Some("art"));
        // A dangling dash leaves the word itself as the head; a plain
        // word has none.
        assert_eq!(hyphen_head("well-"), Some("well"));
        assert_eq!(hyphen_head("-known"), Some("known"));
        assert_eq!(hyphen_head("known"), None);
    }
}
