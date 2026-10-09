//! The tag vocabulary: Penn Treebank tags, their readable word classes, and
//! the surface rules that map a token onto the head the dataset stores.
//!
//! Pure data and pure string rules — no model, no feature gate, no
//! dependencies — so a wasm or embedder build reuses the same answers the
//! tagger's own consumers get instead of restating them.

/// LanguageTool tags are Penn-style but carry extra suffixes (`NN:U`,
/// `IN/that`); strip them so they match this dataset's Penn Treebank tags.
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

/// Same contraction map as the Python notebook: the suffix a token ends in,
/// and the word the dataset stores it under.
pub const ABBREVIATION_MAPPING: [(&str, &str); 7] = [
    ("'m", "am"),
    ("'s", "is"),
    ("'re", "are"),
    ("'ve", "have"),
    ("'d", "had"),
    ("n't", "not"),
    ("'ll", "will"),
];

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

/// The tagger's answer for one word in one sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextPos {
    /// The Penn Treebank tag at this word's position (`NN`, `VB`, ...).
    pub pos: String,
    /// The tagger's lemma for the token, lowercased.
    pub lemma: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_map_to_their_readable_class() {
        assert_eq!(kind_of("NN"), "noun");
        assert_eq!(kind_of("NNS"), "noun");
        assert_eq!(kind_of("NP"), "noun");
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
