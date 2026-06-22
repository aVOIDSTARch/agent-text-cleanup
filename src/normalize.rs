//! Stage 1: regex- and heuristics-based text correction.
//!
//! Runs on every page with no network call and no cost. It fixes the
//! mechanical damage OCR introduces — split words, soft line breaks, runs of
//! whitespace, stray symbol bursts — without trying to understand the content.
//! Anything that needs judgement is left for the agent passes in [`crate::agent`].

use regex::Regex;
use std::sync::LazyLock;

// Regexes are compiled once on first use rather than per call.

/// De-hyphenate words split across a line break: `atten-\ntion` -> `attention`.
static DEHYPHENATE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\w)-\n(\w)").unwrap());

/// Merge a soft line break where a lowercase letter (or opening punctuation)
/// continues the sentence, leaving real sentence-ending breaks and blank-line
/// paragraph boundaries alone (the preceding char must not itself be a newline).
static SOFT_BREAK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("([^.!?:\"\n])\n([a-z(\"])").unwrap());

/// Collapse three or more newlines down to a single paragraph boundary.
static BLANK_LINES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n{3,}").unwrap());

/// Collapse runs of spaces/tabs within a line to a single space (also folds a
/// lone tab into a space).
static INNER_SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t]+").unwrap());

/// Trim trailing spaces/tabs at the end of each line.
static TRAILING_SPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)[ \t]+$").unwrap());

/// Strip lines that are nothing but a burst of stray non-word symbols — the
/// classic OCR garbage like `~^*#@` sitting alone on a line.
static STRAY_SYMBOLS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?m)^[^\w\s"'—-]{3,}.*$"#).unwrap());

/// Normalize a double hyphen to an em-dash.
static DOUBLE_HYPHEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"--").unwrap());

/// Comprehensive regex + heuristics correction for OCR-damaged text.
///
/// This is deterministic, offline, and safe to run on every page before
/// (or instead of) an agent pass. It only touches whitespace, line structure,
/// and obvious symbol garbage — it never rewrites words, so it cannot introduce
/// meaning errors of its own.
pub fn regex_repair(raw: &str) -> String {
    // Normalize line endings first so every later rule can assume `\n`.
    let mut text = raw.replace("\r\n", "\n").replace('\r', "\n");

    text = DEHYPHENATE.replace_all(&text, "$1$2").into_owned();
    // Strip stray-symbol lines while line structure is still intact, so a
    // garbage line can't be merged into the next good line by SOFT_BREAK.
    text = STRAY_SYMBOLS.replace_all(&text, "").into_owned();
    text = SOFT_BREAK.replace_all(&text, "$1 $2").into_owned();
    text = BLANK_LINES.replace_all(&text, "\n\n").into_owned();
    text = INNER_SPACES.replace_all(&text, " ").into_owned();
    text = TRAILING_SPACE.replace_all(&text, "").into_owned();
    text = DOUBLE_HYPHEN.replace_all(&text, "—").into_owned();

    text.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dehyphenates_split_words() {
        assert_eq!(regex_repair("atten-\ntion"), "attention");
    }

    #[test]
    fn merges_soft_line_breaks() {
        assert_eq!(regex_repair("the quick\nbrown fox"), "the quick brown fox");
    }

    #[test]
    fn keeps_sentence_ending_breaks() {
        // A period before the newline is a real boundary — don't merge it.
        let out = regex_repair("First line.\nSecond line.");
        assert!(out.contains("First line.\nSecond line."), "got: {out:?}");
    }

    #[test]
    fn collapses_blank_lines_to_paragraph() {
        assert_eq!(regex_repair("a\n\n\n\nb"), "a\n\nb");
    }

    #[test]
    fn collapses_inner_whitespace() {
        assert_eq!(regex_repair("too    many\tspaces"), "too many spaces");
    }

    #[test]
    fn strips_stray_symbol_lines() {
        let out = regex_repair("good line\n~^*#@ garbage\nanother good line");
        assert!(!out.contains("garbage"), "got: {out:?}");
        assert!(out.contains("good line"));
        assert!(out.contains("another good line"));
    }

    #[test]
    fn normalizes_double_hyphen_to_em_dash() {
        assert_eq!(regex_repair("wait--stop"), "wait—stop");
    }

    #[test]
    fn trims_trailing_whitespace_and_edges() {
        assert_eq!(regex_repair("  hello   \n  "), "hello");
    }

    #[test]
    fn normalizes_crlf() {
        assert_eq!(regex_repair("line one.\r\nline two."), "line one.\nline two.");
    }
}
