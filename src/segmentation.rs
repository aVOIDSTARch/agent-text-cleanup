//! Split → batch → recompile, for documents too large to correct in one call.
//!
//! [`crate::agent`] corrects a whole document in a single Messages request, but
//! a full-sized markdown page can exceed what comfortably fits in one request
//! (and one `max_tokens` worth of output). This module sits in the middle:
//!
//! 1. **Split** the markdown into a [`Vec<Segment>`] that each fit a token
//!    budget, breaking only at structural boundaries (blank lines) and never
//!    inside a fenced code block.
//! 2. **Batch** the segments through the API — either as a real Anthropic
//!    Message Batch ([`BatchMode::Batches`], the default) or as a sequence of
//!    ordinary correction calls ([`BatchMode::Collection`]).
//! 3. **Recompile** the corrected pieces, in order, back into one markdown
//!    string via [`reassemble`].
//!
//! [`run`] (and [`run_file`]) tie all three steps together for callers and the
//! CLI; the individual functions are public so they can be composed or tested
//! on their own.

use crate::agent::{self, AgentError, BatchRequestItem, ClaudeClient, FormatTarget};
use std::collections::HashMap;
use std::path::Path;

/// Default per-segment token budget. Kept well under the 4096 output-token cap
/// so a corrected segment (~1:1 with its input) still fits in one response.
pub const DEFAULT_MAX_TOKENS: usize = 3000;
/// Rough characters-per-token ratio for the heuristic estimator.
const DEFAULT_CHARS_PER_TOKEN: f32 = 4.0;
/// Fraction of the budget actually used, leaving headroom for estimate error.
const DEFAULT_SAFETY_MARGIN: f32 = 0.9;
/// Separator used to join segments when packing and to rejoin when recompiling.
const SEGMENT_SEPARATOR: &str = "\n\n";

/// How segment token counts are measured when deciding where to split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Estimator {
    /// Offline character-count heuristic (`chars / chars_per_token`). The default.
    #[default]
    Heuristic,
    /// Exact counts from the API `count_tokens` endpoint (one call per block).
    Api,
}

/// How corrected segments are produced from the API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum BatchMode {
    /// Submit all segments as one Anthropic Message Batch. The default.
    #[default]
    Batches,
    /// Correct each segment with an ordinary, sequential Messages call.
    Collection,
}

/// Tunables for splitting a document into segments.
#[derive(Debug, Clone)]
pub struct SegmentConfig {
    /// Per-segment token budget before the safety margin is applied.
    pub max_tokens: usize,
    /// Which token estimator to budget against.
    pub estimator: Estimator,
    /// Characters per token for [`Estimator::Heuristic`].
    pub chars_per_token: f32,
    /// Fraction of `max_tokens` a segment is allowed to reach.
    pub safety_margin: f32,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_TOKENS,
            estimator: Estimator::Heuristic,
            chars_per_token: DEFAULT_CHARS_PER_TOKEN,
            safety_margin: DEFAULT_SAFETY_MARGIN,
        }
    }
}

impl SegmentConfig {
    /// The effective per-segment token budget after the safety margin.
    fn budget(&self) -> usize {
        ((self.max_tokens as f32 * self.safety_margin).floor() as usize).max(1)
    }
}

/// One token-bounded piece of the source document, ready to send for correction.
#[derive(Debug, Clone)]
pub struct Segment {
    /// 0-based position in the document; drives reassembly order.
    pub index: usize,
    /// Stable id (`"seg-0001"`) used to correlate batch results back to segments.
    pub custom_id: String,
    /// The segment's markdown text.
    pub content: String,
    /// Estimated token count of `content` under the active estimator.
    pub estimated_tokens: usize,
}

/// A corrected segment coming back from the API.
#[derive(Debug, Clone)]
pub struct SegmentOutput {
    /// The [`Segment::index`] this corresponds to.
    pub index: usize,
    /// The [`Segment::custom_id`] this corresponds to.
    pub custom_id: String,
    /// The corrected markdown for this segment.
    pub corrected: String,
}

/// Errors splitting, batching, or recompiling a document.
#[derive(Debug, thiserror::Error)]
pub enum SegmentationError {
    /// A single line is larger than the budget, so it cannot be placed in any
    /// segment without breaking mid-line.
    #[error(
        "a single line of ~{tokens} tokens exceeds the per-segment budget of {budget}; \
raise --max-tokens"
    )]
    Unsplittable { tokens: usize, budget: usize },

    /// The batch returned no result for a segment we submitted.
    #[error("the batch returned no result for segment {custom_id}")]
    MissingResult { custom_id: String },

    /// The source file could not be read.
    #[error("could not read {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },

    /// An error from the underlying API client.
    #[error(transparent)]
    Agent(#[from] AgentError),
}

/// Estimate the token count of `text` from its character count.
pub fn estimate_tokens_heuristic(text: &str, cfg: &SegmentConfig) -> usize {
    let ratio = if cfg.chars_per_token > 0.0 {
        cfg.chars_per_token
    } else {
        DEFAULT_CHARS_PER_TOKEN
    };
    (text.chars().count() as f32 / ratio).ceil() as usize
}

/// Break markdown into atomic structural blocks, split on blank lines but
/// keeping fenced code blocks (```` ``` ````) whole so a segment never bisects
/// one. Each returned block is trimmed of trailing whitespace.
pub fn split_blocks(markdown: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current = String::new();
    let mut in_fence = false;

    for line in markdown.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
        }
        if line.trim().is_empty() && !in_fence {
            push_nonempty(&mut blocks, &mut current);
        } else {
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
        }
    }
    push_nonempty(&mut blocks, &mut current);
    blocks
}

/// Flush `current` into `blocks` (trimmed) if it holds anything, then clear it.
fn push_nonempty(blocks: &mut Vec<String>, current: &mut String) {
    if !current.trim().is_empty() {
        blocks.push(current.trim_end().to_string());
    }
    current.clear();
}

/// Split a block that overflows the budget into line-aligned pieces, each within
/// `budget`. Errors only if a single line alone exceeds the budget.
fn split_oversized_block(
    block: &str,
    cfg: &SegmentConfig,
    budget: usize,
) -> Result<Vec<String>, SegmentationError> {
    let mut pieces = Vec::new();
    let mut current = String::new();

    for line in block.lines() {
        let line_tokens = estimate_tokens_heuristic(line, cfg);
        if line_tokens > budget {
            return Err(SegmentationError::Unsplittable {
                tokens: line_tokens,
                budget,
            });
        }
        let combined = if current.is_empty() {
            line.to_string()
        } else {
            format!("{current}\n{line}")
        };
        if !current.is_empty() && estimate_tokens_heuristic(&combined, cfg) > budget {
            pieces.push(std::mem::take(&mut current));
            current.push_str(line);
        } else {
            current = combined;
        }
    }
    if !current.is_empty() {
        pieces.push(current);
    }
    Ok(pieces)
}

/// Split markdown into segments using the heuristic estimator (the main, sync
/// entry point). Greedily packs blocks up to the budget, dividing any block that
/// is itself too large.
pub fn segment(markdown: &str, cfg: &SegmentConfig) -> Result<Vec<Segment>, SegmentationError> {
    let budget = cfg.budget();
    let units = atomic_units(markdown, cfg, budget)?;

    let mut segments = Vec::new();
    let mut current = String::new();
    for unit in units {
        let combined = join_segment(&current, &unit);
        if !current.is_empty() && estimate_tokens_heuristic(&combined, cfg) > budget {
            let content = std::mem::replace(&mut current, unit);
            push_segment(&mut segments, content, cfg);
        } else {
            current = combined;
        }
    }
    if !current.is_empty() {
        push_segment(&mut segments, current, cfg);
    }
    Ok(segments)
}

/// Split markdown into segments, budgeting against exact API token counts.
///
/// Counts each block once via [`ClaudeClient::count_tokens`] and packs by the
/// running sum, so the number of API calls is one per block, not per candidate.
pub async fn segment_with_api(
    client: &ClaudeClient,
    markdown: &str,
    cfg: &SegmentConfig,
) -> Result<Vec<Segment>, SegmentationError> {
    let budget = cfg.budget();
    let units = atomic_units(markdown, cfg, budget)?;

    let mut segments = Vec::new();
    let mut current = String::new();
    let mut current_tokens = 0usize;
    for unit in units {
        let unit_tokens = client.count_tokens(&unit).await? as usize;
        if !current.is_empty() && current_tokens + unit_tokens > budget {
            let content = std::mem::take(&mut current);
            push_segment_with_tokens(&mut segments, content, current_tokens);
            current_tokens = 0;
        }
        current = join_segment(&current, &unit);
        current_tokens += unit_tokens;
    }
    if !current.is_empty() {
        push_segment_with_tokens(&mut segments, current, current_tokens);
    }
    Ok(segments)
}

/// Block the document into budget-sized atomic units, dividing oversized blocks.
fn atomic_units(
    markdown: &str,
    cfg: &SegmentConfig,
    budget: usize,
) -> Result<Vec<String>, SegmentationError> {
    let mut units = Vec::new();
    for block in split_blocks(markdown) {
        if estimate_tokens_heuristic(&block, cfg) > budget {
            units.extend(split_oversized_block(&block, cfg, budget)?);
        } else {
            units.push(block);
        }
    }
    Ok(units)
}

/// Join an accumulating segment with the next unit using the segment separator.
fn join_segment(current: &str, unit: &str) -> String {
    if current.is_empty() {
        unit.to_string()
    } else {
        format!("{current}{SEGMENT_SEPARATOR}{unit}")
    }
}

/// Push a finished segment, estimating its tokens with the heuristic.
fn push_segment(segments: &mut Vec<Segment>, content: String, cfg: &SegmentConfig) {
    let estimated_tokens = estimate_tokens_heuristic(&content, cfg);
    push_segment_with_tokens(segments, content, estimated_tokens);
}

/// Push a finished segment with an already-known token estimate.
fn push_segment_with_tokens(segments: &mut Vec<Segment>, content: String, estimated_tokens: usize) {
    let index = segments.len();
    segments.push(Segment {
        custom_id: format!("seg-{index:04}"),
        index,
        content,
        estimated_tokens,
    });
}

/// Format segments into Message Batch requests, one per segment, carrying the
/// same correction prompt the single-shot path uses.
pub fn build_batch_requests(
    segments: &[Segment],
    model: &str,
    target: Option<&FormatTarget>,
    max_tokens: u32,
) -> Vec<BatchRequestItem> {
    segments
        .iter()
        .map(|seg| {
            let prompt = match target {
                Some(target) => agent::correct_to_target_prompt(&seg.content, target),
                None => agent::correct_prompt(&seg.content),
            };
            BatchRequestItem::new(seg.custom_id.clone(), prompt, model, max_tokens)
        })
        .collect()
}

/// Map batch results back onto the original segments, in document order.
///
/// Errors with [`SegmentationError::MissingResult`] if any segment has no
/// corresponding successful result.
pub fn collect_outputs_from_batch(
    results: &[agent::BatchResultItem],
    segments: &[Segment],
) -> Result<Vec<SegmentOutput>, SegmentationError> {
    let mut by_id: HashMap<&str, &str> = results
        .iter()
        .filter_map(|r| r.text.as_deref().map(|text| (r.custom_id.as_str(), text)))
        .collect();

    segments
        .iter()
        .map(|seg| {
            let corrected = by_id.remove(seg.custom_id.as_str()).ok_or_else(|| {
                SegmentationError::MissingResult {
                    custom_id: seg.custom_id.clone(),
                }
            })?;
            Ok(SegmentOutput {
                index: seg.index,
                custom_id: seg.custom_id.clone(),
                corrected: corrected.to_string(),
            })
        })
        .collect()
}

/// Recompile corrected segments into one markdown document, ordered by index and
/// joined with the same separator used when splitting.
pub fn reassemble(outputs: &mut [SegmentOutput]) -> String {
    outputs.sort_by_key(|o| o.index);
    outputs
        .iter()
        .map(|o| o.corrected.trim())
        .collect::<Vec<_>>()
        .join(SEGMENT_SEPARATOR)
}

/// Split, correct, and recompile a markdown string end to end.
///
/// Splits with the estimator in `cfg`, corrects via `mode`, and returns the
/// reassembled document. The optional `target` shapes each segment's output.
pub async fn run(
    client: &ClaudeClient,
    markdown: &str,
    cfg: &SegmentConfig,
    mode: BatchMode,
    target: Option<&FormatTarget>,
) -> Result<String, SegmentationError> {
    let segments = match cfg.estimator {
        Estimator::Heuristic => segment(markdown, cfg)?,
        Estimator::Api => segment_with_api(client, markdown, cfg).await?,
    };
    if segments.is_empty() {
        return Ok(String::new());
    }

    let mut outputs = match mode {
        BatchMode::Batches => {
            let requests = build_batch_requests(
                &segments,
                client.model(),
                target,
                agent::DEFAULT_MAX_TOKENS,
            );
            let results = client.run_batch(requests).await?;
            collect_outputs_from_batch(&results, &segments)?
        }
        BatchMode::Collection => correct_segments_sequentially(client, &segments, target).await?,
    };
    Ok(reassemble(&mut outputs))
}

/// Read a markdown file and run the full split → batch → recompile pipeline.
pub async fn run_file(
    client: &ClaudeClient,
    path: impl AsRef<Path>,
    cfg: &SegmentConfig,
    mode: BatchMode,
    target: Option<&FormatTarget>,
) -> Result<String, SegmentationError> {
    let path = path.as_ref();
    let markdown = std::fs::read_to_string(path).map_err(|source| SegmentationError::Read {
        path: path.display().to_string(),
        source,
    })?;
    run(client, &markdown, cfg, mode, target).await
}

/// Correct each segment with an ordinary, sequential Messages call.
async fn correct_segments_sequentially(
    client: &ClaudeClient,
    segments: &[Segment],
    target: Option<&FormatTarget>,
) -> Result<Vec<SegmentOutput>, SegmentationError> {
    let mut outputs = Vec::with_capacity(segments.len());
    for seg in segments {
        let corrected = match target {
            Some(target) => client.correct_to_target(&seg.content, target).await?,
            None => client.correct(&seg.content).await?,
        };
        outputs.push(SegmentOutput {
            index: seg.index,
            custom_id: seg.custom_id.clone(),
            corrected,
        });
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_tokens: usize) -> SegmentConfig {
        SegmentConfig {
            max_tokens,
            safety_margin: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn heuristic_estimate_grows_with_length() {
        let c = SegmentConfig::default();
        let short = estimate_tokens_heuristic("hello", &c);
        let long = estimate_tokens_heuristic(&"hello ".repeat(100), &c);
        assert!(long > short);
    }

    #[test]
    fn split_blocks_keeps_fenced_code_whole() {
        let md = "Intro paragraph.\n\n```\nline one\n\nline two\n```\n\nOutro.";
        let blocks = split_blocks(md);
        assert_eq!(blocks.len(), 3, "{blocks:?}");
        assert!(blocks[1].contains("line one"));
        assert!(blocks[1].contains("line two"), "blank line inside fence split it");
    }

    #[test]
    fn segment_never_exceeds_budget() {
        // ~5 tokens per paragraph (20 chars / 4); budget 10 -> ~2 per segment.
        let md = (0..10)
            .map(|i| format!("paragraph number {i:02}"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let c = cfg(10);
        let budget = c.budget();
        let segments = segment(&md, &c).unwrap();
        assert!(segments.len() > 1, "expected multiple segments");
        for seg in &segments {
            assert!(
                seg.estimated_tokens <= budget,
                "segment {} over budget: {} > {budget}",
                seg.index,
                seg.estimated_tokens
            );
        }
    }

    #[test]
    fn reassemble_restores_document_order() {
        let mut outputs = vec![
            SegmentOutput {
                index: 2,
                custom_id: "seg-0002".into(),
                corrected: "third".into(),
            },
            SegmentOutput {
                index: 0,
                custom_id: "seg-0000".into(),
                corrected: "first".into(),
            },
            SegmentOutput {
                index: 1,
                custom_id: "seg-0001".into(),
                corrected: "second".into(),
            },
        ];
        assert_eq!(reassemble(&mut outputs), "first\n\nsecond\n\nthird");
    }

    #[test]
    fn segment_round_trips_through_reassemble() {
        let md = "First paragraph here.\n\nSecond paragraph here.\n\nThird one.";
        let segments = segment(md, &cfg(8)).unwrap();
        let mut outputs: Vec<SegmentOutput> = segments
            .iter()
            .map(|s| SegmentOutput {
                index: s.index,
                custom_id: s.custom_id.clone(),
                corrected: s.content.clone(),
            })
            .collect();
        // Reassembling the unchanged segments reproduces the source content.
        assert_eq!(reassemble(&mut outputs), md);
    }

    #[test]
    fn single_oversized_line_is_unsplittable() {
        // One word, no whitespace, far longer than a tiny budget.
        let md = "x".repeat(400);
        let err = segment(&md, &cfg(1)).unwrap_err();
        assert!(matches!(err, SegmentationError::Unsplittable { .. }), "{err}");
    }

    #[test]
    fn collect_outputs_flags_missing_result() {
        let segments = vec![Segment {
            index: 0,
            custom_id: "seg-0000".into(),
            content: "x".into(),
            estimated_tokens: 1,
        }];
        let results = vec![]; // nothing came back
        let err = collect_outputs_from_batch(&results, &segments).unwrap_err();
        assert!(matches!(err, SegmentationError::MissingResult { .. }), "{err}");
    }
}
