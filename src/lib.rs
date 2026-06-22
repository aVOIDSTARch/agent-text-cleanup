//! Two-stage OCR text-cleanup library.
//!
//! * [`normalize`] — offline, deterministic regex + heuristics correction.
//! * [`agent`] — Claude-API corrections for the parts that need judgement,
//!   both a freeform pass and a format/design-target-guided pass.
//! * [`token`] — local, on-disk tracking of token usage across API calls.
//! * [`report`] — render a Markdown report from the token log using a template.
//! * [`api`] — programmatic correction surface (markdown in, corrected out).
//! * [`cli`] — the command-line interface, backed by the modules above.
//!
//! These modules are exposed as a library so they can be driven from the
//! `agent-text-cleanup` binary and from the integration tests in `tests/`.

pub mod agent;
pub mod api;
pub mod cli;
pub mod normalize;
pub mod report;
pub mod token;
