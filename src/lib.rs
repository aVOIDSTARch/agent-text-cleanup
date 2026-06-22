//! Two-stage OCR text-cleanup library.
//!
//! * [`normalize`] — offline, deterministic regex + heuristics correction.
//! * [`agent`] — Claude-API corrections for the parts that need judgement,
//!   both a freeform pass and a format/design-target-guided pass.
//!
//! These modules are exposed as a library so they can be driven from the
//! `agent-text-cleanup` binary and from the integration tests in `tests/`.

pub mod agent;
pub mod normalize;
