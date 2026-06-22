//! Binary entry point: parse the CLI and dispatch.
//!
//! All behavior lives in the library modules (see `cli.rs`); this is just the
//! thin shell that loads `.env`, parses arguments, and returns an exit code.

use agent_text_cleanup::cli::{self, Cli};
use clap::Parser;
use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    // Load ANTHROPIC_API_KEY from a local .env if present (ignored if absent).
    let _ = dotenvy::dotenv();
    cli::run(Cli::parse()).await
}
