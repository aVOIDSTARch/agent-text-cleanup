//! Command-line interface.
//!
//! `clap` provides `-h`/`--help` automatically: `agent-text-cleanup --help`
//! lists the subcommands, and `agent-text-cleanup <command> --help` documents
//! that command's options. The doc comments below are the help text.
//!
//! The `correct` command is backed by [`crate::api`], so the CLI and the
//! programmatic surface share one code path.

use crate::api::{self, CorrectionApi};
use crate::{agent::ClaudeClient, normalize, report, token};
use clap::{Args, Parser, Subcommand};
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Correct OCR-damaged text and markdown via regex heuristics and the Claude API.
#[derive(Debug, Parser)]
#[command(name = "agent-text-cleanup", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Apply offline regex + heuristics correction (no API calls, no cost).
    ///
    /// Reads one file (or stdin with `-`), fixes mechanical OCR damage
    /// (split words, soft line breaks, stray symbols, whitespace), and writes
    /// the result to a file or stdout.
    Normalize(NormalizeArgs),

    /// Correct markdown/text files with the Claude agent.
    ///
    /// Without `--target`, the agent freely repairs the text and layout. With
    /// `--target <FILE>` (a JSON "OutputFormat"/FormatTarget file) it shapes the
    /// output to match that target. Multiple inputs are corrected in turn.
    Correct(CorrectArgs),

    /// Render a token-usage report from a Markdown template.
    ///
    /// Fills the template's `{{placeholders}}` from the token log. See
    /// `src/template.md` for the available values.
    Report(ReportArgs),

    /// Print token-usage totals (this month and all time).
    Usage(UsageArgs),
}

#[derive(Debug, Args)]
pub struct NormalizeArgs {
    /// Input file to normalize, or `-` for stdin.
    pub input: PathBuf,
    /// Write the result here instead of stdout.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct CorrectArgs {
    /// Markdown/text files to correct. Use a single `-` to read stdin.
    #[arg(required = true)]
    pub inputs: Vec<PathBuf>,

    /// JSON FormatTarget ("OutputFormat") file describing the desired shape.
    #[arg(short, long, value_name = "FILE")]
    pub target: Option<PathBuf>,

    /// Write corrected files into this directory (keeping their base names).
    #[arg(short = 'd', long, value_name = "DIR", conflicts_with = "stdout")]
    pub output_dir: Option<PathBuf>,

    /// Print corrected output to stdout instead of writing files.
    #[arg(long)]
    pub stdout: bool,

    /// Override the model (default: claude-opus-4-8).
    #[arg(long, value_name = "MODEL")]
    pub model: Option<String>,
}

#[derive(Debug, Args)]
pub struct ReportArgs {
    /// Report template (default: src/token-tracking.md).
    #[arg(short, long, value_name = "FILE")]
    pub template: Option<PathBuf>,
    /// Token log to read (default: token-usage.json).
    #[arg(short, long, value_name = "FILE")]
    pub log: Option<PathBuf>,
    /// Write the rendered report here instead of stdout.
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct UsageArgs {
    /// Token log to read (default: token-usage.json).
    #[arg(short, long, value_name = "FILE")]
    pub log: Option<PathBuf>,
}

/// Errors surfaced to the user (one line on stderr, exit code 1).
#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error("{0}")]
    Io(#[from] io::Error),
    #[error(transparent)]
    Api(#[from] api::ApiError),
    #[error(transparent)]
    Agent(#[from] crate::agent::AgentError),
    #[error(transparent)]
    Report(#[from] report::ReportError),
    #[error(transparent)]
    Token(#[from] token::TokenError),
}

/// Parse-free dispatch: run an already-parsed [`Cli`] and map the result to a
/// process exit code.
pub async fn run(cli: Cli) -> ExitCode {
    let result = match cli.command {
        Command::Normalize(args) => cmd_normalize(args),
        Command::Correct(args) => cmd_correct(args).await,
        Command::Report(args) => cmd_report(args),
        Command::Usage(args) => cmd_usage(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_normalize(args: NormalizeArgs) -> Result<(), CliError> {
    let input = read_input(&args.input)?;
    let cleaned = normalize::regex_repair(&input);
    write_output(args.output.as_deref(), &cleaned)
}

async fn cmd_correct(args: CorrectArgs) -> Result<(), CliError> {
    let mut client = ClaudeClient::from_env()?;
    if let Some(model) = &args.model {
        client = client.with_model(model);
    }
    let surface = CorrectionApi::new(client);

    // Load the optional target once and reuse it for every input.
    let target = match &args.target {
        Some(path) => Some(api::load_target(path)?),
        None => None,
    };

    let to_stdout = args.stdout || args.output_dir.is_none() && is_stdin(&args.inputs);
    let multiple = args.inputs.len() > 1;

    for input in &args.inputs {
        let corrected = if is_stdin_path(input) {
            let markdown = read_stdin()?;
            surface.correct_markdown(&markdown, target.as_ref()).await?
        } else {
            surface
                .correct_markdown_file(input, target.as_ref())
                .await?
                .corrected
        };

        if to_stdout {
            if multiple {
                println!("==> {} <==", input.display());
            }
            print_text(&corrected)?;
        } else {
            let out_path = output_path_for(input, args.output_dir.as_deref());
            if let Some(parent) = out_path.parent()
                && !parent.as_os_str().is_empty()
            {
                fs::create_dir_all(parent)?;
            }
            fs::write(&out_path, &corrected)?;
            eprintln!("corrected {} -> {}", input.display(), out_path.display());
        }
    }
    Ok(())
}

fn cmd_report(args: ReportArgs) -> Result<(), CliError> {
    let template = args
        .template
        .unwrap_or_else(|| PathBuf::from(report::DEFAULT_TEMPLATE_PATH));
    let log = args
        .log
        .unwrap_or_else(|| PathBuf::from(token::DEFAULT_LOG_PATH));
    let rendered = report::generate_report(template, log)?;
    write_output(args.output.as_deref(), &rendered)
}

fn cmd_usage(args: UsageArgs) -> Result<(), CliError> {
    let log_path = args
        .log
        .unwrap_or_else(|| PathBuf::from(token::DEFAULT_LOG_PATH));
    let log = token::read_log(&log_path)?;
    let month = log.this_month();
    let all = &log.all_time;
    println!(
        "this month: {} requests, {} in / {} out ({} total)",
        month.requests,
        month.input_tokens,
        month.output_tokens,
        month.total_tokens()
    );
    println!(
        "all time:   {} requests, {} in / {} out ({} total)",
        all.requests,
        all.input_tokens,
        all.output_tokens,
        all.total_tokens()
    );
    println!("events:     {}", log.events.len());
    Ok(())
}

// --- I/O helpers -----------------------------------------------------------

fn is_stdin_path(path: &Path) -> bool {
    path == Path::new("-")
}

fn is_stdin(inputs: &[PathBuf]) -> bool {
    inputs.iter().any(|p| is_stdin_path(p))
}

fn read_stdin() -> io::Result<String> {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}

fn read_input(path: &Path) -> io::Result<String> {
    if is_stdin_path(path) {
        read_stdin()
    } else {
        fs::read_to_string(path)
    }
}

/// Write to a file, or to stdout when `output` is `None` or `-`.
fn write_output(output: Option<&Path>, content: &str) -> Result<(), CliError> {
    match output {
        Some(path) if !is_stdin_path(path) => Ok(fs::write(path, content)?),
        _ => print_text(content).map_err(CliError::Io),
    }
}

/// Print text to stdout, ensuring a trailing newline for terminal friendliness.
fn print_text(content: &str) -> io::Result<()> {
    let mut out = io::stdout().lock();
    out.write_all(content.as_bytes())?;
    if !content.ends_with('\n') {
        out.write_all(b"\n")?;
    }
    out.flush()
}

/// Where a corrected file is written: into `output_dir` with the same base
/// name, or beside the input as `<stem>.corrected.<ext>`.
fn output_path_for(input: &Path, output_dir: Option<&Path>) -> PathBuf {
    if let Some(dir) = output_dir {
        let name = input
            .file_name()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("corrected.out"));
        return dir.join(name);
    }
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("corrected");
    let name = match input.extension().and_then(|e| e.to_str()) {
        Some(ext) => format!("{stem}.corrected.{ext}"),
        None => format!("{stem}.corrected"),
    };
    input.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        // Catches clap configuration errors (duplicate flags, bad conflicts) at
        // test time rather than first run.
        Cli::command().debug_assert();
    }

    #[test]
    fn output_path_into_dir_keeps_basename() {
        let p = output_path_for(Path::new("docs/page.md"), Some(Path::new("out")));
        assert_eq!(p, PathBuf::from("out/page.md"));
    }

    #[test]
    fn output_path_sibling_inserts_corrected() {
        let p = output_path_for(Path::new("docs/page.md"), None);
        assert_eq!(p, PathBuf::from("docs/page.corrected.md"));
    }

    #[test]
    fn output_path_sibling_handles_no_extension() {
        let p = output_path_for(Path::new("NOTES"), None);
        assert_eq!(p, PathBuf::from("NOTES.corrected"));
    }
}
