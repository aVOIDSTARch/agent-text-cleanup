//! Programmatic correction surface.
//!
//! This is the in-process API behind the `correct` CLI command — call it
//! directly instead of shelling out to the binary. For now it does one thing:
//! take markdown documents (optionally guided by an "OutputFormat" file — a
//! JSON-serialized [`FormatTarget`]), correct them with the Claude agent, and
//! return the corrected markdown.
//!
//! Markdown is the interchange format: it round-trips layout (headings, lists,
//! emphasis, tables) as plain text the model handles natively.

use crate::agent::{AgentError, ClaudeClient, FormatTarget};
use std::path::{Path, PathBuf};

/// Errors from the correction surface.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("could not read {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("could not parse target file {path}: {source}")]
    TargetParse {
        path: String,
        source: serde_json::Error,
    },
    #[error(transparent)]
    Agent(#[from] AgentError),
}

/// A corrected document and the path it was read from.
#[derive(Debug, Clone)]
pub struct CorrectedDocument {
    /// The input path this came from.
    pub source: PathBuf,
    /// The corrected markdown.
    pub corrected: String,
}

/// The correction surface. Wraps a configured [`ClaudeClient`].
#[derive(Debug, Clone)]
pub struct CorrectionApi {
    client: ClaudeClient,
}

impl CorrectionApi {
    /// Build the surface over an existing client (set the model on the client).
    pub fn new(client: ClaudeClient) -> Self {
        Self { client }
    }

    /// Build the surface reading `ANTHROPIC_API_KEY` from the environment.
    pub fn from_env() -> Result<Self, AgentError> {
        Ok(Self::new(ClaudeClient::from_env()?))
    }

    /// Correct raw markdown, optionally toward a format/design target.
    ///
    /// With a target, the corrected output is shaped to match it; without one,
    /// the agent just repairs the text and layout as best it can.
    pub async fn correct_markdown(
        &self,
        markdown: &str,
        target: Option<&FormatTarget>,
    ) -> Result<String, ApiError> {
        let corrected = match target {
            Some(target) => self.client.correct_to_target(markdown, target).await?,
            None => self.client.correct(markdown).await?,
        };
        Ok(corrected)
    }

    /// Correct a markdown file on disk, returning the corrected document.
    pub async fn correct_markdown_file(
        &self,
        path: impl AsRef<Path>,
        target: Option<&FormatTarget>,
    ) -> Result<CorrectedDocument, ApiError> {
        let path = path.as_ref();
        let markdown = read_to_string(path)?;
        let corrected = self.correct_markdown(&markdown, target).await?;
        Ok(CorrectedDocument {
            source: path.to_path_buf(),
            corrected,
        })
    }

    /// Correct several markdown files with the same optional target.
    ///
    /// Returns one result per input, in order, so a single bad file doesn't
    /// abort the batch — inspect each entry.
    pub async fn correct_markdown_files(
        &self,
        paths: &[PathBuf],
        target: Option<&FormatTarget>,
    ) -> Vec<Result<CorrectedDocument, ApiError>> {
        let mut out = Vec::with_capacity(paths.len());
        for path in paths {
            out.push(self.correct_markdown_file(path, target).await);
        }
        out
    }
}

/// Load a [`FormatTarget`] from a JSON file (the "OutputFormat" file).
pub fn load_target(path: impl AsRef<Path>) -> Result<FormatTarget, ApiError> {
    let path = path.as_ref();
    let contents = read_to_string(path)?;
    serde_json::from_str(&contents).map_err(|source| ApiError::TargetParse {
        path: path.display().to_string(),
        source,
    })
}

fn read_to_string(path: &Path) -> Result<String, ApiError> {
    std::fs::read_to_string(path).map_err(|source| ApiError::Read {
        path: path.display().to_string(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::OutputFormat;

    #[test]
    fn load_target_reads_partial_json() {
        // A target file may set only some fields; the rest default.
        let dir = std::env::temp_dir();
        let path = dir.join(format!("atc-target-{}.json", std::process::id()));
        std::fs::write(
            &path,
            r#"{ "output_format": "markdown", "description": "a tidy README" }"#,
        )
        .unwrap();

        let target = load_target(&path).unwrap();
        assert_eq!(target.output_format, OutputFormat::Markdown);
        assert_eq!(target.description, "a tidy README");
        assert!(target.examples.is_empty());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_target_reports_bad_json() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("atc-target-bad-{}.json", std::process::id()));
        std::fs::write(&path, "not json").unwrap();

        let err = load_target(&path).unwrap_err();
        assert!(matches!(err, ApiError::TargetParse { .. }), "{err}");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_target_missing_file_is_read_error() {
        let err = load_target("/no/such/target-file.json").unwrap_err();
        assert!(matches!(err, ApiError::Read { .. }), "{err}");
    }
}
