//! Stage 2: agent-based correction via the Claude API.
//!
//! Where [`crate::normalize`] fixes mechanical OCR damage with no judgement,
//! this module hands the text to Claude for the corrections that need
//! understanding — restoring garbled words, fixing layout, inferring structure.
//!
//! There is no official Anthropic Rust SDK, so calls go straight to the
//! Messages API (`POST /v1/messages`) over `reqwest`.
//!
//! Two entry points share one connection:
//! * [`ClaudeClient::correct`] — no formatting guidance; Claude just does its
//!   best to repair the text and layout.
//! * [`ClaudeClient::correct_to_target`] — same connection, but the caller
//!   passes a [`FormatTarget`] describing the shape the output should take.
//!
//! This is a reusable module; some of its public surface (e.g. `with_model`,
//! `OutputFormat::Markdown`) isn't exercised by the demo binary, so dead-code
//! analysis for the binary target is suppressed here.
#![allow(dead_code)]

use crate::token;
use serde::Serialize;
use std::path::PathBuf;

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MODEL: &str = "claude-opus-4-8";
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Errors that can occur talking to the Claude API.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// `ANTHROPIC_API_KEY` was not set when constructing via [`ClaudeClient::from_env`].
    #[error("ANTHROPIC_API_KEY environment variable is not set")]
    MissingApiKey,

    /// The transport itself failed (connection, TLS, timeout, decode).
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// The API returned a non-2xx status. Body is included for diagnosis.
    #[error("Claude API returned status {status}: {body}")]
    Api { status: u16, body: String },

    /// A 2xx response that contained no `text` content block.
    #[error("Claude API response contained no text block")]
    NoTextBlock,
}

/// A reusable connection to the Claude Messages API.
///
/// Holds a single `reqwest::Client` (which pools connections), the API key,
/// and the model to use. Clone it freely — the underlying client is cheap to
/// clone and shares its connection pool.
#[derive(Debug, Clone)]
pub struct ClaudeClient {
    http: reqwest::Client,
    api_key: String,
    model: String,
    token_log_path: PathBuf,
}

impl ClaudeClient {
    /// Build a client with an explicit API key, defaulting to [`DEFAULT_MODEL`]
    /// and the default token-log path ([`token::DEFAULT_LOG_PATH`]).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            token_log_path: PathBuf::from(token::DEFAULT_LOG_PATH),
        }
    }

    /// Build a client reading the API key from `ANTHROPIC_API_KEY`.
    pub fn from_env() -> Result<Self, AgentError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| AgentError::MissingApiKey)?;
        Ok(Self::new(api_key))
    }

    /// Override the model (e.g. `"claude-sonnet-4-6"` for cheaper batch work).
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override where token usage is logged (defaults to
    /// [`token::DEFAULT_LOG_PATH`] in the working directory).
    #[must_use]
    pub fn with_token_log(mut self, path: impl Into<PathBuf>) -> Self {
        self.token_log_path = path.into();
        self
    }

    /// Correct OCR-damaged text with **no** formatting guidance.
    ///
    /// Claude is told only that the text is OCR-damaged and asked to do its
    /// best to repair the words and layout, returning the corrected text and
    /// nothing else.
    pub async fn correct(&self, text: &str) -> Result<String, AgentError> {
        let prompt = format!(
            "The following text was extracted from a scanned document via OCR and is damaged: \
garbled words, broken line wrapping, stray characters, and inconsistent spacing.\n\n\
Correct it to the best of your ability. Fix misrecognized words using context, repair the \
line and paragraph layout, and remove obvious OCR artifacts — but do not invent content or \
change the meaning. Return only the corrected text, with no preamble or commentary.\n\n\
--- BEGIN TEXT ---\n{text}\n--- END TEXT ---"
        );
        self.send_messages(prompt, "correct").await
    }

    /// Correct OCR-damaged text toward a specific format/design target.
    ///
    /// Uses the same connection as [`ClaudeClient::correct`], but injects a
    /// description of the desired output shape so Claude knows what it is
    /// aiming for (a JSON schema, a Markdown layout, a tone, examples, ...).
    pub async fn correct_to_target(
        &self,
        text: &str,
        target: &FormatTarget,
    ) -> Result<String, AgentError> {
        let prompt = format!(
            "The following text was extracted from a scanned document via OCR and is damaged: \
garbled words, broken line wrapping, stray characters, and inconsistent spacing.\n\n\
Correct it and reshape it to match the target described below. Fix misrecognized words using \
context and infer structure from the target, but do not invent content or change the meaning. \
Return only the result, with no preamble or commentary.\n\n\
{target_block}\n\n\
--- BEGIN TEXT ---\n{text}\n--- END TEXT ---",
            target_block = target.describe(),
        );
        self.send_messages(prompt, "correct_to_target").await
    }

    /// The single shared API path both public methods route through.
    ///
    /// `operation` labels the recorded token event so the on-disk log shows
    /// which call (freeform vs. target-guided) spent the tokens.
    async fn send_messages(&self, prompt: String, operation: &str) -> Result<String, AgentError> {
        let request = MessagesRequest {
            model: &self.model,
            max_tokens: DEFAULT_MAX_TOKENS,
            messages: vec![Message {
                role: "user",
                content: prompt,
            }],
        };

        let response = self
            .http
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(AgentError::Api {
                status: status.as_u16(),
                body,
            });
        }

        let parsed: MessagesResponse = response.json().await?;

        // Record a token event to the local log. A logging failure must not
        // sink an otherwise-successful correction, so warn rather than error.
        if let Err(e) = token::record(&self.token_log_path, &self.model, operation, &parsed.usage) {
            eprintln!("warning: failed to record token usage: {e}");
        }

        parsed
            .content
            .into_iter()
            .find_map(|block| match block {
                ContentBlock::Text { text } => Some(text),
                ContentBlock::Other => None,
            })
            .ok_or(AgentError::NoTextBlock)
    }
}

/// The output format the corrected text should take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    /// Plain prose, no markup.
    #[default]
    PlainText,
    /// Markdown with headings, lists, emphasis as appropriate.
    Markdown,
    /// A JSON document (pair with `schema` to pin the shape).
    Json,
}

impl OutputFormat {
    fn label(self) -> &'static str {
        match self {
            OutputFormat::PlainText => "plain text (no markup)",
            OutputFormat::Markdown => "Markdown",
            OutputFormat::Json => "JSON",
        }
    }
}

/// A description of what the corrected output should look like.
///
/// This replaces the hard-coded schema the original TypeScript baked in: the
/// caller now describes the target generically, and [`describe`](FormatTarget::describe)
/// renders the non-empty fields into a prompt block.
#[derive(Debug, Clone, Default, Serialize)]
pub struct FormatTarget {
    /// What the content is, and what it should become.
    pub description: String,
    /// The output format Claude should produce.
    pub output_format: OutputFormat,
    /// Optional JSON schema, or a freeform description of the desired shape.
    pub schema: Option<String>,
    /// Optional layout guidance: sections, headings, columns, ordering.
    pub layout_notes: Option<String>,
    /// Optional few-shot examples of the desired output.
    pub examples: Vec<String>,
    /// Optional tone/voice guidance.
    pub tone: Option<String>,
}

impl FormatTarget {
    /// Render the target into a human-readable prompt block, omitting any
    /// fields the caller left empty.
    pub fn describe(&self) -> String {
        let mut out = String::from("--- TARGET FORMAT ---\n");
        out.push_str(&format!("Output format: {}\n", self.output_format.label()));

        if !self.description.is_empty() {
            out.push_str(&format!("Goal: {}\n", self.description));
        }
        if let Some(schema) = &self.schema {
            out.push_str(&format!("Schema / shape:\n{schema}\n"));
        }
        if let Some(layout) = &self.layout_notes {
            out.push_str(&format!("Layout: {layout}\n"));
        }
        if let Some(tone) = &self.tone {
            out.push_str(&format!("Tone: {tone}\n"));
        }
        if !self.examples.is_empty() {
            out.push_str("Examples of the desired output:\n");
            for (i, ex) in self.examples.iter().enumerate() {
                out.push_str(&format!("Example {}:\n{ex}\n", i + 1));
            }
        }
        out.push_str("--- END TARGET FORMAT ---");
        out
    }
}

// --- Wire types for the Messages API ---------------------------------------

#[derive(Serialize)]
struct MessagesRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<Message>,
}

#[derive(Serialize)]
struct Message {
    role: &'static str,
    content: String,
}

#[derive(serde::Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    #[serde(default)]
    usage: token::Usage,
}

/// A content block in the response. We only care about `text`; everything else
/// (thinking, tool_use, ...) is collapsed to `Other` and skipped.
#[derive(serde::Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(other)]
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_omits_empty_fields() {
        let target = FormatTarget {
            description: "a daily journal entry".to_string(),
            output_format: OutputFormat::Markdown,
            ..Default::default()
        };
        let desc = target.describe();
        assert!(desc.contains("Markdown"));
        assert!(desc.contains("a daily journal entry"));
        assert!(!desc.contains("Schema"));
        assert!(!desc.contains("Examples"));
    }

    #[test]
    fn describe_includes_all_set_fields() {
        let target = FormatTarget {
            description: "structured entries".to_string(),
            output_format: OutputFormat::Json,
            schema: Some(r#"{ "date": "string", "body": "string" }"#.to_string()),
            layout_notes: Some("one object per entry".to_string()),
            examples: vec![r#"{ "date": "March 9", "body": "..." }"#.to_string()],
            tone: Some("neutral".to_string()),
        };
        let desc = target.describe();
        assert!(desc.contains("JSON"));
        assert!(desc.contains("Schema"));
        assert!(desc.contains("Layout"));
        assert!(desc.contains("Tone"));
        assert!(desc.contains("Example 1"));
    }

    #[test]
    fn with_model_overrides_default() {
        let client = ClaudeClient::new("test-key").with_model("claude-sonnet-4-6");
        assert_eq!(client.model, "claude-sonnet-4-6");
    }
}
