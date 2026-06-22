//! End-to-end tests that exercise the full pipeline against the live Claude API.
//!
//! These make real network calls and consume tokens. They read the key from the
//! environment or a local `.env` (var: `ANTHROPIC_API_KEY`). When no key is
//! available the tests **skip** rather than fail, so the suite still passes in
//! CI or on a fresh checkout — run them locally with the key present:
//!
//! ```sh
//! cargo test --test e2e
//! ```

use agent_text_cleanup::agent::{ClaudeClient, FormatTarget, OutputFormat};
use agent_text_cleanup::normalize;

/// Raw OCR-damaged text: split word, soft-wrapped sentence, garbled theme
/// title, a stray-symbol line, and a closing negation.
const OCR_SAMPLE: &str = "March 9th\n\
Wea week! ory Doctors\n\n\
The   power of atten-\ntion is the most\nimportant skill we can culti-\nvate.\n\n\
~^*#@eae garbage line\n\n\
Negation: I will not be ruled by distraction.";

/// Build a client from `.env`/environment, or `None` to signal a skip.
fn client_or_skip() -> Option<ClaudeClient> {
    let _ = dotenvy::dotenv();
    match ClaudeClient::from_env() {
        Ok(client) => Some(client),
        Err(_) => {
            eprintln!("skipping live e2e test: ANTHROPIC_API_KEY not set");
            None
        }
    }
}

/// Extract the JSON object from a model response, tolerating ```json fences or
/// stray prose around it by slicing from the first `{` to the last `}`.
fn extract_json(raw: &str) -> &str {
    match (raw.find('{'), raw.rfind('}')) {
        (Some(start), Some(end)) if end >= start => &raw[start..=end],
        _ => raw,
    }
}

/// Stage 1 + Stage 2: regex cleanup, then a freeform agent correction with no
/// formatting guidance. The agent must repair what regex can't decide.
#[tokio::test]
async fn freeform_correction_repairs_text() {
    let Some(client) = client_or_skip() else {
        return;
    };

    // Feed the *raw* OCR through the agent so it has real damage to fix
    // (regex alone wouldn't rejoin "culti-\nvate" + "important" wrapping cleanly).
    let corrected = client
        .correct(OCR_SAMPLE)
        .await
        .expect("correct() should return text");

    assert!(
        !corrected.trim().is_empty(),
        "correction was empty: {corrected:?}"
    );

    let lower = corrected.to_lowercase();
    // Content the model should recover/preserve, not destroy.
    assert!(
        lower.contains("attention"),
        "expected 'attention' to survive correction, got: {corrected}"
    );
    assert!(
        lower.contains("cultivate"),
        "expected 'cultivate' to survive correction, got: {corrected}"
    );
    // The split-word hyphenation must be gone.
    assert!(
        !corrected.contains("atten-"),
        "hyphenation was not repaired: {corrected}"
    );
    // The stray-symbol garbage line should not survive.
    assert!(
        !corrected.contains("~^*#@"),
        "stray OCR symbols leaked through: {corrected}"
    );
}

/// Stage 1 + Stage 3: regex cleanup, then a target-guided agent correction that
/// must emit JSON matching the supplied [`FormatTarget`].
#[tokio::test]
async fn target_correction_emits_expected_json() {
    let Some(client) = client_or_skip() else {
        return;
    };

    let cleaned = normalize::regex_repair(OCR_SAMPLE);

    let target = FormatTarget {
        description: "a single daily meditation entry with a date, a theme title, body text, \
            and a closing negation line"
            .to_string(),
        output_format: OutputFormat::Json,
        schema: Some(
            r#"{ "date": "string", "theme": "string", "body": "string", "negation": "string" }"#
                .to_string(),
        ),
        layout_notes: Some(
            "one JSON object; infer the garbled theme title from context".to_string(),
        ),
        ..Default::default()
    };

    let out = client
        .correct_to_target(&cleaned, &target)
        .await
        .expect("correct_to_target() should return text");

    let value: serde_json::Value =
        serde_json::from_str(extract_json(&out)).unwrap_or_else(|e| {
            panic!("response was not valid JSON ({e}): {out}");
        });

    for key in ["date", "theme", "body", "negation"] {
        assert!(
            value.get(key).is_some(),
            "missing key {key:?} in JSON output: {out}"
        );
    }
}

/// A bad API key must surface as `AgentError::Api` with the HTTP status, not a
/// silent success or a panic — confirms the auth/header wiring end to end.
#[tokio::test]
async fn invalid_key_surfaces_api_error() {
    // Skip when there's no environment to talk to a real endpoint meaningfully;
    // this test still issues a real request, so only run it alongside the others.
    if client_or_skip().is_none() {
        return;
    }

    let bogus = ClaudeClient::new("sk-ant-invalid-key-for-testing");
    let err = bogus
        .correct("hello")
        .await
        .expect_err("a bogus key must not succeed");

    match err {
        agent_text_cleanup::agent::AgentError::Api { status, .. } => {
            assert!(
                status == 401 || status == 403,
                "expected 401/403 auth error, got {status}"
            );
        }
        other => panic!("expected AgentError::Api, got: {other}"),
    }
}
