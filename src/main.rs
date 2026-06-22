//! Demo entry point for the OCR text-cleanup pipeline.
//!
//! Stage 1 ([`normalize::regex_repair`]) always runs — it is offline and free.
//! Stages 2 and 3 (the agent corrections) only run when `ANTHROPIC_API_KEY`
//! is set, since they hit the Claude API.

mod agent;
mod normalize;

use agent::{ClaudeClient, FormatTarget, OutputFormat};

const SAMPLE: &str = "March 9th\n\
Wea week! ory Doctors\n\n\
The   power of atten-\ntion is the most\nimportant skill we can culti-\nvate.\n\n\
~^*#@eae garbage line\n\n\
Negation: I will not be ruled by distraction.";

#[tokio::main]
async fn main() {
    // Stage 1: regex + heuristics (always available).
    let cleaned = normalize::regex_repair(SAMPLE);
    println!("=== Stage 1: regex_repair ===\n{cleaned}\n");

    // Stages 2 & 3 need an API key.
    let client = match ClaudeClient::from_env() {
        Ok(client) => client,
        Err(_) => {
            println!("(set ANTHROPIC_API_KEY to run the agent correction stages)");
            return;
        }
    };

    // Stage 2: agent correction, no formatting guidance.
    match client.correct(&cleaned).await {
        Ok(corrected) => println!("=== Stage 2: agent correct ===\n{corrected}\n"),
        Err(e) => eprintln!("stage 2 failed: {e}"),
    }

    // Stage 3: agent correction toward a format/design target.
    let target = FormatTarget {
        description: "a single daily meditation entry with a date, a theme title, body text, \
            and a closing negation line"
            .to_string(),
        output_format: OutputFormat::Json,
        schema: Some(
            r#"{ "date": "string", "theme": "string", "body": "string", "negation": "string" }"#
                .to_string(),
        ),
        layout_notes: Some("one JSON object; correct the garbled theme title from context".to_string()),
        ..Default::default()
    };
    match client.correct_to_target(&cleaned, &target).await {
        Ok(structured) => println!("=== Stage 3: agent correct_to_target ===\n{structured}"),
        Err(e) => eprintln!("stage 3 failed: {e}"),
    }
}
