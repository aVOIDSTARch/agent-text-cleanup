//! Render a token-usage report from a Markdown template.
//!
//! This module imports [`crate::token`] and fills a user-editable Markdown
//! template (by default `src/token-tracking.md`) with values read from the
//! token log (`token-usage.json`). The template uses `{{placeholder}}` tokens;
//! see `src/template.md` for the full list and how to customize the layout.

use crate::token::{self, Totals, TokenLog};
use chrono::Utc;
use std::path::Path;

/// Conventional location of the report template.
pub const DEFAULT_TEMPLATE_PATH: &str = "src/token-tracking.md";

/// Errors generating a report.
#[derive(Debug, thiserror::Error)]
pub enum ReportError {
    #[error("could not read report template: {0}")]
    Template(std::io::Error),
    #[error(transparent)]
    Token(#[from] token::TokenError),
}

/// Generate a report by filling `template_path` with values from the token log
/// at `log_path`.
pub fn generate_report(
    template_path: impl AsRef<Path>,
    log_path: impl AsRef<Path>,
) -> Result<String, ReportError> {
    let template = std::fs::read_to_string(template_path).map_err(ReportError::Template)?;
    let log = token::read_log(log_path)?;
    Ok(render(&template, &log))
}

/// Fill a template string with values from an in-memory [`TokenLog`].
///
/// Block placeholders (`{{events_table}}`, `{{monthly_table}}`) are expanded
/// first, then scalar placeholders. Text that isn't a recognized placeholder is
/// left untouched.
pub fn render(template: &str, log: &TokenLog) -> String {
    let now = Utc::now();
    let mut out = template.to_string();

    // Block placeholders (whole Markdown tables) first.
    out = out.replace("{{events_table}}", &events_table(log));
    out = out.replace("{{monthly_table}}", &monthly_table(log));

    // Top-level scalars.
    out = out.replace("{{generated_at}}", &now.to_rfc3339());
    out = out.replace("{{month_key}}", &now.format("%Y-%m").to_string());
    out = out.replace("{{event_count}}", &log.events.len().to_string());

    // Totals groups.
    replace_totals(&mut out, "all_time", &log.all_time);
    replace_totals(&mut out, "this_month", &log.this_month());

    out
}

/// Replace the `{{<prefix>.<field>}}` scalar placeholders for one totals group.
fn replace_totals(out: &mut String, prefix: &str, t: &Totals) {
    let mut set = |field: &str, val: String| {
        *out = out.replace(&format!("{{{{{prefix}.{field}}}}}"), &val);
    };
    set("requests", t.requests.to_string());
    set("input_tokens", t.input_tokens.to_string());
    set("output_tokens", t.output_tokens.to_string());
    set("total_tokens", t.total_tokens().to_string());
    set(
        "cache_creation_input_tokens",
        t.cache_creation_input_tokens.to_string(),
    );
    set(
        "cache_read_input_tokens",
        t.cache_read_input_tokens.to_string(),
    );
}

/// Render the per-event log as a Markdown table (or a note when empty).
fn events_table(log: &TokenLog) -> String {
    if log.events.is_empty() {
        return "_No events recorded yet._".to_string();
    }
    let mut s = String::from(
        "| Timestamp | Model | Operation | Input | Output | Cache write | Cache read |\n\
         | --- | --- | --- | ---: | ---: | ---: | ---: |\n",
    );
    for e in &log.events {
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            e.timestamp,
            e.model,
            e.operation,
            e.input_tokens,
            e.output_tokens,
            e.cache_creation_input_tokens,
            e.cache_read_input_tokens,
        ));
    }
    s.trim_end().to_string()
}

/// Render the monthly rollup as a Markdown table (or a note when empty).
fn monthly_table(log: &TokenLog) -> String {
    if log.monthly.is_empty() {
        return "_No monthly totals yet._".to_string();
    }
    let mut s = String::from(
        "| Month | Requests | Input | Output | Total |\n| --- | ---: | ---: | ---: | ---: |\n",
    );
    for (month, t) in &log.monthly {
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            month,
            t.requests,
            t.input_tokens,
            t.output_tokens,
            t.total_tokens(),
        ));
    }
    s.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::TokenEvent;
    use std::collections::BTreeMap;

    fn sample_log() -> TokenLog {
        let totals = Totals {
            requests: 2,
            input_tokens: 300,
            output_tokens: 60,
            ..Default::default()
        };
        let mut monthly = BTreeMap::new();
        monthly.insert("2026-06".to_string(), totals.clone());
        TokenLog {
            all_time: totals,
            monthly,
            events: vec![TokenEvent {
                timestamp: "2026-06-22T10:00:00+00:00".to_string(),
                model: "claude-opus-4-8".to_string(),
                operation: "correct".to_string(),
                input_tokens: 300,
                output_tokens: 60,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            }],
        }
    }

    #[test]
    fn fills_scalar_placeholders() {
        let out = render(
            "reqs={{all_time.requests}} total={{all_time.total_tokens}} events={{event_count}}",
            &sample_log(),
        );
        assert_eq!(out, "reqs=2 total=360 events=1");
    }

    #[test]
    fn renders_events_table() {
        let out = render("{{events_table}}", &sample_log());
        assert!(out.contains("| Timestamp |"), "{out}");
        assert!(out.contains("claude-opus-4-8"));
        assert!(out.contains("correct"));
    }

    #[test]
    fn renders_monthly_table() {
        let out = render("{{monthly_table}}", &sample_log());
        assert!(out.contains("| Month |"), "{out}");
        assert!(out.contains("2026-06"));
    }

    #[test]
    fn empty_log_tables_show_placeholder_text() {
        let out = render("{{events_table}}\n{{monthly_table}}", &TokenLog::default());
        assert!(out.contains("No events recorded"), "{out}");
        assert!(out.contains("No monthly totals"), "{out}");
    }

    #[test]
    fn unknown_text_is_left_untouched() {
        let out = render("# My Report\nNothing to fill here.", &sample_log());
        assert_eq!(out, "# My Report\nNothing to fill here.");
    }

    #[test]
    fn generate_report_reads_template_and_log_files() {
        // Write a temp log and a temp template, then render through the file API.
        let dir = std::env::temp_dir();
        let log_path = dir.join(format!("atc-report-log-{}.json", std::process::id()));
        let tpl_path = dir.join(format!("atc-report-tpl-{}.md", std::process::id()));
        std::fs::write(&log_path, serde_json::to_string(&sample_log()).unwrap()).unwrap();
        std::fs::write(&tpl_path, "requests={{all_time.requests}}").unwrap();

        let out = generate_report(&tpl_path, &log_path).unwrap();
        assert_eq!(out, "requests=2");

        std::fs::remove_file(&log_path).ok();
        std::fs::remove_file(&tpl_path).ok();
    }
}
