//! Local token-usage tracking with a per-event log.
//!
//! Every successful Claude API call appends a [`TokenEvent`] node (timestamp,
//! model, operation, and input/output/cache token counts) to a JSON log on
//! disk, and folds its usage into a running **monthly** total and an **all-time**
//! total. [`crate::agent`] calls [`record`] wherever it talks to the API.
//!
//! On-disk shape:
//! ```json
//! {
//!   "all_time": { "requests": 2, "input_tokens": 547, "output_tokens": 117, ... },
//!   "monthly":  { "2026-06": { "requests": 2, ... } },
//!   "events": [
//!     { "timestamp": "2026-06-22T10:00:00+00:00", "model": "claude-opus-4-8",
//!       "operation": "correct", "input_tokens": 300, "output_tokens": 60, ... }
//!   ]
//! }
//! ```
//!
//! The read-modify-write is serialized within the process by a global lock, so
//! concurrent API calls (e.g. parallel async tests) accumulate correctly.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Default path for the on-disk log (relative to the working directory).
pub const DEFAULT_LOG_PATH: &str = "token-usage.json";

/// Errors recording or reading the token log.
#[derive(Debug, thiserror::Error)]
pub enum TokenError {
    #[error("token log I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("token log parse error: {0}")]
    Parse(#[from] serde_json::Error),
}

/// Token usage for a single response, as reported by the Messages API `usage`
/// object. Cache fields default to zero when absent.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: u64,
}

/// One recorded API call.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TokenEvent {
    /// RFC 3339 timestamp of when the event was recorded.
    pub timestamp: String,
    /// Model that served the request (e.g. `"claude-opus-4-8"`).
    pub model: String,
    /// Which agent operation produced it (e.g. `"correct"`).
    pub operation: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
}

/// Aggregated token totals over some set of events.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct Totals {
    /// Number of recorded API requests.
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
}

impl Totals {
    /// Input + output tokens (cache accounting is reported separately).
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    fn add(&mut self, usage: &Usage) {
        self.requests += 1;
        self.input_tokens += usage.input_tokens;
        self.output_tokens += usage.output_tokens;
        self.cache_creation_input_tokens += usage.cache_creation_input_tokens;
        self.cache_read_input_tokens += usage.cache_read_input_tokens;
    }
}

/// The full on-disk log: per-event nodes plus monthly and all-time rollups.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct TokenLog {
    /// Running total across every recorded event.
    #[serde(default)]
    pub all_time: Totals,
    /// Running totals keyed by `"YYYY-MM"` (UTC), ordered chronologically.
    #[serde(default)]
    pub monthly: BTreeMap<String, Totals>,
    /// Every recorded event, in the order they happened.
    #[serde(default)]
    pub events: Vec<TokenEvent>,
}

impl TokenLog {
    /// Totals for the current UTC month, or the default (all-zero) totals if
    /// nothing has been recorded this month.
    pub fn this_month(&self) -> Totals {
        let key = Utc::now().format("%Y-%m").to_string();
        self.monthly.get(&key).cloned().unwrap_or_default()
    }
}

// Serializes the read-modify-write of the log within this process.
static LOG_LOCK: Mutex<()> = Mutex::new(());

/// Append an event for one API call to the log at `path` (created if missing),
/// update the monthly and all-time totals, and return the updated log.
///
/// `model` and `operation` are stored on the event as "other details".
pub fn record(
    path: impl AsRef<Path>,
    model: &str,
    operation: &str,
    usage: &Usage,
) -> Result<TokenLog, TokenError> {
    record_at(path, Utc::now(), model, operation, usage)
}

/// [`record`] with an explicit timestamp — the seam used by tests so monthly
/// bucketing is deterministic.
fn record_at(
    path: impl AsRef<Path>,
    now: DateTime<Utc>,
    model: &str,
    operation: &str,
    usage: &Usage,
) -> Result<TokenLog, TokenError> {
    // Recover from a poisoned lock: the data behind it is just `()`.
    let _guard = LOG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = path.as_ref();

    let mut log = read_log(path)?;

    log.all_time.add(usage);
    log.monthly
        .entry(now.format("%Y-%m").to_string())
        .or_default()
        .add(usage);
    log.events.push(TokenEvent {
        timestamp: now.to_rfc3339(),
        model: model.to_string(),
        operation: operation.to_string(),
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cache_creation_input_tokens: usage.cache_creation_input_tokens,
        cache_read_input_tokens: usage.cache_read_input_tokens,
    });

    write_log(path, &log)?;
    Ok(log)
}

/// Read the current log, returning the default (empty) log if the file does not
/// exist or is empty.
pub fn read_log(path: impl AsRef<Path>) -> Result<TokenLog, TokenError> {
    match std::fs::read_to_string(path.as_ref()) {
        Ok(contents) if contents.trim().is_empty() => Ok(TokenLog::default()),
        Ok(contents) => Ok(serde_json::from_str(&contents)?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(TokenLog::default()),
        Err(e) => Err(e.into()),
    }
}

/// Write the log atomically (write to a temp file, then rename into place) so a
/// crash mid-write can't leave a half-written, unparseable log.
fn write_log(path: &Path, log: &TokenLog) -> Result<(), TokenError> {
    let json = serde_json::to_string_pretty(log)?;
    let mut tmp: PathBuf = path.to_path_buf();
    tmp.set_extension("json.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "atc-token-{tag}-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn at(rfc3339: &str) -> DateTime<Utc> {
        rfc3339.parse().unwrap()
    }

    #[test]
    fn appends_event_nodes_with_details() {
        let path = temp_path("events");
        let usage = Usage {
            input_tokens: 300,
            output_tokens: 60,
            ..Default::default()
        };

        let log = record_at(
            &path,
            at("2026-06-22T10:00:00Z"),
            "claude-opus-4-8",
            "correct",
            &usage,
        )
        .unwrap();

        assert_eq!(log.events.len(), 1);
        let ev = &log.events[0];
        assert_eq!(ev.model, "claude-opus-4-8");
        assert_eq!(ev.operation, "correct");
        assert_eq!(ev.input_tokens, 300);
        assert_eq!(ev.output_tokens, 60);
        assert!(ev.timestamp.starts_with("2026-06-22T10:00:00"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rolls_up_all_time_and_monthly() {
        let path = temp_path("rollup");
        let usage = Usage {
            input_tokens: 100,
            output_tokens: 40,
            ..Default::default()
        };

        // Two events in June, one in July.
        record_at(&path, at("2026-06-10T08:00:00Z"), "m", "correct", &usage).unwrap();
        record_at(&path, at("2026-06-20T08:00:00Z"), "m", "correct", &usage).unwrap();
        let log = record_at(&path, at("2026-07-01T08:00:00Z"), "m", "correct_to_target", &usage)
            .unwrap();

        // All-time spans every event.
        assert_eq!(log.all_time.requests, 3);
        assert_eq!(log.all_time.input_tokens, 300);
        assert_eq!(log.all_time.total_tokens(), 420);
        assert_eq!(log.events.len(), 3);

        // Monthly buckets are split by month and ordered.
        assert_eq!(log.monthly.len(), 2);
        let june = &log.monthly["2026-06"];
        assert_eq!(june.requests, 2);
        assert_eq!(june.total_tokens(), 280);
        let july = &log.monthly["2026-07"];
        assert_eq!(july.requests, 1);
        assert_eq!(july.total_tokens(), 140);

        // Persisted log round-trips identically.
        let reread = read_log(&path).unwrap();
        assert_eq!(reread.all_time, log.all_time);
        assert_eq!(reread.monthly, log.monthly);
        assert_eq!(reread.events.len(), 3);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_file_reads_as_default() {
        let path = temp_path("missing");
        let log = read_log(&path).unwrap();
        assert_eq!(log.all_time, Totals::default());
        assert!(log.monthly.is_empty());
        assert!(log.events.is_empty());
    }

    #[test]
    fn tracks_cache_tokens() {
        let path = temp_path("cache");
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 5,
            cache_creation_input_tokens: 7,
            cache_read_input_tokens: 3,
        };
        let log = record_at(&path, at("2026-06-22T10:00:00Z"), "m", "correct", &usage).unwrap();
        assert_eq!(log.all_time.cache_creation_input_tokens, 7);
        assert_eq!(log.all_time.cache_read_input_tokens, 3);
        assert_eq!(log.events[0].cache_read_input_tokens, 3);
        std::fs::remove_file(&path).ok();
    }
}
