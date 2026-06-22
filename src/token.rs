//! Local token-usage tracking.
//!
//! Every successful Claude API call returns a `usage` block. [`record`] folds
//! that usage into a running ledger persisted as JSON on disk, so total token
//! spend survives across runs. [`crate::agent`] calls [`record`] wherever it
//! talks to the API.
//!
//! The read-modify-write is serialized within the process by a global lock, so
//! concurrent API calls (e.g. parallel async tests) accumulate correctly rather
//! than clobbering each other.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Default path for the on-disk ledger (relative to the working directory).
pub const DEFAULT_LOG_PATH: &str = "token-usage.json";

/// Errors recording or reading the token ledger.
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

/// Cumulative token totals persisted to the local log file.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct TokenLedger {
    /// Number of recorded API requests.
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
}

impl TokenLedger {
    /// Input + output tokens across all recorded requests (excludes cache-only
    /// accounting, which is reported separately).
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

// Serializes the read-modify-write of the ledger within this process.
static LOG_LOCK: Mutex<()> = Mutex::new(());

/// Fold one request's `usage` into the ledger at `path` (created if missing)
/// and return the updated totals.
pub fn record(path: impl AsRef<Path>, usage: &Usage) -> Result<TokenLedger, TokenError> {
    // Recover from a poisoned lock: the data behind it is just `()`.
    let _guard = LOG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = path.as_ref();
    let mut ledger = read_ledger(path)?;
    ledger.add(usage);
    write_ledger(path, &ledger)?;
    Ok(ledger)
}

/// Read the current ledger, returning the default (all-zero) ledger if the file
/// does not exist or is empty.
pub fn read_ledger(path: impl AsRef<Path>) -> Result<TokenLedger, TokenError> {
    match std::fs::read_to_string(path.as_ref()) {
        Ok(contents) if contents.trim().is_empty() => Ok(TokenLedger::default()),
        Ok(contents) => Ok(serde_json::from_str(&contents)?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(TokenLedger::default()),
        Err(e) => Err(e.into()),
    }
}

/// Write the ledger atomically (write to a temp file, then rename into place) so
/// a crash mid-write can't leave a half-written, unparseable ledger.
fn write_ledger(path: &Path, ledger: &TokenLedger) -> Result<(), TokenError> {
    let json = serde_json::to_string_pretty(ledger)?;
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
        let unique = format!(
            "atc-token-{tag}-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        );
        p.push(unique);
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn records_and_accumulates() {
        let path = temp_path("accum");
        let usage = Usage {
            input_tokens: 100,
            output_tokens: 40,
            ..Default::default()
        };

        let after_first = record(&path, &usage).unwrap();
        assert_eq!(after_first.requests, 1);
        assert_eq!(after_first.total_tokens(), 140);

        let after_second = record(&path, &usage).unwrap();
        assert_eq!(after_second.requests, 2);
        assert_eq!(after_second.input_tokens, 200);
        assert_eq!(after_second.output_tokens, 80);
        assert_eq!(after_second.total_tokens(), 280);

        // Persisted value matches what record() returned.
        assert_eq!(read_ledger(&path).unwrap(), after_second);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_file_reads_as_default() {
        let path = temp_path("missing");
        assert_eq!(read_ledger(&path).unwrap(), TokenLedger::default());
    }

    #[test]
    fn accumulates_cache_tokens() {
        let path = temp_path("cache");
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 5,
            cache_creation_input_tokens: 7,
            cache_read_input_tokens: 3,
        };
        let ledger = record(&path, &usage).unwrap();
        assert_eq!(ledger.cache_creation_input_tokens, 7);
        assert_eq!(ledger.cache_read_input_tokens, 3);
        std::fs::remove_file(&path).ok();
    }
}
