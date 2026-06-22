# Customizing the token-usage report

`token-tracking.md` is the **template** used to render a token-usage report. It
is plain Markdown with `{{placeholder}}` tokens that are replaced with values
from `token-usage.json`. Edit it however you like — reorder sections, delete
placeholders you don't care about, add prose, change headings or styling.
Anything that isn't a recognized placeholder is left untouched, so the file
stays valid Markdown whether or not it's been rendered.

## Generating a report

```rust
use agent_text_cleanup::report;
use agent_text_cleanup::token;

let markdown = report::generate_report(
    report::DEFAULT_TEMPLATE_PATH, // "src/token-tracking.md"
    token::DEFAULT_LOG_PATH,       // "token-usage.json"
)?;
std::fs::write("token-report.md", markdown)?;
```

`generate_report(template_path, log_path)` reads the template, reads the log,
and returns the filled Markdown. A missing log file is treated as empty (all
zeros); a missing template file is an error.

## Scalar placeholders

Each is replaced wherever it appears (use any of them zero or more times).

| Placeholder | Value |
| --- | --- |
| `{{generated_at}}` | Report generation time (RFC 3339) |
| `{{month_key}}` | Current month, `YYYY-MM` |
| `{{event_count}}` | Number of logged events |
| `{{all_time.requests}}` | Total API requests, all time |
| `{{all_time.input_tokens}}` | Input tokens, all time |
| `{{all_time.output_tokens}}` | Output tokens, all time |
| `{{all_time.total_tokens}}` | Input + output, all time |
| `{{all_time.cache_creation_input_tokens}}` | Cache-write tokens, all time |
| `{{all_time.cache_read_input_tokens}}` | Cache-read tokens, all time |
| `{{this_month.requests}}` | Requests in the current month |
| `{{this_month.input_tokens}}` | Input tokens, current month |
| `{{this_month.output_tokens}}` | Output tokens, current month |
| `{{this_month.total_tokens}}` | Input + output, current month |
| `{{this_month.cache_creation_input_tokens}}` | Cache-write tokens, current month |
| `{{this_month.cache_read_input_tokens}}` | Cache-read tokens, current month |

(`total_tokens` is input + output; cache tokens are reported separately.)

## Table placeholders

These expand into a whole Markdown table. Put each on its own line.

| Placeholder | Expands to |
| --- | --- |
| `{{monthly_table}}` | One row per month: Month, Requests, Input, Output, Total |
| `{{events_table}}` | One row per event: Timestamp, Model, Operation, Input, Output, Cache write, Cache read |

When there's no data yet, a table placeholder renders a short italic note
(e.g. `_No events recorded yet._`) instead of an empty table.

## Adding a new value

The placeholder set lives in [`src/report.rs`](report.rs) — the `render`
function (scalars) and the `events_table` / `monthly_table` helpers. To expose a
new value, add a `replace` call in `render`, then document the placeholder in
the table above.
