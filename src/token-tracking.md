<!--
Token-usage report template.

This is plain Markdown with {{placeholder}} tokens that get filled from
token-usage.json when a report is generated. Edit it freely — reorder sections,
delete placeholders you don't need, change headings, add prose. Anything that
isn't a recognized placeholder is left exactly as written.

See template.md (same folder) for the full list of available placeholders.
-->

# Token Usage Report

_Generated: {{generated_at}}_

## This month ({{month_key}})

- Requests: {{this_month.requests}}
- Input tokens: {{this_month.input_tokens}}
- Output tokens: {{this_month.output_tokens}}
- **Total: {{this_month.total_tokens}}**

## All time

- Requests: {{all_time.requests}}
- Input tokens: {{all_time.input_tokens}}
- Output tokens: {{all_time.output_tokens}}
- Cache (write / read): {{all_time.cache_creation_input_tokens}} / {{all_time.cache_read_input_tokens}}
- **Total: {{all_time.total_tokens}}**

## Monthly breakdown

{{monthly_table}}

## Per-event log ({{event_count}} events)

{{events_table}}
