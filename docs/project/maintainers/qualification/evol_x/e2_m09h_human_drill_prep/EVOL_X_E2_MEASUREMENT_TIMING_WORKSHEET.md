# EVOL-X E2 — Measurement timing worksheet (human)

**Policy:** OD-EVOL-08-M1 · Units: **seconds** (finite, ≥ 0) · Timestamps: **UTC ISO-8601**

## Definitions

| Symbol | Starts | Stops | SLA |
|--------|--------|-------|-----|
| `tool_seconds` | Tool invocation begins (`perf_counter` or UTC pair) | Tool output written | ≤ **120** per case |
| `attribution_seconds` | Evaluator receives **complete** diagnostic packet | Evaluator submits diagnosis **before** oracle | median(C1–C3) ≤ **900** |

Tool time **must not** be included in attribution.

## Per-case log (fill during drill)

| Case | tool_started_utc | tool_completed_utc | tool_seconds | packet_received_utc | diagnosis_submitted_utc | attribution_seconds | submitted_cause_identities |
|------|------------------|--------------------|--------------|---------------------|-------------------------|---------------------|----------------------------|
| C1 | | | | | | | |
| C2 | | | | | | | |
| C3 | | | | | | | |

## Medians (after C1–C3)

| Metric | Value | Computed from |
|--------|-------|-------------|
| `median_tool_seconds` | | median(C1,C2,C3 tool) |
| `median_attribution_seconds` | | median(C1,C2,C3 attribution) |

## Acceptance (human + Mode A)

- All three `tool_seconds` ≤ 120
- Median attribution ≤ 900 (strict `>` fails)
- All diagnoses match oracle (operator review post-seal)
- Adversarial AC-M09H-01 pass
- Worksheet JSON committed; digest matches drill artifact

## JSON export

Transcribe into [artifacts/evol_m09_human_drill_worksheet.v1.template.json](./artifacts/evol_m09_human_drill_worksheet.v1.template.json) — replace every `PENDING_HUMAN_EXECUTION` with measured values. Incorrect diagnoses stay recorded; do not substitute oracle values retroactively.
