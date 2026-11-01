# EVOL-X M-09H — Human attribution drill runbook

**Policy:** `OD-EVOL-08-M1` (OPERATOR APPROVED, see [`EVOL_X_OPERATOR_DECISION_REGISTER_W1.md`](EVOL_X_OPERATOR_DECISION_REGISTER_W1.md)).  
**Contract:** `evol_m09_human_drill.v2` — [`EVOL_X_EXPERIMENT_AND_METRIC_SPEC.md`](EVOL_X_EXPERIMENT_AND_METRIC_SPEC.md).  
**Finding fixed:** `F-EVOLX-P0-M09H-ATTRIBUTION-SLA-FALSE-PASS` (v1 applied the 900 s median to tool time).

## Who / when

- Performed **only by the human evaluator** (independent of the implementer/agent; OD-EVOL-11). No agent-produced, synthetic or demo timings are permitted anywhere.
- Evidence lifecycle: **no** completed drill, worksheet or M-09H `MEASURED` in the TECH publication. They enter only in the FINAL evidence commit.

## Two separate measurements

| Field | What is measured | Threshold |
|-------|------------------|-----------|
| `tool_seconds` | Runtime of the qualified dossier/drill tool on the seed path (as EVOL-M-09T). Measured separately from the human. | **≤ 120 s per case** |
| `attribution_seconds` | Human clock: starts when the evaluator **receives the complete diagnostic packet** for the case; stops when the evaluator **submits the diagnosis, before** the expected / canonical answer is shown. | **median(C1,C2,C3) ≤ 900 s** (equality OK) |

Tool runtime is **not** part of `attribution_seconds`; the 900 s median is **never** applied to tool timing.

## Procedure (per case C1, C2, C3 from `evol_m09_drill_case_registry.v1.json`)

1. Run the tool on the seed path @ `test_execution_sha`; record `tool_seconds`.
2. Hand the complete diagnostic packet to the evaluator; start the attribution clock.
3. Evaluator submits the diagnosis (attributed cause identities); stop the clock → `attribution_seconds`. Only then reveal the expected answer.
4. Record `correct` and `attributed_cause_identities` (must equal registry `expected_cause_identities`).
5. Run the adversarial control; record `adversarial_control_passed`.
6. Complete the worksheet (`evol_m09_human_drill_worksheet.v1.json`) and compute its SHA-256 into `worksheet_digest`.

## Drill artifact `evol_m09_human_drill.v2.json`

- `schema_version`: exactly `evol_m09_human_drill.v2`; `measurement_policy_id`: exactly `OD-EVOL-08-M1`.
- `source_commit_sha` = `test_execution_sha` = NEW_TES; `registry_digest` = SHA-256 of committed registry bytes.
- `cases[]`: `case_id`, `correct`, `tool_seconds`, `attribution_seconds`, `attributed_cause_identities` (all seconds finite, ≥ 0).
- `median_tool_seconds`, `median_attribution_seconds` must equal the computed case medians (±0.001).
- `drill_started_at_utc` ≤ `drill_completed_at_utc`; non-empty `evaluator_id`, `evaluator_display_name`.

## Verification boundary

`verify-evidence-publish` (Mode A, final) checks schema literal, policy id, bindings, arithmetic, per-case tool SLA and median attribution SLA. It **cannot** prove a human actually performed the drill or that the clock was measured honestly — that attestation is the independent B6 auditor's responsibility.
