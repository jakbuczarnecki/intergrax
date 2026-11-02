# EVOL-X E2 — Automated diagnostic qualification record

**Classification:** `AUTOMATED_DIAGNOSTIC` — **not** M-09H human drill evidence.

**Disposition:** `AUTOMATED DIAGNOSTIC QUALIFICATION — PASS` @ active TEST_EXECUTION_SHA `2b165516485154583a4118ba1ecb2e25549bb5d0`.

| Field | Value |
|-------|-------|
| Active TES | `2b165516485154583a4118ba1ecb2e25549bb5d0` |
| Active TECH | `1e175ff8f678d605aa841ae52362486a56ed006d` |
| W6-04 CI | run `41327408027`, 288 PASS |
| Run ID | `cd51347d65424b75aaf5693bee6da05d` |
| Machine artifact | [`artifacts/evol_m09_automated_diagnostic_qualification.v1.json`](./artifacts/evol_m09_automated_diagnostic_qualification.v1.json) |
| Diagnostic outputs | [`artifacts/automated_qualification/cd51347d65424b75aaf5693bee6da05d/`](./artifacts/automated_qualification/cd51347d65424b75aaf5693bee6da05d/) |
| Session runner (replay) | `.tmp/session/evol-x-e2-automated/run_e2_automated_batch.py` |
| Session log | `.tmp/session/evol-x-e2-automated/logs/run2.log`, `pytest-focused.log` |

## Explicit non-goals (certification boundary)

- **EVOL-M-09H (human):** remains **NOT_MEASURED** in P03 `evol_metrics.v1.json`.
- **EVIDENCE_FINAL_PUBLICATION_SHA:** **PENDING** (no `evol_m09_human_drill.v2.json` / worksheet).
- **B6 independent acceptance:** **PENDING** (OD-EVOL-11 auditor not named; no human drill package).
- **EVOL-X parent closure:** **OPEN**.

## C1–C3 machine results (summary)

| Case | Seed | Registry agree | `tool_seconds` | Cause identity (hex) |
|------|------|------------------|----------------|----------------------|
| C1 | `intergrax/contracts/execution_identity.py` | true | 0.505 | `3fc41379…b091d` |
| C2 | `testing_support/change_safety/graph/impact.py` | true | 0.521 | `f1decda5…f83e` |
| C3 | `testing_support/change_safety/contracts/enums.py` | true | 0.656 | `82364099…ca55` |

Median machine `tool_seconds`: **0.521** (SLA ≤120 s/case — PASS). **No** `attribution_seconds` recorded.

## Automated adversarial subset

Controls **AC-AUTO-01..08:** PASS (determinism, wrong digest/kind, unknown seed, oracle mismatch, honeytoken separation, wrong TES inventory, tampered node id).

**AC-M09H-01a/b/c:** **NOT_AUTOMATED** (human attestation required per [EVOL_X_E2_ADVERSARIAL_CONTROL_PROCEDURE.md](./EVOL_X_E2_ADVERSARIAL_CONTROL_PROCEDURE.md)).

## Qualification tests (Batch F)

Focused pytest @ publication HEAD (pre-commit): `test_evol_x_m09_cause_identity.py` + `test_m09_human_drill_negative_cases` — **40 passed** (see session log). No qualified-code change → **no** new TES / **no** VL3 full re-run.

## Substitution analysis

See [EVOL_X_E2_PROPOSED_M09H_AUTOMATION_SUBSTITUTION_AMENDMENT.md](./EVOL_X_E2_PROPOSED_M09H_AUTOMATION_SUBSTITUTION_AMENDMENT.md) (**PROPOSED / NOT APPROVED**).
