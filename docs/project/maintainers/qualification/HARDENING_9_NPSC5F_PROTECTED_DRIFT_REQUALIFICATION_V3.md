# HARDENING-9 — NPSC-5F Protected Drift Requalification V3

**Task:** `HARDENING_9_NPSC5F_PROTECTED_DRIFT_REQUALIFICATION_V3`  
**Status:** `REQUALIFIED / RE-FROZEN / PASS`  
**Audited HEAD (`origin/development`):** `6b3744f356c6e725fc124843f1da4a8fd5417120`  
**Qualified OBS evolution (R1 feature + remediation + integrated R2 bounds):** `36de9ed76` + `4c809e84a` + `c588d04b1` (final state @ audited HEAD)  
**Prior sentinel (V2):** `22c4793da4ba751fff6c93f780a7f6848650a5d9`

## Executive summary

After V2 @ `22c4793da`, integrated `development` gained **OBS-RUNTIME-HISTORY-BOUNDS-R1** (`36de9ed76`) and mandatory **identity probe remediation** (`4c809e84a`). The feature commit alone introduced disallowed raw `mint_*` identity in composition validation; qualification applies only to the **paired final state** at audited HEAD, not to `36de9ed76` in isolation.

**Classification:** `QUALIFIED_OBS_EVOLUTION` + `REMEDIATION_OF_ARCHITECTURE_REGRESSION` (probe IDs).  
**UNKNOWN:** `0` · **UNQUALIFIED_PROTECTED_DRIFT:** `0` · **REAL_ARCHITECTURE_REGRESSION:** `0`

**Production code in this requalification commit:** NO — sentinel baselines + qualification evidence only.

## Baseline re-freeze (V3)

| Sentinel | Old | New |
|----------|-----|-----|
| `R1_POST_R2_QUALIFIED_BASELINE_SHA` | `22c4793da4ba751fff6c93f780a7f6848650a5d9` | `6b3744f356c6e725fc124843f1da4a8fd5417120` |
| `R2_POST_QUALIFIED_BASELINE_SHA` | `22c4793da4ba751fff6c93f780a7f6848650a5d9` | **unchanged** (no R2-protected drift in range) |
| `NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA` | `22c4793da4ba751fff6c93f780a7f6848650a5d9` | `6b3744f356c6e725fc124843f1da4a8fd5417120` |

`NPSC5F_R3_H1_QUALIFIED_BASELINE_SHA` / `NPSC5F_R3_H1_QUALIFICATION_RECORD_SHA`: **unchanged** (H1 integrated-head pin — out of V3 scope).

## Pre-fix drift (@ `22c4793da..19a16cd1`)

| Family | Protected paths |
|--------|-----------------|
| R1 | `intergrax/runtime/events/event_bus.py` |
| R2 | *(none)* |
| Final Evidence Plane (BREAKING) | `event_bus.py`, `runtime_event_history.py`, `runtime_event_history_validation.py`, `runtime_event_metric_scope.py` |

Protected drift trio before re-freeze: **1/3 PASS** (R2 clean; R1/Final drift via Final sentinel tests).

## Protected path ledger

| Family | Path | Introducing | Remediation | Classification |
|--------|------|-------------|-------------|----------------|
| R1 | `event_bus.py` | `36de9ed76` | — | QUALIFIED_OBS_EVOLUTION (`event_count`; history append; durable path preserved) |
| Final | `event_bus.py` | `36de9ed76` | — | QUALIFIED_OBS_EVOLUTION |
| Final | `runtime_event_history.py` | `36de9ed76` | — | QUALIFIED_OBS_EVOLUTION (retention contract; bounded/disabled) |
| Final | `runtime_event_history_validation.py` | `36de9ed76` | `4c809e84a` | REMEDIATION_OF_ARCHITECTURE_REGRESSION → QUALIFIED @ HEAD |
| Final | `runtime_event_metric_scope.py` | `c588d04b1` | — | QUALIFIED_OBS_EVOLUTION (contract `RuntimeEventMetricScope`; scoped counters) |

**Adjacent (non-BREAKING / explicit unrelated):** `intergrax/contracts/runtime_event_history.py`, `intergrax/contracts/runtime_event_metric.py` (contract lifts), `nexus_loop.py` / `task_finisher.py` (scoped metrics wiring; Final classifier unrelated prefix).

## Commit ledger (`22c4793da..HEAD` touching `intergrax/runtime/events/`)

| SHA | Message | Protected impact |
|-----|---------|------------------|
| `36de9ed76` | OBS-RUNTIME-HISTORY-BOUNDS-R1 | R1 + Final paths above; contract `RuntimeEventHistoryRetention` |
| `4c809e84a` | probe ID remediation | `runtime_event_history_validation.py` only |
| `c588d04b1` | OBS-RUNTIME-HISTORY-BOUNDS-R2 | hard retention + `runtime_event_metric_scope.py`; bus/history extensions |

Docs-only commits `63608aae` (GR-7-A7), `1349f5071`, `19a16cd1d` do not touch Evidence Plane BREAKING surfaces.

## `36de9ed76` architecture map

```text
RuntimeEventHistoryBuffer (contract)
        ↓
resolve_runtime_event_history_buffer / plugins
        ↓
validate_runtime_event_history_buffer()  [composition]
        ↓
RuntimeEventBus (history append + event_count)
        ↓
Nexus TaskExecutionMetrics.runtime_events (delta event_count)
```

## `4c809e84a` remediation audit

- `mint_run_id` / `mint_attempt_id` / `mint_execution_id` in validator: **0** @ HEAD  
- Probe IDs: `validate_task_id` / `validate_run_id` / `validate_attempt_id` / `validate_execution_id` / `validate_event_id` with deterministic `history-retention-probe` namespace  
- No second authority, no `testing_support` import, conformance-only synthetic event

## Architecture proofs (@ audited HEAD)

| Audit | Result |
|-------|--------|
| Contract-first history buffer / retention | **PASS** |
| Plugin replaceability (`test_runtime_event_history_bounds.py`) | **PASS** |
| Retention `bounded` \| `disabled` fail-closed | **PASS** |
| EE-A2 / EE-A2-H2 / single authority | **PASS** (gate suite) |
| Root Admission / Resume Lineage / H6 | **PASS** (NPSC5F matrix) |
| H3 / production→testing_support | **PASS** |
| R1 / R2 / Final Evidence Plane gates | **PASS** |
| RC-01 / RC-02 / RC-03 | **CLOSED** (ancestry + gates) |
| Test weakening | **0** |

## Regression evidence (post re-freeze)

- Protected drift trio: **3/3 PASS** (recorded in V3 session log under `.tmp/session/npsc5f-protected-drift-v3/`)
- `test_npsc5f_final_mandatory_regression_matrix_passes`: session log (H1 may remain separate blocker)

## GitHub audit note

> Wprowadzone zmiany muszą zostać zaudytowane na podstawie kodu znajdującego się aktualnie na GitHub.

Post-push: verify V3 qualification commit, sentinel SHAs, H1 untouched, trio **3/3 PASS**, no production files in V3 commit.
