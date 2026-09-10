# NPSC-5E/R2 — Checkpoint & Durable Resume Hardening

**Status:** `CORRECTION REQUIRED` → closed by R2-H1 (`NPSC_5E_R2_H1_AUTHORITATIVE_RESUME_AUTHORITY_STALE_CHECKPOINT_CLOSURE.md`)

**Date:** 2026-09-09

**Branch:** `development`

---

## Purpose

Harden canonical checkpoint/resume so durable resume after process interruption is identity-safe, lineage-safe, terminal-safe, authority-safe, governance-safe, version-safe, stale-safe, and cross-process-safe — without a second checkpoint framework, runtime, or scheduler.

---

## Pre-flight drift reconciliation

| Label | SHA |
| ----- | --- |
| R1 Final | `76603ed266f9f54106bf4718fe8886180a826351` |
| Relevant post-R1 lineage hardening | `a18e65c077ed57bf3bb64ef015b47ee1f3ceb6bf` |
| R2 qualified start baseline (`origin/development`) | `98dcc2603713188098d6199303eaaf99196d39a5` |

**Drift classification (`76603ed..origin/development`):**

| Class | Files |
| ----- | ----- |
| B — execution/lineage/child | `child.py`, `active_lineage.py`, `admission.py`, DG_001 lineage tests/docs |
| C — unrelated VPI | `platform_proofs/.../verified_product_identification/*` |
| D/E — unrelated adapters/docs | strict tool adapters + tests |

**PRE-FLIGHT LINEAGE REQUALIFICATION:** `PASS` (R1 final, P0A, DG_001 nested-child suite)

---

## Implementation scope

| Artifact | Role |
| -------- | ---- |
| `checkpoint_resume_validation.py` | Canonical resume eligibility validator |
| `runtime_checkpoint.py` | `runtime_checkpoint.v2` schema gate in `validate_canonical` |
| `coordinator.py` | Persist gate + restore eligibility + narrowed authority |
| `test_npsc5e_r2_checkpoint_durable_resume_hardening.py` | R2 qualification gate |

---

## Qualification matrix (summary)

| Area | Result |
| ---- | ------ |
| Schema version gate | PASS |
| Wrong task/run/attempt/root/tenant | BLOCKED |
| Stale checkpoint | BLOCKED |
| Terminal / cancel after checkpoint | BLOCKED |
| Lineage cross-validation | PASS |
| Missing required durable lineage | BLOCKED |
| Authority narrowing on restore | PASS |
| Authority expansion via checkpoint | BLOCKED |
| **R2-H1 defect:** checkpoint authority fallback on restore (`restored.execution_authority`) | **FIXED** (H1) |
| **R2-H1 defect:** timestamp-only stale ordering | **FIXED** (H1 physical sequence; superseded by H2 logical revision CAS) |
| Cross-process SQLite resume | PASS |
| Concurrent scheduler claim (one winner) | PASS |
| Completed node output retained | PASS |
| Unknown side-effect blind replay | BLOCKED |
| Provider-neutral coordinator | PASS |
| No second checkpoint framework | PASS |
| R1 final regression | PASS |
| P0A regression | PASS |
| Checkpoint store restore merge (pre-existing defect) | FIXED |

---

## Known pre-existing (unchanged by R2)

| Test | Note |
| ---- | ---- |
| `test_partial_results::test_build_task_progress_view_aggregates_checkpoints` | `human_request_expires_at` hydration unrelated to resume gate |
| `test_graph_runner_resilience` | Task pydantic validation (recorded R1) |
| `test_budget_ticks` | identity binding (recorded R1) |

---

## R2-H1 correction provenance

| Defect | Correction |
| ------ | ---------- |
| `coordinator.py` restored checkpoint `execution_authority` when current task authority was `None` | Removed; `validate_checkpoint_resume_authority` + `resolve_resume_execution_authority` only narrow authoritative current |
| `validate_checkpoint_not_stale` allowed equal/missing timestamp ambiguity | H1: `store_sequence` interim ordering; **H2:** `checkpoint_revision` canonical ordering + CAS |
| Stale writer race (late physical insert resurrects old state) | **FIXED** (R2-H2 durable revision CAS — `NPSC_5E_R2_H2_DURABLE_CHECKPOINT_REVISION_STALE_WRITER_PROTECTION.md`) |

## Next

`NPSC-5E/R2 Final` — audit freeze after operator review.
