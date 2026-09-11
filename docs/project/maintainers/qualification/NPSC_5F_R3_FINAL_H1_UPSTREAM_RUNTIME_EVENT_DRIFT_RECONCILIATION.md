# NPSC-5F/R3 Final H1 — Upstream RuntimeEvent Drift Reconciliation

## Purpose

Qualify integrated post-R3 changes to the runtime event surface (especially `RuntimeEventType.EXECUTION_FAILED` and failure-evidence wiring) as **compatible** with frozen NPSC-5F/R1, R2, and R3 export-security contracts, then reconcile the R1 P0 drift sentinel baseline without modifying production code.

## Provenance

| Milestone | SHA |
|-----------|-----|
| R1 implementation | `455d3b216f0ad56ea9cdf9db6e0f760b50063a81` |
| R1 Final | `455c09f342f995ac0a6fcb03ffef2f4d3e36a447` |
| R2 implementation | `632507420f0ab8360aede43a2740e8fccc44efb4` |
| R2 Final | `76c92847f67da22d97943b55896a88c814d7e39d` |
| R3 implementation | `0346face3ef68d8f21504822a26f8f45f2384cf9` |
| Qualified `EXECUTION_FAILED` enum (DIAG R2) | `40cc8c11e0b57ed4cf0d99ed1b9b297820c6eaa8` |
| H1 gate completion (`origin/development`) | `5aeaaca4aeb78fd620d5994b7f08f5daa20b7db2` |
| R3 Final integrated `development` | `2965f2fcfed27162f06625c4b8bcd84b18d27704` |

## Current integrated HEAD

`2965f2fcfed27162f06625c4b8bcd84b18d27704` (`origin/development` at R3 Final freeze sign-off; no R3-protected export contract drift since `0346face`).

## Post-R3 remote drift

Under `intergrax/runtime/events/` since `0346face`:

| Path | Class | Commit |
|------|-------|--------|
| `runtime_event.py` | A — enum identity | `40cc8c11e` |
| `event_catalog.py` | G — catalog extension | `40cc8c11e` |
| `payload_registry.py` | H — registry extension | `40cc8c11e` |
| `payloads/canonical.py` | I — new payload type | `40cc8c11e` |
| `payloads/__init__.py` | K — re-export | `40cc8c11e` |
| `spine_consolidation.py` | J — spine budget +1 | `40cc8c11e` |

## Local working-tree drift

Parallel sessions may hold uncommitted execution/diagnostics/resilience changes. H1 qualification gates run against the integrated tree at `HEAD`; H1 commit contains **only** qualification artifacts (tests, `testing_support`, this doc, sentinel baseline).

## Changed event surfaces

See table above. No changes to R1-protected `persistence_contract.py`, `event_bus.py`, or `evidence_durability.py` since R2 implementation.

## RuntimeEvent exact diff

Since R3 implementation, `runtime_event.py` adds one enum member only:

```text
+    EXECUTION_FAILED = "execution_failed"
```

No `RuntimeEvent` model field, serializer, equality, or tenant routing changes.

## R1 compatibility assessment

| Invariant | Result |
|-----------|--------|
| Mandatory evidence fail-closed | PASS (behavioral) |
| Persist-before-history / subscriber | PASS |
| Tenant route/event exact match | PASS |
| EventId idempotency / conflict | PASS |
| Cross-tenant isolation | PASS |
| Provider-neutral persistence | PASS |

**Impact:** NONE on frozen R1 contract semantics; additive enum value only.

## R2 compatibility assessment

| Invariant | Result |
|-----------|--------|
| RunJournalReadPage / cursor | PASS |
| snapshot_through / after / through | PASS |
| ExecutionEventPosition tenant+run local | PASS |
| TaskRuntimeEventRuns / complete journal | PASS |

**Impact:** NONE.

## R3 forward-compatibility assessment

`ExecutionFailurePayloadV1` fields (`failure_kind`, `safe_summary`, `failure_code`) are **not** in `_SAFE_RUNTIME_EVENT_PAYLOAD_KEYS`. Export surfaces omit them; deny-by-default holds.

**Impact:** NONE (no automatic raw export).

## Event catalog changes

`EXECUTION_FAILED` mapped to `ExecutionPhase.STEP_EXECUTION`, retention via existing catalog rules, `ops:alert` hint. Catalog membership does not imply export permission.

## Payload registry changes

`EVENT_TYPE_PREFERRED_SCHEMA` maps `EXECUTION_FAILED` → `execution_failure.v1`. Registration ≠ export allowlist.

## Failure evidence changes

Failure evidence records canonical `RuntimeEvent` via `runtime_event_recorder.py` (execution adjacency). No logger/raw bypass in export path.

## Sentinel failures

| Sentinel | Baseline | Trigger | Owner |
|----------|----------|---------|-------|
| P0 R1 protected drift | was `R2_IMPLEMENTATION_SHA` | `runtime_event.py` post-R2 | R1 evidence |
| R2 Final drift | `63250742` | (none) | R2 journal |
| R3 protected drift | `0346face` | (none) | R3 export |

**Resolution:** Advance R1 sentinel window to `R1_POST_R2_QUALIFIED_BASELINE_SHA` (`40cc8c11e`) after behavioral proof.

## Behavioral qualification

`tests/unit/runtime/architecture/test_npsc5f_r3_final_h1_upstream_runtime_event_drift_reconciliation.py`

## Flaky/budget investigation

`spine_consolidation` target max 56→57 is additive catalog alignment; run `tests/unit/runtime/events/test_spine_consolidation.py` in full event suite gate.

## Baseline reconciliation decision

**BASELINE ADVANCED:** YES — to qualified enum commit `40cc8c11e` (immutable). Integrated HEAD `2965f2fcfed27162f06625c4b8bcd84b18d27704` recorded as H1 qualification pin (R3 Final sign-off on `origin/development`).

**Justification:** QUALIFIED COMPATIBLE DRIFT

## Changed qualification files

- `testing_support/npsc5f_r1_protected_drift.py`
- `testing_support/npsc5f_r3_h1_upstream_event_drift.py`
- `tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py`
- `tests/unit/runtime/architecture/test_npsc5f_r3_final_h1_upstream_runtime_event_drift_reconciliation.py`
- This document

## Final verdict

Compatible upstream drift; R3 Final predecessor blocker removed pending full gate matrix PASS on integrated `development`.

## Next step

NPSC-5F/R3 Final — Governed Evidence Export Qualification and Freeze (do not mark R3 FROZEN in H1).
