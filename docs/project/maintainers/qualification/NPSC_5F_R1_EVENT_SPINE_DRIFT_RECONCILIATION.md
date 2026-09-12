# NPSC-5F/R1 — Event Spine Drift Reconciliation

## Purpose

Qualify integrated post-H1 changes to R1-owned event spine surfaces (`event_bus.py`, `runtime_event.py`) as **compatible** with frozen NPSC-5F/R1 durable evidence contracts, then advance the R1 P0 drift sentinel baseline without altering persistence semantics.

## Provenance

| Milestone | SHA |
|-----------|-----|
| R1 implementation | `455d3b216f0ad56ea9cdf9db6e0f760b50063a81` |
| Qualified `EXECUTION_FAILED` enum (H1) | `40cc8c11e0b57ed4cf0d99ed1b9b297820c6eaa8` |
| Pre-reconciliation integrated `development` | `ec23b24d4bb398c83ed310c9deee7d462d73a7e0` |
| R1 sentinel after reconciliation | `ec23b24d4bb398c83ed310c9deee7d462d73a7e0` |

## Drift window (`40cc8c11` → `ec23b24d`)

| Path | Class | Summary |
|------|-------|---------|
| `intergrax/runtime/events/runtime_event.py` | A | Additive `RuntimeEventType.EXTERNAL_OPERATION_FAILED` |
| `intergrax/runtime/events/event_bus.py` | B | Optional `EventSinkPort` delivery after durable commit; W5-B close/drain |

Related non-R1-protected wiring (catalog, payloads, registry) supports the enum extension only.

## Contract assessment

| Area | Result | Notes |
|------|--------|-------|
| Event identity (`EventId`) | PASS | No ID minting or equality changes |
| `RuntimeEvent` fields / versioning | PASS | Additive enum member only |
| Ordering (`ExecutionEventPosition`) | PASS | Unchanged; persistence store owns order |
| `publish` / `record` semantics | PASS | Still fact publication; persist-before-history/subscribers |
| Persistence boundary | PASS | `_commit_durable_evidence` unchanged; sink is observability-only |
| Tenant isolation | PASS | Behavioral gate on memory store |

## Publication spine budget

`RuntimeEventType` count increased to **58** with `EXTERNAL_OPERATION_FAILED` (qualified additive enum). `_PUBLICATION_SPINE_TARGET_MAX` advanced to **58** with behavioral proof in `test_publication_spine_budget_within_target` — not a performance regression.

## Tests

| Gate | Path |
|------|------|
| R1 event-spine reconciliation | `tests/unit/runtime/architecture/test_npsc5f_r1_event_spine_drift_reconciliation.py` |
| R1 protected drift classifier | `tests/unit/testing_support/test_npsc5f_r1_event_spine_drift.py` |
| R1 Final / P0 / R2 Final / R3 Final | Existing NPSC-5F Final qualification modules |

## Decision

**PASS** — Event spine technical changes preserve frozen R1 durable-fact meaning. R1 sentinel baseline advanced to `ec23b24d4bb398c83ed310c9deee7d462d73a7e0`.

**Production code changed in reconciliation task:** YES — publication spine budget constant only (`spine_consolidation.py`).
