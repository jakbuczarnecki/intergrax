# NPSC-5F/R1 — Durable Evidence Commit & Tenant Integrity

## Purpose

Close OBS-01 (mandatory evidence fail-open on bus) and OBS-05 (routing tenant may diverge from `RuntimeEvent.tenant_id`) without introducing a second evidence framework.

## Provenance

| Milestone | SHA |
| --------- | --- |
| NPSC-5E Final | `fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7` |
| NPSC-5F/P0 | `7811371da1069b661987b050a4c9bf42c02bda69` |
| R1 integrated baseline | `development` at qualification time |

## Parallel-session ownership

Session A owns R1 production surfaces. Sessions B/C/D inventories and qualification runners were not modified.

## Production surface

- `intergrax/runtime/events/evidence_durability.py` — `EvidencePersistenceRequirement`, `evidence_persistence_requirement`
- `intergrax/runtime/events/persistence_contract.py` — tenant routing validation, typed errors
- `intergrax/runtime/events/event_bus.py` — persist-before-history/subscribers; mandatory fail-closed

## Durability taxonomy

| Value | Meaning |
| ----- | ------- |
| `NOT_PERSISTED` | `should_persist_event` is false (sampling / gate) |
| `MANDATORY` | Durable commit required; persistence failure propagates |
| `BEST_EFFORT` | Persist when gated; failure logged, boundary continues |

## Mandatory evidence classes

Spine and platform kinds with `RetentionClass.OPERATIONAL` or `RetentionClass.AUDIT` when `should_persist_event` is true (execution lifecycle, tools, HITL, retry, terminal, validation audit types, etc.).

## Best-effort classes

`RetentionClass.DEBUG` (currently `TASK_PROGRESS` spine) when selected for persistence by sampling.

## Persistence failure semantics

- **Mandatory:** `MandatoryEvidencePersistenceError` (chained from store error); no subscriber dispatch; no bus history append on failure.
- **Best-effort:** logged exception; history and subscribers proceed.

## Tenant resolution matrix

| Case | Result |
| ---- | ------ |
| event `T1`, route `T1` | Accept `T1` |
| event `T1`, route `T2` | `EvidenceTenantRoutingMismatchError`, zero write |
| event `T1`, route absent | `T1` |
| event absent, route `T1` | `T1` (explicit host scope) |
| both absent | `""` (legacy global/test scope) |

## Idempotency interaction

Unchanged: `reconcile_idempotent_event_acceptance`. Tenant mismatch on duplicate route is blocked before write.

## Store adapter consistency

Enforced via shared `resolve_persistence_scope` / `resolve_event_tenant_id` in memory, SQLite, document-backed, validating wrapper, and null test store.

## Cross-process durability

SQLite cross-instance read qualification retained (P0 + R1).

## Concurrency invariants

Per-run position allocation unchanged (R1 does not alter store locking).

## Frozen 5E interaction

No execution lifecycle, retry, checkpoint, lineage, or governance code changes.

## Known out-of-scope P0 blockers

OBS-03 export bypass (R3), OBS-04 journal truncation (R2), task-global ordering (R2), as-of/bitemporal (R4).

## Regression matrix

| Suite | Role |
| ----- | ---- |
| `test_npsc5f_r1_durable_evidence_commit_tenant_integrity.py` | R1 gate |
| `test_npsc5f_p0_execution_evidence_architecture_reconciliation.py` | P0 + updated OBS-01/05 |
| `tests/unit/runtime/events/**` | Event bus, idempotency, stores |

## Static quality

`ruff` / `pyright` on changed paths — required clean at commit.

## Final verdict

Recorded at commit time after full regression PASS.
