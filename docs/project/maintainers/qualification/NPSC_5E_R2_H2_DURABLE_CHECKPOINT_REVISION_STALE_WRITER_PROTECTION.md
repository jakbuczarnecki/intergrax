# NPSC-5E/R2-H2 — Durable Checkpoint Revision & Stale Writer Protection

**Status:** `PASS`

**Date:** 2026-09-10

**Branch:** `development`

**Predecessors:** R2-H1 `7bc0c651ca536ffe1b06ac88e7180b4b62380010`

---

## Purpose

Close stale-write correctness in canonical checkpoint persistence via durable logical revision and atomic compare-and-set so a delayed writer based on superseded state cannot become the new canonical checkpoint solely because it persisted later.

**Guarantee:**

> A checkpoint writer based on superseded logical revision cannot commit a new canonical checkpoint revision.

Claims: optimistic concurrency, single successor revision, stale writer rejection, idempotent duplicate commit reconciliation — not distributed exactly-once.

---

## Concurrency model

| Dimension | Contract |
| --- | --- |
| Stream key | `(tenant_id, task_id)` — not `run_id`; retry attempts do not reset revision |
| Revision representation | `TaskCheckpoint.revision` (`int >= 1` when persisted) |
| First revision | `1` on first successful save (`expected_revision=None`) |
| Expected revision | Writer presents predecessor revision; `None` only for empty stream |
| Next revision | `N + 1` when CAS succeeds |
| Atomicity boundary | `TaskCheckpointPersistence.save` (SQLite: `BEGIN IMMEDIATE` → read max revision → validate → insert) |
| Conflict behavior | `StaleCheckpointWriteError` with `task_id`, `tenant_id`, `expected_revision`, `actual_revision` |
| Duplicate write | Same `checkpoint_id` + identical payload → idempotent return |
| Same ID conflict | Same `checkpoint_id` + different payload → `CheckpointIdConflictError` |
| Physical sequence | `store_sequence` / `rowid` — diagnostics only |

---

## Implementation scope

| Artifact | Role |
| -------- | ---- |
| `checkpoint_revision.py` | Typed errors (`StaleCheckpointWriteError`, etc.) |
| `models.py` | `TaskCheckpoint.revision` |
| `persistence_contract.py` | `save(..., expected_revision=...)` port |
| `store.py` | SQLite CAS + migration + `ORDER BY checkpoint_revision DESC` |
| `coordinator.py` | Derives `expected_revision` from `TaskOrchestrationState.checkpoint_revision` |
| `checkpoint_resume_validation.py` | Stale comparison via logical revision |
| `task_contract.py` | `TaskOrchestrationState.checkpoint_revision` orchestration fact |

---

## Qualification gate

`tests/unit/runtime/architecture/test_npsc5e_r2_h2_checkpoint_revision_stale_writer_protection.py`

| Area | Result |
| ---- | ------ |
| First write → revision 1 | PASS |
| Successor write CAS | PASS |
| Stale writer | BLOCKED |
| Concurrent same-predecessor writers | ONE success / ONE stale |
| Revision fork / skip | BLOCKED |
| Late stale writer (blocker reproduction) | BLOCKED |
| Timestamp variants (same/missing/manipulated) | PASS |
| Higher rowid / lower revision | Logical revision wins |
| `get_latest` / `get_by_token` | Highest logical revision |
| Old token resume | `REJECT_STALE` |
| Idempotent duplicate / unknown commit retry | PASS |
| Tenant / task stream isolation | PASS |
| Cross-process / store reopen | PASS |
| H1 authority gate | PASS |
| R2 original gate | PASS |

---

## Ownership (unchanged)

```text
Checkpoint logical revision  → TaskCheckpointPersistence
Scheduler fence              → resume ownership (separate)
Attempt generation           → AttemptLifecycleService
Lineage                      → ExecutionLineagePersistence
Authority                    → authority plane
```

---

## Next

`NPSC-5E/R2 Final` — Checkpoint & Durable Resume Qualification and Freeze
