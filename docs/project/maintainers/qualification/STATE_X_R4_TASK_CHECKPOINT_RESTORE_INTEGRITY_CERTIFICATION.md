# STATE-X-R4 — Task Checkpoint Restore Integrity & Recovery Authority

| Field | Value |
| --- | --- |
| PRE_AUDIT_HEAD / START_HEAD (R4) | `716ed746f8b463681db536804235cedc86adc162` |
| R4-R1 START_HEAD | `6a8252548a8fb81830afac8534acbd5183f9df95` |
| Scope | TaskCheckpoint → validated restore → Task materialization → execution re-entry |
| Canonical restore validator | `intergrax/runtime/long_running/checkpoint_resume_validation.py` |
| Canonical persistence owner | `TaskCheckpointPersistence` |
| Status (Cursor) | **READY FOR AUDIT** — not CLOSED |

## Prior accepted reconciliation

Independent acceptance @ `716ed746f8b463681db536804235cedc86adc162`:

- STATE-X-R3-R3 / SX-F12 — CLOSED (bookkeeping)
- STATE-X-R3-R4 / SX-F13 — CLOSED (bookkeeping)
- STATE-X-R3-R5 / SX-F14 — CLOSED (bookkeeping)

## Before / after restore graph

**Before:** checkpoint → `Task.model_validate(task_snapshot)` → optional `is_checkpoint_resumable`.

**After:** checkpoint → `assert_checkpoint_resume_materialization_eligible` (structural) → `Task.model_validate` → execution paths with `assert_checkpoint_resume_eligible` where current authority applies (`restore_if_resuming`, orchestration).

## Closed-world restore inventory

See `tests/qualification/state_x/_r4_task_checkpoint_restore_support.py` (`RESTORE_CONSUMER_INVENTORY`).

## Q2-D1 disposition

**PASS (local):** `terminal_capability_from_task_checkpoint_store()` in `execution_terminal/persistence.py` replaces ad-hoc `isinstance` narrowing in `nexus_loop.py`. No persistence ownership change.

## Backup / restore responsibility (next child only)

| Store | Responsibility |
| --- | --- |
| `TaskCheckpointPersistence` / SQLite default | BACKEND/OPERATOR WITH EXPLICIT PLATFORM CONSISTENCY REQUIREMENTS |
| Execution terminal via checkpoint capability | component of checkpoint store when capability implemented |
| Global backup engine | NOT ENTERED — STATE-X-R5 DR child |

## FRZ mapping (scoped evidence only)

Direct: FRZ-STA-01/02/04/05/06/08; FRZ-REC-01/02/03/04/06/09/10; FRZ-TEN-08.  
Not claimed: FRZ-REC-05, FRZ-REC-08, global FRZ PASS, STATE-X parent CLOSED.

## STATE-X-R4-R1 — Closed-World Restore Consumer Convergence

Canonical structural reader: `validated_task_snapshot_from_checkpoint()` in `checkpoint_resume_validation.py` (composes schema, identity binding, snapshot integrity). Execution/resume paths retain `assert_checkpoint_resume_materialization_eligible` / `assert_checkpoint_resume_eligible`.

Consumer inventory: `tests/qualification/state_x/_r4_r1_restore_consumer_support.py` (`R4_R1_TASK_CHECKPOINT_CONSUMERS`). Raw `Task.model_validate(checkpoint.task_snapshot)` outside allowlist: **0** (`find_raw_task_snapshot_parsers()`).

Worker recovery: **YES** — checkpoint `run_id`/`attempt_id` continue after `restore_if_resuming`; incoming worker envelope is recovery; tenant from `execution_identity.tenant_id`; cross-task/tenant checkpoint rejected before activation.

Prior report **Q47**: documentation numbering error (DG-001 lineage matrix), not STATE-X-R4 Q01–Q40.

## Tests

`uv run --with cryptography pytest tests/qualification/state_x -k r4_r1 -p no:xdist -q`
`uv run --with cryptography pytest tests/qualification/state_x/test_state_x_r4_task_checkpoint_restore.py tests/qualification/state_x -p no:xdist -q`
