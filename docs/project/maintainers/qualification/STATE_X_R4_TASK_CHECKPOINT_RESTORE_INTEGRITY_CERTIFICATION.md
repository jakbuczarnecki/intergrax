# STATE-X-R4 — Task Checkpoint Restore Integrity & Recovery Authority

| Field | Value |
| --- | --- |
| PRE_AUDIT_HEAD / START_HEAD | `716ed746f8b463681db536804235cedc86adc162` |
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

## Tests

`uv run --with cryptography pytest tests/qualification/state_x/test_state_x_r4_task_checkpoint_restore.py tests/qualification/state_x -p no:xdist -q`
