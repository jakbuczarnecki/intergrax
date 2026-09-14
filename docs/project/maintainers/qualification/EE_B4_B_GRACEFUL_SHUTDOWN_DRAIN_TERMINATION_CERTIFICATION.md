# EE-B4-B — Graceful Shutdown, Drain & Termination Certification

**Task:** EE-B4-B  
**Branch:** `development`

## Provenance

| Field | Value |
|-------|-------|
| **START_HEAD** | `cd0217ef0cbf2386f5f6134c30cfb80adf6ecddb` |
| **START_ORIGIN** | `cd0217ef0cbf2386f5f6134c30cfb80adf6ecddb` |

## Deliverables

| Artifact | Path |
| -------- | ---- |
| Shutdown model | `docs/project/maintainers/architecture/EXECUTION_ENGINE_GRACEFUL_SHUTDOWN_DRAIN_TERMINATION_MODEL.md` |
| Reference surface | `testing_support/shutdown/` |
| Gate tests | `tests/unit/runtime/architecture/test_ee_b4_b_*.py` |

**PRODUCTION CODE CHANGED:** NO (`intergrax/` unchanged).

## Ownership matrix

| Concern | Canonical owner | Mechanism | Duplicate? |
| ------- | --------------- | --------- | ---------- |
| Stop new root work | Host + execution contract | `ExecutionRuntimeShutdownPhase.STOP_ACCEPTING_NEW_WORK` | No |
| Active execution tracking | Host active-work port / execution pool | counters, tasks | No |
| Capacity permits | `ExecutionCapacityAdmissionPort` | acquire/release | No |
| Worker drain | Hosting `ShutdownExecutor` + execution work | bounded wait/cancel | No |
| Cancellation | asyncio / host cancel phase | `CancelledError` + permit release | No |
| Evidence flush | Runtime event persistence | mandatory flush port | No |
| Final state | Host/runtime persistence | `PERSIST_FINAL_STATE` | No |
| Retry during shutdown | Recovery plane (NPSC-5E) | no new root admission | No |
| Worker termination | pool / boundary close | `TERMINATE_WORKERS` | No |
| Exporter close | observability wiring | best-effort after persist | No |

## Tests

| Module | Focus |
| ------ | ----- |
| `test_ee_b4_b_stop_accepting_new_work.py` | Reject roots after stop |
| `test_ee_b4_b_active_execution_drain.py` | Drain admitted work |
| `test_ee_b4_b_shutdown_admission_race.py` | Event-sync race |
| `test_ee_b4_b_worker_failure_during_drain.py` | Contained worker fault |
| `test_ee_b4_b_cancellation_during_drain.py` | Cancel + permit release |
| `test_ee_b4_b_mandatory_evidence_flush.py` | Fail-closed evidence |
| `test_ee_b4_b_final_state_persistence.py` | Order before terminate |
| `test_ee_b4_b_capacity_release.py` | Permits + double release |
| `test_ee_b4_b_worker_task_leak.py` | Zero managed leaks |
| `test_ee_b4_b_shutdown_idempotency.py` | Safe second call |
| `test_ee_b4_b_concurrent_shutdown.py` | Single effective lifecycle |
| `test_ee_b4_b_compound_shutdown_failure.py` | Primary vs secondary |
| `test_ee_b4_b_root_vs_child_drain.py` | Root vs child admission |
| `test_ee_b4_b_shutdown_health_semantics.py` | EE-B4-A mapping |
| `test_ee_b4_b_shutdown_architecture_gate.py` | Docs + forbidden symbols |

## NPSC-5F HANDOFF

**NONE** — protected surfaces not modified.

## Regression matrix

EE-A1, EE-A2, NPSC-4.2, NPSC-5B, NPSC-5E recovery slice, W5-A bounded sink tests where applicable, EE-B1.1–B1.3, EE-B2, EE-B3-A/C, EE-B4-A, EE-B4-B — see session pytest log.

## Final verdict

**PASS** when EE-B4-B suite (2×) and regression matrix complete with zero new static errors.
