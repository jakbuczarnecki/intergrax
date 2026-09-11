# Enterprise Execution Scale & Resilience — W3 Final Qualification

| Field | Value |
|-------|-------|
| **Status** | W3-C2 decision event append + snapshot CAS **implemented**; W3-C3 **`DECISION_DURABLE` recovery start admission** wired; W3-C4 **canonical durable recovery wiring** + production bypass gate |
| **Date** | 2026-09-11 |
| **Baseline** | `development` |

## Scope delivered (W3-C2)

| Capability | Evidence |
|------------|----------|
| Decision event compare-and-append | `intergrax/contracts/decision_event_append.py`, `sqlite_decision_event_append_persistence.py` |
| Event idempotency (`event_id`) | `DuplicateDecisionEventError` + idempotent replay |
| Snapshot revision CAS | `snapshot_revision` on `decision_checkpoints`, `StaleDecisionCheckpointWriteError` |
| Multi-writer conflict tests | `tests/unit/runtime/execution/test_decision_event_append_and_snapshot_cas.py` (20 writers) |
| `DECISION_DURABLE` recovery admission | `RecoveryKind.DECISION_DURABLE`, `decision_durable_recovery_handoff`, `test_local_recovery_admission.py`, `test_decision_durable_recovery_admission.py` |

## Scope delivered (W3-C4)

| Capability | Evidence |
|------------|----------|
| Canonical `DECISION_DURABLE` recovery entry | `resume_decision_from_durable_state_with_recovery_admission` (`decision_durable_recovery_handoff.py`) |
| Production bypass forbidden | `test_w3_c4_decision_durable_recovery_canonical_wiring.py` |
| Qualification workers wired | `testing_support/decision_e2e/canonical_decision_durable_resume.py`, `docker_worker.py` |
| Admission lifecycle matrix | `test_decision_durable_recovery_w3_c4.py` (storm / cancel / CAS conflict permit release) |

```text
DECISION_DURABLE recovery (production):
  handoff.acquire → materialize → handoff.release (finally)
```

## Explicit non-goals (unchanged)

- Full decision recovery orchestration from event stream
- Distributed event store / Kafka
- Global lock or retry managers
- Merging decision plane with execution recovery (`HostTaskExecution`, `LongRunningCoordinator`, `FanOutPartialRecoveryService` untouched)

## Plane separation (qualified)

```text
Decision Event History  ≠  Checkpoint Snapshot  ≠  Execution Recovery
```

Event append conflicts (`StaleDecisionEventAppendError`) and snapshot CAS conflicts (`StaleDecisionCheckpointWriteError`) are **distinct** architectural signals — snapshot CAS does not substitute for authoritative event sequencing.

## Regression gates (W3-C2 closeout)

Run before merge:

- W1 admission: `tests/unit/runtime/resilience/test_local_recovery_admission.py` (recovery admission contract sibling set includes W1/W3 paths)
- W2 dependency isolation inventory / admission unit tests
- W2-C resilience composition tests under `tests/unit/runtime/resilience/`
- W3 recovery admission: `tests/unit/runtime/resilience/test_local_recovery_admission.py`
- W3-C4 canonical wiring: `tests/unit/runtime/architecture/test_w3_c4_decision_durable_recovery_canonical_wiring.py`, `tests/unit/runtime/resilience/test_decision_durable_recovery_w3_c4.py`
- R3 partial recovery qualification tests (`test_npsc5e_r3_*`)
- DG_001 lineage tests in `tests/unit/runtime/architecture/test_npsc5e_r2_h2_q1_frozen_regression_closure.py` (lineage marker)
