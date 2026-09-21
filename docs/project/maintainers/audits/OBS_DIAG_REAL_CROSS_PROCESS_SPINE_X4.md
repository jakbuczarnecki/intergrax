# OBS-DIAG-X4 — Real Cross-Process Async & Recovery Spine

**Verdict:** `PASS — REAL CROSS-PROCESS ASYNC & RECOVERY SPINE QUALIFIED` (requires real Kafka broker at `localhost:9092` for Phase A gate execution).

## Proof level distinction

| Capability | Before X4 | After X4 |
| --- | --- | --- |
| In-process async worker spine | P3 | P3 (unchanged) |
| Kafka transport only | P4 (existing integration proofs) | P4 |
| Kafka → worker → execution → OBS → DIAG | P2/P3 | **P4** — `test_obs_universal_spine_cross_process_x4_e2e.py` (marker `obs_coverage_p4`, `external_proof`) |
| HITL durable runtime rebuild | P3 — `test_obs_universal_spine_hitl_restart_e2e.py` | P3 (unchanged semantics) |
| HITL OS process pause → new process resume | NOT_PROVEN | **P4** — same X4 module, `CrossProcessSpineHarness.run_hitl_cross_process_proof` |

## Phase A (Kafka)

- **Broker:** repo Docker profile `infra/integration` Kafka (`localhost:9092`).
- **Boundary:** parent process enqueues via `open_kafka_producer`; worker subprocess (`testing_support.cross_process_spine.kafka_worker_cli`) consumes and enters canonical `UnifiedTaskRunner` / diagnostic stack (no direct `DiagnosticOrchestrator` in worker).
- **Delivery:** at-least-once Kafka semantics; idempotency via platform worker/idempotency keys where configured in scenarios.
- **Fresh read:** new `SqliteFileDocumentStore` + `build_diagnostic_read_service` in parent after worker exit.

## Phase B (HITL)

- **Process A:** pause + durable checkpoint + continuation export JSON.
- **Process B:** new host runtime, restore continuation backing, resume with checkpoint; `process_pid` differs (enforced in tests).
- **Platform fix:** `resolve_root_task_identity` restores checkpoint root `execution_id` when resuming without explicit override (governed HITL approval alignment).
- **X4A closure (successor):** [`OBS_DIAG_HITL_FAILURE_DIAGNOSTIC_CLOSURE_X4A.md`](OBS_DIAG_HITL_FAILURE_DIAGNOSTIC_CLOSURE_X4A.md) — cross-process approved resume + post-terminal violation → durable Problem visible via fresh read; human `REJECT` remains governed FAILED without false Problem.

## X4A note (honest gap history)

X4 did not originally assert durable Problem on HITL resume failure paths; X4A adds `test_hitl_cross_process_resume_injected_violation_creates_durable_problem` and renames human rejection test to reflect semantics (no diagnostic Problem expected).

## Harness location

Reusable support: `testing_support/cross_process_spine/` (`CrossProcessSpineHarness`, typed Pydantic scenario configs, subprocess CLIs).

## Out of scope (X5+)

Broker outage, partition recovery, external HITL vendor qualification, multi-provider persistence failover matrices.

## Tests (representative)

```text
tests/integration/runtime/test_obs_universal_spine_cross_process_x4_e2e.py
tests/integration/runtime/test_obs_universal_spine_hitl_restart_e2e.py  # P3 rebuild (in-process)
tests/integration/runtime/test_obs_universal_spine_async_e2e.py         # P3 async
tests/unit/runtime/execution/test_orchestration.py                      # checkpoint execution_id restore
```
