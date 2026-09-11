# Enterprise Execution Scale & Resilience — W4-C Distributed Cancellation Inventory

**Task:** W4-C — Distributed cancellation semantics & durable external operation termination  
**Status:** W4-C FINAL · contract frozen · qualification matrix extended  
**Production runtime changed:** YES

Companion: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md`](../architecture/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md).

## Principle

**Cancellation ≠ termination.** Intent (`CANCELLATION_REQUESTED` … `TERMINATED`) is separate from physical provider state (`RUNNING` … `UNKNOWN`). No central `CancellationManager` / `GlobalOperationManager` — small ports + durable ownership per `operation_id`.

## External operation inventory

| Operation | Owner | Durable state | Cancellation capability |
|-----------|-------|---------------|-------------------------|
| LLM call | `LLMAdapter` + optional `ExternalOperationCancellationPort` | `ExternalOperationStateStore` (process-local default) | Provider-dependent; noop port → intent-only; status → `UNKNOWN` when unsupported |
| Tool execution | `RuntimeToolInvoker` + `DependencyAttemptExecutionBoundary` | Same store (optional inject) | Cooperative + durable intent; permit released only after terminal physical record |
| HTTP integration | Integration adapter boundary | No W4-C store by default | Client disconnect / timeout; physical may stay `UNKNOWN` |
| Background workers | Host / queue worker | Checkpoint + optional external op store | Cooperative metadata (W4-A); orphan reconcile marks `UNKNOWN` |
| Recovery action | `resume_decision_from_durable_state_with_recovery_admission` | Checkpoint CAS (W3) + external op gate | Reconcile `RUNNING` without owner before resume |
| External side effects | Tool idempotency ledger + external op identity | Idempotency + `operation_id` stable across retries | Cancel before submit suppresses external call |

## Contracts

| Artifact | Path |
|----------|------|
| Identity | `intergrax/contracts/external_operation_identity.py` |
| Cancellation / status ports | `intergrax/contracts/external_operation_cancellation.py` |
| Durable CAS store | `intergrax/runtime/external_operations/external_operation_state_store.py` |
| Ownership helpers | `intergrax/runtime/external_operations/external_operation_ownership.py` |
| Tool seam | `intergrax/runtime/external_operations/tool_external_operation_attempt.py` |
| LLM seam | `intergrax/runtime/external_operations/llm_external_operation_attempt.py` |
| Recovery gate | `intergrax/runtime/external_operations/recovery_external_operation_gate.py` |

## Invariants

| ID | Invariant |
|----|-----------|
| I1 | Stable `operation_id` across physical retries (`mint_stable_operation_id`) |
| I2 | Permit: acquire → physical execution → terminal durable state → release |
| I3 | Lost worker: `RUNNING` without active owner → `UNKNOWN` + reconcile before recovery |
| I4 | CAS: stale terminal write → `StaleExternalOperationStateError` (preserve terminal) |

## Qualification matrix

See `tests/unit/runtime/architecture/test_enterprise_scale_resilience_w4_c_distributed_cancellation_qualification.py`.
