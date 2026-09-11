# LLM External Operation — Diagnostic Integration Architecture R2

| Field | Value |
|-------|-------|
| **Status** | Accepted (R2) |
| **Date** | 2026-09-11 |
| **Depends on** | Admission R1, Execution Spine, Central Diagnostic Engine |

## Decision

External operations remain **adapters** on the Execution Runtime Spine. Diagnostic truth flows through **RuntimeEvent** → **DiagnosticOrchestrator**; no second execution authority or local Problem store.

## Identity ownership

| Identifier | Authority |
|----------|-----------|
| `execution_id`, `attempt_id`, `run_id`, `task_id`, `tenant_id` | Execution Runtime |
| `operation_attempt_id` | External operation admission (retry mints new) |
| `operation_intent_id` | External operation intent |

`ExternalOperationExecutionContext` binds platform identity at execution time. `ExternalOperationAttempt.bind_platform_execution` enforces `execution_id` immutability.

## Evidence flow

```text
Provider failure
  → ExternalOperationFailureKind (taxonomy)
  → RuntimeEvent EXTERNAL_OPERATION_FAILED + ExternalOperationFailurePayloadV1
  → ExecutionReconstruction
  → ExternalOperationFailureAnalyzer
  → DiagnosticFinding EXTERNAL_OPERATION_FAILED
  → ProblemGrouping / FailureBoundaryAnalysis
```

`ExternalOperationEvidenceContributor` emits extension evidence only (no `create_problem`).

## Failure semantics

- No `exception.message` → Problem mapping.
- Provider exceptions classified to namespaced `ExternalOperationFailureKind`.
- Parent execution is not marked `EXECUTION_FAILED` for isolated provider faults.

## Diagnostic integration

- `DiagnosticFindingKind.EXTERNAL_OPERATION_FAILED` with `DiagnosticPrecision.EXTERNAL_BOUNDARY`.
- `DiagnosticInvestigationView` exposes `external_operation_context` and `external_operation_failures`.

## Predictive integration

`collect_external_operation_failure_history` supplies read-only failure history for risk analyzers (recommend mitigation; does not stop providers).

## Security

- Tenant enforced at admission and event persistence.
- Audit/evidence payloads sanitized; secrets rejected in qualification tests.
- Typed payloads bounded (`max_length` on envelope fields).

## Limitations

- Retry helper does not schedule platform retries automatically.
- Predictive history is task-scoped via runtime event store APIs.
- Extension contributor snapshots are opt-in for non-event evidence.
