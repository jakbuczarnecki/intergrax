# ERL-QUAL-004 — Tracing integration

**Qualification:** ERL-QUAL-004 (`enterprise_payment_uncertainty_recovery`)  
**Scope:** Scenario execution observability on the platform diagnostic spine (no new telemetry framework).

## Business need

Payment uncertainty cases must be reconstructible end-to-end: checkout intent, external capture, UNKNOWN admission, reconciliation, evidence, resolution, governance, and recovery. Operators and proof evaluators need one correlated timeline—not proof-only logs.

## Ownership

| Layer | Owns |
| --- | --- |
| **Platform** | `TraceEvent`, `DiagnosticPayload` (`intergrax.contracts.tracing`), execution identity (`execution_id`, `run_id`), W3C `trace_id` |
| **Scenario** | Business step semantics (`ScenarioExecutionTraceStepId`), `ErlQual004LifecycleStepDiagV1` payload fields |
| **Proof** | Projects `ScenarioExecutionProofResult.execution_trace_events` into evidence; does not emit parallel trace models |

## Trace lifecycle

```text
scenario_execution_started
  → payment_workflow_started
  → external_payment_effect_created
  → unknown_detected
  → reliability_case_created
  → reconciliation_executed
  → evidence_evaluated
  → resolution_decided
  → governance_evaluated
  → recovery_executed
  → scenario_completed
```

Each step is a canonical `TraceEvent` with typed diagnostic payload `platform_proofs.erl_qual_004.lifecycle_step.v1` carrying `trace_id`, `correlation_id`, `execution_id`, `scenario_id`, `step_id`, `outcome`, and `component_identity`.

## Example timeline (Variant A — success)

| Seq | step_id | outcome | component |
| --- | --- | --- | --- |
| 1 | `scenario_execution_started` | started | `application.execution.runner` |
| 2 | `payment_workflow_started` | started | `application.enterprise_payment_workflow` |
| 3 | `external_payment_effect_created` | unknown | `external_payment.capture_service` |
| 4 | `unknown_detected` | unknown | `external_payment.integration_channel` |
| 5 | `reliability_case_created` | unknown | `intergrax.runtime.enterprise_reliability.admission` |
| 6 | `reconciliation_executed` | probe_executed | `intergrax.runtime.enterprise_reliability.reconciliation` |
| 7 | `evidence_evaluated` | ready_for_decision | `erl_integration.plugins.payment_evidence_evaluator` |
| 8 | `resolution_decided` | continue | `erl_integration.plugins.payment_resolution_strategy` |
| 9 | `governance_evaluated` | approval_required | `erl_integration.plugins.payment_governance_policy` |
| 10 | `recovery_executed` | continue | `erl_integration.plugins.payment_recovery_strategy` |
| 11 | `scenario_completed` | continue | `application.execution.runner` |

Variant B surfaces `resolution_decided` → `stop` and `recovery_executed` → `terminate`. Variant C surfaces `recovery_executed` → `escalate` and insufficient reconciliation evidence in step detail.

## Relation to audit evidence

Proof packaging should reference the same `correlation_id` and `evidence_ref` values already present on platform `ResolutionDecision`, `GovernanceDecision`, and `RecoveryDecision`. The trace timeline is the operator-facing spine; ERL semantic facts (`UncertaintyAdmissionFact`, `ReconciliationAttemptFact`, etc.) remain the long-term audit contract when platform emission wiring is enabled.

## Implementation map

- Port: `application/tracing/port.py`
- Recorder: `application/tracing/recorder.py` (`RecordingScenarioExecutionTrace` for lab runs)
- Wiring: `application/execution/runner.py`, `application/execution/composition.py`
- Tests: `tests/unit/platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/test_erl_qual_004_execution_tracing.py`

## Public contract boundary

Applications, scenarios, and proof results import Plane B trace types from **`intergrax.contracts.tracing`** only. Runtime Nexus (`intergrax.runtime.nexus.tracing.trace_models`) re-exports the same types for legacy runtime call sites; it is not an application-facing boundary.

### Trace value typing

Public trace fields (`TraceEvent.tags`, `DiagnosticPayload.to_dict()`, `ToolCallTrace.arguments`, `ToolCallTrace.raw_trace`) use **`TraceValue` / `TraceObject`** from the same package — deterministic JSON-safe data validated at contract boundaries (`intergrax.contracts.structured_json_value`). Arbitrary Python objects, bytes, datetimes, and non-finite floats are rejected explicitly. `ToolCallTrace` remains public as the API-facing tool-call artifact (RuntimeAnswer); it is not a scenario-local DTO.

Dependency direction:

```text
scenario / application / proof
  → intergrax.contracts.tracing
  → runtime implementations (emitters, stores, bridges)
```

## Abstraction and adapter boundary

`EnterprisePaymentScenarioExecutor` and the payment / ERL orchestration layers depend only on `ScenarioExecutionTracePort` (`begin_execution`, `emit_lifecycle_step`, `snapshot`). They must not reference `RecordingScenarioExecutionTrace` or other concrete adapters.

The lab composition entry point wires the trace port explicitly:

```text
Caller
  → build_lab_execution_composition(execution_trace=<ScenarioExecutionTracePort | None>)
  → composition selects RecordingScenarioExecutionTrace when execution_trace is omitted
  → EnterprisePaymentScenarioExecutor uses ScenarioExecutionTracePort only
```

Callers may pass any `ScenarioExecutionTracePort` implementation (test double, no-op adapter, alternate recorder). When omitted, composition mints the default in-memory lab recorder. Orchestration layers remain adapter-agnostic; the concrete recorder is a lab adapter, not a business contract.
