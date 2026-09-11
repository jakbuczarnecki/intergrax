# DIAGNOSTIC-ENGINE-DECISION-EXECUTION — Audit R1

**Task:** `DIAGNOSTIC-ENGINE-DECISION-EXECUTION-DIAGNOSTIC-LINEAGE-R1` (Etap 1)

## Decision identity (as-built)

| Artifact | Location | Notes |
| -------- | -------- | ----- |
| `DecisionId` | `intergrax/contracts/decision_identity.py` | Canonical `decision_` + 32 hex |
| `DecisionVersion` | same | Positive int ≥ 1; no separate `DecisionAttemptId` type |
| `DecisionIdentity` | same | `decision_id`, `version`, `scope`, `tenant_id`, `execution` |
| `DecisionExecutionLineage` | same | `task_id`, `run_id`, `attempt_id`, optional `execution_id` |
| Decision lifecycle | `intergrax/contracts/decision_lifecycle.py` | Stages + transitions |
| Lifecycle observability | `intergrax/runtime/decision_lifecycle_observability.py` | RuntimeEvent payload with decision + execution ids |
| Lifecycle projection | `intergrax/runtime/diagnostics/decision_lifecycle_projection.py` | Observational snapshot only |

**`decision_attempt_id` (R4 contract):** maps to canonical `AttemptId` on `DecisionExecutionLineage` — execution attempt binding, not a new identity authority.

## Execution correlation (as-built before R4 slice)

| Link | Status |
| ---- | ------ |
| `DecisionId` → `DecisionExecutionLineage` on `DecisionIdentity` | **Present** |
| Durable immutable `DecisionExecutionCorrelationRecord` store | **GAP** → R4 contract + in-memory port |
| `DiagnosticReadService` decision enrichment | **GAP** → `DecisionContextView` + optional provider |
| Decision as diagnostic root cause | **Absent** (required) |

No workaround was added by copying execution lineage into Decision System or minting cross-system identity.

## Central diagnostics (unchanged authority)

- `ProblemLifecycleEngine` — sole Problem authority
- `DiagnosticReadService` — sole operator read composition
- `DiagnosticAssessmentBuilder` — execution/lifecycle evidence only; no decision-cause findings

## GAP closure (this task)

- `intergrax/contracts/decision_execution_correlation.py`
- `intergrax/runtime/diagnostics/decision_context_*`
- Qualification: `tests/unit/runtime/diagnostics/test_decision_execution_lineage_r4_qualification.py`
