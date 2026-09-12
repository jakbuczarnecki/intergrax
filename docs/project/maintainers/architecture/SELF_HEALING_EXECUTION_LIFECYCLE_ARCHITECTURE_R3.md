# Self-healing execution lifecycle (R3)

## Authority

Self-healing **never executes** repairs. It detects, plans, approves, requests execution via External Operation Spine, observes, validates, and learns.

| Layer | Role |
|-------|------|
| `SelfHealingLifecycleEngine` | R3 lifecycle state, correlation, observation → validation → rollback orchestration |
| `SelfHealingWorkflowOrchestrator` | R2 step plan + spine delegation |
| `GovernedSelfHealingOrchestrator` | Governance + `ExternalOperationExecutionGate` |
| Diagnostic / predictive read models | Projections only |

## Lifecycle (R3)

`CREATED` → `APPROVAL_PENDING` → `APPROVED` → `EXECUTION_REQUESTED` → `EXECUTING` → `OBSERVING` → `VALIDATING` → `COMPLETED` | `ROLLBACK_PENDING` → `ROLLED_BACK` | `FAILED`

## Execution correlation

`SelfHealingExecutionContext` is immutable and serializable. `execution_ids` and `operation_attempt_ids` are collected from spine audit/attempt records only — no healing-local execution ID minting.

## Observation

`SelfHealingObservationProvider` supplies `ObservationResult` (metrics, `evidence_refs`, confidence). Validation pipeline requires observation evidence.

## Validation

`SelfHealingValidationPipeline`: registry + `SelfHealingValidator` plugins → `SelfHealingValidationDecision` (no bare `validation=true` without evidence).

## Rollback

`SelfHealingRollbackCoordinator`: policy → rollback plan → spine admission (reuse R2 rollback SPI).

## Strategy selection & learning

`SelfHealingStrategySelector` SPI; default `HighestConfidenceStrategySelector`. `SelfHealingStrategyPerformance` feeds selection; quality store remains single authority for outcome loop integration.

## Read model

`DiagnosticInvestigationView.healing_execution_timeline` — operator timeline (problem → outcome), no authority.
