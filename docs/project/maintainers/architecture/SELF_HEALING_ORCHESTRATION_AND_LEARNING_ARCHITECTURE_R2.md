# Self-Healing Orchestration and Learning (R2)

**Task:** `AUTONOMOUS-ENTERPRISE-SELF-HEALING-ORCHESTRATION-R2`

**Status:** Architecture frozen

**Invariant:** `WORKFLOW_IS_NOT_EXECUTION` — workflow orchestrates lifecycle; External Operation spine executes.

---

## Authority model

| Layer | Role |
| ----- | ---- |
| `SelfHealingStrategy` | Decision only (R1) |
| `SelfHealingPlanBuilder` | Declarative plan — no I/O |
| `SelfHealingWorkflowOrchestrator` | Lifecycle, step coordination, audit |
| `GovernedSelfHealingOrchestrator` | Safety → governance → `ExternalOperationExecutionGate` |
| `SelfHealingValidationProvider` | Evidence-based validation |
| `SelfHealingRollbackProvider` | Rollback directives → spine |
| Central Diagnostic Engine | Evidence source — no healing execution |

## Workflow lifecycle

States: `CREATED` → `PLANNED` → (`WAITING_APPROVAL`) → `APPROVED` → `EXECUTING` → `VALIDATING` → `SUCCEEDED` | `ROLLBACK_REQUIRED` → `ROLLING_BACK` → `FAILED` | `ESCALATED`.

Transitions are audited (`SelfHealingWorkflowAuditEntry`), tenant-scoped.

## Plugin boundaries

Registries: plan builder, validation, rollback (strategy registry remains R1). Plugins declare `SelfHealingWorkflowPluginDescriptor` (id, version, namespace, priority, capabilities, tenant_scope, timeout). Failures surface as `PLUGIN_FAILED` — no platform halt, no fake success.

## Governance flow

`HealingPlan` → safety (R1) → `SelfHealingAdmissionGate` → `ExternalOperationAdmission` → execution. Confidence and validation confidence are **not** authorization.

## Execution reuse

Forbidden: workflow-local executor, scheduler, retry engine, event bus, Problem store, diagnostic engine, direct infrastructure calls from plugins.

Required: `ExternalOperationIntent` via `SelfHealingActionProvider`, existing admission gates.

## Learning

`SelfHealingWorkflowOutcome` feeds `SelfHealingWorkflowOutcomeEngine` → `SelfHealingStrategyQualityProfile` (extends R1 store).

## Operator read model

`DiagnosticInvestigationView.healing_workflow_history` — `RelatedSelfHealingWorkflowHistoryEntryView` trail (strategy → plan → approval → execution → validation → rollback → outcome).

## Contracts / runtime layout

- `intergrax/contracts/self_healing/workflow/`
- `intergrax/runtime/self_healing/workflow/`
