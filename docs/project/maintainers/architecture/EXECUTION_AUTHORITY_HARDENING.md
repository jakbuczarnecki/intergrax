# Execution authority hardening (HARDENING-6)

## Authority model

```text
Diagnostic / Prediction / Prevention / Self-Healing decision / Knowledge / Autonomy control
        |
        v
Decision authority (governance, admission, policy)
        |
        v
Execution authorization (AutonomyExecutionGuard, external-operation admission)
        |
        v
Execution boundary (ExecutionAdmissionHook, identity binding)
        |
        v
Execution spine (external-operation gate, unified task runner — outside this audit scope)
        |
        v
Real executor / provider
```

Decision and intelligence layers produce **diagnosis, prediction, recommendations, plans, and authorization** only.
They must not call graph executors, task runners, or provider executors directly.

Regression gate: `tests/unit/runtime/architecture/test_hardening_6_execution_authority_gate.py`.

Related gates (unchanged by this pass): NPSC-4.1 execution boundary intake, EE-A2 identity authority, R6.3 guard/boundary integration tests, R6.4 qualification (`direct_executor_access`).

## Forbidden flows

| Flow | Why forbidden |
|------|----------------|
| Diagnostic orchestrator → `GraphExecutor` / task runner | Diagnostics are read-only reconstruction and evidence |
| Predictive / preventive analyzers → remediation execution | Analysis proposes; admission decides |
| Strategy / adaptive / R5 knowledge services → `attempt_execution` | R1/R5 remain pure decision or knowledge |
| Autonomy control engine / guard → action provider or UAEP | Guard is authorization, not executor |
| Decision layer → `ExternalOperationExecutionGate` | Only composition roots (self-healing / preventive action orchestrators) |
| Evidence / checkpoint persistence → side-effect execution | Persistence records authority outcomes; it does not perform work |
| `direct_executor_access_enabled=True` in autonomy qualification | Explicit bypass flag fails R6.4 qualification |

Dynamic bypass (`getattr` / `setattr` on execution surfaces) and `dict[str, Any]` execution contracts remain out of scope for new code; existing serialization seams are not expanded here.

## Approved flows

| Domain | Role | Execution path |
|--------|------|----------------|
| Diagnostic Engine | Orchestration, grouping, evidence projection | No runtime execution imports except identity correlation (`ExecutionIdentityBinding`) and decision finalization persistence |
| Predictive Intelligence | Forecasting, governance, outcome resolution | Contract + runtime prediction tree — no execution spine imports |
| Preventive Intelligence | Analyzers, recommendations, admission contracts | `PreventiveActionProposal` never exposes `execute()`; action orchestrator uses external-operation gate |
| Self-Healing R1 strategy | `SelfHealingStrategy.evaluate` | Pure; `GovernedSelfHealingOrchestrator` is the sole R1 execution composition root |
| Self-Healing R2 workflow | `SelfHealingWorkflowOrchestrator` | Delegates steps via `GovernedSelfHealingOrchestrator.attempt_execution` only |
| Self-Healing R3 lifecycle | State / validation pipeline | No direct gate imports in lifecycle engine |
| R5 Knowledge Evolution | Learning, recommendation, governance audit | Knowledge and recommendation ports only |
| R6 Autonomy | `AutonomyControlEngine`, `AutonomyExecutionGuard`, `DefaultAutonomyExecutionBoundary` | Guard authorizes; boundary exposes `ExecutionAdmissionHook` to spine — no executor imports in control modules |
| Identity / context | `RuntimeExecutionContext`, active execution authority | Minting and binding remain on frozen execution plane (EE-A2); diagnostics consume bindings read-only |

## Legacy and exceptions

| Item | Notes |
|------|--------|
| `intergrax/runtime/prevention/actions/orchestrator.py` | Preventive remediation composition root — mirrors self-healing external-operation spine |
| `intergrax/runtime/self_healing/action_providers/**` | Provider plugins translate intents; invoked only from `GovernedSelfHealingOrchestrator` |
| `terminal_execution_diagnostic_bridge.py` | ONE-SPINE-3 identity-aligned diagnostic evidence — not an execution bypass |
| `decision_lifecycle_projection.py` | Reads decision finalization persistence for operator projections |
| Autonomy qualification `direct_executor_access_enabled` | Diagnostic flag for hosts that declare a bypass — fails qualification when true |
| Nexus / task / execution runtime internals | Governed by NPSC-4.1 and EE-A2; not modified in HARDENING-6 |

## HARDENING-6 audit outcome

No production bypass was found in decision or intelligence layers. This pass adds the architecture gate and documents the authority model; execution contracts and spine behaviour are unchanged.
