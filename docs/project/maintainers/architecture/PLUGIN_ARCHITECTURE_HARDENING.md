# Plugin architecture hardening (HARDENING-5)

## Target model

```text
Stable contract (Protocol / immutable models)
        |
        v
Plugin interface
        |
   +----+----+
   v         v
Impl A    Impl B
```

Core orchestration (decision, workflow, lifecycle, adaptive recommendation, R5/R6 services) must depend only on **contracts**, not on in-memory or platform-default implementations.

## Plugin points audited

| Domain | Contract | Typical implementations |
|--------|----------|-------------------------|
| Self-healing strategies | `SelfHealingStrategy`, `SelfHealingStrategyRegistry` | Platform defaults, test fakes |
| Workflow plugins | `SelfHealingPlanBuilderRegistry`, `SelfHealingValidationRegistry`, `SelfHealingRollbackRegistry` | `InMemory*` registries, platform bootstrap |
| Adaptive intelligence | `SelfHealingStrategyRankingRegistry`, `SelfHealingConfidenceEvaluatorRegistry` | Platform ranking / confidence plugins |
| R5 quality | `StrategyQualityEvaluator`, `StrategyQualityAssessor` | Basic / weighted evaluators, evaluation service |
| R5 learning / recommendation | `StrategyLearningEngine`, `StrategyRecommendationEngine` | Basic learning, quality-based recommendation |
| R5 governance hook | `StrategyKnowledgeEvolutionGovernanceRecorder` | Governance service |
| R6 autonomy | `AutonomyPolicy`, `AutonomyRiskEvaluator`, `HumanApprovalRequirementResolver`, `AutonomyExecutionGuardRule`, `AutonomySafetyCheck` | Default platform plugins via DI |
| Performance / knowledge persistence | `StrategyPerformanceMemoryRepository`, `StrategyKnowledgeRepository`, audit ports | In-memory and test doubles |

## Fixes in HARDENING-5

| Issue | Resolution |
|-------|------------|
| `SelfHealingDecisionEngine` typed on `InMemorySelfHealingStrategyRegistry` | Constructor uses `SelfHealingStrategyRegistry` port. |
| Workflow / lifecycle orchestration typed on in-memory workflow registries | Fields use `SelfHealingPlanBuilderRegistry`, `SelfHealingValidationRegistry`, `SelfHealingRollbackRegistry`. |
| `AdaptiveSelfHealingEngine` defaulted in-memory registries and concrete confidence evaluator | Requires registry ports plus explicit `fallback_confidence_evaluator` (`SelfHealingConfidenceEvaluator`). |
| R5 evolution / recommendation services imported concrete runtime facades | Optional quality path uses `StrategyQualityAssessor`; governance hook uses `StrategyKnowledgeEvolutionGovernanceRecorder`. |
| New assessor / governance recorder ports | `contracts/self_healing/quality_evaluation/assessor.py`, `contracts/.../governance/recorder.py`. |

Regression gate: `tests/unit/runtime/architecture/test_hardening_5_plugin_architecture_gate.py`.

## Composition roots (allowed concrete imports)

Platform wiring remains in **bootstrap** modules only, for example:

- `runtime/self_healing/workflow/bootstrap.py` — registers platform plan / validation / rollback plugins.
- `runtime/self_healing/adaptive/bootstrap.py` — registers platform ranking and confidence plugins.
- `runtime/self_healing/*/registries.py`, `strategy_registry.py`, `in_memory_*.py` — adapter implementations.

`GovernedSelfHealingOrchestrator` and external-operation spine are unchanged; execution authority is not modified by this pass.

## Legacy / deferred

| Item | Notes |
|------|--------|
| `SelfHealingPlanBuilderRegistry.register` vs in-memory `strategy_id` keyword | In-memory adapter extends registration; orchestration uses `resolve*` only. Unify register signature only if a shared registration contract is needed. |
| `execution/context.py` `to_serializable_dict() -> dict[str, Any]` | Serialization seam, not a plugin contract; left unchanged. |
| Global platform plugin discovery (`intergrax.core.plugins`) | Separate track; DS-PLUGIN gates remain authoritative for Decision plugins. |

## Extension checklist

1. Add or extend a **Protocol** under `intergrax/contracts/self_healing/`.
2. Register implementation via bootstrap, DI, or test fixture — not via import inside orchestration core.
3. Add a gate test row if a new core module must stay implementation-agnostic.
