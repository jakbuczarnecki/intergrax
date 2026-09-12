# Self-Healing Strategy Knowledge Evolution (R5.4)

**Status:** Foundation (R5.4) — derived knowledge only; no execution authority.

## Purpose

`StrategyKnowledgeEvolution` projects **versioned, derived knowledge** from R5.1 performance experiences (and optional R5.2 quality snapshots). It answers: *what durable knowledge should we retain about strategy effectiveness in a context?*

**Knowledge ≠ decision.** Profiles describe historical effectiveness; they never command execution.

## Out of scope

- ML, reinforcement learning, automatic strategy selection
- Lifecycle, orchestrator, selector, or execution engine changes
- Vendor-specific persistence in domain contracts
- Critical-path workflow wiring (event processor is provided but not integrated)

## Contracts

| Artifact | Role |
|----------|------|
| `StrategyKnowledgeProfile` | Immutable derived knowledge row (versioned per tenant/strategy/context) |
| `StrategyKnowledgeRevision` | Append-only audit bundle |
| `StrategyKnowledgeRepository` | Storage port |
| `StrategyLearningEngine` | Plugin SPI — updates knowledge from experiences |
| `StrategyMetricProvider` | Plugin SPI — measurement separate from learning |
| `StrategyComparisonPolicy` | Plugin SPI — strategy contrast without selection authority |
| `StrategyKnowledgeEvolutionService` | Composition root — read R5.1, evolve, persist |
| `SelfHealingWorkflowCompleted` / `KnowledgeEvolutionProcessor` | Async hook shapes (not wired to workflow) |

## Plugin points

```text
StrategyKnowledgeEvolutionService
        ├── StrategyPerformanceMemoryRepository (R5.1)
        ├── StrategyKnowledgeRepository
        ├── StrategyLearningEngine  → BasicStrategyLearningEngine (default)
        ├── StrategyMetricProvider  → SuccessRateMetricProvider (default)
        └── StrategyComparisonPolicy (optional)
```

## Persistence

Domain uses `StrategyKnowledgeRepository` only. `InMemoryStrategyKnowledgeRepository` is a test double under `intergrax/runtime/self_healing/knowledge_evolution/`.

## Integration pattern

```text
R5.1 experiences (source of truth)
        ↓ query
StrategyKnowledgeEvolutionService.evolve(context)
        ↓ append_revision
StrategyKnowledgeRepository
```

Optional async path:

```text
SelfHealingWorkflowCompleted → WorkflowCompletedKnowledgeEvolutionProcessor
```

Idempotent: duplicate trigger + `trigger_refs` → no new revision.

## Related docs

- Blueprint: [`SELF_HEALING_STRATEGY_KNOWLEDGE_EVOLUTION_ARCHITECTURE_R5.md`](SELF_HEALING_STRATEGY_KNOWLEDGE_EVOLUTION_ARCHITECTURE_R5.md)
- R5.1–R5.3 foundations in sibling `SELF_HEALING_STRATEGY_*_R5.md` files
