# Self-Healing Contextual Knowledge Optimization (R5.5)

**Status:** Foundation (R5.5) — richer derived knowledge only; no execution authority.

## What is “context” here?

**Context** describes the conditions under which strategy knowledge was observed: problem type, environment, source, execution conditions, and constraints. It is stored on `StrategyKnowledgeProfile.operating_context` as `StrategyKnowledgeOperatingContext`.

R5.4 keeps `StrategyKnowledgeContext` as the **evolution scope key** (tenant, strategy, fingerprint, refs). R5.5 adds a separate **operating context** model so quality scores are not flattened into a single number without situational metadata.

## Context is not a decision

```text
Contextual Knowledge  →  Better recommendation / comparison input
```

Context must never:

- route execution,
- change selectors or lifecycle,
- disable or auto-switch strategies.

## Extension of R5.4

| R5.4 artifact | R5.5 extension |
|---------------|----------------|
| `StrategyKnowledgeProfile` | Optional `operating_context` |
| `StrategyKnowledgeEvolutionContext` | `resolved_operating_context`, `freshness_policy_id` |
| `StrategyComparisonScope` / `StrategyComparisonSubject` | Optional operating context and freshness scores |
| `StrategyKnowledgeEvolutionService` | Optional `context_providers`, `freshness_policy` |

Persistence remains **only** via `StrategyKnowledgeRepository` — no vendor or database calls in domain contracts.

## Plugin points

```text
StrategyContextProvider
        ├── EvolutionRefsContextProvider (default helper)
        └── future: DiagnosticContextProvider, EnvironmentContextProvider, …

KnowledgeFreshnessPolicy
        ├── NoDecayKnowledgeFreshnessPolicy
        └── TimeWeightedKnowledgeFreshnessPolicy
```

`StrategyComparisonPolicy` implementations may read scope/subject context and freshness; success-rate dominance is unchanged unless rates are equal (freshness tie-break only).

## Related docs

- R5.4: [`SELF_HEALING_STRATEGY_KNOWLEDGE_EVOLUTION_R5.md`](SELF_HEALING_STRATEGY_KNOWLEDGE_EVOLUTION_R5.md)
- Blueprint: [`SELF_HEALING_STRATEGY_KNOWLEDGE_EVOLUTION_ARCHITECTURE_R5.md`](SELF_HEALING_STRATEGY_KNOWLEDGE_EVOLUTION_ARCHITECTURE_R5.md)
