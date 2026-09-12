# Self-healing adaptive intelligence (R4)

## Authority boundaries

Adaptive intelligence **recommends** strategy ordering and confidence only. It does not execute repairs, call External Operation Spine, perform rollback, or mutate lifecycle state.

| Authority | Owner |
|-----------|--------|
| Execution | External Operation Spine + governance gate |
| Diagnostics | Central Diagnostic Engine |
| Lifecycle | `SelfHealingLifecycleEngine` |
| Adaptive ranking | `AdaptiveSelfHealingEngine` (advisory) |

## Lifecycle position

```text
AdaptiveSelfHealingEngine → AdaptiveHealingRecommendation
        ↓
SelfHealingStrategySelector (R3 + R4 adaptive wrapper)
        ↓
SelfHealingLifecycleEngine → Governance → Execution Spine
```

## SPI

- `SelfHealingStrategyRankingProvider` — custom ranking algorithms
- `SelfHealingConfidenceEvaluator` — custom confidence aggregation

Registries support `priority`, `namespace`, `version`, `tenant_scope`, `timeout_seconds`, and `metadata`.

## Plugin model

Contract → SPI → Registry → Provider implementation. Platform defaults: `PlatformHistoricalRankingProvider`, `AdaptiveConfidenceEvaluator`.

## Failure model

Plugin failures are isolated per provider. Outcomes:

- `PLUGIN_UNAVAILABLE` — no usable ranking output
- `DEGRADED_ADAPTIVE_INTELLIGENCE` — partial plugin failure or missing evidence-backed confidence

Platform must not crash; confidence is never fabricated without `evidence_refs`.

## Diagnostic integration

`AdaptiveHealingInsightView` attaches to `DiagnosticInvestigationView.adaptive_healing_insights` via `project_adaptive_healing_insights`. Read-only enrichment only.

## Persistence

`AdaptiveHealingLearningRepository` port; default tests use `InMemoryAdaptiveHealingLearningRepository`. No Problem/Incident/Root Cause stores.

## Limitations

- No cross-tenant learning federation in R4
- No automatic strategy mutation — recommendation only
- Ranking timeout is per-plugin descriptor

## Roadmap

- Persistent learning repository adapters
- Operator-approved promotion of custom ranking plugins
- Deeper context-similarity features on diagnostic read models
