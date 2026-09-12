# Self-Healing Strategy Recommendation Engine (R5.3)

**Status:** Foundation (R5.3) — advisory recommendations only.

## Purpose

The recommendation layer answers: **which strategy looks best for this situation, based on historical quality?** It consumes R5.2 `StrategyQualityAssessment` values and returns a `StrategyRecommendation` (ranked candidates, rationale, confidence label).

## Authority boundary

The recommendation engine is a **counselor**, not an executor.

| Allowed | Forbidden |
|---------|-----------|
| Read performance memory via R5.2 | Invoke strategy executors |
| Rank strategies from explicit candidate lists | Change lifecycle or workflow |
| Emit immutable recommendation records | Auto-select or run strategies |

Decision authority remains with upstream consumers (future R5.4 / R6 controls).

## Architecture

```text
StrategyPerformanceMemoryRepository
        ↓
StrategyQualityEvaluationService  (R5.2)
        ↓
StrategyRecommendationService
        ↓
StrategyRecommendationEngine  →  QualityBased / future plugins
        ↓
StrategyRecommendation
```

No persistence is added in R5.3; recommendations are computed on demand.

## Contracts

| Artifact | Role |
|----------|------|
| `StrategyRecommendationRequest` | Tenant, diagnostic scope, candidate strategy ids |
| `StrategyRecommendationContext` | Assessments bound to candidates |
| `StrategyRecommendationEngine` | Pluggable ranking SPI |
| `StrategyRecommendation` | Advisory output |

Compose `StrategyRecommendationService` with a configured R5.2 evaluation service and engine implementation. Do not wire recommendations into the existing strategy selector or orchestrator in R5.3.
