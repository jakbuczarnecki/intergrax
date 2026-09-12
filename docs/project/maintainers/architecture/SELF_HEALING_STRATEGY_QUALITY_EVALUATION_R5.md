# Self-Healing Strategy Quality Evaluation (R5.2)

**Status:** Foundation (R5.2) — read-only historical quality analysis.

## Purpose

`StrategyQualityEvaluation` answers: **how well did a strategy perform in the past?** It consumes episodic records from R5.1 `StrategyPerformanceMemoryRepository`, aggregates factual outcomes, and produces a `StrategyQualityAssessment` (counts, recovery-time statistics, evaluator-specific `quality_score`).

## Out of scope (later R5 stages)

- Strategy recommendation or automatic selection (R5.3)
- Continuous learning or adaptive optimization (R5.4)
- Changes to lifecycle, orchestrator, execution engine, or strategy selector
- Machine-learning scoring pipelines

## Separation from Recommendation Engine

| Layer | Question |
|-------|----------|
| R5.2 Quality Evaluation | How good was strategy X historically? |
| R5.3 Recommendation | Which strategy should run next? |

Quality evaluation **never** returns priorities, ranked lists, or execution directives.

## Architecture

```text
StrategyPerformanceMemoryRepository
        ↓ (query)
StrategyQualityEvaluationService
        ↓ (delegate)
StrategyQualityEvaluator  →  Basic / Weighted / future plugins
        ↓
StrategyQualityAssessment
```

Persistence stays behind `StrategyPerformanceMemoryRepository`; evaluators remain vendor-agnostic and operate on in-memory experience tuples.

## Contracts

| Artifact | Role |
|----------|------|
| `StrategyQualityEvaluationCriteria` | Tenant + strategy read scope |
| `StrategyQualityAssessment` | Immutable analysis result |
| `StrategyQualityEvaluator` | Pluggable scoring SPI |
| `summarize_strategy_performance_experiences` | Pure history aggregation |

## Integration

Callers compose `StrategyQualityEvaluationService` with a configured repository adapter and evaluator implementation. No hooks are added to workflow or lifecycle code in R5.2.
