# Self-Healing Strategy Performance Memory (R5.1)

**Status:** Foundation (R5.1) — append-only factual history.

## Purpose

`StrategyPerformanceMemory` records **what strategy ran**, **in which diagnostic context**, and **what happened** after a self-healing workflow completes. It answers historical questions only.

## Out of scope (later R5 stages)

- Strategy ranking, scoring, or quality models
- Recommendations or automatic strategy selection
- Machine learning pipelines
- Changes to execution authority, lifecycle control, or workflow orchestration

## Contracts

| Artifact | Role |
|----------|------|
| `SelfHealingStrategyPerformanceExperience` | Immutable per-run fact record |
| `StrategyPerformanceMemoryRepository` | Storage port (DB / events / analytics later) |
| `StrategyPerformanceMemoryQuery` | Tenant-scoped read filters |
| `StrategyPerformanceMemoryRecorder` | Observes `SelfHealingWorkflowOutcome` + `SelfHealingExecutionContext` |

R3 `SelfHealingStrategyPerformance` remains the **aggregated** selection feedback profile. R5.1 memory is **episodic** and does not replace R3 aggregates.

## Integration pattern

```text
Self-Healing Lifecycle → workflow outcome + execution context
        ↓ (observe only)
StrategyPerformanceMemoryRecorder → Repository
```

Callers compose the recorder **after** lifecycle completion; the lifecycle engine is unchanged.

## Future R5 extension

Later steps may project aggregates, similarity features, or adaptive inputs from stored experiences — always via new consumers of `StrategyPerformanceMemoryRepository`, not by extending experience records with decision fields.
