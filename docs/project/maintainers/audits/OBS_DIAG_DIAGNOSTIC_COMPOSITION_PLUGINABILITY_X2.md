# OBS-DIAG-X2 — Diagnostic Composition Pluginability

> **Maintainer audit / qualification evidence — not architecture SSOT.**
> **CURRENT AUTHORITY:** [`DIAGNOSTICS.md`](../../architecture/DIAGNOSTICS.md) · [`OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md)
> **Historical predecessor:** [`OBS_DIAG_UNIVERSAL_ENTERPRISE_GAP_BASELINE_X1.md`](OBS_DIAG_UNIVERSAL_ENTERPRISE_GAP_BASELINE_X1.md)
> (X1 recorded host composition replaceability as **PARTIAL** at that SHA — preserved.)

| Field | Value |
| ----- | ----- |
| **Program** | OBS-DIAG-X2 |
| **Branch** | `development` |
| **Verdict** | `PASS — DIAGNOSTIC HOST COMPOSITION IS CONTRACT-DRIVEN AND PLUGINABLE` |

## Architecture

```text
Application / Scenario / Worker
        ↓
Shared Diagnostic Composition
        ↓
DiagnosticCompositionOverrides (typed) + platform defaults
        ↓
resolve_diagnostic_persistence_composition  ← one write/read persistence resolver
        ↓
resolve_diagnostic_composition
        ↓
ONE DiagnosticOrchestrator + ONE ProblemLifecycleEngine
```

Primary module: `intergrax/applications/_shared/diagnostic_composition.py`

Injection surface: `ApplicationCompositionContext.diagnostic_composition_overrides`
(and optional explicit `overrides=` on shared wiring helpers).
`ApplicationBuildContext` remains free of runtime diagnostic providers.

## Classification

| Mechanism | Classification |
| --------- | -------------- |
| DiagnosticOrchestrator | HARD INVARIANT |
| ProblemLifecycleEngine | HARD INVARIANT |
| ProblemGroupingEngine validation / identity | HARD INVARIANT |
| LifecycleAnomalyAnalyzer | NOT PLUGGABLE BY DESIGN |
| DiagnosticAssessmentBuilder | NOT PLUGGABLE BY DESIGN |
| ProblemPersistence | PLUGGABLE PROVIDER |
| ProblemOccurrencePersistence | PLUGGABLE PROVIDER |
| CausalEvidencePersistence | PLUGGABLE PROVIDER |
| ExecutionReconstructionReader | PLUGGABLE PROVIDER |
| ProblemGroupingStrategy | PLUGGABLE STRATEGY |

## Proofs

| Proof | Module |
| ----- | ------ |
| Default + custom persistence / reconstruction / grouping | `tests/unit/applications/_shared/test_obs_diag_x2_diagnostic_composition_pluginability.py` |
| Duplicate strategy ID fail-closed | same |
| STRICT missing durable ≠ InMemory fallback | same |
| Read/write same override instances | same |
| Host-owned close / borrowed not closed | same |
| AST: no vendor imports / getattr / concrete branching / service locator | same |
| HARDEN 1D / 4B / 4D + OBS-DIAG port isolation | regression matrix |

## Explicit non-goals (remain for later Xn)

- Real Mongo/Kafka/Postgres/OTLP provider qualification (X7 / X4 / X9)
- Universal entry-path zero-bypass (X3)
- Second DiagnosticOrchestrator / alternate diagnostic authority — **forbidden**
