# Central Diagnostic Engine — Operator Experience (R7 / R1)

**Task:** `DIAGNOSTIC-ENGINE-OPERATOR-EXPERIENCE-R1`

## Invariant

```text
Evidence Producers → Central Diagnostic Engine → ProblemLifecycleEngine
        → DiagnosticReadService → Operator Experience (projection)
```

Operator Experience **must not** implement diagnostics, Problem authority, lineage stores, or causal inference.

## Canonical investigation read model

`DiagnosticReadService.get_investigation` returns `DiagnosticInvestigationResult` with:

| Projection | Role |
| ---------- | ---- |
| `DiagnosticInvestigationView` | Single bounded operator bundle |
| `FailureInvestigationSummary` | Boundary vs impact vs **unknown** root cause |
| `DiagnosticTimeline` | Chronological evidence only |
| `DiagnosticEvidenceExplanation` | PROVEN / SUPPORTED / UNKNOWN |
| `DiagnosticImpactGraph` | Lineage-derived health — no graph store |
| `DiagnosticRecommendation` | Text recommendations only |
| `DiagnosticStructuredInvestigationPayload` | Future AI assistant input — LLM not authority |

## Contracts (`intergrax/contracts/diagnostic_investigation.py`)

Shared enums: evidence confidence, root cause status (default `UNKNOWN`), severity, impact node health, recommendation kinds.

## Timeline rules

- Merge runtime events, failure evidence refs, decision correlation labels, extension evidence, problem occurrence time.
- Sort by timestamp / stable secondary key.
- **Forbidden:** infer causality from order.

## Remediation

`DiagnosticRecommendation` is advisory. No automatic execution hooks in this slice.

## AI assistant preparation

```text
Diagnostic Engine → Structured Investigation View → AI Assistant → explanation
```

Assistant consumes `DiagnosticStructuredInvestigationPayload`; it does not mint Problems or boundaries.

## Quality gates

| Gate | Expectation |
| ---- | ----------- |
| `ONE_DIAGNOSTIC_ENGINE` | Single runtime diagnostics authority |
| `ONE_PROBLEM_AUTHORITY` | Problem store unchanged |
| `READ_MODEL_ONLY` | Projections not persisted as truth |
| `NO_NEW_LINEAGE` | Impact graph from existing lineage view |
| `NO_NEW_CAUSALITY` | Decisions + timeline non-causal |
| `NO_HEURISTIC_ROOT_CAUSE` | `DiagnosticRootCauseStatus.UNKNOWN` default |
| `TENANT_ISOLATION` | `get_investigation` tenant scoped |
| `BOUNDED_READS` | Reuses occurrence limits |
| `PLUGIN_SAFE` | Extension enrichment unchanged |

## Implementation map

| Module | Responsibility |
| ------ | -------------- |
| `diagnostic_operator_investigation_read_models.py` | Frozen DTOs |
| `diagnostic_operator_investigation_projection.py` | Deterministic projectors |
| `diagnostic_read_service.py` | `get_investigation` orchestration |
