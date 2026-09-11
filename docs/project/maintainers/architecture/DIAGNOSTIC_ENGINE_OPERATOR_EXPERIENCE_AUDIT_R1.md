# Central Diagnostic Engine — Operator Experience audit (R7 / R1)

**Task:** `DIAGNOSTIC-ENGINE-OPERATOR-EXPERIENCE-R1` · **Etap 1**

## Scope

Audit of existing operator read surfaces before R7 composition. No new diagnostic authority.

## DiagnosticReadService

| Capability | Status |
| ---------- | ------ |
| Bounded `list_problems` / `get_problem` | **Present** |
| Tenant-scoped reads | **Present** |
| Occurrence reconstruction (DIAG-2→4) | **Present** |
| Unified investigation entry | **GAP → `get_investigation` (R7)** |

## DiagnosticProblemView (list + detail)

| Field / behavior | Status |
| ---------------- | ------ |
| `DiagnosticProblemSummary` | **Present** |
| `DiagnosticProblemDetail` + bounded occurrences | **Present** |
| Operator “what happened” narrative | **GAP → `DiagnosticInvestigationView`** |

## DiagnosticOccurrenceView

| Field | Status |
| ----- | ------ |
| `read_status` / `unavailable_reason` | **Present** |
| `assessment` (`DiagnosticAssessment`) | **Present** |
| Grouping provenance on occurrence | **Present** |

## ExecutionLineageView

| Field | Status |
| ----- | ------ |
| `DiagnosticExecutionLineageView` on occurrence | **Present** (DG-001 projection) |
| Impact health labels | **GAP → `DiagnosticImpactGraph` (R7)** |

## DecisionContextView

| Field | Status |
| ----- | ------ |
| Optional provider on read service | **Present** (R4) |
| Causal disclaimers | **Present** |
| Timeline placement | **GAP → `DiagnosticTimeline` (R7)** |

## Extension enrichment

| Field | Status |
| ----- | ------ |
| `DiagnosticExtensionOccurrenceEnrichment` | **Present** (R5) |
| Evidence explanation confidence | **GAP → `DiagnosticEvidenceExplanation` (R7)** |

## FailureBoundaryAnalysis

| Field | Status |
| ----- | ------ |
| On `DiagnosticAssessment.failure_boundary_analysis` | **Present** (R3) |
| Operator failure vs cause separation | **GAP → `FailureInvestigationSummary` (R7)** |

## Conclusion

Reuse existing read DTOs; add **composition-only** projections (`DiagnosticInvestigationView`, timeline, impact graph, recommendations, assistant payload) on `DiagnosticReadService.get_investigation`.

## Quality gate pre-check

| Gate | Pre-R7 |
| ---- | ------ |
| `ONE_DIAGNOSTIC_ENGINE` | PASS |
| `ONE_PROBLEM_AUTHORITY` | PASS |
| `READ_MODEL_ONLY` | PASS (assessment already derived) |
| Operator six-question bundle | **GAP** |
