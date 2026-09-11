# Central Diagnostic Engine — Extension SPI (R5 / R1 delivery)

**Task:** `DIAGNOSTIC-ENGINE-EXTENSION-SPI-ARCHITECTURE-R1`

## Invariant

```text
DIAGNOSTIC_AUTHORITY_COUNT = 1
Extensions → typed evidence + bounded finding candidates only
ProblemLifecycleEngine → sole Problem authority
```

## Target flow

```text
Application / Agent / Domain Module
        |
        v
Typed Diagnostic Extension SPI (contributors + analyzers + taxonomy)
        |
        v
DiagnosticExtensionRegistry (deterministic order, validation)
        |
        v
DiagnosticExtensionService (collect + analyze, fault isolation)
        |
        v
Central Diagnostic Engine + DiagnosticReadService (projection)
        |
        v
ProblemLifecycleEngine (unchanged)
```

## Contracts (frozen names)

| Port | Responsibility | Forbidden |
| ---- | -------------- | --------- |
| `DiagnosticEvidenceContributor` | `collect(context)` → immutable `DiagnosticExtensionEvidence` | Problem mint, root cause, lifecycle, private problem store |
| `DiagnosticAnalyzer` | `analyze(evidence)` → `DiagnosticFindingCandidate` | `ProblemPersistence`, global certainty, mint `ExecutionId` |
| `DiagnosticTaxonomyContributor` | Namespaced kind catalog (`company.sap.*`) | Global mega-enums (`GLOBAL_TIMEOUT`) |
| `DiagnosticExecutionContext` | `tenant_id`, run/attempt/execution, `time_budget_ms` | Global state, plugin `ContextVar`, singleton cache |

Ordering key: **`priority` → `namespace` → stable id** (never import/filesystem order).

Plugin failure: **`PLUGIN_UNAVAILABLE`**, read status **`DEGRADED`**, central diagnostics continue.

## Registry

`DiagnosticExtensionRegistry` — discovery at bootstrap, duplicate stable id → `DiagnosticExtensionConfigurationError`.

## Read model

`DiagnosticExtensionFindingView` + `DiagnosticExtensionOccurrenceEnrichment` on `DiagnosticProblemOccurrenceView.extension_enrichment`.

Problem remains canonical; extension output is **additional evidence / SUPPORTED interpretation**, not a second Problem.

## Persistence

`DiagnosticExtensionEvidenceStore` (append-only evidence port; `InMemoryDiagnosticExtensionEvidenceStore` for qualification).

## Quality gates

| Gate | Expectation |
| ---- | ----------- |
| `ONE_DIAGNOSTIC_ENGINE` | Single `intergrax.runtime.diagnostics` orchestration |
| `ONE_PROBLEM_AUTHORITY` | Only `ProblemLifecycleEngine` writes Problems |
| `NO_LOCAL_DIAGNOSTICS` | No `ApplicationDiagnosticEngine` symbols |
| `NO_PLUGIN_PROBLEM_WRITE` | SPI ports exclude persistence |
| `NO_LLM_AUTHORITY` | Unchanged |
| `NO_DUPLICATE_LINEAGE` | Unchanged |
| `TENANT_ISOLATION` | Evidence + findings scoped by `tenant_id` |
| `BOUNDED_EXECUTION` | `time_budget_ms` on extension context |

Audit: [`DIAGNOSTIC_ENGINE_EXTENSION_SPI_AUDIT_R1.md`](../qualification/DIAGNOSTIC_ENGINE_EXTENSION_SPI_AUDIT_R1.md)

Qualification: [`DIAGNOSTIC_ENGINE_EXTENSION_SPI_QUALIFICATION_R1.md`](../qualification/DIAGNOSTIC_ENGINE_EXTENSION_SPI_QUALIFICATION_R1.md)
