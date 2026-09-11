# DIAGNOSTIC-ENGINE-EXTENSION-SPI — Audit R1

**Task:** `DIAGNOSTIC-ENGINE-EXTENSION-SPI-ARCHITECTURE-R1` (Etap 1)

## Existing extension surfaces (reused)

| Surface | Location | Role | R5 disposition |
| ------- | -------- | ---- | -------------- |
| `DiagnosticEvidenceContributor` | `intergrax/contracts/diagnostic_evidence_contributor.py` | Frozen SPI name (SINGLE_AUTHORITY §13) | **Extended** with `collect()` + `priority` |
| `DecisionEvidenceContributor` | same | Decision facts marker | **Unchanged** |
| `DiagnosticScopeDiscoveryProviderRegistry` | `diagnostic_scope_discovery_provider.py` | Deterministic provider order, duplicate rejection | **Pattern reused** for `DiagnosticExtensionRegistry` |
| `FunctionalDiagnosticAnalyzer` | `functional_diagnostic_analyzer.py` | One generic functional analyzer | **Not duplicated** — domain plugins use `DiagnosticAnalyzer` |
| `PlatformFunctionalEvidence` | `functional_evidence.py` | Pipeline functional facts | **Separate store** — extension evidence uses `DiagnosticExtensionEvidence` |
| `ProblemGroupingStrategyRegistry` | `problem_grouping.py` | Explicit strategy registration | Reference only |
| `DecisionContextProvider` | `decision_context_provider.py` | Optional read enrichment | **Parallel** optional `DiagnosticExtensionService` on read path |

## Gaps found (addressed in R5)

- No `collect()` on evidence contributor — **added**
- No `DiagnosticAnalyzer` / `DiagnosticFindingCandidate` contract — **added**
- No `DiagnosticTaxonomyContributor` — **added**
- No central `DiagnosticExtensionRegistry` — **added**
- No bounded `DiagnosticExecutionContext` — **added** (`contracts/diagnostic_extension_evidence.py`)
- Read model lacked extension findings — **`DiagnosticExtensionOccurrenceEnrichment`**

## Explicit non-goals (unchanged)

- Application-local diagnostic engines
- Plugin writes to `ProblemPersistence`
- Import-order / filesystem discovery for plugins

## Verdict

Compatible extension points exist; R5 formalizes SPI + registry + read enrichment without second diagnostic authority.
