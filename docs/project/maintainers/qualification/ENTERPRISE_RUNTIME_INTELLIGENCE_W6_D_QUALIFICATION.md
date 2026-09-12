# Enterprise Runtime Intelligence — W6-D qualification

**Wave:** W6-D — analyzer orchestration foundation  
**Status:** Implementation qualification (multi-analyzer coordination)  
**Depends on:** W6-B contract freeze, W6-C context builder + reference analyzer

---

## Scope delivered

| Deliverable | Location | Role |
|-------------|----------|------|
| Analyzer orchestrator | `intergrax/runtime/runtime_intelligence/analyzer_orchestrator.py` | Explicit ordering, isolated invocation, outcome aggregation |
| Orchestrated lifecycle | `intergrax/runtime/runtime_intelligence/analysis_request.py` | `run_runtime_intelligence_orchestrated_analysis` |
| Conformance tests | `tests/unit/runtime/runtime_intelligence/test_w6_d_runtime_intelligence_orchestration.py` | Ordering, isolation, empty set, immutability |

**Out of scope (later waves):** `RuntimeIntelligencePort` facade, policy correlation, `AdaptivePolicySignalPort`, ML/external adapters.

---

## Ownership

| Concern | Owner |
|---------|--------|
| Fact collection | Caller assembles `RuntimeIntelligenceFacts` (unchanged) |
| Context projection | `RuntimeIntelligenceContextBuilder` (unchanged) |
| Analyzer logic / evidence / recommendations | Each `RuntimeIntelligenceAnalyzerPort` plugin |
| Ordering, invocation lifecycle, failure containment, aggregation | `RuntimeIntelligenceAnalyzerOrchestrator` |
| Execution / recovery / cancel | Unchanged — intelligence remains advisory |

---

## Lifecycle

```text
create RuntimeIntelligenceOrchestratedAnalysisRequest (facts + analyzers tuple + builder)
        ↓
RuntimeIntelligenceContextBuilder.build(facts)
        ↓
RuntimeIntelligenceAnalyzerOrchestrator.orchestrate(context, analyzers)
        ↓
per analyzer: run_runtime_intelligence_analyzer_isolated (W6-B)
        ↓
RuntimeIntelligenceOrchestratedAnalysisResponse
        ↓
caller releases request-scoped resources
```

**Forbidden in W6-D:** singleton orchestrator, registry, scheduler, daemon, global mutable orchestration state.

---

## Plugin extension model

```text
RuntimeIntelligenceAnalyzerPort
        │
        ├── DeterministicRuntimeIntelligenceAnalyzer (W6-C)
        ├── Future ML analyzer
        └── Future external analyzer
```

Caller supplies `tuple[RuntimeIntelligenceAnalyzerPort, ...]` in explicit order. Orchestrator does not branch on analyzer implementation type.

---

## Failure model

| Event | Behavior |
|-------|----------|
| Invalid facts at build | `InvalidIntelligenceContextError` at `build()` — no orchestration |
| Analyzer failure | `ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE` for that analyzer only |
| Invalid context at analyze boundary | `ANALYZER_OUTCOME_INVALID_CONTEXT` for that analyzer |
| Healthy analyzers | `ANALYZER_OUTCOME_OK` with full `RuntimeIntelligenceResult` |

No swallowed exceptions; no hidden fallback; no partial invalid results. Aggregated response always lists one outcome per requested analyzer (empty tuple → zero outcomes).

---

## Determinism

Same immutable context + same analyzer tuple (ids + versions) + same plugin behavior → identical aggregated outcomes and ordering. Ordering follows the caller-supplied tuple only (no discovery or map iteration).

---

## Quality gates (W6-D)

| Gate | Requirement |
|------|-------------|
| Unit tests | W6-D orchestration tests + W6-C + W6-B contract tests |
| Ruff / Pyright | Clean on touched modules |
| No god components | No `*Manager`, registry, or global mutable intelligence state |

---

## Verdict

| Gate | W6-D |
|------|------|
| Multi-analyzer orchestration | **PASS** |
| Deterministic ordering | **PASS** |
| Failure isolation | **PASS** (W6-B isolated runner per plugin) |
| Immutable context preserved | **PASS** |
| Docs | **PASS** (hub + this qualification) |
