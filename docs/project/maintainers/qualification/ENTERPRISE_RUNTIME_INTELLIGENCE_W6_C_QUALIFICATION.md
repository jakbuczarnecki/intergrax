# Enterprise Runtime Intelligence — W6-C qualification

**Wave:** W6-C — local deterministic analyzer + context builder  
**Status:** Implementation qualification (builder + reference analyzer)  
**Depends on:** W6-B contract freeze (`intergrax/contracts/runtime_intelligence/`)

---

## Scope delivered

| Deliverable | Location | Role |
|-------------|----------|------|
| Read-only fact snapshot | `intergrax/runtime/runtime_intelligence/runtime_facts.py` | Request-scoped inputs from integration layer |
| Context builder | `intergrax/runtime/runtime_intelligence/context_builder.py` | `RuntimeIntelligenceFacts` → immutable `RuntimeIntelligenceContext` |
| Deterministic analyzer | `intergrax/runtime/runtime_intelligence/deterministic_analyzer.py` | Reference `RuntimeIntelligenceAnalyzerPort` (no LLM, no side effects) |
| Request lifecycle | `intergrax/runtime/runtime_intelligence/analysis_request.py` | build → analyze → response (no background worker) |
| Conformance tests | `tests/unit/runtime/runtime_intelligence/test_w6_c_runtime_intelligence.py` | Contract, ownership, isolation, plugin swap, determinism |

**Out of scope (later waves):** `RuntimeIntelligencePort` facade, analysis engine orchestration, ML/external adapters, `AdaptivePolicySignalPort`.

---

## Ownership

| Concern | Owner |
|---------|--------|
| Fact collection | Existing W1–W5 planes (events, checkpoint, terminal, recovery audit) — caller assembles `RuntimeIntelligenceFacts` |
| Context projection | `RuntimeIntelligenceContextBuilder` (integration layer) |
| Analysis | `RuntimeIntelligenceAnalyzerPort` plugins (`DeterministicRuntimeIntelligenceAnalyzer` in W6-C) |
| Execution / recovery / cancel | Unchanged — intelligence never mutates runtime |
| Failure containment | W6-B `run_runtime_intelligence_analyzer_isolated` + typed `InvalidIntelligenceContextError` at build boundary |

---

## Lifecycle

```text
create RuntimeIntelligenceAnalysisRequest (facts + analyzer + builder)
        ↓
RuntimeIntelligenceContextBuilder.build(facts)
        ↓
run_runtime_intelligence_analyzer_isolated(analyzer, context)
        ↓
RuntimeIntelligenceAnalysisResponse
        ↓
caller releases request-scoped facts / response
```

**Forbidden in W6-C:** global registry, manager, scheduler, daemon, shared mutable cache.

---

## Dependency direction

```text
integration (read ports) → RuntimeIntelligenceFacts
        ↓
runtime/runtime_intelligence/context_builder → contracts RuntimeIntelligenceContext
        ↓
runtime/runtime_intelligence/deterministic_analyzer → contracts RuntimeIntelligenceResult
```

Contracts do not import runtime modules. Analyzers depend on context contracts only.

---

## Failure model

| Failure | Behavior | Execution plane |
|---------|----------|-----------------|
| Invalid facts / empty projection | `InvalidIntelligenceContextError` at `build()` | Caller catches; no analyzer invoke |
| Invalid context at analyze boundary | `ANALYZER_OUTCOME_INVALID_CONTEXT` via isolated runner | Fail-soft |
| Analyzer failure | `ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE` | Fail-soft |
| Analyzer identity mismatch | `PLUGIN_UNAVAILABLE` | Fail-soft |

No hidden fallbacks; no partial `RuntimeIntelligenceResult` on plugin failure.

---

## Deterministic analyzer signals

Observed signals are projected into context as stable `intelligence_signal:{kind}:{intensity}` fact pointers (read-only). Rules evaluate:

- execution instability
- repeated failures
- retry pressure
- recovery signals
- resource pressure

Same context + same analyzer version → identical `RuntimeIntelligenceResult` (stable ids via content hash).

---

## Quality gates (W6-C)

| Gate | Requirement |
|------|-------------|
| Unit tests | `tests/unit/runtime/runtime_intelligence/` + existing W6-B contract tests |
| Ruff / Pyright | Clean on touched modules |
| No new god components | No `*Manager`, registry, or global mutable intelligence state |

---

## Verdict

| Gate | W6-C |
|------|------|
| Context builder | **PASS** (request-scoped, immutable output) |
| Deterministic analyzer | **PASS** (SPI conformant, advisory only) |
| Failure isolation | **PASS** (typed errors + W6-B isolated runner) |
| Plugin swap | **PASS** (analyzer injected per request) |
| Docs | **PASS** (hub + this qualification) |
