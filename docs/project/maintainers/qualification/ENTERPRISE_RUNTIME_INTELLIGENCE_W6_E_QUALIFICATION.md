# Enterprise Runtime Intelligence — W6-E qualification

**Wave:** W6-E — execution runtime integration boundary  
**Status:** Implementation qualification (advisory port + service + execution call site)  
**Depends on:** W6-B contracts, W6-C context builder, W6-D orchestration

---

## Scope delivered

| Deliverable | Location | Role |
|-------------|----------|------|
| Integration port + fail-soft invoke | `intergrax/contracts/runtime_intelligence/integration.py` | Dependency inversion; execution hot-path isolation |
| Intelligence service | `intergrax/runtime/runtime_intelligence/service.py` | Implements port; delegates to facade + analyzer SPI |
| Execution call site | `intergrax/runtime/execution/runtime_intelligence_advisory.py` | Optional wiring; no lifecycle ownership |
| Facade (direct API) | `intergrax/runtime/runtime_intelligence/facade.py` | Non-execution callers |
| Conformance tests | `tests/unit/runtime/runtime_intelligence/test_w6_e_*.py` | Boundary, isolation, ownership, lifecycle, plugins |

**Out of scope:** `AdaptivePolicySignalPort`, automatic execution hooks, global registry, background schedulers.

---

## Boundary

```text
Execution Runtime
        │
        ▼
RuntimeIntelligenceRuntimeIntegrationPort.analyze_advisory
        │
        ▼
RuntimeIntelligenceService → RuntimeIntelligenceFacade → Analyzer SPI
```

Runtime Intelligence **observes** fact pointers and returns advisory envelopes only.

---

## Ownership

| Concern | Owner |
|---------|--------|
| Execution lifecycle, retries, cancel, checkpoints | Execution Runtime (W1–W4) — unchanged |
| Fact pointer assembly at boundary | Execution caller |
| Context projection | `RuntimeIntelligenceContextBuilder` |
| Analysis + recommendations | `RuntimeIntelligenceAnalyzerPort` plugins |
| Integration fail-soft semantics | `invoke_runtime_intelligence_integration_isolated` |

Runtime Intelligence does **not** own execution state transitions or admission decisions.

---

## Lifecycle

```text
RuntimeIntelligenceFactsInput (immutable)
        ↓
analyze_advisory (request-scoped)
        ↓
RuntimeIntelligenceAdvisoryResponse
        ↓
caller releases scope
```

No persistent mutable intelligence state; no background workers.

---

## Failure model

| Event | Behavior |
|-------|----------|
| Invalid facts / context at build | `INTEGRATION_OUTCOME_INVALID_INPUT` |
| `RuntimeIntelligenceError` at port | `INTEGRATION_OUTCOME_UNAVAILABLE` |
| Analyzer failure | Per-analyzer `PLUGIN_UNAVAILABLE` inside advisory |
| Unwired port at execution | `None` — execution proceeds |

Execution must not receive intelligence exceptions on the hot path.

---

## What Runtime Intelligence does not do

- Control or mutate execution lifecycle
- Execute recommendations or bypass governance
- Register global analyzers or run schedulers
- Block or fail runs when intelligence is unavailable

---

## Quality gates (W6-E)

| Gate | Requirement |
|------|-------------|
| Unit tests | `test_w6_e_runtime_intelligence_integration_boundary.py`, `test_w6_e_runtime_intelligence_facade.py` |
| Ruff / Pyright | Clean on touched modules |
| No god components | No `*Manager`, registry, singleton, or global mutable state |

---

## Verdict

| Gate | W6-E |
|------|------|
| Integration boundary | **PASS** |
| Failure isolation | **PASS** |
| Ownership (advisory only) | **PASS** |
| Request-scoped lifecycle | **PASS** |
| Plugin compatibility | **PASS** |
| Docs | **PASS** |
