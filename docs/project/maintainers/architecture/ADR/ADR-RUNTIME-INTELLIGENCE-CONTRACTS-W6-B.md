# ADR-RUNTIME-INTELLIGENCE-CONTRACTS-W6-B

| Field | Value |
| ----- | ----- |
| **Status** | Accepted (W6-B contract freeze) |
| **Date** | 2026-09-12 |
| **Parent** | [`ADR_ENTERPRISE_RUNTIME_INTELLIGENCE_ARCHITECTURE.md`](../ADR_ENTERPRISE_RUNTIME_INTELLIGENCE_ARCHITECTURE.md) |

---

## Problem

W6-A froze architecture without production contracts. W6-B must deliver a **stable Enterprise SPI** for analyzers without introducing managers, registries, or execution authority.

---

## Decision

1. Package `intergrax/contracts/runtime_intelligence/` owns immutable context, result, evidence, recommendation, typed errors, and `RuntimeIntelligenceAnalyzerPort`.
2. **No** `RuntimeIntelligenceManager`, **no** global registry, **no** `if analyzer_type` dispatch in contracts.
3. Fail-soft containment lives in `run_runtime_intelligence_analyzer_isolated` (per-analyzer boundary); orchestration engines compose tuples at wiring roots in later waves.
4. `RuntimeIntelligencePort` (facade analyze API) remains a **future** W6-C+ runtime module — W6-B freezes analyzer SPI and data envelopes only.

---

## Consequences

- Plugin authors implement `RuntimeIntelligenceAnalyzerPort` only.
- Execution, diagnostics, and governance owners stay unchanged.
- Conformance tests gate immutability, plugin shapes, and failure isolation at contract level.
