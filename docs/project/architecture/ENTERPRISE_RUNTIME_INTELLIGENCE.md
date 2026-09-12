<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Enterprise Runtime Intelligence

**Enterprise Runtime Intelligence (ERI)** is the planned platform layer that **explains, correlates, and scores** runtime execution using **canonical facts** already produced by W1–W5 — without becoming a second diagnostic engine, without owning execution truth, and without introducing god components.

> [!NOTE]
> **W6-A:** Architecture inventory and qualification. **W6-B:** Contract freeze in `intergrax/contracts/runtime_intelligence/` (SPI + immutable envelopes). **W6-C:** Context builder + deterministic analyzer in `intergrax/runtime/runtime_intelligence/`. **W6-D:** Multi-analyzer orchestration in `intergrax/runtime/runtime_intelligence/analyzer_orchestrator.py`. See [`ADR_ENTERPRISE_RUNTIME_INTELLIGENCE_ARCHITECTURE.md`](../maintainers/architecture/ADR_ENTERPRISE_RUNTIME_INTELLIGENCE_ARCHITECTURE.md), [`ADR-RUNTIME-INTELLIGENCE-CONTRACTS-W6-B.md`](../maintainers/architecture/ADR/ADR-RUNTIME-INTELLIGENCE-CONTRACTS-W6-B.md), [`ENTERPRISE_RUNTIME_INTELLIGENCE_W6_A_QUALIFICATION.md`](../maintainers/qualification/ENTERPRISE_RUNTIME_INTELLIGENCE_W6_A_QUALIFICATION.md), [`ENTERPRISE_RUNTIME_INTELLIGENCE_W6_C_QUALIFICATION.md`](../maintainers/qualification/ENTERPRISE_RUNTIME_INTELLIGENCE_W6_C_QUALIFICATION.md), and [`ENTERPRISE_RUNTIME_INTELLIGENCE_W6_D_QUALIFICATION.md`](../maintainers/qualification/ENTERPRISE_RUNTIME_INTELLIGENCE_W6_D_QUALIFICATION.md).

**Primary audience:** Principal / Staff architects and platform engineers extending execution reliability, observability, and adaptive operations.

**Related foundations:**

| Wave | Capability | Hub / qualification |
|------|------------|---------------------|
| W1 | Execution reliability | Execution scale & resilience architecture |
| W2 | Scale & resilience | W2 final qualification |
| W3 | Checkpoint & recovery | W3-A inventory, checkpoint ADR |
| W4 | Cancellation & external operations | ERL + external operations contracts |
| W5 | Observability platform | W5-H final qualification |
| **W6** | **Runtime intelligence** | **This document** |

**Diagnostic authority (unchanged):** [`DIAGNOSTICS.md`](DIAGNOSTICS.md) — ERI is **advisory** relative to Problems.

---

## Purpose

ERI answers operator and automation questions that cross-cut execution planes:

1. **Execution intelligence** — Why did this run succeed or fail (attempts, dependencies, recovery, terminal)?
2. **Runtime diagnostics** — What bounded automated findings can be surfaced without minting Problems?
3. **Decision intelligence** — How were resume, cancel, reconcile, or self-heal decisions taken and how should they be scored?
4. **Adaptive policy foundation** — What signals indicate whether policy changes are **safe to recommend** (action still via governance)?

---

## Architectural position

```text
Intergrax Platform
        │
   Agent / Application Layer
        │
        ▼
   Execution Runtime (W1) — lifecycle, identity, terminal
        │
        ├── Resilience & admission (W2)
        ├── Checkpoint & recovery (W3)
        ├── Cancellation & external ops (W4)
        └── Events & observability export (W5)
        │
        ▼
   Canonical facts (RuntimeEvent, checkpoint, terminal, lineage, admission audit)
        │
        ├── Central Diagnostics → Problem authority
        ├── Observability export → derived telemetry
        └── Enterprise Runtime Intelligence (W6) → advisory envelopes
                    │
                    └── (governance) → existing policy / recovery / self-heal ports
```

---

## Design rules (frozen at W6-A)

| Rule | Detail |
|------|--------|
| **Port → adapter → policy/engine** | No `RuntimeIntelligenceManager` |
| **Contract first** | `intergrax/contracts/runtime_intelligence/` before code |
| **Plugin analyzers** | Local, ML, external service — composed registry |
| **Fail-soft** | Intelligence errors do not fail runs |
| **No new canonical store** | Read projections from existing durable facts |
| **Separation** | `checkpoint ≠ evidence ≠ terminal ≠ Problem ≠ intelligence envelope` |

---

## Distinction from neighboring layers

| Layer | Question | Authority |
|-------|----------|-----------|
| **Central Diagnostics** | What recurring Problem did the platform detect? | **Canonical** (Problems) |
| **Prediction** | What risk might happen next? | Advisory (risk signals) |
| **Adaptive harness (`runtime/adaptive`)** | How should harness profiles evolve (L4)? | Governed profile lifecycle |
| **Enterprise Runtime Intelligence** | Why did execution behave this way; what safe adaptation is suggested? | **Advisory** (envelopes + refs) |
| **Self-healing / prevention** | What automated remediation runs? | Action with admission |

---

## Contract model (W6-B)

```text
intergrax/contracts/runtime_intelligence/
        │
        ├── RuntimeIntelligenceContext (+ fact refs, metadata)
        ├── RuntimeIntelligenceResult (+ confidence, evidence, recommendations)
        ├── IntelligenceEvidence / IntelligenceRecommendation
        ├── RuntimeIntelligenceAnalyzerPort (plugin SPI)
        └── run_runtime_intelligence_analyzer_isolated (per-analyzer fail-soft)
```

| Artifact | Role |
|----------|------|
| `RuntimeIntelligenceContext` | Immutable snapshot of runtime IDs + observed fact pointers |
| `RuntimeIntelligenceResult` | Versioned analysis envelope (advisory) |
| `IntelligenceEvidence` | Traceable refs into canonical stores (not a second authority) |
| `IntelligenceRecommendation` | Recommend-only output — no execution hooks |
| `RuntimeIntelligenceAnalyzerPort` | Local / ML / external analyzers behind one Protocol |

**Deferred (post W6-D):** `RuntimeIntelligencePort` facade, `AdaptivePolicySignalPort`.

---

## Runtime integration (W6-C + W6-D)

```text
intergrax/runtime/runtime_intelligence/
        │
        ├── RuntimeIntelligenceFacts (read-only request inputs)
        ├── RuntimeIntelligenceContextBuilder → RuntimeIntelligenceContext
        ├── DeterministicRuntimeIntelligenceAnalyzer (reference plugin)
        ├── RuntimeIntelligenceAnalyzerOrchestrator (multi-plugin coordination)
        ├── run_runtime_intelligence_analysis (single-analyzer lifecycle)
        └── run_runtime_intelligence_orchestrated_analysis (multi-analyzer lifecycle)
```

Signal observations from read ports are projected into context as stable fact pointers (`intelligence_signal:{kind}:{intensity}`) so W6-B context shape stays unchanged.

---

## Ownership (W6-B + W6-C + W6-D)

| Concern | Owner |
|---------|--------|
| Fact assembly | Integration caller (`RuntimeIntelligenceFacts` from read ports) |
| Context projection | `RuntimeIntelligenceContextBuilder` (`runtime/runtime_intelligence/`) |
| Analysis | `RuntimeIntelligenceAnalyzerPort` implementations |
| Orchestration (ordering, isolation, aggregation) | `RuntimeIntelligenceAnalyzerOrchestrator` |
| Result envelope | Runtime Intelligence contract plane |
| Recommendations | Runtime Intelligence contract plane (governance decides action) |
| Execution / retry / cancel / checkpoints | Existing W1–W4 owners — **unchanged** |

---

## Lifecycle (W6-B + W6-C + W6-D)

```text
create request → build context → analyze (single or orchestrated) → return response → release request resources
```

- **Request-scoped** — no W6 managers, schedulers, or daemons.
- Analyzers are **injected per request** (composition root / caller), not discovered via a central registry.
- Multi-analyzer paths use an explicit `tuple` order; orchestrator delegates each slot to `run_runtime_intelligence_analyzer_isolated`.
- No global mutable intelligence state in contracts or W6 runtime modules.

---

## Plugin model

```text
RuntimeIntelligenceAnalyzerPort
        │
        ├── Local deterministic adapter (W6-C)
        ├── ML model adapter
        └── External HTTP/service adapter
```

Contracts do **not** branch on analyzer kind; each adapter implements the same Protocol.

---

## Failure isolation

```text
Invalid facts at build → InvalidIntelligenceContextError (caller boundary)
Analyzer failure / invalid context at analyze
        ↓
run_runtime_intelligence_analyzer_isolated → PLUGIN_UNAVAILABLE or INVALID_CONTEXT
        ↓
RuntimeIntelligenceAnalyzerOrchestrator aggregates per-analyzer outcomes (W6-D)
        ↓
Execution hot path continues unchanged
```

Typed errors: `RuntimeIntelligenceError`, `AnalyzerExecutionError`, `InvalidIntelligenceContextError` — never generic execution exceptions on the hot path.

---

## Versioning

- Context: `RUNTIME_INTELLIGENCE_CONTEXT_SCHEMA_VERSION`
- Result: `RUNTIME_INTELLIGENCE_RESULT_SCHEMA_VERSION`
- Per plugin: `analyzer_id` + `analyzer_version` on every `RuntimeIntelligenceResult`

Optional future port: `AdaptivePolicySignalPort` (recommend-only signals).
