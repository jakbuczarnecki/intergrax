<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Enterprise Runtime Intelligence

**Enterprise Runtime Intelligence (ERI)** is the planned platform layer that **explains, correlates, and scores** runtime execution using **canonical facts** already produced by W1–W5 — without becoming a second diagnostic engine, without owning execution truth, and without introducing god components.

> [!NOTE]
> **W6-A status:** Architecture inventory and qualification only. See [`ADR_ENTERPRISE_RUNTIME_INTELLIGENCE_ARCHITECTURE.md`](../maintainers/architecture/ADR_ENTERPRISE_RUNTIME_INTELLIGENCE_ARCHITECTURE.md) and [`ENTERPRISE_RUNTIME_INTELLIGENCE_W6_A_QUALIFICATION.md`](../maintainers/qualification/ENTERPRISE_RUNTIME_INTELLIGENCE_W6_A_QUALIFICATION.md). **Not** a claim of production implementation.

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

## Planned contracts (W6-B — not implemented in W6-A)

- `RuntimeIntelligencePort` — analyze request → versioned envelope
- `RuntimeIntelligenceAnalyzerPort` — plugin SPI
- `RuntimeIntelligenceContext` — immutable fact snapshot
- `RuntimeIntelligenceEvidenceRef` — stable pointers into canonical stores
- Optional: `AdaptivePolicySignalPort` — recommend-only signals

See ADR for lifecycle, failure model, and versioning strategy.
