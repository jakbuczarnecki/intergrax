# Adaptive Harness Intelligence — Implementation Plan

**Architecture (1:1):** [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/plan/` satellites on demand).

**Last updated:** 2026-06-20 — **P2-ARCH-10** AHI governance boundary.

---

## Architecture doc alignment (P2-ARCH)

| ID | Task | Status |
|----|------|--------|
| **P2-ARCH-10** | Clarify AHI governance boundary and production auto-apply rule | **Done** (2026-06-20) |

Architecture: [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md#governance-boundary).

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/plan/ADAPTIVE_HARNESS_INTELLIGENCE_audit_history.md`](plan/plan/ADAPTIVE_HARNESS_INTELLIGENCE_audit_history.md) | audit history |

> **Cursor context budget:** read this hub + **at most one** satellite per session.


---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §25 · baseline **32/32 L3** · W-ADAPT **70/70 Done**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — L4 runtime exists; gaps = production evidence + marketplace readiness

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-6.2 | §6 LLM | Live cost/latency/quality routing (shared LLM_ADAPTERS) | P2 | **Partial** — runtime adapter swap: [M-LLM-X.5](plan/LLM_ADAPTERS.md) |
| AUDIT-IDEAL-9.3 | §9 Orchestration | Dynamic execution strategy selection (shared ORCHESTRATION) | P2 | **Done** |
| AUDIT-IDEAL-12.2 | §12 Skills | Dynamic skill selection L4 hook (shared SKILLS) | P2 | **Done** |
| AUDIT-IDEAL-24.2 | §24 Cost | Automated cost optimization recommendations (shared UAEP) | P2 | **Done** |
| AUDIT-IDEAL-AHI.1 | §25 AHI | 30-day L4 closed-loop evidence on ≥3 golden scenarios (real deploy) | P1 | **Done** |
| AUDIT-IDEAL-AHI.2 | §25 AHI | Bounded policy learning without governance drift | P2 | **Done** |
| AUDIT-IDEAL-AHI.3 | §25 AHI | Capability marketplace readiness (trust, certification, billing) | P3 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---
