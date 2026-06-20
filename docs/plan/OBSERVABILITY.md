# Observability — Implementation Plan

**Architecture (1:1):** [`architecture/OBSERVABILITY.md`](../architecture/OBSERVABILITY.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/plan/` satellites on demand).

**Cross-plan — Agent layer (ACP):** Dual observability planes (architecture §31) — `AgentRunTrace` on `AgentRunResult` (Plane B) and `ApplicationRunSummary` on Task completion (Plane A). Delivered in [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) **Wave 3** (`ACP-OBS-1`, `ACP-OBS-2`) and **Wave 7** redaction (`ACP-PROD-8`). Trace spine changes MUST keep step records compatible with `AgentStepRecord` tool/RAG/LLM fields.

**Cross-plan — Event catalog (OBS-EVOL-9 · P1-ARCH-02):** Layered spine + `event_kind` (architecture §4.4 · ADR-OBS-003). Developers extend via `emit_domain_signal`, not new `RuntimeEventType`. Pre-release spine consolidation before publication.

**Last updated:** 2026-06-20 — **P2-ARCH-07** event spine canon.

---

## Architecture documentation (P2)

| ID | Task | Status |
|----|------|--------|
| **P2-ARCH-07** | Clarify observability event spine and event ownership | **Done** (2026-06-20) |

Architecture: [`OBSERVABILITY.md`](../architecture/OBSERVABILITY.md#observability-event-spine).

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/plan/OBSERVABILITY_audit_history.md`](plan/plan/OBSERVABILITY_audit_history.md) | audit history |

> **Cursor context budget:** read this hub + **at most one** satellite per session.


---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.9, §11 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (2026-06-09) — AUDIT-IDEAL observability rows closed

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-5.3 | §5 Policy | Governance health dashboard (GOV-PROD.1) | P4 | **Done** |
| AUDIT-IDEAL-21.1 | §21 Observability | Causal diagnostics beyond trace bridge (ops tooling) | P2 | **Done** |
| AUDIT-IDEAL-21.2 | §21 Observability | Quality / governance / cost health dashboard contracts | P2 | **Done** |
| AUDIT-IDEAL-21.3 | §21 Observability | Unified product observability dashboard | P4 | **Done** |
| AUDIT-IDEAL-30.2 | §30 Ops | Real deploy SLO window evidence (prod `W_OPS_RELEASE_CYCLES`) | P1 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---
