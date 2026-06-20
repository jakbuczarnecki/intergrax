# Skills — Implementation Plan

**Architecture (1:1):** [`architecture/SKILLS.md`](../architecture/SKILLS.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/plan/` satellites on demand).

**Last updated:** 2026-06-17 — **Full Harness LC** (re-validates 2026-06-08 closeout); SK-EXP through SK-EXP5 **Done** (149 skills · 41 bundles); SK-BRIDGE.1/2 **Done**.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/plan/SKILLS_audit_history.md`](plan/plan/SKILLS_audit_history.md) | audit history |

> **Cursor context budget:** read this hub + **at most one** satellite per session.


---

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.6 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-12.1 | §12 Skills | LangGraph-compatible skill pack import path | P2 | **Done** |
| AUDIT-IDEAL-12.2 | §12 Skills | Dynamic skill selection L4 hook (AHI) | P2 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---
