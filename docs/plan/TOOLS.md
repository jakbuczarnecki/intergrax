# Tools — Implementation Plan

**Architecture (1:1):** [`architecture/TOOLS.md`](../architecture/TOOLS.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/plan/` satellites on demand).

**Queue status (2026-06-12):** Phase **TOOL-ENG** **closed** (36/36) · [§Layer completion final audit](#layer-completion-final-audit-2026-06-12). Catalog expansion (Phase O / T-EXPAND) **closed**. Default harness queue → **gate maintenance** in [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md).

**Layer completion mode (2026-06-12):** [§Layer completion audit](#layer-completion-audit-2026-06-12) · [§Layer completion sprints](#layer-completion-sprints-2026-06-12) · [§Final audit](#layer-completion-final-audit-2026-06-12)

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/TOOLS_audit_history.md`](plan/TOOLS_audit_history.md) | audit history |

> **Cursor context budget:** read this hub + **at most one** satellite per session.


---

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.6 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (catalog layer) — engine gaps tracked in **Phase TOOL-ENG** (2026-06-10)

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-11.1 | §11 Tools | Sandboxed execution for code / side-effectful tools | P1 | **Done** |
| AUDIT-IDEAL-11.2 | §11 Tools | MCP / function-schema export for shipped tool catalog | P2 | **Done** |
| AUDIT-IDEAL-11.3 | §11 Tools | Oversized-tool lint enforcement in CI (adoption sweep) | P2 | **Done** |

**Follow-on (engine, not AUDIT-IDEAL id):** TOOL-ENG-1–10 — see [Phase TOOL-ENG](#phase-tool-eng--tool-engine-hardening-2026-06-10-audit).

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---
