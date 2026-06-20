# Agent Contracts And Assembly — Implementation Plan

**Architecture (1:1):** [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/plan/` satellites on demand).

---

## Agent architecture completion — executive summary (2026-06-11)

**Phases ACP + ACP-CLOSE + ACP-FINISH + AUDIT-IDEAL (§12–§20):** **Done** (2026-06-13) — platform runtime, fleet migration, production gates, token budget depth, CI matrix, registry snapshot, cap-graph blast-radius, lifecycle on-call.  
**Parallel track:** [Phase AUDIT-IDEAL](#phase-audit-ideal--ideal-architecture-gap-register-2026-06-09) — **10/10 Done** (incl. 19.1 · 20.1 · 31.1).

| Track | Scope | Status | Remaining IDs |
|-------|-------|--------|---------------|
| **ACP runtime depth** | §25.4–§25.5 token usage, limits, reactions | **Done** | — |
| **Architecture doc truth** | §28.3 GAP register · §36.4 · §40.13 tables | **Done** | **ACP-FINISH-DOC-1** **Done** (2026-06-13) |
| **AUDIT-IDEAL (§12–§20)** | Registry snapshot · cap-graph CI · lifecycle owner | **Done** | — |
| **Gate maintenance** | §6.1 continuous | **Active** | `pytest -m gate` on every PR |

**Architecture-complete DoD (ACP-FINISH):** GAP-ACP-36/37 **Closed** · §28.3 **37 Closed · 0 Open** · ACP-TOK-* green · one implementation journal entry · domain audit prompt regenerated.

**Ordered queue:** [§6.1bc](#61bc-harness-implementation-queue--acp-finish-closed) — **Done** (2026-06-13).

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/AGENT_CONTRACTS_AND_ASSEMBLY_audit_history.md`](plan/AGENT_CONTRACTS_AND_ASSEMBLY_audit_history.md) | audit history |

> **Cursor context budget:** read this hub + **at most one** satellite per session.


---

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §17–§19, §31 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (2026-06-13) — 10/10 **Done** · master register [`AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) synced

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-17.1 | §17 Prompts | Prompt approval workflow (beyond registry metadata) | P2 | **Done** |
| AUDIT-IDEAL-17.2 | §17 Prompts | Prompt diff / compare API for all managed prompts | P2 | **Done** |
| AUDIT-IDEAL-18.1 | §18 Assembly | `ModalityProfile` mandatory on certified agents | P1 | **Done** |
| AUDIT-IDEAL-18.2 | §18 Assembly | Cross-host agent reuse certification test suite | P2 | **Done** |
| AUDIT-IDEAL-19.1 | §19 Registry | Durable cross-host registry snapshot store (DEBT-19-01) | **P0** | **Done** |
| AUDIT-IDEAL-19.2 | §19 Registry | Capability negotiation at runtime resolve | P2 | **Done** |
| AUDIT-IDEAL-20.1 | §20 Cap. graph | Product CI blast-radius check on tool/skill changes | P1 | **Done** |
| AUDIT-IDEAL-20.2 | §20 Cap. graph | Policy change impact visualization CLI | P2 | **Done** |
| AUDIT-IDEAL-31.1 | §31 Lifecycle | Owner/on-call mandatory on all certified agents | P1 | **Done** |
| AUDIT-IDEAL-31.2 | §31 Lifecycle | Evaluation required before production promotion (enforce) | P1 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---
