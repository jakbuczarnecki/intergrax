# Memory — Implementation Plan

**Architecture (1:1):** [`architecture/MEMORY.md`](../architecture/MEMORY.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/plan/` satellites on demand).

**Cross-plan — Agent layer (ACP):** Per-agent `memory_view` and `memory_scope` (user vs org §30.9) resolve in `merge_environment` — [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) **Wave 2** (`ACP-DX-2`). Agent session state (`AcpSessionState`) is separate from LTM namespaces; do not store secrets in `acp.state.v1` (architecture §25.2).

**Last updated:** 2026-06-17 — **Full Harness LC** (re-validates layer completion + MEM-VEC/MEM-DEPTH closeout).

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (MEMORY plan).

- **Implement / audit default:** Hub §6 · [`plan/plan/`](plan/plan/) satellites on demand. **On demand (one max):** [`plan/plan/MEMORY_appendices.md`](plan/plan/MEMORY_appendices.md) · [`plan/plan/MEMORY_audit_history.md`](plan/plan/MEMORY_audit_history.md). Phase AUDIT-IDEAL — **Planned** / open rows only. §6.1 maintenance queues — open P0/P1 only
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/MEMORY.md`](../architecture/MEMORY.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/MEMORY.md`](../guides/audit_slices/MEMORY.md).
- **Satellites:** at most **one** `plan/plan/` file per session unless RESUME cites more.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/plan/MEMORY_appendices.md`](plan/plan/MEMORY_appendices.md) | appendices |
| [`plan/plan/MEMORY_audit_history.md`](plan/plan/MEMORY_audit_history.md) | audit history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §7, §16 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** — memory-routed rows 14.1–14.2, 15.1–15.3, 16.1–16.2 closed in master register (16.x implemented in CONTEXT_ENGINEERING)

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-14.1 | §14 RAG | Graph RAG as default production retrieval profile | P1 | **Done** |
| AUDIT-IDEAL-14.2 | §14 RAG | Retrieval poisoning defense live on product hosts | P1 | **Done** |
| AUDIT-IDEAL-15.1 | §15 Memory | Org memory 2.5 (organizational LTM scope) | **P0** | **Done** |
| AUDIT-IDEAL-15.2 | §15 Memory | Episodic / semantic / procedural taxonomy (`MemoryKind` uplift) | P1 | **Done** |
| AUDIT-IDEAL-15.3 | §15 Memory | Entity graph memory ship (MEM-DEPTH-5.1 beyond RFC) | P2 | **Done** |
| AUDIT-IDEAL-16.1 | §16 Context | Online context drift monitoring + alerts | P1 | **Done** — owner [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) §11 |
| AUDIT-IDEAL-16.2 | §16 Context | Semantic compression in production profiles | P2 | **Done** — owner CE §11 (`semantic_compression_enabled`) |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---
