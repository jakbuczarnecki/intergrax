## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/CONTEXT_ENGINEERING_audit_history.md`](plan/satellites/CONTEXT_ENGINEERING_audit_history.md) | audit history |
| [`plan/satellites/CONTEXT_ENGINEERING_embedded_detail.md`](plan/satellites/CONTEXT_ENGINEERING_embedded_detail.md) | embedded detail |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


# Context Engineering — Implementation Plan

**Architecture (1:1):** [`architecture/CONTEXT_ENGINEERING.md`](../architecture/CONTEXT_ENGINEERING.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)  
**ADR:** [`ADR-CTX-001`](../adr/entries/2026-06-12/ADR-CTX-001.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites/` satellites on demand).

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (CONTEXT_ENGINEERING plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites/`](plan/satellites/) satellites on demand. **On demand (one max):** [`plan/satellites/CONTEXT_ENGINEERING_audit_history.md`](plan/satellites/CONTEXT_ENGINEERING_audit_history.md) · [`plan/satellites/CONTEXT_ENGINEERING_embedded_detail.md`](plan/satellites/CONTEXT_ENGINEERING_embedded_detail.md). §6.1 maintenance queues — open P0/P1 only
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/CONTEXT_ENGINEERING.md`](../architecture/CONTEXT_ENGINEERING.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/CONTEXT_ENGINEERING.md`](../guides/audit_slices/CONTEXT_ENGINEERING.md).
- **Satellites:** at most **one** `plan/satellites/` file per session unless RESUME cites more.

---

## Status summary (2026-06-17)

| Phase | Scope | Status |
|-------|-------|--------|
| **CTX** (control plane closeout) | `context_runtime_bridge`, `context_wiring`, Appendix L | **Done** (2026-06-02) |
| **R-Context** | Budget API, `CONTEXT_*` events (graph path) | **Done** |
| **MEM-DEPTH-1.\*** | `ContextCompiler`, `DegradationLadder`, preflight **modules** | **Done** — library + tests; **hot-path wiring = CE-3.9** (post ACP-CLOSE) |
| **CE-DOC** | Domain split + architecture + plan + FAUDIT refresh | **Done** (CE-DOC.7 closes 2026-06-12 audit) |
| **CE-EXT** | Plugin engine + hot-path compiler + step-aware + codebase preset | **Done** (S0–S12, 2026-06-12) |
| **CE-DOC.8** | Architecture ↔ implementation sync post CE-EXT | **Done** (2026-06-12) |
| **CE-ALIGN** | Post-audit implementation alignment (GAP-CTX-15..19) | **Done** (A0–A6, 2026-06-12) |
| **CE-PROV-WIRE** | Builtin stub providers → legacy collectors on `assemble()` path | **Done** (B1–B4, 2026-06-14) |
| **CE-DOC.9** | FAUDIT 2026-06-12 deep audit — GAP-CTX-15..19 + CE-ALIGN sprint register | **Done** (2026-06-12) |
| **CE-DOC.10** | CE-ALIGN closeout audit + architecture sync | **Done** (2026-06-12) |
| **CE-HANDLE-FILL** | RuntimeState → provider metadata sync on nexus context steps | **Done** (2026-06-14) |
| **P2-ARCH-05** | Add context path unification rules (approved / disallowed / transitional paths + Cursor checklist) | **Done** (2026-06-20) |

**As-built maturity:** L3+ engine / L3 control plane — CE-PROV-WIRE closed GAP-CTX-20; Layer Completion iteration III (2026-06-17) confirms **Architecturally Mature** — no P0/P1; **Full Harness LC** (2026-06-17); see architecture §3.

**Delivery rule:** One **CE-\*** ID per PR → update master table + gap register → `pytest -m gate` + domain CI scripts green.

---
