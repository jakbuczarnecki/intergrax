## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/CODE_CRAFT_audit_history.md`](plan/satellites/CODE_CRAFT_audit_history.md) | audit history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


# Ephemeral Code Craft — Implementation Plan

**Architecture (1:1):** [`architecture/CODE_CRAFT.md`](../../architecture/CODE_CRAFT.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)
**ADR:** [`adr/entries/2026-06-10/ADR-CODECRAFT-001.md`](../../technical/adr/entries/2026-06-10/ADR-CODECRAFT-001.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

**Status:** **ECC-0…ECC-6 Done** · **S7–S11 post-closeout** (2026-06-13) · **Full Harness LC** (2026-06-17)  
**Last updated:** 2026-06-20 — **P2-ARCH-12** CodeCraft safety boundary.  
**Default queue:** Phase **ECC** **closed** (2026-06-13); default gate maintenance continues in [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md).

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (CODE_CRAFT plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites`](plan/satellites) satellites on demand. **On demand (one max):** [`plan/satellites/CODE_CRAFT_audit_history.md`](plan/satellites/CODE_CRAFT_audit_history.md). §6.1 maintenance queues — open P0/P1 only
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/CODE_CRAFT.md`](../../architecture/CODE_CRAFT.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/CODE_CRAFT.md`](../../technical/guides/audit_slices/CODE_CRAFT.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## Architecture doc alignment (P2-ARCH)

| ID | Scope | Status |
|----|-------|--------|
| **P2-ARCH-12** | Clarify CodeCraft safety boundary and promotion rules | **Done** (2026-06-20) |

---

## Delivery rules

1. **One ECC phase per PR** (or one cohesive sub-slice within a phase) → gate green → update this plan row.
2. **Contract first** — Pydantic models + Protocol before orchestrator wiring.
3. **Trace** — every state transition emits `CODECRAFT_*` (+ `RuntimeEvent` / `TraceEvent` where wired).
4. **Tests** — unit + integration; deterministic; no network in gate tests (mock sandbox).
5. **Reuse Tier-0** — extend sandbox, ToolRuntime, CVL; no parallel exec stacks.
6. **Fail closed** — deny paths must have policy tests.
7. **No product scope creep** — ECC harness only; no K.1/K.2 agents without §6.3 decision.

---
