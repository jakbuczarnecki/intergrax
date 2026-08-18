# Critic Verification — Implementation Plan

**Architecture (1:1):** [`architecture/CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

**Last updated:** 2026-06-20 — **P2-ARCH-08** verification safety boundaries.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (CRITIC_VERIFICATION plan).

- **Implement / audit default:** AUDIT-IDEAL · §CVL-4 backlog · audit_history satellite. **On demand (one max):** [`plan/satellites/CRITIC_VERIFICATION_appendices.md`](plan/satellites/CRITIC_VERIFICATION_appendices.md) · [`plan/satellites/CRITIC_VERIFICATION_audit_history.md`](plan/satellites/CRITIC_VERIFICATION_audit_history.md). Phase AUDIT-IDEAL — **Planned** / open rows only. §6.1 maintenance queues — open P0/P1 only
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md) read-scope block only.
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## Architecture doc alignment (P2-ARCH)

| ID | Scope | Status |
|----|-------|--------|
| **P2-ARCH-08** | Clarify verification safety boundaries | **Done** (2026-06-20) |

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/CRITIC_VERIFICATION_appendices.md`](plan/satellites/CRITIC_VERIFICATION_appendices.md) | appendices |
| [`plan/satellites/CRITIC_VERIFICATION_audit_history.md`](plan/satellites/CRITIC_VERIFICATION_audit_history.md) | audit history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §18 · baseline **32/32 L3**
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-25.1 | §25 Evaluation | Shadow eval path automation (DEBT-25-01) | P1 | **Done** |
| AUDIT-IDEAL-25.2 | §25 Evaluation | Human review sample queue (beyond CLI) | P2 | **Done** |
| AUDIT-IDEAL-25.3 | §25 Evaluation | Context/RAG eval blocking product release CI | P1 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

---

## Cross-domain phase registers (canonical elsewhere)

Foreign **Platform / ORCH / FLOW / FAUDIT** registers were removed from this hub.

| Need | Canonical source |
|------|------------------|
| Platform gate maintenance | [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) §6.1 |
| ORCH closeout | [`ORCHESTRATION.md`](ORCHESTRATION.md) |
| FAUDIT-32 | [`plan/satellites/PLATFORM_FOUNDATION_phase_closeout.md`](plan/satellites/PLATFORM_FOUNDATION_phase_closeout.md) |
| FLOW depth | [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) |

**Audit history (CVL-1…3, LC closeout):** [`plan/satellites/CRITIC_VERIFICATION_audit_history.md`](plan/satellites/CRITIC_VERIFICATION_audit_history.md)

---
## Audit §CVL-4 — Backlog (P2–P4, non-blocking)

| ID | Priority | Item | Notes |
|----|----------|------|-------|
| CVL-BACKLOG-01 | P2 | LLM trajectory judge in runtime path | **Documented** — `eval.trajectory_judge` skill; `eval.trajectory` stays heuristic (CVL-LC-4) |
| CVL-BACKLOG-02 | P2 | Test isolation for critic graph suite | **Done** — `register_default_tools` idempotent override (CVL-LC-3) |
| CVL-BACKLOG-03 | P2 | `NexusEvalRunner.from_nexus_loop` auto-wire semantic client | **Done** — CVL-LC-2 |
| CVL-BACKLOG-04 | P3 | Duplicate CRIT-V master register removed | **CVL-LC-1** doc cleanup |
| CVL-BACKLOG-05 | P4 | L4 adaptive critic thresholds in CI | AHIA / `VerificationLoop` extension |
| CVL-BACKLOG-06 | P4 | FLOW-8 product reference host with critic demo | §6.3 deferred |

---
