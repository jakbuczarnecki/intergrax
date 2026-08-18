# Agent Contracts And Assembly — Implementation Plan

**Architecture (1:1):** [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

**Cross-feature — Token Optimization:** feature architecture [`features/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md) · feature plan [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md). ACP may later expose agent-level output/context compactness hints, but agents must not manually assemble prompts or import token optimization internals.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (AGENT_CONTRACTS_AND_ASSEMBLY plan).

- **Implement / audit default:** §6.1bc ACP-FINISH status · AUDIT-IDEAL §12–§20 table (**Done** skip unless cited) · [`plan/satellites/AGENT_CONTRACTS_AND_ASSEMBLY_implementation_history.md`](plan/satellites/AGENT_CONTRACTS_AND_ASSEMBLY_implementation_history.md) on demand
- **Token Optimization:** read feature pair + row `TOKEN-ACP-1` only when adding agent-level hints; do not implement until TOKEN-UER-2 runtime output policy exists.
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) read-scope block only.
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## Agent architecture completion — executive summary (2026-06-11)

**Phases ACP + ACP-CLOSE + ACP-FINISH + AUDIT-IDEAL (§12–§20):** **Done** (2026-06-13) — platform runtime, fleet migration, production gates, token budget depth, CI matrix, registry snapshot, cap-graph blast-radius, lifecycle on-call.  
**Parallel track:** [Phase AUDIT-IDEAL](.#phase-audit-ideal--ideal-architecture-gap-register-2026-06-09) — **10/10 Done** (incl. 19.1 · 20.1 · 31.1).

| Track | Scope | Status | Remaining IDs |
|-------|-------|--------|---------------|
| **ACP runtime depth** | §25.4–§25.5 token usage, limits, reactions | **Done** | — |
| **Architecture doc truth** | §28.3 GAP register · §36.4 · §40.13 tables | **Done** | **ACP-FINISH-DOC-1** **Done** (2026-06-13) |
| **AUDIT-IDEAL (§12–§20)** | Registry snapshot · cap-graph CI · lifecycle owner | **Done** | — |
| **Gate maintenance** | §6.1 continuous | **Active** | `pytest -m gate` on every PR |

**Architecture-complete DoD (ACP-FINISH):** GAP-ACP-36/37 **Closed** · §28.3 **37 Closed · 0 Open** · ACP-TOK-* green · one implementation journal entry · domain audit prompt regenerated.

**Ordered queue:** [§6.1bc](.#61bc-harness-implementation-queue--acp-finish-closed) — **Done** (2026-06-13).

### Protocol v2 remediation — STRATEGIC_HARNESS_MODEL (2026-08-18)

**Audit:** [`docs/audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md) · campaign [`README`](../../audit_results/2026-08-18/README.md)
**Status:** ACCEPTED findings — **PLANNED** remediation only. **Not implemented** by audit persistence task AUDIT-20260818-INIT.

| Block | Status | Findings | Scope |
|-------|--------|----------|-------|
| **SHM-FIX-A** | ACCEPTED / PLANNED | [`-01`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md), [`-02`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md), [`-03`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md), [`-04`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md) | One canonical governed execution semantics across direct, Nexus, normal, and resume paths |
| **SHM-FIX-B** | ACCEPTED / PLANNED | [`-06`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md), [`-08`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md), [`-09`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md) | Typed critical author/runtime boundary; Task/Run/Attempt continuity |
| **SHM-FIX-C** | ACCEPTED / PLANNED | [`-05`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md), [`-07`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md) | Explicit production host/profile requirements; product-neutral result transport |
| **SHM-FIX-D** | ACCEPTED / PLANNED | [`-10`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md) | Maturity recertification after A–C verification |

**Remediation rules:**

- Revalidate each finding against then-current `development` HEAD before implementation.
- If parallel work already fixed a finding, do not duplicate — independently verify before lifecycle advancement.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** only after campaign/remediation rollup confirms closure ([`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md)).

---

## Phase TOKEN-ACP — Optional token optimization agent hints (Deferred)

**Feature:** [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md)
**Architecture:** [`features/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md)
**Priority:** P3 / deferred until TOKEN-UER-2 and TOKEN-CE-1 exist  
**Delivery rule:** agent contracts may declare hints only; runtime resolves effective policy.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **TOKEN-ACP-1** | Contract | P3 | Deferred | Optional agent-level hints for desired output profile, context compactness, non-compressible sources, and citation/evidence sensitivity | Hints are declarative; agents do not manually assemble prompts; runtime/profile policy can override hints; no Tier-2 imports from `runtime/token_optimization`; tests prove existing agents remain compatible |

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/AGENT_CONTRACTS_AND_ASSEMBLY_implementation_history.md`](plan/AGENT_CONTRACTS_AND_ASSEMBLY_implementation_history.md) | implementation history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

---

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §17–§19, §31 · baseline **32/32 L3**
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

**Delivery rule:** One **AUDIT-IDEAL-*** ID per PR → update this table + master register → gate green.

---
