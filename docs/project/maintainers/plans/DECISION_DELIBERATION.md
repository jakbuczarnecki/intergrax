# Decision Deliberation — Implementation Plan

**Architecture (1:1):** [`architecture/DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Parent:** [`DECISION_SYSTEM.md`](DECISION_SYSTEM.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> **DS-DOC (2026-08-30):** Deliberation / Council strategy architecture **FROZEN**. **Council runtime NOT STARTED.**

**Last updated:** 2026-08-30

---

## Cursor read scope (token budget)

- **Default:** Strategy contract rows **P0/P1** only.
- **Architecture:** [`DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md) read-scope block.
- **Skip** Council implementation detail until DS-DELIB-01 lands.

---

## Architecture frozen vs implementation planned

| Layer | Status |
| ----- | ------ |
| **Target architecture** | **FROZEN** |
| **DecisionStrategy contract** | **PLANNED** |
| **Council strategy** | **PLANNED** — not started |
| **CURRENT production** | Single-model agent/graph paths only |

---

## Phase DS-DELIB — Strategy foundation (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-DELIB-01 | P0 | `DecisionStrategy` protocol + registry | **Planned** |
| DS-DELIB-02 | P0 | Single Model strategy (baseline) | **Planned** |
| DS-DELIB-03 | P1 | Disagreement artifact typed contract | **Planned** |
| DS-DELIB-04 | P1 | Participant role configuration model | **Planned** |
| DS-DELIB-05 | P1 | Context visibility policy per role | **Planned** |
| DS-DELIB-06 | P2 | Rule-Based strategy | **Planned** |
| DS-DELIB-07 | P2 | Hybrid strategy composition | **Planned** |

---

## Phase DS-COUNCIL — Council strategy (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-COUNCIL-01 | P1 | Council strategy — parallel proposals | **Planned** |
| DS-COUNCIL-02 | P1 | Structured disagreement capture | **Planned** |
| DS-COUNCIL-03 | P1 | Synthesis candidate emission (no majority erasure) | **Planned** |
| DS-COUNCIL-04 | P1 | Bounded rounds under Nexus budget | **Planned** |
| DS-COUNCIL-05 | P2 | Deadlock → Adjudication / UNRESOLVED routing | **Planned** |

---

## Open requirements (shared)

| ID | Priority | Status | Notes |
|----|----------|--------|-------|
| **DS-VER-PRODUCER-INDEPENDENCE** | P0/P1 | ACCEPTED / PLANNED | Participant profile separation |
| **DS-DEC-EXECUTION-IDENTITY-BINDING** | P0/P1 | ACCEPTED / PLANNED | Participant + proposal identity |

---

## Explicit non-goals (this plan)

- Separate Council Runtime — **FORBIDDEN** by architecture.
- Council-owned verification — candidates feed Verification Pipeline only.
- Private chain-of-thought persistence — **FORBIDDEN**.
- Mandatory Council for all decisions — **FORBIDDEN**.

---

## Delivery rule

Council slices land **after** DS-CORE lifecycle foundation ([`DECISION_SYSTEM.md`](DECISION_SYSTEM.md)).
