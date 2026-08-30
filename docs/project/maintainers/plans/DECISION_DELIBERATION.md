# Decision Deliberation — Implementation Plan

**Architecture (1:1):** [`architecture/DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Parent:** [`DECISION_SYSTEM.md`](DECISION_SYSTEM.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> **DS-DOC-CLEAN (2026-08-30):** Deliberation / Council strategy architecture **FROZEN**. **Council runtime NOT STARTED.**

**Last updated:** 2026-08-30 — DS-DOC-CLEAN plan consolidation.

---

## Cursor read scope (token budget)

- **Default:** hub status + phase index only.
- **Detail rows:** phase sections below — one phase per session max.
- **Architecture:** [`DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md) read-scope block.
- **Extended depth:** [`architecture/satellites/DECISION_DELIBERATION_extended_depth.md`](../../architecture/satellites/DECISION_DELIBERATION_extended_depth.md) on demand.
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

## Phase index

| Phase | Status | Section |
| ----- | ------ | ------- |
| DS-DELIB | PLANNED | [below](#phase-ds-delib--strategy-foundation) |
| DS-COUNCIL | PLANNED | [below](#phase-ds-council--council-strategy) |

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
| DS-COUNCIL-03 | P1 | Synthesis candidate emission | **Planned** |
| DS-COUNCIL-04 | P1 | Bounded rounds under Nexus budget | **Planned** |
| DS-COUNCIL-05 | P2 | Deadlock → Adjudication / UNRESOLVED routing | **Planned** |

---

## Explicit non-goals (this plan)

- Separate Council Runtime — **FORBIDDEN** by architecture.
- Council-owned verification — candidates feed Verification Pipeline only.
- Private chain-of-thought persistence — **FORBIDDEN**.
- Mandatory Council for all decisions — **FORBIDDEN**.

---

## Delivery rule

Council slices land **after** DS-CORE lifecycle foundation ([`DECISION_SYSTEM.md`](DECISION_SYSTEM.md)).
