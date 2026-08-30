# Decision Deliberation — Implementation Plan

**Architecture (1:1):** [`architecture/DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Parent:** [`DECISION_SYSTEM.md`](DECISION_SYSTEM.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> **DS-DOC-HARDEN (2026-08-30):** Deliberation / Council strategy architecture **FROZEN**. **Council runtime NOT STARTED.**

**Last updated:** 2026-08-30

---

## Cursor read scope (token budget)

- **Default:** hub status + phase index only.
- **Detail rows:** [`satellites/DECISION_DELIBERATION_implementation_strategies.md`](satellites/DECISION_DELIBERATION_implementation_strategies.md) — one satellite per session max.
- **Architecture:** [`DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md) read-scope block.
- **Skip** Council implementation detail until DS-DELIB-01 lands.

---

## Implementation satellite

| Satellite | Contents |
| --------- | -------- |
| [`satellites/DECISION_DELIBERATION_implementation_strategies.md`](satellites/DECISION_DELIBERATION_implementation_strategies.md) | DS-DELIB · DS-COUNCIL strategy rows |

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

| Phase | Status | Satellite |
| ----- | ------ | --------- |
| DS-DELIB | PLANNED | [`implementation_strategies`](satellites/DECISION_DELIBERATION_implementation_strategies.md) |
| DS-COUNCIL | PLANNED | same satellite |

---

## Explicit non-goals (this plan)

- Separate Council Runtime — **FORBIDDEN** by architecture.
- Council-owned verification — candidates feed Verification Pipeline only.
- Private chain-of-thought persistence — **FORBIDDEN**.
- Mandatory Council for all decisions — **FORBIDDEN**.

---

## Delivery rule

Council slices land **after** DS-CORE lifecycle foundation ([`DECISION_SYSTEM.md`](DECISION_SYSTEM.md)).
