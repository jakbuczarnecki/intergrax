# Decision Deliberation - Implementation Plan

**Architecture (1:1):** [`architecture/DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Parent:** [`DECISION_SYSTEM.md`](DECISION_SYSTEM.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> **DS-ROADMAP-REALITY-SYNC (2026-09-04):** Deliberation / Council strategy architecture **FROZEN**. **CouncilStrategy implemented** (DS-COUNCIL enterprise closed). Separate Council Runtime **forbidden**. Real multi-provider qualification remains **DS-E2E-02**.

**Last updated:** 2026-09-04 - DS-COUNCIL enterprise closeout.

---

## Cursor read scope (token budget)

- **Default:** hub status + phase index only.
- **Detail rows:** phase sections below - one phase per session max.
- **Architecture:** [`DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md) read-scope block.
- **Extended depth:** [`architecture/satellites/DECISION_DELIBERATION_extended_depth.md`](../../architecture/satellites/DECISION_DELIBERATION_extended_depth.md) on demand.
- **Skip** Council implementation detail until DS-DELIB-01 lands.

---

## Architecture frozen vs implementation reality

| Layer | Status |
| ----- | ------ |
| **Target architecture** | **FROZEN** |
| **DecisionStrategy contract + foundation strategies** | **DONE** - DS-DELIB |
| **CouncilStrategy** | **DONE** - `CouncilStrategy implements DecisionStrategy`; `intergrax/contracts/council_strategy.py` + `intergrax/runtime/execution/council_deliberation.py`; `tests/unit/contracts/test_council_strategy.py` + `tests/unit/runtime/execution/test_council_deliberation.py` |
| **Production paths** | Single-model / rule-based / hybrid / council contract paths available; real multi-provider Council **DS-E2E-02** open |

---

## Phase index

| Phase | Status | Section |
| ----- | ------ | ------- |
| DS-DELIB | Done | [below](#phase-ds-delib--strategy-foundation) |
| DS-COUNCIL | Done | [below](#phase-ds-council--council-strategy) |

---

## Phase DS-DELIB - Strategy foundation (Done)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-DELIB-01 | P0 | `DecisionStrategy` protocol + canonical domain registry | **Done** - `intergrax/contracts/decision_strategy.py`; `tests/unit/contracts/test_decision_strategy.py` |
| DS-DELIB-02 | P0 | Single Model strategy (baseline) | **Done** - profile-bound inference via `intergrax/runtime/execution/inference_profile.py`; `intergrax/contracts/single_model_strategy.py`; `intergrax/runtime/execution/single_model_deliberation.py`; `tests/unit/runtime/execution/test_inference_profile_resolution.py` |
| DS-DELIB-03 | P1 | Disagreement artifact typed contract | **Done** - identity hardening via `DecisionProposalRef`; `intergrax/contracts/decision_disagreement.py`; `tests/unit/contracts/test_decision_disagreement.py` |
| DS-DELIB-04 | P1 | Participant role configuration model | **Done** - `intergrax/contracts/decision_participants.py`; `tests/unit/contracts/test_decision_participants.py` |
| DS-DELIB-05 | P1 | Context visibility policy per role | **Done** - `intergrax/contracts/decision_context_visibility.py`; `tests/unit/contracts/test_decision_context_visibility.py` |
| DS-DELIB-06 | P2 | Rule-Based strategy | **Done** - `intergrax/contracts/rule_based_strategy.py`; `tests/unit/contracts/test_rule_based_strategy.py` |
| DS-DELIB-07 | P2 | Hybrid strategy composition | **Done** - `intergrax/contracts/hybrid_strategy.py`; `tests/unit/contracts/test_hybrid_strategy.py` |

---

## Phase DS-COUNCIL - Council strategy (Done)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-COUNCIL-01 | P1 | Council strategy - parallel proposals | **Done** |
| DS-COUNCIL-02 | P1 | Structured disagreement capture | **Done** |
| DS-COUNCIL-03 | P1 | Synthesis candidate emission | **Done** |
| DS-COUNCIL-04 | P1 | Bounded rounds under hosting Execution budget | **Done** |
| DS-COUNCIL-05 | P2 | Deadlock → Adjudication / UNRESOLVED routing | **Done** |

---

## Explicit non-goals (this plan)

- Separate Council Runtime - **FORBIDDEN** by architecture.
- Council-owned verification - candidates feed Verification Pipeline only.
- Private chain-of-thought persistence - **FORBIDDEN**.
- Mandatory Council for all decisions - **FORBIDDEN**.

---

## Delivery rule

Council slices land **after** DS-CORE lifecycle foundation ([`DECISION_SYSTEM.md`](DECISION_SYSTEM.md)).
