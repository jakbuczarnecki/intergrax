# Decision Verification - Implementation Plan

**Architecture (1:1):** [`architecture/DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Parent:** [`DECISION_SYSTEM.md`](DECISION_SYSTEM.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> **DS-ROADMAP-REALITY-SYNC (2026-09-04):** Target Verification Pipeline architecture **FROZEN**. Verification Pipeline implementation **migrated and active**. Legacy Critic verification runtime **retired**. Remaining trust hardening and production qualification tracked in plan.

**Last updated:** 2026-09-04 - DS-ROADMAP-REALITY-SYNC.

---

## Cursor read scope (token budget)

- **Default:** hub status + open P0/P1 summary only.
- **Detail rows:** phase sections below - one phase per session max.
- **Architecture:** [`DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md) read-scope block.
- **Extended depth:** [`architecture/satellites/DECISION_VERIFICATION_extended_depth.md`](../../architecture/satellites/DECISION_VERIFICATION_extended_depth.md) on demand.
- **Lifecycle context:** [`DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md) - version binding on demand.
- **CURRENT code:** Decision Verification runtime under `intergrax/runtime/decision_verification*` — one module per session.

---

## Architecture frozen vs implementation reality

| Layer | Status |
| ----- | ------ |
| **Target architecture** | **FROZEN** |
| **Verification Pipeline runtime** | **DONE** |
| **Verification production composition** | **DONE / ENTERPRISE CLOSED** |
| **Production qualification** | **PLANNED** - DS-E2E |

---

## Phase index

| Phase | Status | Section |
| ----- | ------ | ------- |
| DS-VER-PIPE | **DONE** | [below](#phase-ds-ver-pipe--pipeline-foundation) |
| DS-VER-STAGES | **DONE** | [below](#phase-ds-ver-stages--stage-migration-from-cvl) |
| DS-VER-PROD-COMP | **DONE** | [below](#phase-ds-ver-prod-comp--production-composition) |

---

## Phase DS-VER-PIPE - Pipeline foundation (DONE)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-VER-PIPE-01 | P0 | Verification Pipeline orchestrator contract | **Done** |
| DS-VER-PIPE-02 | P0 | Stage plugin interface + registration | **Done** |
| DS-VER-PIPE-03 | P0 | Verification Result + Challenge typed contracts | **Done** |
| DS-VER-PIPE-04 | P0 | Deterministic-before-probabilistic ordering | **Done** |
| DS-VER-PIPE-05 | P1 | Challenge → Lifecycle handoff (no in-place mutation) | **Done** |
| DS-VER-PIPE-06 | P1 | Fail-closed unavailable required stage | **Done** |
| DS-VER-PIPE-07 | P2 | Stage telemetry → Observability | **Done** |

---

## Phase DS-VER-STAGES - Stage migration from CVL (DONE)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-VER-STAGE-L0 | P0 | Structural/deterministic stage from `L0Gateway` | **Done** |
| DS-VER-STAGE-SEM | P1 | Semantic stage from `L1Gateway` / `eval.judge` | **Done** |
| DS-VER-STAGE-TRAJ | P1 | Trajectory stage from `eval.trajectory` | **Done** |
| DS-VER-STAGE-EVID | P1 | Evidence verification stage | **Done** |
| DS-VER-STAGE-GR | P2 | Guardrail merge from `guardrail_l0` | **Done** |
| DS-VER-STAGE-DOM | P2 | Independent/domain verifier stage | **Done** |

Enterprise hardening (typed validators, exact rubric-ref integrity, SHARED_PROFILE truthfulness) closed in probabilistic/domain contracts — no new stage IDs.

---

## Phase DS-VER-PROD-COMP - Production composition (DONE)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-VER-PROD-COMP | P0 | Neutral ToolWiring eval adapters + pipeline factory for semantic/trajectory production composition | **Done / ENTERPRISE CLOSED** - dynamic capability availability (no stale boolean); `intergrax/runtime/decision_verification_composition.py`; `tests/unit/runtime/test_decision_verification_composition.py` |

---

## Open requirements (migrated from Critic audit)

| ID | Priority | Status |
|----|----------|--------|
| DS-VER-RUBRIC-PROVENANCE-INTEGRITY | P0/P1 | **DONE** |
| DS-VER-PRODUCER-INDEPENDENCE | P0/P1 | **IMPLEMENTED / QUALIFICATION OPEN** |
| DS-VER-ADVERSARIAL-SEMANTIC | P1 | **DONE / ENTERPRISE CLOSED** - typed trust contracts · canonical candidate JSON envelope · `build_eval_judge_messages()` · adversarial unit tests |
| DS-VER-RESULT-COHERENCE | P1/P2 | **DONE** |

---

## Explicit non-goals (this plan)

- L2 Human verification stage - **DELETE** from verification; use HITL via Lifecycle.
- `policy_bridge` verdict → action mapping - **SPLIT** to Policy + Lifecycle.
- Offline/shadow eval ownership - remains **OUTSIDE** pipeline ([`CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md) eval boundary).

---

## Delivery rule

One **DS-VER-\*** ID per PR → update phase row in this hub → parent [`DECISION_SYSTEM.md`](DECISION_SYSTEM.md) disposition when Critic capability retired.
