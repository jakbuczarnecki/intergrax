# Decision Verification - Implementation Plan

**Architecture (1:1):** [`architecture/DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Parent:** [`DECISION_SYSTEM.md`](DECISION_SYSTEM.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> **DS-DOC-CLEAN (2026-08-30):** Target Verification Pipeline architecture **FROZEN**. Runtime still uses `CriticOrchestrator` until migration.

**Last updated:** 2026-08-30 - DS-DOC-CLEAN plan consolidation.

---

## Cursor read scope (token budget)

- **Default:** hub status + open P0/P1 summary only.
- **Detail rows:** phase sections below - one phase per session max.
- **Architecture:** [`DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md) read-scope block.
- **Extended depth:** [`architecture/satellites/DECISION_VERIFICATION_extended_depth.md`](../../architecture/satellites/DECISION_VERIFICATION_extended_depth.md) on demand.
- **Lifecycle context:** [`DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md) - version binding on demand.
- **CURRENT code:** `intergrax/runtime/critic/**` - migration audit only; one module per session.

---

## Architecture frozen vs implementation planned

| Layer | Status |
| ----- | ------ |
| **Target architecture** | **FROZEN** |
| **Verification Pipeline runtime** | **PLANNED** |
| **CURRENT production** | `CriticOrchestrator` + L0/L1/L2 gateways |

---

## Phase index

| Phase | Status | Section |
| ----- | ------ | ------- |
| DS-VER-PIPE | IN PROGRESS | [below](#phase-ds-ver-pipe--pipeline-foundation) |
| DS-VER-STAGES | PLANNED | [below](#phase-ds-ver-stages--stage-migration-from-cvl) |

---

## Phase DS-VER-PIPE - Pipeline foundation (IN PROGRESS)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-VER-PIPE-01 | P0 | Verification Pipeline orchestrator contract | **Done** |
| DS-VER-PIPE-02 | P0 | Stage plugin interface + registration | **Done** |
| DS-VER-PIPE-03 | P0 | Verification Result + Challenge typed contracts | **Done** |
| DS-VER-PIPE-04 | P0 | Deterministic-before-probabilistic ordering | **Done** |
| DS-VER-PIPE-05 | P1 | Challenge → Lifecycle handoff (no in-place mutation) | **Done** |
| DS-VER-PIPE-06 | P1 | Fail-closed unavailable required stage | **Done** |
| DS-VER-PIPE-07 | P2 | Stage telemetry → Observability | **Planned** |

---

## Phase DS-VER-STAGES - Stage migration from CVL (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-VER-STAGE-L0 | P0 | Structural/deterministic stage from `L0Gateway` | **Planned** |
| DS-VER-STAGE-SEM | P1 | Semantic stage from `L1Gateway` / `eval.judge` | **Planned** |
| DS-VER-STAGE-TRAJ | P1 | Trajectory stage from `eval.trajectory` | **Planned** |
| DS-VER-STAGE-EVID | P1 | Evidence verification stage | **Planned** |
| DS-VER-STAGE-GR | P2 | Guardrail merge from `guardrail_l0` | **Planned** |
| DS-VER-STAGE-DOM | P2 | Independent/domain verifier stage | **Planned** |

---

## Open requirements (migrated from Critic audit)

| ID | Priority | Status |
|----|----------|--------|
| DS-VER-RUBRIC-PROVENANCE-INTEGRITY | P0/P1 | ACCEPTED / PLANNED |
| DS-VER-PRODUCER-INDEPENDENCE | P0/P1 | ACCEPTED / PLANNED |
| DS-VER-ADVERSARIAL-SEMANTIC | P1 | ACCEPTED / PLANNED |
| DS-VER-RESULT-COHERENCE | P1/P2 | ACCEPTED / PLANNED |

---

## Explicit non-goals (this plan)

- L2 Human verification stage - **DELETE** from verification; use HITL via Lifecycle.
- `policy_bridge` verdict → action mapping - **SPLIT** to Policy + Lifecycle.
- Offline/shadow eval ownership - remains **OUTSIDE** pipeline ([`CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md) eval boundary).

---

## Delivery rule

One **DS-VER-\*** ID per PR → update phase row in this hub → parent [`DECISION_SYSTEM.md`](DECISION_SYSTEM.md) disposition when Critic capability retired.
