<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Decision / Approval / Governance — Implementation Plan (MP-4R)

**Status:** **MP-4R3 CLOSED** · **MP-4R4 — READY_FOR_FINAL_INDEPENDENT_CLOSURE_AUDIT** (live PostgreSQL qualification **PASSED**) · **MP-4R2** closed · **MP-4R1** closed · **MP-4R0** closed · legacy MP-4A `SUPERSEDED_BY_MP4R0` · MP-4B `RETIRED` · MP-4C `RETIRED` (MP-4R2) · MP-4D `RETIRED` (MP-4R2) · **MP-4R5 NOT STARTED**
**Architecture (1:1):** [`../../architecture/DECISION_APPROVAL_GOVERNANCE.md`](../../architecture/DECISION_APPROVAL_GOVERNANCE.md)
**ADR:** [ADR-MP-009](../../technical/adr/entries/2026-09-15/ADR-MP-009.md) · [ADR-MP-005](../../technical/adr/entries/2026-09-08/ADR-MP-005.md) (historical)
**Feature coordination:** [`MULTIPLAYER_AI`](../../capabilities/plan/MULTIPLAYER_AI.md)

---

## Cursor read scope (token budget)

1. Architecture hub — canonical ownership + MP-4R0 inventory sections only.
2. This file — **active MP-4R slice row only**.
3. ADR-MP-009 for rebase decisions; ADR-MP-005 for historical MP-4A context only.

---

## MP-4R0 — Core rebase & supersession gate

| Field | Value |
|-------|-------|
| **ID** | MP-4R0 |
| **Priority** | P0 |
| **Status** | **CLOSED** |
| **Purpose** | Rebase Multiplayer MP-4 on canonical Decision / Execution / Governance / Evidence / Diagnostics; inventory legacy MP-4B–D; supersede ADR-MP-005 ownership table; freeze MP-4R roadmap and architecture gates |
| **Dependencies** | Canonical Decision System; Execution Engine; ADR-GR-5-001; Evidence/Diagnostics single authority docs |
| **Exact scope** | ADR-MP-009; architecture/plan/MULTIPLAYER_AI sync; caller inventory; `test_mp4r0_multiplayer_rebase_architecture_gates.py`; **no** production deletion |
| **REUSED** | Existing MP-4B–D gates; DS-CORE contracts; `ExecutionContinuationPort` documentation |
| **NEW** | Supersession ADR; MP-4R1…R8 plan rows; collaborative_work Nexus/authority gates |
| **Explicit out of scope** | MP-4R1+ implementation; repositories; HITL bridge runtime; legacy module deletion |
| **Acceptance** | Ownership table frozen; legacy classification documented; ADR index updated; docs pair check green; MP-4R0 gates pass |
| **Proof requirements** | `git diff --check`; `python scripts/docs/check_docs_domain_pairs.py`; `pytest tests/unit/runtime/architecture/test_mp4r0_multiplayer_rebase_architecture_gates.py` |
| **Next step** | **CLOSED** — independent closure audit passed |

---

## MP-4R1 — Decision contract convergence

| Field | Value |
|-------|-------|
| **ID** | MP-4R1 |
| **Status** | **CLOSED** |
| **Purpose** | Converge `intergrax/contracts/decision.py` onto canonical Decision System (`decision_identity`, lifecycle, resolution, finalization); resolve **MP-4R1 convergence debt** (`decision/` package shadowing sibling `decision.py`, `importlib` dynamic MP-4B export bridge) |
| **Dependencies** | MP-4R0 closed |
| **Acceptance** | No duplicate `DecisionId` in production paths; Decision Integration SPI preserved; `test_mp4r1_decision_authority_convergence_gates.py` green |
| **Proof requirements** | `pytest tests/unit/runtime/architecture/test_mp4r1_decision_authority_convergence_gates.py`; decision integration + canonical decision suites |
| **Next step** | **CLOSED** — independent audit passed |

---

## MP-4R2 — Human review / Approval convergence

| Field | Value |
|-------|-------|
| **ID** | MP-4R2 |
| **Status** | **CLOSED** |
| **Purpose** | Retire duplicate MP-4C/D Approval authority; canonical human judgment = `decision_human_review` + Governance/HITL |
| **Dependencies** | MP-4R1 closed |
| **Acceptance** | Legacy `approval.py` / `intergrax/approval/**` removed; `test_mp4r2_human_review_approval_convergence_gates.py` green; stale proposal binding proven |
| **Proof requirements** | Caller proof (zero production consumers); pytest MP-4R0/1/2 gates; `test_decision_human_review.py`; `test_decision_flow.py` governance/HITL cases |
| **Next step** | **CLOSED** — independent audit passed |

## MP-4R3 — Execution continuation integration

| Field | Value |
|-------|-------|
| **ID** | MP-4R3 |
| **Status** | **READY_FOR_FINAL_INDEPENDENT_CLOSURE_AUDIT** |
| **Purpose** | All execution HITL lifecycle effects via `ExecutionContinuationPort` — zero direct Nexus dependency from Multiplayer |
| **Dependencies** | MP-4R2 closed |
| **Acceptance** | No Multiplayer continuation lifecycle/store; no Nexus imports in `collaborative_work`; `test_mp4r3_execution_continuation_integration_gates.py` green; canonical two-phase resume + CAS + identity qualification proven |
| **Proof requirements** | MP-4R0/1/2/3 architecture gates; `test_gr5_r2_canonical_pause_resume.py`; `test_execution_continuation.py`; governed continuation bridge/HITL suites as in validation matrix |
| **Scope note** | **MP-4R3 establishes/adopts boundary and qualification; no artificial Multiplayer continuation runtime was required** — production `collaborative_work` has zero direct `ExecutionContinuationPort` callers today; future integration must use platform contracts only; independent technical audit **PASS** (formal slice closure pending final re-audit) |
| **Next step** | Final independent MP-4R3 closure audit → MP-4R4 **NOT STARTED** |

---

## MP-4R4 — Collaborative decision binding

| Field | Value |
|-------|-------|
| **ID** | MP-4R4 |
| **Status** | **READY_FOR_FINAL_INDEPENDENT_CLOSURE_AUDIT** |
| **Purpose** | `CollaborativeDecisionBinding` — immutable Multiplayer-owned association to exact `DecisionProposalRef` (+ optional `WorkArtifactVersionRef`); **no** Decision lifecycle, Governance authorization, Execution state, or Evidence ownership |
| **Dependencies** | MP-4R3 closed |
| **Acceptance** | `CollaborativeDecisionBindingService` → `CollaborativeDecisionBindingRepository` protocol only; PostgreSQL = configured provider; `test_mp4r4_collaborative_decision_binding_gates.py` green; live PostgreSQL qualification **8 passed, 0 skipped** |
| **Proof requirements** | `pytest tests/integration/collaborative_work/test_postgresql_decision_binding_qualification.py -m "integration and network"` (real PostgreSQL via `infra/docker/postgresql/docker-compose.yml`; DSN `INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN` — not committed); `test_decision_binding_service.py`; `test_decision_proposal_ref_wire.py` |
| **Live PostgreSQL qualification (PASSED)** | Backend: real PostgreSQL 16.x via repo Docker compose (`postgresql` service, host port **5434**, database `intergrax`). Coverage: create/read round-trip; exact `DecisionProposalRef`; tenant/workspace isolation; idempotent replay; idempotency conflict; semantic dedup; concurrent semantic duplicate (2 bundles / 2 connections, `COUNT(*)=1`); concurrent idempotency conflict (1 success, 1 `CollaborativeDecisionBindingIdempotencyConflict`, 1 row). No SQLite, in-memory, mock, or fake backend for this suite. |
| **Next step** | Final independent MP-4R4 closure audit on GitHub — **MP-4R5 NOT STARTED** |

---

## MP-4R5 — Evidence Plane adoption

| Field | Value |
|-------|-------|
| **ID** | MP-4R5 |
| **Status** | **NOT STARTED** |
| **Purpose** | Multiplayer emits/links canonical Evidence Plane facts — no Multiplayer evidence store |
| **Dependencies** | MP-4R4 (or parallel per gate) |

---

## MP-4R6 — Legacy MP-4 removal & migration

| Field | Value |
|-------|-------|
| **ID** | MP-4R6 |
| **Status** | **NOT STARTED** |
| **Purpose** | After caller proof: delete duplicate contracts/services/tests/docs |
| **Dependencies** | MP-4R1…R5 |

---

## MP-4R7 — Enterprise integration qualification

| Field | Value |
|-------|-------|
| **ID** | MP-4R7 |
| **Status** | **NOT STARTED** |
| **Purpose** | E2E proof: Collaborative Work → canonical Decision → Governance/HITL → `ExecutionContinuationPort` → Execution → Evidence → Diagnostics |
| **Dependencies** | MP-4R6 |

---

## MP-4R8 — Final closure audit

| Field | Value |
|-------|-------|
| **ID** | MP-4R8 |
| **Status** | **NOT STARTED** |
| **Purpose** | Independent architecture/code audit; closes MP-4 program |
| **Dependencies** | MP-4R7 |

---

## Legacy MP-4 rows (frozen — do not extend)

| Slice | Status |
|-------|--------|
| MP-4A | SUPERSEDED_BY_MP4R0 |
| MP-4B | RETIRED (MP-4R1) |
| MP-4C | RETIRED (MP-4R2) |
| MP-4D | RETIRED (MP-4R2) |
| MP-4E | CANCELLED_BEFORE_START |
| MP-4F | CANCELLED_REPLACED_BY_EVIDENCE_ADOPTION |
| MP-4G | CANCELLED_IN_OLD_FORM |
| MP-4H | REPLACED_BY_MP4R8 |

Historical implementation notes for MP-4B–D remain in git history and contract modules; see architecture hub inventory.

---

## Out of scope (MP-4R0)

- Deleting `intergrax/contracts/decision.py`, `approval.py`, or `intergrax/approval/**`
- MP-4R1+ runtime work
- Collaborative Work schema changes
- Execution / Decision core changes

---

## Next step

Final independent MP-4R4 closure audit.
MP-4R5 NOT STARTED.
