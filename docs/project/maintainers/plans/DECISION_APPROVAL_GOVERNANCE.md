<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Decision / Approval / Governance — Implementation Plan (MP-4R)

**Status:** **MP-4R1 — READY_FOR_INDEPENDENT_AUDIT** · **MP-4R0** closed · legacy MP-4A `SUPERSEDED_BY_MP4R0` · MP-4B `RETIRED` · MP-4C `FROZEN_PENDING_CONVERGENCE` · MP-4D `FROZEN_PENDING_AUTHORITY_REBASE` · **MP-4R2 NOT STARTED**
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
| **Status** | **CLOSURE FIX / READY_FOR_REAUDIT** |
| **Purpose** | Rebase Multiplayer MP-4 on canonical Decision / Execution / Governance / Evidence / Diagnostics; inventory legacy MP-4B–D; supersede ADR-MP-005 ownership table; freeze MP-4R roadmap and architecture gates |
| **Dependencies** | Canonical Decision System; Execution Engine; ADR-GR-5-001; Evidence/Diagnostics single authority docs |
| **Exact scope** | ADR-MP-009; architecture/plan/MULTIPLAYER_AI sync; caller inventory; `test_mp4r0_multiplayer_rebase_architecture_gates.py`; **no** production deletion |
| **REUSED** | Existing MP-4B–D gates; DS-CORE contracts; `ExecutionContinuationPort` documentation |
| **NEW** | Supersession ADR; MP-4R1…R8 plan rows; collaborative_work Nexus/authority gates |
| **Explicit out of scope** | MP-4R1+ implementation; repositories; HITL bridge runtime; legacy module deletion |
| **Acceptance** | Ownership table frozen; legacy classification documented; ADR index updated; docs pair check green; MP-4R0 gates pass |
| **Proof requirements** | `git diff --check`; `python scripts/docs/check_docs_domain_pairs.py`; `pytest tests/unit/runtime/architecture/test_mp4r0_multiplayer_rebase_architecture_gates.py` |
| **Next step** | Independent MP-4R0 **closure** audit → MP-4R1 (NOT STARTED) |

---

## MP-4R1 — Decision contract convergence

| Field | Value |
|-------|-------|
| **ID** | MP-4R1 |
| **Status** | **READY_FOR_INDEPENDENT_AUDIT** |
| **Purpose** | Converge `intergrax/contracts/decision.py` onto canonical Decision System (`decision_identity`, lifecycle, resolution, finalization); resolve **MP-4R1 convergence debt** (`decision/` package shadowing sibling `decision.py`, `importlib` dynamic MP-4B export bridge) |
| **Dependencies** | MP-4R0 closed |
| **Acceptance** | No duplicate `DecisionId` in production paths; Decision Integration SPI preserved; `test_mp4r1_decision_authority_convergence_gates.py` green |
| **Proof requirements** | `pytest tests/unit/runtime/architecture/test_mp4r1_decision_authority_convergence_gates.py`; decision integration + canonical decision suites |
| **Next step** | Independent MP-4R1 audit → MP-4R2 (NOT STARTED) |

---

## MP-4R2 — Human review / Approval convergence

| Field | Value |
|-------|-------|
| **ID** | MP-4R2 |
| **Status** | **NOT STARTED** |
| **Purpose** | Prove whether independent MP-4 Approval primitive is needed vs `decision_human_review` + Governance/HITL |
| **Dependencies** | MP-4R1 |
| **Acceptance** | Duplicate Approval semantics removed or justified with ADR |

---

## MP-4R3 — Execution continuation integration

| Field | Value |
|-------|-------|
| **ID** | MP-4R3 |
| **Status** | **NOT STARTED** |
| **Purpose** | All execution HITL lifecycle effects via `ExecutionContinuationPort` — zero direct Nexus dependency from Multiplayer |
| **Dependencies** | MP-4R2 |

---

## MP-4R4 — Collaborative decision binding

| Field | Value |
|-------|-------|
| **ID** | MP-4R4 |
| **Status** | **NOT STARTED** |
| **Purpose** | Optional `WorkItemDecisionBinding` (or equivalent) — immutable typed binding to canonical `DecisionId` / `DecisionVersion`; **no** Decision lifecycle |
| **Dependencies** | MP-4R3 |

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
| MP-4C | FROZEN_PENDING_CONVERGENCE |
| MP-4D | FROZEN_PENDING_AUTHORITY_REBASE |
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
