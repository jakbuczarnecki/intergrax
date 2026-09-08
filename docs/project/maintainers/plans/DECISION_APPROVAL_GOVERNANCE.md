<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Decision / Approval / Governance — Implementation Plan (MP-4)

**Status:** **MP-4A — APPROVED / CLOSED** · **MP-4B — READY_FOR_REVIEW** · **MP-4C — READY_FOR_REVIEW** · **MP-4D — READY_FOR_REVIEW** · MP-4E…MP-4H **NOT STARTED**
**Architecture (1:1):** [`../../architecture/DECISION_APPROVAL_GOVERNANCE.md`](../../architecture/DECISION_APPROVAL_GOVERNANCE.md)
**ADR:** [ADR-MP-005](../../technical/adr/entries/2026-09-08/ADR-MP-005.md)
**Feature coordination:** [`MULTIPLAYER_AI`](../../capabilities/plan/MULTIPLAYER_AI.md)
**Anchor domain (reference boundary):** [`COLLABORATIVE_WORK`](COLLABORATIVE_WORK.md)

---

## Cursor read scope (token budget)

1. [`../../architecture/DECISION_APPROVAL_GOVERNANCE.md`](../../architecture/DECISION_APPROVAL_GOVERNANCE.md) — ownership + anti-substitution sections only.
2. This file — **active MP-4 slice row only**.
3. ADR-MP-005 when implementing or reviewing gates.
4. Minimal source files required by the active slice (contracts only after MP-4B opens).

Do not implement runtime, persistence, or workflow engine in MP-4A.

---

## Planning model

MP-4 delivers collaborative Decision, Approval, and Governance semantics as **extensions of the Multiplayer plane**, distinct from Collaborative Work artifact lifecycle and Unified Execution runtime state.

Implementation rows below are **architecture/contract gates** until the relevant slice opens. Domain-specific Python production code begins at **MP-4B** (Decision contracts) subject to ADR-MP-005 acceptance.

---

## MP-4A — Ownership + contracts freeze

| Field | Value |
|-------|-------|
| **ID** | MP-4A |
| **Priority** | P1 |
| **Status** | **APPROVED / CLOSED** |
| **Purpose** | Freeze Decision / Approval / Governance ownership, dependency direction, anti-substitution rules, and contract reuse audit before any MP-4 implementation |
| **Dependencies** | MP-1 **CLOSED**; MP-2 **APPROVED / CLOSED**; MP-3 ownership **FROZEN** (ADR-MP-004) |
| **Exact scope** | Architecture hub; plan decomposition MP-4A…MP-4H; ADR-MP-005; MULTIPLAYER_AI + COLLABORATIVE_WORK boundary sync; validation grep + docs checks |
| **REUSED** | MP-1 Principal/authority; `ExecutionProvenanceRef`; Governed Execution policy/HITL (reference); Evidence reference patterns |
| **NEW** | Ownership freeze documentation; ADR-MP-005; roadmap decomposition only |
| **Explicit out of scope** | Python production contracts; repositories; services; persistence; workflow engine; Nexus wiring; Collaborative Work schema changes |
| **Acceptance** | Ownership table frozen; anti-substitution documented; dependency direction frozen; reuse audit complete; ADR-MP-005 **Accepted**; MP-4B…H rows present; no CW/UER leakage in docs; `check_docs_domain_pairs.py` green |
| **Proof requirements** | `git diff --check`; `python scripts/docs/check_docs_domain_pairs.py`; leakage greps (see architecture hub) |
| **Next step** | MP-4B — Decision contracts |

---

## MP-4B — Decision contracts

| Field | Value |
|-------|-------|
| **ID** | MP-4B |
| **Priority** | P1 |
| **Status** | **READY_FOR_REVIEW** |
| **Purpose** | Typed Decision identity, lifecycle, and outcome contracts under `intergrax/contracts/decision.py` |
| **Dependencies** | MP-4A **Accepted / CLOSED** |
| **Exact scope** | DecisionId; Decision lifecycle states; DecisionOutcome; references to WorkItem / WorkArtifactVersion / ExecutionProvenanceRef — no artifact or execution lifecycle ownership |
| **REUSED** | `CollaborativePrincipal`; `ExecutionProvenanceRef`; MP-1 authority patterns; tenant/workspace scoping from CW conventions |
| **NEW** | Decision-specific contract types and invariants |
| **Explicit out of scope** | Approval lifecycle; persistence; services; Nexus imports; WorkItem/WorkArtifact mutation |
| **Acceptance** | Decision ≠ Artifact invariant enforced in contracts; no forbidden imports; contract tests at slice gate |
| **Next step** | MP-4C |

---

## MP-4C — Approval / HITL contracts

| Field | Value |
|-------|-------|
| **ID** | MP-4C |
| **Priority** | P1 |
| **Status** | **READY_FOR_REVIEW** |
| **Purpose** | Approval request, approval lifecycle, human action contracts; explicit HITL **bridge** contract (not HITL ownership) |
| **Dependencies** | MP-4B approved |
| **Exact scope** | `ApprovalRequest`; `ApprovalId`; `ApprovalLifecycleState`; `HumanApprovalAction`; `ApprovalOutcome` in `intergrax/contracts/approval.py` — references Decision and neutral CW/UER refs only |
| **REUSED** | Governed Execution HITL pause/resume semantics (bridge only); MP-1 authority; Decision references from MP-4B |
| **NEW** | Approval-specific contracts; HITL bridge interface (vendor-neutral) |
| **Explicit out of scope** | `WorkArtifactVersion.status`; `ExecutionState.WAITING_FOR_HUMAN` as approval store; Nexus orchestration imports |
| **Acceptance** | Approval references WorkArtifactVersion without owning artifact lifecycle; HITL bridge is optional and explicit; Decision ≠ HITL invariant preserved |
| **Next step** | MP-4D |

---

## MP-4D — Authority integration

| Field | Value |
|-------|-------|
| **ID** | MP-4D |
| **Priority** | P1 |
| **Status** | **READY_FOR_REVIEW** |
| **Purpose** | Wire Decision/Approval mutations through MP-1 effective authority and platform policy composition |
| **Dependencies** | MP-4C approved |
| **Exact scope** | Authority gates for approval create and human action mutations; reuse `EffectiveAuthorityRequest` / `PolicyDecision` via `CollaborativeWorkEnforcementGate`; `intergrax/approval/service.py` |
| **REUSED** | MP-1 authority resolver; `CollaborativeWorkEnforcementGate`; `EffectiveAuthorityRequest` |
| **NEW** | `approval.request.create` / `approval.action.execute` trusted operations; `ApprovalService`; `ApprovalAuthorizationDenied`; `approval_resource_scope` |
| **Explicit out of scope** | Second ACL engine; artifact publication authority redefinition; persistence |
| **Acceptance** | All meaningful Approval mutations pass MP-1 authority; no duplicate Principal types; static architecture gates pass |
| **Next step** | MP-4E |

---

## MP-4E — Persistence boundary

| Field | Value |
|-------|-------|
| **ID** | MP-4E |
| **Priority** | P1 |
| **Status** | **NOT STARTED** |
| **Purpose** | Repository ports and persistence direction for Decision/Approval/Governance aggregates |
| **Dependencies** | MP-4D approved |
| **Exact scope** | Port definitions; in-memory reference adapter pattern; SQLite → PostgreSQL direction mirroring CW — no implementation until slice opens |
| **REUSED** | CW repository port patterns (revision/CAS/idempotency intent); tenant/workspace isolation |
| **NEW** | Decision/Approval/Governance repository ports |
| **Explicit out of scope** | Collaborative Work repository changes; Nexus stores |
| **Acceptance** | Persistence boundary documented; no cross-domain table ownership |
| **Next step** | MP-4F |

---

## MP-4F — Evidence / provenance integration

| Field | Value |
|-------|-------|
| **ID** | MP-4F |
| **Priority** | P1 |
| **Status** | **NOT STARTED** |
| **Purpose** | Stable linkage from Decision/Approval/Governance to platform evidence without evidence ownership transfer |
| **Dependencies** | MP-4E approved (or parallel with MP-4E per gate decision) |
| **Exact scope** | Governance evidence references; optional proof/evidence handles on Decision/Approval records |
| **REUSED** | Proof Receipts / observability reference patterns; `ExecutionProvenanceRef` |
| **NEW** | Collaborative governance evidence linkage contracts only |
| **Explicit out of scope** | ProofReceipt as Decision identity; mutable evidence fields substituting outcomes |
| **Acceptance** | Evidence references are optional and immutable post-record; Proof Receipts remains owner |
| **Next step** | MP-4G |

---

## MP-4G — Qualification

| Field | Value |
|-------|-------|
| **ID** | MP-4G |
| **Priority** | P1 |
| **Status** | **NOT STARTED** |
| **Purpose** | Qualification contracts and live proof for Decision/Approval/Governance runtime paths |
| **Dependencies** | MP-4F approved |
| **Exact scope** | Qualification profiles; authorization/isolation tests; HITL bridge integration proof; idempotency/concurrency tests |
| **REUSED** | MP-1/MP-2/MP-3 qualification patterns |
| **NEW** | MP-4 qualification evidence |
| **Explicit out of scope** | MP-6 Activity; LKW product adoption |
| **Acceptance** | Qualification green per slice gate; anti-substitution audit pass |
| **Next step** | MP-4H |

---

## MP-4H — Final closure audit

| Field | Value |
|-------|-------|
| **ID** | MP-4H |
| **Priority** | P1 |
| **Status** | **NOT STARTED** |
| **Purpose** | Final MP-4 independent review and closure gate |
| **Dependencies** | MP-4A…MP-4G approved per rollout policy |
| **Exact scope** | ADR-MP-005 compliance; anti-substitution audit; docs sync; architecture gates; no MP-6 leakage; LKW remains consumer |
| **REUSED** | MP-3 final review pattern |
| **NEW** | MP-4 closure evidence |
| **Explicit out of scope** | MP-5+ implementation |
| **Acceptance** | All MP-4 acceptance criteria met; ownership ADR compliance; authority reuse; no CW/UER/Nexus coupling violations; documentation checks green |
| **Proof requirements** | Focused regression suite; `check_docs_domain_pairs.py`; leakage greps; live qualification evidence |
| **Next step** | MP-5 architecture gate (when scheduled) |

---

## Out of scope (current phase)

- MP-4B…MP-4H implementation until respective slice opens
- Python production code in MP-4A
- Collaborative Work WorkArtifact lifecycle changes
- Unified Execution runtime state machine changes
- MP-6 Activity projection
- LKW product adoption (MP-7)
