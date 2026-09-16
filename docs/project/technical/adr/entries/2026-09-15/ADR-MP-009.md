# ADR-MP-009: Multiplayer MP-4 core rebase on canonical enterprise core

| Field | Value |
|-------|-------|
| **Status** | Accepted — architecture supersession gate (MP-4R0); no production deletion in this ADR |
| **Date** | 2026-09-15 |
| **Deciders** | Intergrax platform architecture (MP-4R0 supersession gate) |
| **Related** | [ADR-MP-005](../2026-09-08/ADR-MP-005.md) (partially superseded) · [ADR-GR-5-001](../2026-09-15/ADR-GR-5-001.md) · [ADR-DECISION-001](../2026-09-14/ADR-DECISION-001.md) · [`DECISION_APPROVAL_GOVERNANCE`](../../../../architecture/DECISION_APPROVAL_GOVERNANCE.md) · [`MULTIPLAYER_AI`](../../../../capabilities/architecture/MULTIPLAYER_AI.md) |

## Context

MP-4A (ADR-MP-005) froze collaborative Decision, Approval, and Governance ownership **before** the enterprise core finalized:

- canonical **Decision System** (`intergrax/contracts/decision_*`, lifecycle hosted by Execution),
- canonical **Execution Engine** (Task → Run → Attempt → Execution → Event; Nexus as **internal** orchestration strategy only),
- canonical **Governance / HITL** with **`ExecutionContinuationPort`** as the public continuation boundary,
- single **Evidence Plane** and **Diagnostics** interpretation authority.

Subsequent MP-4B–D work introduced `intergrax/contracts/decision.py`, `intergrax/contracts/approval.py`, and `intergrax/approval/` aligned with the **pre-rebase** MP-4 ownership table. That surface risks **duplicate Decision identity/lifecycle** and **duplicate Approval/HITL semantics** relative to the canonical core.

## Problem

Without formal supersession, Multiplayer could become a parallel source of truth for Decision, Approval/HITL, Execution continuation, Evidence, or Diagnostics — violating single-authority platform invariants.

## Decision

1. **Multiplayer collaborative plane** owns only:

   - WorkItem / Assignment (MP-2),
   - WorkArtifact / WorkArtifactVersion (MP-3),
   - future **collaborative bindings/projections** that **reference** canonical platform primitives (MP-4R4+),
   - MP-1 Principal / membership / delegation substrate (reuse).

2. **Multiplayer does not own:**

   - Decision identity, version, lifecycle, resolution, finalization, or authoritative outcome,
   - platform Approval/HITL lifecycle or execution authorization,
   - Execution pause/wait/resume or orchestration,
   - Evidence factual truth or diagnostic interpretation.

3. **Integration rule:** Multiplayer integrates with Decision, Governance/HITL, Execution, Evidence, and Diagnostics **only through stable platform contracts/ports** — never through Nexus internals or concrete runtime services as public integration surfaces.

4. **Legacy MP-4B–D code** remains in-tree **frozen** pending caller-proof convergence (MP-4R1–R6). MP-4R0 performs inventory, classification, and gates — **no production deletion**.

## Supersedes (partial) — ADR-MP-005

| ADR-MP-005 element | MP-4R0 disposition |
|--------------------|-------------------|
| Decision identity/lifecycle/outcome owned by **Decision Domain (MP-4)** | **Superseded** — canonical Decision System owns Decision truth; Multiplayer may bind/reference only |
| Approval request/lifecycle/human actions owned by **Approval Domain (MP-4)** | **Superseded** — canonical Decision human review + Governance/HITL own authorization semantics; MP-4 Approval contracts are duplicate pending MP-4R2 convergence |
| **Governance Domain (collaborative)** as separate MP-4 authority for execution authorization | **Superseded** for authorization truth — Governance/HITL owns authorization; collaborative governance reduces to linkage/posture references where needed |
| Anti-substitution invariants (Decision ≠ Artifact; Approval ≠ Artifact status; Decision ≠ HITL; approval/evidence ≠ execution authorization) | **Retained** — reassigned to canonical owners listed in MP-4R0 ownership table |
| Forbidden Nexus imports in MP-4 contracts/services | **Retained and extended** — no Multiplayer production Nexus dependency |
| MP-1 Principal/authority reuse | **Retained** |
| Dependency direction contracts-first | **Retained** — strengthened to contract/port-only integration |

ADR-MP-005 remains **historical record** of MP-4A; its ownership table for Decision/Approval lifecycle is **not** authoritative after MP-4R0.

## Consequences

- **MP-4A** → `SUPERSEDED_BY_MP4R0` (documentation status).
- **MP-4B / MP-4C** → `FROZEN_PENDING_CONVERGENCE` — classify `REPLACE_WITH_CANONICAL` / `REMOVE_AFTER_CALLER_PROOF`.
- **MP-4D** → `FROZEN_PENDING_AUTHORITY_REBASE` — Approval mutations must converge on Governance/HITL + MP-1 collaborative enforcement, not a second HITL engine.
- **MP-4E–H (old roadmap)** → cancelled/replaced by **MP-4R1–MP-4R8** roadmap.
- New architecture gates protect collaborative_work and future Multiplayer modules from Nexus coupling and duplicate authority classes.

## References

- Canonical Decision: `intergrax/contracts/decision_identity.py`, `decision_lifecycle.py`, `decision_human_review.py`, …
- Continuation: `intergrax/contracts/execution_continuation.py` (`ExecutionContinuationPort`)
- Governance: `docs/project/architecture/GOVERNED_EXECUTION.md`, ADR-GR-5-001
