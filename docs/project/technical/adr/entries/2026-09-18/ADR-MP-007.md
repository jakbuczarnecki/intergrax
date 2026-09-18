# ADR-MP-007: Collaborative Activity & Provenance ownership and contract boundary

| Field | Value |
|-------|-------|
| **Status** | Accepted — architecture and ownership gate only; MP-6B+ runtime persistence **NOT STARTED** |
| **Date** | 2026-09-18 |
| **Deciders** | Intergrax platform architecture (MP-6A ownership freeze) |
| **Related** | [`architecture/COLLABORATIVE_WORK.md`](../../../../architecture/COLLABORATIVE_WORK.md) · [`plan/COLLABORATIVE_WORK.md`](../../../../maintainers/plans/COLLABORATIVE_WORK.md) · [`capabilities/architecture/MULTIPLAYER_AI.md`](../../../../capabilities/architecture/MULTIPLAYER_AI.md) · [ADR-MP-006](../2026-09-17/ADR-MP-006.md) · [`DECISION_APPROVAL_GOVERNANCE.md`](../../../../architecture/DECISION_APPROVAL_GOVERNANCE.md) · [`OBSERVABILITY.md`](../../../../architecture/OBSERVABILITY.md) · [`PROOF_RECEIPTS.md`](../../../../architecture/PROOF_RECEIPTS.md) |

## Context

Multiplayer AI MP-6 requires **Collaborative Activity & Provenance**: append-oriented, typed semantic history of meaningful collaborative actions with reference-only evidence links — without copying domain payloads or coupling to a specific event store, database, or broker.

MP-0 provisionally listed `OBSERVABILITY`, `PROOF_RECEIPTS`, and `UNIFIED_EXECUTION_RUNTIME` as likely MP-6 owners. Repository canon already states **MP-INV-25** (`Collaborative Activity != Runtime Trace`), semantic channel separation in COLLABORATIVE_WORK, and OBSERVABILITY / PROOF_RECEIPTS ownership of execution evidence and proof receipts respectively.

## Decision (summary)

**COLLABORATIVE_WORK** (Multiplayer **MP-6**) is the single semantic owner of collaborative activity records, actor/target attribution, ordering semantics for activity history, activity classification, correlation identifiers, and query contracts. It **references** — does not own — run/step trace, Decision/Approval semantics, artifact/memory/UCL payloads, authorization source of truth, `RuntimeEvent`, `AgentRunTrace`, and `ProofReceipt` bodies.

**Integration:** source domains publish `CollaborativeActivityPublication` via `CollaborativeActivityPublicationPort` (neutral contract in `intergrax/contracts/collaborative_activity.py`); MP-6 implementation appends idempotently; read via `CollaborativeActivityReadPort`. **Forbidden:** repository wrap inference, log parsing, `dict` payload authority, source → store implementation imports.

**Idempotency:** `ActivityIdempotencyKey(source_domain, source_stable_id, activity_type)`; `activity_id = mint_collaborative_activity_id(...)`.

**Ordering:** no global total order; per-workspace cursor timeline on `(occurred_at, activity_id)`; append-only with correction via `ACTIVITY_CORRECTION`.

**Anti-substitution:** `Collaborative Activity != Runtime Trace`; Activity ≠ observability telemetry; Activity ≠ proof receipt storage.

## Status

**MP-6A — APPROVED / CLOSED**. **MP-6 ownership — FROZEN**. **MP-6B — NEXT.**

See [`COLLABORATIVE_WORK.md`](../../../../architecture/COLLABORATIVE_WORK.md) § Collaborative Activity (MP-6) for diagrams, threat model, and roadmap.
