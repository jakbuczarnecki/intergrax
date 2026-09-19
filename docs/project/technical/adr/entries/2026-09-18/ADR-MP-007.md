# ADR-MP-007: Collaborative Activity & Provenance ownership and contract boundary

| Field | Value |
|-------|-------|
| **Status** | Accepted â€” **MP-6A â€” CLOSED / RECERTIFIED**; **MP-6B â€” CLOSED / RECERTIFIED**; **MP-6C â€” CLOSED / RECERTIFIED**; **MP-6D — CLOSED / CERTIFIED**; **MP-6E — NEXT** (subject to independent audit) |
| **Date** | 2026-09-18 |
| **Deciders** | Intergrax platform architecture (MP-6A ownership freeze) |
| **Related** | [`architecture/COLLABORATIVE_WORK.md`](../../../../architecture/COLLABORATIVE_WORK.md) Â· [`plan/COLLABORATIVE_WORK.md`](../../../../maintainers/plans/COLLABORATIVE_WORK.md) Â· [`capabilities/architecture/MULTIPLAYER_AI.md`](../../../../capabilities/architecture/MULTIPLAYER_AI.md) Â· [ADR-MP-006](../2026-09-17/ADR-MP-006.md) Â· [`DECISION_APPROVAL_GOVERNANCE.md`](../../../../architecture/DECISION_APPROVAL_GOVERNANCE.md) Â· [`OBSERVABILITY.md`](../../../../architecture/OBSERVABILITY.md) Â· [`PROOF_RECEIPTS.md`](../../../../architecture/PROOF_RECEIPTS.md) |

## Context

Multiplayer AI MP-6 requires **Collaborative Activity & Provenance**: append-oriented, typed semantic history of meaningful collaborative actions with reference-only evidence links â€” without copying domain payloads or coupling to a specific event store, database, or broker.

MP-0 provisionally listed `OBSERVABILITY`, `PROOF_RECEIPTS`, and `UNIFIED_EXECUTION_RUNTIME` as likely MP-6 owners. Repository canon already states **MP-INV-25** (`Collaborative Activity != Runtime Trace`), semantic channel separation in COLLABORATIVE_WORK, and OBSERVABILITY / PROOF_RECEIPTS ownership of execution evidence and proof receipts respectively.

## Decision (summary)

**COLLABORATIVE_WORK** (Multiplayer **MP-6**) is the single semantic owner of collaborative activity records, actor/target attribution, ordering semantics for activity history, activity classification, correlation identifiers, and query contracts. It **references** â€” does not own â€” run/step trace, Decision/Approval semantics, artifact/memory/UCL payloads, authorization source of truth, `RuntimeEvent`, `AgentRunTrace`, and `ProofReceipt` bodies.

**Integration:** source domains publish `CollaborativeActivityPublication` via `CollaborativeActivityPublicationPort` (neutral contract in `intergrax/contracts/collaborative_activity.py`); MP-6 resolves policy at the ingestion boundary, builds CollaborativeActivityAppendIntent, and appends via CollaborativeActivityAppendStore.append_idempotent(intent); read via `CollaborativeActivityReadPort` with authority resolved outside the store (MP-6E). **Forbidden:** repository wrap inference, log parsing, `dict` payload authority, source â†’ store implementation imports.

**Idempotency (MP-6A-C1):** `ActivityIdempotencyKey(tenant_id, workspace_id, source, source_stable_id, activity_type)` where `source` is `CollaborativeActivitySourceId` and `activity_type` is `CollaborativeActivityTypeId`; `activity_id = mint_collaborative_activity_id(...)` over frozen `activity-id/v1` length-prefixed hash material (SHA-256 truncated to 32 hex). Key tenant/workspace must match publication/activity scope. No global idempotency without proven global source-stable uniqueness.

**Activity type extensibility:** namespaced plugin types (`CollaborativeActivityTypeId.for_extension`); platform built-ins (`CollaborativeActivityBuiltinType`); reserved `platform` / `intergrax` namespaces for platform ownership; no closed enum as sole extension mechanism.

**Source producer identity:** namespaced `CollaborativeActivitySourceId` (built-ins via `CollaborativeActivityBuiltinSource`); external plugins use distinct namespaces â€” no generic `PLUGIN` bucket collapsing producers.

**Ordering:** no global total order. **Event time** (`occurred_at`, source-owned) and **materialization time** (`recorded_at`, assigned at durable append-store acceptance on materialized `CollaborativeActivity` only â€” not on `CollaborativeActivityPublication`) are distinct from **pagination continuation**, which uses opaque provider-neutral `CollaborativeActivityPageCursor` (append/snapshot position). Display/event-time ordering may use `(occurred_at, activity_id)` tie-break; forward reads must not use `occurred_at` alone as a lossy watermark under at-least-once / late arrival. **`append_position`** on materialized activities is assigned **atomically at the persistence append boundary** (`CollaborativeActivityAppendStore.append_idempotent(intent)`) for **new** activities only â€” by the configured append-store implementation under the platform append contract (not producers). **`append_position` is monotonic and unique within `(tenant_id, workspace_id)`** (not global across tenants/workspaces). **Duplicate idempotency key** delivery returns the **original** materialized activity **without allocating a new append position**. **Contiguous gapless sequence is not required.** Append-only with `platform.activity.correction` supersession.

**Delegation:** canonical delegation attribution on `CollaborativeActivityActorRef` (`delegation_id` â†” `delegator_principal_id` paired); no parallel root `authority_delegation_id`.

**Anti-substitution:** `Collaborative Activity != Runtime Trace`; Activity â‰  observability telemetry; Activity â‰  proof receipt storage.

## Status

**MP-6A-C1 â€” CLOSED** (correction applied; subject to independent audit). **MP-6A â€” CLOSED / RECERTIFIED**. **MP-6 ownership â€” FROZEN**. **MP-6B â€” CLOSED / RECERTIFIED**. **MP-6C â€” CLOSED / RECERTIFIED**. **MP-6D — CLOSED / CERTIFIED.** **MP-6E — NEXT.**

See [`COLLABORATIVE_WORK.md`](../../../../architecture/COLLABORATIVE_WORK.md) Â§ Collaborative Activity (MP-6) for diagrams, threat model, and roadmap.
