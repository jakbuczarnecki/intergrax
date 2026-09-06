<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Multiplayer AI - Multi-layer Feature Plan

**Status:** **MP-1 — CLOSED / FINAL INDEPENDENT REVIEW PASS** — **MP-2 — IMPLEMENTATION IN PROGRESS** (ADR-MP-003 Accepted)
**Feature architecture (1:1):** [`../architecture/MULTIPLAYER_AI.md`](../architecture/MULTIPLAYER_AI.md)
**Primary anchor domain:** [`COLLABORATIVE_WORK`](../../architecture/COLLABORATIVE_WORK.md) (MP-1 ownership frozen - ADR-MP-001)
**Related domains:** `UNIFIED_EXECUTION_RUNTIME`, `ORCHESTRATION`, `UNIFIED_CONTEXT_LIFECYCLE`, `CONTEXT_ENGINEERING`, `MEMORY`, `RAG`, `RELIABILITY_FAILURE_AND_HITL`, `NEXUS_EXECUTION_FLOW`, `OBSERVABILITY`, `PROOF_RECEIPTS`, `INTEGRATIONS`, `AGENT_CONTRACTS_AND_ASSEMBLY`, `APPLICATION_HOSTING`
**Current active task:** **COLLAB-WORK-2D** (SQLite durability parity)
**Next task after MP-2 gate:** COLLAB-WORK-2D — SQLite durability parity

---

## Cursor read scope (token budget)

Do not read the whole repository.

Default read scope for Multiplayer AI work:

1. [`../architecture/MULTIPLAYER_AI.md`](../architecture/MULTIPLAYER_AI.md) - read-scope block + active `MP-*` phase summary only.
2. This file - read-scope block + **active MP-* section only**.
3. For **MP-1+** implementation: the affected domain architecture/plan pair for the current MP slice after bounded ownership check.
4. The minimal source files required by that domain plan item.

Do not create `docs/project/maintainers/plans/MULTIPLAYER_AI.md`. This is a multi-layer feature plan, not a domain-layer plan.

**Satellites:** none for MP-0. At most **one** `plan/satellites` file per session when a future phase needs a cross-domain register.

---

## Planning model

This file coordinates cross-layer delivery. Concrete implementation rows belong in owning **domain** plan files once a phase becomes actionable.

| Step | When |
|------|------|
| 1. Bounded ownership check | Before each `MP-1+` phase starts |
| 2. Domain architecture sync | Confirm or add MP-owned concepts in `docs/project/architecture/<DOMAIN>.md` |
| 3. Domain plan rows | Add concrete `MP-*` rows to `docs/project/maintainers/plans/<DOMAIN>.md` |
| 4. Implement | Smallest domain-owned slice |
| 5. Gate | Acceptance criteria for the active MP phase |

**MP-0 performs the documentation step only** (feature hub pair; README only
if its status requires synchronization). No domain plan edits.

### Capability labeling (mandatory in future MP rows)

| Label | Use when |
|-------|----------|
| **REUSED EXISTING CAPABILITY** | An existing domain or application row is integrated as-is or extended without becoming the MP primitive owner |
| **NEW CAPABILITY REQUIRED** | A platform Multiplayer primitive must be introduced; adjacent rows cannot substitute |

### Anti-substitution (plan-level)

The plan uses the same canonical anti-substitution semantics as the
architecture hub:

1. `LKW-CONVERSATION-CONTEXT-*` is not the MP-1 architectural anchor.
2. `CONVERSATION-CHANNEL-1` is not the foundation of Principal / Membership /
   Delegation.
3. Slack shared-conversation rows are not MP-2; Slack is a channel adapter,
   not Shared Work owner.
4. `LKW-HYBRID-ASK-*` is not MP-3; Hybrid Ask may use WorkArtifacts later,
   but WorkArtifact is a platform primitive.
5. Slack vertical rows are not MP-4; Slack may surface Decisions; Decision is
   platform-owned; HITL remains execution pause/resume.
6. `TOKEN-10E-*` is not Multiplayer MP-5 implementation; Multiplayer may
   reuse UCL/Token Optimization work.
7. Do not force-map existing adjacent rows to MP phases merely because
   concepts are related.
8. Existing capabilities are `REUSED EXISTING CAPABILITY`; missing primitives
   are `NEW CAPABILITY REQUIRED`. Never substitute adjacent rows for missing
   primitives.

**LKW note:** LKW is the first reference consumer (MP-7), not owner of Multiplayer mechanics.

---

## Roadmap sequence

```text
MP-0 (docs) → MP-1 (identity & authority) → MP-2 (shared work)
→ MP-3 (work artifacts) → MP-4 (decisions + HITL bridge)
→ MP-5 (context view) → MP-6 (activity & evidence)
→ MP-7 (LKW adoption) → MP-8 (agent directory & external agents)
→ MP-9 (advanced UX / notifications / optional realtime)
```

---

## MP-0 - Canonical architecture and implementation roadmap

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | READY_FOR_REVIEW |
| **Purpose** | Establish canonical Multiplayer AI architecture, MP-0…MP-9 roadmap, capability classification, anti-substitution rules, and provisional domain ownership map. |
| **Owning domain plan** | Feature plan + feature architecture hub only - **no domain plan edits in MP-0** |
| **Dependencies** | None |
| **Exact scope** | `docs/project/capabilities/architecture/MULTIPLAYER_AI.md`, `docs/project/capabilities/plan/MULTIPLAYER_AI.md`, `docs/project/capabilities/README.md` index row |
| **Explicit out of scope** | Satellites (unless genuinely required), domain architecture/plan edits, code, tests, scripts, MP-1+ implementation |
| **Architecture/ADR gate** | Canonical architecture rules, invariants, and ADR-MP-001…ADR-MP-007 register are present; all are marked REQUIRED BEFORE RELEVANT IMPLEMENTATION |
| **Pre-implementation domain-sync gate** | Not applicable to MP-0; no domain plan synchronization or implementation |
| **Acceptance criteria** | Feature pair exists 1:1; roadmap MP-0…MP-9 matches canonical sequence; canonical anti-substitution semantics, capability labels, authority boundaries, and required execution fields are present; provisional ownership uses `OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION` where not proven; no incorrect mapping of LKW/Slack/TOKEN/UCL rows to MP phases |
| **Expected proof/evidence** | Documentation review; phase/order and invariant cross-check; scope check; `git diff --check` |
| **User-visible outcome** | Reviewable Multiplayer roadmap; no runtime change |

---

## MP-1 - Principal, WorkspaceMembership and Delegation / effective authority

| Field | Value |
|-------|-------|
| **Priority** | P0 (after MP-0) |
| **Status** | **CLOSED / FINAL INDEPENDENT REVIEW PASS** |
| **Purpose** | Platform collaborative identity, workspace membership, and delegation with effective authority. |
| **Owning domain plan** | [`COLLABORATIVE_WORK.md`](../../maintainers/plans/COLLABORATIVE_WORK.md) - frozen by ADR-MP-001 / ADR-MP-002 |
| **Dependencies** | MP-0 accepted |
| **Exact scope** | Principal semantics; WorkspaceMembership; Delegation; effective-authority evaluation and fail-closed enforcement boundary |
| **REUSED EXISTING CAPABILITY** | Request-context principal propagation; LKW application principal docs (consumer reference only) |
| **NEW CAPABILITY REQUIRED** | Principal (collaborative), WorkspaceMembership, Delegation / effective authority |
| **Explicit out of scope** | Using `LKW-CONVERSATION-CONTEXT-*` or `CONVERSATION-CHANNEL-1` as MP-1 anchor |
| **Architecture/ADR gate** | ADR-MP-001 and ADR-MP-002 **Accepted**; COLLAB-WORK-1A…1J-R2 **CLOSED** per [`COLLABORATIVE_WORK.md`](../../maintainers/plans/COLLABORATIVE_WORK.md) |
| **Pre-implementation domain-sync gate** | **Done** — MP-1 final independent review pass closed |
| **User-visible outcome** | Governed multi-principal identity and authority model |
| **Acceptance criteria** | Meaningful mutations resolve an effective Principal; membership is explicit where required; delegation cannot amplify authority; tenant/workspace identifiers alone cannot authorize; agent authority remains distinct and failures are closed |
| **Expected proof/evidence** | Contract tests; isolation/authorization tests; fail-closed tests; delegation non-amplification tests; idempotency tests for membership/invite and delegation mutations |

---

## MP-2 - Shared Work: WorkItem, Assignment, lifecycle and concurrency

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | **IMPLEMENTATION IN PROGRESS** — COLLAB-WORK-2A APPROVED / CLOSED; COLLAB-WORK-2B APPROVED / CLOSED; COLLAB-WORK-2C APPROVED / CLOSED |
| **Purpose** | Platform-owned shared work primitives with lifecycle and concurrency semantics. |
| **Owning domain plan** | [`COLLABORATIVE_WORK.md`](../../maintainers/plans/COLLABORATIVE_WORK.md) — frozen by ADR-MP-003 |
| **Reused domain capabilities** | `ORCHESTRATION` (explicit bridge), `UNIFIED_EXECUTION_RUNTIME` / NEXUS (execution identities), `BACKGROUND_TASKS` (associated execution), `OBSERVABILITY` / `PROOF_RECEIPTS` (provenance) |
| **Dependencies** | MP-1 accepted |
| **Exact scope** | WorkItem; Assignment; collaborative lifecycle; explicit optimistic concurrency and idempotency semantics |
| **REUSED EXISTING CAPABILITY** | MP-1 authority, repository concurrency/idempotency patterns; Nexus task/run identities as execution references only |
| **NEW CAPABILITY REQUIRED** | WorkItem, Assignment, shared-work lifecycle and concurrency |
| **Explicit out of scope** | Slack shared-conversation or any channel adapter as Shared Work owner; WorkArtifact (MP-3); Decision (MP-4); Activity (MP-6) |
| **Architecture/ADR gate** | WorkItem/Task separation and concurrency direction accepted; **ADR-MP-003 Accepted** |
| **Pre-implementation domain-sync gate** | **Done** — bounded ownership check closed; COLLAB-WORK-2A…2G rows registered |
| **User-visible outcome** | Addressable shared work units assignable to principals and agents |
| **Acceptance criteria** | WorkItems are durable and independently addressable; WorkItemState is not TaskState; multiple tasks/runs may relate to one WorkItem; stale authoritative mutations fail explicitly; Nexus does not own WorkItem lifecycle |
| **Expected proof/evidence** | Contract tests; lifecycle tests; assignment authorization tests; concurrency/conflict tests; idempotency tests; provenance linkage to real `task_id`/`run_id` |
| **Next implementation row** | **COLLAB-WORK-2D** |

---

## MP-3 - WorkArtifact and WorkArtifactVersion

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Durable collaborative outputs with versioning and provenance. |
| **Likely owning domain plans** | `UNIFIED_CONTEXT_LIFECYCLE.md`, `PROOF_RECEIPTS.md`, `MEMORY.md` - **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-2 accepted (may overlap MP-1 for authority on artifacts) |
| **Exact scope** | WorkArtifact; authoritative WorkArtifactVersion; publication, version pointer, lineage, and provenance |
| **REUSED EXISTING CAPABILITY** | UCL artifact lifecycle patterns; receipt/provenance models |
| **NEW CAPABILITY REQUIRED** | WorkArtifact, WorkArtifactVersion |
| **Explicit out of scope** | `LKW-HYBRID-ASK-*` as WorkArtifact owner |
| **Architecture/ADR gate** | WorkArtifact is separated from UCL OptimizationArtifact and authoritative version semantics are accepted; ADR-MP-004 completed |
| **Pre-implementation domain-sync gate** | Bounded ownership check → domain architecture/plan sync with MP-3 rows |
| **User-visible outcome** | Versioned collaborative artifacts with lineage |
| **Acceptance criteria** | A WorkArtifactVersion is the authoritative collaborative output; versions remain addressable after executions end; publication preserves principal/work/execution lineage; current-version updates detect stale writes |
| **Expected proof/evidence** | Contract tests; authorization/isolation tests; version/concurrency tests; idempotent publication tests; provenance/evidence integration proof |

---

## MP-4 - Decision / DecisionResponse or Approval semantics + HITL bridge

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Collaborative decision and approval semantics with explicit bridge to Nexus HITL when execution must pause. |
| **Likely owning domain plans** | `RELIABILITY_FAILURE_AND_HITL.md`, `NEXUS_EXECUTION_FLOW.md`, `UNIFIED_EXECUTION_RUNTIME.md` - **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-1 accepted; MP-2 recommended |
| **Exact scope** | Decision; DecisionResponse/Approval semantics; policy-gated response; explicit bridge to existing Nexus HITL pause/resume |
| **REUSED EXISTING CAPABILITY** | Nexus HITL pause/resume; policy evaluation hooks |
| **NEW CAPABILITY REQUIRED** | Decision / DecisionResponse (or Approval) collaborative primitive; HITL bridge contract |
| **Explicit out of scope** | Slack vertical rows as Decision owner; conflating Decision records with HITL machinery |
| **Architecture/ADR gate** | Decision/HITL separation, approval non-authorization, and explicit bridge semantics accepted; ADR-MP-004 completed |
| **Pre-implementation domain-sync gate** | Bounded ownership check → domain architecture/plan sync with MP-4 rows |
| **User-visible outcome** | Explicit collaborative approvals that can pause and resume governed execution |
| **Acceptance criteria** | A Decision can exist without an active task; a pause bridge uses existing HITL only; responses are principal- and policy-authorized; approval/evidence alone does not authorize execution; decision responses are idempotent |
| **Expected proof/evidence** | Contract tests; authorization/isolation tests; HITL bridge integration proof; decision concurrency tests; idempotency tests; provenance/evidence linkage |

---

## MP-5 - Principal-scoped ContextView

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Principal-scoped context view composing UCL, Context Engineering, Memory, and Knowledge. |
| **Likely owning domain plans** | `UNIFIED_CONTEXT_LIFECYCLE.md`, `CONTEXT_ENGINEERING.md`, `MEMORY.md`, `RAG.md` - **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-1 accepted |
| **Exact scope** | Principal-scoped ContextView policy and composition over UCL, Context Engineering, Memory, and Knowledge |
| **REUSED EXISTING CAPABILITY** | UCL, Context Engineering, Memory, RAG/Knowledge, Token Optimization (`TOKEN-10E-*` etc.) |
| **NEW CAPABILITY REQUIRED** | ContextView contract and principal-scope composition policy |
| **Explicit out of scope** | Relabeling `TOKEN-10E-*` or UCL rows as MP-5 implementation |
| **Architecture/ADR gate** | Principal-specific context, private/shared memory boundary, least-context external access, and UCL ownership accepted; ADR-MP-006 completed |
| **Pre-implementation domain-sync gate** | Bounded ownership check → domain architecture/plan sync with MP-5 rows |
| **User-visible outcome** | Membership-aware context visible to each collaborative principal |
| **Acceptance criteria** | Context visibility is principal-specific and membership/policy-aware; private memory is not automatically shared; shared state is not automatically model context; external agents receive minimum required context and resources; UCL remains lifecycle authority |
| **Expected proof/evidence** | Context contract tests; isolation/authorization tests; private-to-shared promotion tests; least-context external-agent tests; integration and provenance proof |

---

## MP-6 - Collaborative Activity + provenance / evidence linkage

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Collaborative activity stream linked to provenance and evidence. |
| **Likely owning domain plans** | `OBSERVABILITY.md`, `PROOF_RECEIPTS.md`, `UNIFIED_EXECUTION_RUNTIME.md` - **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-2, MP-3, MP-4 recommended |
| **Exact scope** | Collaborative Activity projection and stable linkage to runtime Trace/Evidence without authority transfer |
| **REUSED EXISTING CAPABILITY** | Traces, receipts, attempt ledger, existing provenance fields |
| **NEW CAPABILITY REQUIRED** | Activity model with collaborative semantics and evidence linkage |
| **Explicit out of scope** | Activity feeds, indexes, or projections as permission, decision, artifact, or lifecycle authorities |
| **Architecture/ADR gate** | Activity/Runtime Trace separation and evidence lineage accepted; relevant ADR register decisions completed |
| **Pre-implementation domain-sync gate** | Bounded ownership check → domain architecture/plan sync with MP-6 rows |
| **User-visible outcome** | Auditable collaborative activity tied to evidence |
| **Acceptance criteria** | Activity is product-facing and linked to runtime evidence; runtime Trace/Evidence remains execution truth; projections cannot authorize or mutate authoritative state; lineage remains queryable through real execution identities |
| **Expected proof/evidence** | Contract tests; projection non-authority tests; provenance/evidence integration proof; authorization/isolation tests; consistency/rebuild proof |

---

## MP-7 - LKW reference-product adoption

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Adopt platform Multiplayer primitives in LKW as first reference consumer. |
| **Likely owning plans** | Tier-3 LKW application implementation plan (consumer); platform primitives in Tier-0/Tier-1 domain plans - **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-1…MP-6 platform primitives accepted for the adopted subset |
| **Exact scope** | LKW consumer integration for an explicitly selected subset of platform Multiplayer primitives |
| **REUSED EXISTING CAPABILITY** | Prior LKW conversation, Ask, channel capabilities until explicitly integrated |
| **NEW CAPABILITY REQUIRED** | LKW integration rows per adopted primitive (consumer-side only) |
| **Explicit out of scope** | Transferring platform primitive ownership to LKW |
| **Architecture/ADR gate** | Platform ownership and non-migration of current LKW Workspace are accepted; ADR-MP-005 completed |
| **Pre-implementation domain-sync gate** | Bounded ownership check → LKW plan sync; no substitution of LKW-local rows for missing primitives |
| **User-visible outcome** | LKW demonstrates end-to-end Multiplayer on platform contracts |
| **Acceptance criteria** | LKW consumes the selected platform contracts without redefining them; current LKW Workspace is not moved by this phase; ownership and authority boundaries remain enforceable end to end |
| **Expected proof/evidence** | Integration proof; consumer contract tests; isolation/authorization tests; provenance/evidence proof; regression proof for existing LKW behavior |

---

## MP-8 - AgentDirectory / external-agent interoperability

| Field | Value |
|-------|-------|
| **Priority** | P3 |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Agent discovery/registry, ExternalWork reuse, future A2A adapter at integration boundary. |
| **Likely owning domain plans** | `AGENT_CONTRACTS_AND_ASSEMBLY.md`, `INTEGRATIONS.md`, `UNIFIED_EXECUTION_RUNTIME.md` - **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-1, MP-2 recommended |
| **Exact scope** | AgentDirectory identity, capability/trust/discovery direction, governed external participation, and adapter boundary |
| **REUSED EXISTING CAPABILITY** | `ExternalWorkIntegration`, governed external work host lifecycle |
| **NEW CAPABILITY REQUIRED** | AgentDirectory, interoperability policy, external-agent adapter boundary |
| **Explicit out of scope** | Transport types (HTTP/A2A/REST) in core Multiplayer contracts |
| **Architecture/ADR gate** | AgentDirectory/AgentRegistry separation, least-context/authority boundary, and adapter-only transport policy accepted; ADR-MP-007 completed |
| **Pre-implementation domain-sync gate** | Bounded ownership check → domain architecture/plan sync with MP-8 rows |
| **User-visible outcome** | Governed participation of internal and external agents in shared work |
| **Acceptance criteria** | Collaborative/external agent identity is distinct from execution registry identity; external participation is policy-gated; minimum context/resources/authority are enforced; transport/provider types remain outside canonical contracts |
| **Expected proof/evidence** | Contract tests; trust and authorization tests; least-authority external-agent tests; adapter integration proof; idempotency and provenance/evidence proof |

---

## MP-9 - Advanced collaborative UX, notifications, subscriptions, optional realtime

| Field | Value |
|-------|-------|
| **Priority** | P3 |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Product-facing collaboration UX, notifications/subscriptions, optional realtime or generative UI when justified. |
| **Likely owning domain plans** | `APPLICATION_HOSTING.md`, `INTEGRATIONS.md` - **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-1…MP-8 as needed per UX slice |
| **Exact scope** | Product-facing subscriptions, notifications, and optional realtime/generative UI justified by prior Multiplayer primitives |
| **REUSED EXISTING CAPABILITY** | `notification_channel`, `conversation_channel`, hosting presentation |
| **NEW CAPABILITY REQUIRED** | Subscription model tied to Activity/WorkItem/Decision where generic notifications are insufficient |
| **Explicit out of scope** | Using UX/realtime as substitute for missing MP-1…MP-6 primitives |
| **Architecture/ADR gate** | Justified UX slice preserves platform ownership, policy gating, and projection non-authority; relevant ADR register decisions completed |
| **Pre-implementation domain-sync gate** | Justification review + bounded ownership check → domain plan sync |
| **User-visible outcome** | Optional rich collaboration surfaces without owning core primitives |
| **Acceptance criteria** | Each surface consumes authoritative Multiplayer events and respects principal/workspace policy; no notification, subscription, or realtime projection becomes an authority source; optional realtime is justified by a bounded use case |
| **Expected proof/evidence** | Integration proof; authorization/isolation tests; subscription/idempotency tests; projection non-authority tests; user-visible workflow proof |

---

## Domain plan row template (MP-1+ only)

When adding rows to an owning domain plan after ownership confirmation:

```text
MP-<n>-<slice> - <title>
  Classification: NEW CAPABILITY REQUIRED | REUSED EXISTING CAPABILITY
  Feature coordination: docs/project/capabilities/plan/MULTIPLAYER_AI.md §MP-<n>
  Owning domain: <DOMAIN>
  Dependencies: <prior MP or domain rows>
  Out of scope: <explicit anti-substitution items>
  Acceptance: <testable criteria>
```

**Not used in MP-0.**
