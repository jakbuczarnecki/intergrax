<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Multiplayer AI - Multi-layer Feature Plan

**Status:** **MP-1 — CLOSED** — **MP-2 — APPROVED / CLOSED** — **MP-3 — ENTERPRISE CERTIFIED / CLOSED** — **MP-3A…MP-3H — APPROVED / CLOSED** — **MP-4 implementation — FORMALLY CLOSED** (MP-4R0…MP-4R8 **CLOSED**; ADR-MP-009) — **MP-4 documentation certification — CLOSED** (MP-4D1–D8) — **MP-5A — APPROVED / CLOSED** — **MP-5B — APPROVED / CLOSED** — **MP-5C — APPROVED / CLOSED** — **MP-5 ownership — FROZEN**
**Feature architecture (1:1):** [`../architecture/MULTIPLAYER_AI.md`](../architecture/MULTIPLAYER_AI.md)
**Primary anchor domain:** [`COLLABORATIVE_WORK`](../../architecture/COLLABORATIVE_WORK.md) (MP-1…MP-3) · [`DECISION_APPROVAL_GOVERNANCE`](../../architecture/DECISION_APPROVAL_GOVERNANCE.md) (MP-4R)
**Related domains:** `UNIFIED_EXECUTION_RUNTIME`, `ORCHESTRATION`, `UNIFIED_CONTEXT_LIFECYCLE`, `CONTEXT_ENGINEERING`, `MEMORY`, `RAG`, `RELIABILITY_FAILURE_AND_HITL`, `NEXUS_EXECUTION_FLOW`, `GOVERNED_EXECUTION`, `OBSERVABILITY`, `PROOF_RECEIPTS`, `INTEGRATIONS`, `AGENT_CONTRACTS_AND_ASSEMBLY`, `APPLICATION_HOSTING`
**Current active task:** **MP-6 — NEXT** (active slice **MP-6F — NEXT**; **MP-6E — CLOSED / RECERTIFIED**; **MP-6D — CLOSED / RECERTIFIED**; **MP-6C — CLOSED**; **MP-6B — CLOSED**). **MP-6A — CLOSED / RECERTIFIED** (**MP-6A-C1**) (**MP-6 ownership — FROZEN**, ADR-MP-007). **MP-5 — ENTERPRISE CERTIFIED / CLOSED** (**MP-5H-D1 — CLOSED / CERTIFIED**; historical **MP-5H — CLOSED / FINAL CERTIFICATION PASSED** at `d0aee066` — [`MP-5H-D1_POST_B4_DELTA_ENTERPRISE_RECERTIFICATION.md`](../../maintainers/qualification/MP-5H-D1_POST_B4_DELTA_ENTERPRISE_RECERTIFICATION.md), [`MP-5H_FINAL_ENTERPRISE_CERTIFICATION.md`](../../maintainers/qualification/MP-5H_FINAL_ENTERPRISE_CERTIFICATION.md)). **MP-5G — ENTERPRISE E2E / ISOLATION QUALIFICATION CERTIFIED / CLOSED** (`tests/unit/collaborative_work/test_mp5g_context_view_e2e_qualification.py`, `mp5g_e2e_harness.py`). **MP-5F — ENTERPRISE SOURCE INTEGRATION CERTIFIED / CLOSED** (**MP-5F-B1…B4 — CLOSED**; **MP-5F-B5 — CLOSED** — `intergrax/collaborative_work/context_view_source_adapters.py`, `context_view_source_wiring.py`). MP-5E **`ContextViewComposer`** / **`DefaultContextViewComposer`** — **APPROVED / CLOSED** (`intergrax/contracts/context_view_composition.py`).
**Previous:** **MP-5C — APPROVED / CLOSED** — principal visibility policy (`intergrax/contracts/context_view_visibility_policy.py`)

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
5. Slack vertical rows are not **MP-4R**; Slack may surface canonical Decisions
   but never owns them. **Decision System** owns Decision. **Governance/HITL**
   owns authorization / human approval semantics. **`ExecutionContinuationPort`**
   is the canonical public execution continuation boundary. Nexus remains
   **internal** to the Execution Engine orchestration strategy only; Multiplayer
   may bind/project only.
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
→ MP-3 (work artifacts) → MP-4R (canonical Decision / Governance / Execution integration) — **implementation CLOSED**
→ MP-4D (enterprise documentation & proof closure — active)
→ MP-5 (context view) → MP-6 (activity & evidence)
→ MP-7 (LKW adoption) → MP-8 (agent directory & external agents)
→ MP-9 (advanced UX / notifications / optional realtime)
```

**Historical (superseded):** pre-rebase **MP-4** (decisions + public Nexus HITL bridge) — replaced by **MP-4R** and ADR-MP-009.

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
| **Status** | **APPROVED / CLOSED** — COLLAB-WORK-2A…2G APPROVED / CLOSED; ADR-MP-003 implementation **COMPLETE** |
| **Purpose** | Platform-owned shared work primitives with lifecycle and concurrency semantics. |
| **Owning domain plan** | [`COLLABORATIVE_WORK.md`](../../maintainers/plans/COLLABORATIVE_WORK.md) — frozen by ADR-MP-003 |
| **Reused domain capabilities** | `ORCHESTRATION` (explicit WorkItem context consumption), `UNIFIED_EXECUTION_RUNTIME` (canonical execution identities), `NEXUS` (internal orchestration only — not Collaborative Work contract dependency), `BACKGROUND_TASKS` (associated execution), `OBSERVABILITY` / `PROOF_RECEIPTS` (provenance) |
| **Dependencies** | MP-1 accepted |
| **Exact scope** | WorkItem; Assignment; collaborative lifecycle; explicit optimistic concurrency and idempotency semantics |
| **REUSED EXISTING CAPABILITY** | MP-1 authority, repository concurrency/idempotency patterns; canonical `TaskId`/`RunId`/`AttemptId`/`ExecutionId` via neutral `ExecutionProvenanceRef` |
| **NEW CAPABILITY REQUIRED** | WorkItem, Assignment, shared-work lifecycle and concurrency |
| **Explicit out of scope** | Slack shared-conversation or any channel adapter as Shared Work owner; WorkArtifact (MP-3); Decision (MP-4); Activity (MP-6) |
| **Architecture/ADR gate** | WorkItem/Task separation and concurrency direction accepted; **ADR-MP-003 Accepted** |
| **Pre-implementation domain-sync gate** | **Done** — bounded ownership check closed; COLLAB-WORK-2A…2G rows registered |
| **User-visible outcome** | Addressable shared work units assignable to principals and agents |
| **Acceptance criteria** | WorkItems are durable and independently addressable; WorkItemState is not TaskState; multiple tasks/runs may relate to one WorkItem; stale authoritative mutations fail explicitly; Nexus does not own WorkItem lifecycle |
| **Expected proof/evidence** | Contract tests; lifecycle tests; assignment authorization tests; concurrency/conflict tests; idempotency tests; provenance linkage to real four-part `ExecutionProvenanceRef` |
| **Next implementation row** | **MP-5F — NEXT** (MP-5E **APPROVED / CLOSED**) |

---

## MP-3 - WorkArtifact and WorkArtifactVersion

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | **ENTERPRISE CERTIFIED / CLOSED** — ADR-MP-004 **Accepted**; **architecture decomposition — APPROVED / CLOSED**; **MP-3A…MP-3H — APPROVED / CLOSED** |
| **Purpose** | Durable collaborative outputs with versioning and provenance. |
| **Owning domain plan** | [`COLLABORATIVE_WORK.md`](../../maintainers/plans/COLLABORATIVE_WORK.md) — frozen by ADR-MP-004; full slice rows § COLLAB-WORK-3 |
| **Reused domain capabilities** | UCL (consumption only); Memory indexing/retrieval; Proof Receipts attestation; MP-1 authority; MP-2 repository/CAS/idempotency patterns; optional `ExecutionProvenanceRef` |
| **Dependencies** | MP-2 accepted (MP-1 authority on artifacts) |
| **Exact scope** | WorkArtifact; authoritative immutable WorkArtifactVersion; publication; CAS-protected current-version pointer; `ArtifactPublicationRepository` transactional boundary for **both** atomic initial creation (`create_artifact_with_initial_version(...)`) and subsequent publication (`publish_version(...)`); lineage; principal provenance; optional execution provenance; neutral `ArtifactContentRef` |
| **REUSED EXISTING CAPABILITY** | MP-1 effective authority; MP-2 persistence/concurrency patterns; neutral execution provenance contracts |
| **NEW CAPABILITY REQUIRED** | WorkArtifact, WorkArtifactVersion runtime contracts and services (slices MP-3A…MP-3H) |
| **Explicit out of scope** | `LKW-HYBRID-ASK-*` as WorkArtifact owner; UCL/Memory/Proof Receipts as owners; MP-4 Decision state; MP-6 Activity projection; MP-3B+ runtime until scheduled |
| **Architecture/ADR gate** | **Done** — WorkArtifact separated from UCL OptimizationArtifact; immutable version + CAS current pointer accepted; **ADR-MP-004 Accepted** |
| **Pre-implementation domain-sync gate** | **Done** — MP-3 bounded ownership check closed; decomposition **APPROVED / CLOSED**; MP-3A…MP-3H rows registered |
| **User-visible outcome** | Versioned collaborative artifacts with lineage |
| **Acceptance criteria** | A WorkArtifactVersion is the authoritative collaborative output; versions remain addressable after executions end; publication preserves principal/work/execution lineage; current-version updates detect stale writes; atomic initial creation and subsequent publication via dedicated port (no dangling `current_version_id`, no orphan initial version) |
| **Expected proof/evidence** | Contract tests; authorization/isolation tests; version/concurrency tests; idempotent initial create tests; idempotent publication tests; cross-process publication proof (MP-3E); provenance/evidence integration proof (MP-3G) |
| **Next implementation row** | **MP-5F — NEXT** (**MP-5E — APPROVED / CLOSED**) |

### MP-3 architectural implementation slices

Decomposition **APPROVED / CLOSED** — canonical rows in [`COLLABORATIVE_WORK.md`](../../maintainers/plans/COLLABORATIVE_WORK.md) § COLLAB-WORK-3.

| Slice | Scope | Status |
|-------|-------|--------|
| MP-3A | Contracts + invariants + `ArtifactContentRef` | APPROVED / CLOSED |
| MP-3B | Ports + in-memory + `ArtifactPublicationRepository` (atomic initial create + publish) | APPROVED / CLOSED |
| MP-3C | Publication service + MP-1 authority | APPROVED / CLOSED |
| MP-3D | SQLite transactional persistence | APPROVED / CLOSED |
| MP-3E | PostgreSQL + qualification | APPROVED / CLOSED |
| MP-3F | Content storage adapters (after 3E) | APPROVED / CLOSED |
| MP-3G | Execution/evidence integration | APPROVED / CLOSED |
| MP-3H | Final enterprise certification | APPROVED / CLOSED |

---

## MP-4R — Decision integration (canonical core rebase)

**Domain plan (1:1):** [`DECISION_APPROVAL_GOVERNANCE.md`](../../maintainers/plans/DECISION_APPROVAL_GOVERNANCE.md) · **ADR:** [ADR-MP-009](../../technical/adr/entries/2026-09-15/ADR-MP-009.md)

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | **MP-4 — FORMALLY CLOSED** · **MP-4R0…MP-4R8 CLOSED**; MP-4B/MP-4C/MP-4D **RETIRED** |
| **Purpose** | Multiplayer **bindings/projections** over canonical Decision, Governance/HITL, Execution continuation, Evidence, and Diagnostics — no duplicate authorities |
| **Owning domain** | [`DECISION_APPROVAL_GOVERNANCE`](../../architecture/DECISION_APPROVAL_GOVERNANCE.md) + Collaborative Work for work primitives |
| **Dependencies** | MP-1 **CLOSED**; MP-2 **CLOSED**; MP-3 ownership **FROZEN**; canonical Decision + GR-5 continuation |
| **REUSED EXISTING CAPABILITY** | Decision System; Decision human review; Governance/HITL; `ExecutionContinuationPort`; Evidence Plane; Diagnostics |
| **NEW CAPABILITY REQUIRED (delivered)** | Collaborative Decision binding / projection — **MP-4R4 CLOSED** |
| **Explicit out of scope (SUPERSEDED / HISTORICAL)** | Legacy MP-4 program rows treating Decision, Approval, `DecisionResponse`, or public Nexus HITL bridge as **NEW Multiplayer primitives** |
| **Architecture/ADR gate** | ADR-MP-009 **Accepted** at MP-4R0 |
| **User-visible outcome** | Collaborative work associated with canonical decisions and governed execution without parallel decision/approval stores |

### MP-4R slice status

| Slice | Scope | Status |
|-------|-------|--------|
| **MP-4R0** | Core rebase & supersession gate | **CLOSED** |
| MP-4R1 | Decision contract convergence | **CLOSED** |
| MP-4R2 | Human review / Approval convergence | **CLOSED** |
| MP-4R3 | Execution continuation integration | **CLOSED** |
| MP-4R4 | Collaborative decision binding | **CLOSED** |
| MP-4R5 | Evidence Plane adoption | **CLOSED** |
| MP-4R6 | Legacy removal & migration | **CLOSED** |
| MP-4R7 | Enterprise integration qualification | **CLOSED** |
| MP-4R8 | Final closure audit | **CLOSED** |

### MP-4D — Enterprise documentation & proof closure

**Does not reopen MP-4 implementation.** Canonical architecture: [`DECISION_APPROVAL_GOVERNANCE`](../../architecture/DECISION_APPROVAL_GOVERNANCE.md).

| Stage | Status | Summary |
|-------|--------|---------|
| **MP-4D1** | **CLOSED** | Synchronize documentation state with closed implementation |
| **MP-4D2** | **CLOSED** | Consolidate canonical architecture entry point |
| **MP-4D3** | **CLOSED** | Visual architecture layer (Mermaid in architecture SSOT) |
| **MP-4D4** | **CLOSED** | E2E proof / invariant-to-test matrix |
| **MP-4D5** | **CLOSED** | Provider/persistence qualification boundaries |
| **MP-4D6** | **CLOSED** | Enterprise boundary & pluginability certification |
| **MP-4D7** | **CLOSED** | Documentation regression gates |
| **MP-4D8** | **CLOSED** | Final enterprise documentation audit |

### Legacy MP-4 (historical)

| Slice | Status |
|-------|--------|
| MP-4A | SUPERSEDED_BY_MP4R0 |
| MP-4B | RETIRED (MP-4R1) |
| MP-4C | RETIRED (MP-4R2) |
| MP-4D | RETIRED (MP-4R2) |
| MP-4E–H | CANCELLED / REPLACED by MP-4R* |

---

## MP-5 - Principal-scoped ContextView

**MP-5 ownership — FROZEN** — [`COLLABORATIVE_WORK`](../../architecture/COLLABORATIVE_WORK.md) · [ADR-MP-006](../../technical/adr/entries/2026-09-17/ADR-MP-006.md) **Accepted**.

| Slice | Purpose | Status |
|-------|---------|--------|
| MP-5A | Ownership, contracts architecture, ADR, docs sync | **APPROVED / CLOSED** |
| MP-5B | Core typed Principal-scoped ContextView contracts | **APPROVED / CLOSED** |
| MP-5C | Principal-scope visibility policy | **APPROVED / CLOSED** |
| MP-5D | Source composition ports | **APPROVED / CLOSED** |
| MP-5E | Default composition implementation | **CLOSED** |
| MP-5F | Source adapters / integration | **CLOSED** |
| MP-5G | E2E / isolation qualification | **CLOSED** |
| MP-5H | Final MP-5 enterprise certification | **CLOSED / FINAL CERTIFICATION PASSED** |
| MP-5 | Principal-scoped ContextView capability | **ENTERPRISE CERTIFIED / CLOSED** |
| MP-6 | Collaborative Activity + provenance | **IN PROGRESS** (**MP-6F — NEXT**) |

### MP-5A — ContextView ownership & contract architecture gate

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | **APPROVED / CLOSED** |
| **Purpose** | Freeze semantic owner, public contract boundary, anti-substitution, and MP-5 decomposition before runtime. |
| **Owning domain** | [`COLLABORATIVE_WORK`](../../architecture/COLLABORATIVE_WORK.md) |
| **Dependencies** | MP-1 **CLOSED** |
| **REUSED EXISTING CAPABILITY** | MP-1 authority; UCL; CE; Memory; RAG/Knowledge; Token Optimization |
| **NEW CAPABILITY REQUIRED** | None at MP-5A (contracts land in MP-5B) |
| **Explicit out of scope** | Runtime resolver, adapters, storage, providers, Nexus contract surface |
| **Architecture/ADR gate** | **ADR-MP-006 Accepted** |
| **Acceptance criteria** | Single owner; `ContextView ≠ storage`; dependency direction frozen; threat model documented |
| **Next step** | **MP-5F — NEXT** |

### MP-5B — Core ContextView contracts

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | **MP-5B — APPROVED / CLOSED** |
| **Purpose** | Typed, extra-forbid Principal-scoped ContextView contracts in Collaborative Work namespace |
| **Owning domain plan** | [`plan/COLLABORATIVE_WORK.md`](../../maintainers/plans/COLLABORATIVE_WORK.md) |
| **Dependencies** | MP-5A **CLOSED** |
| **REUSED EXISTING CAPABILITY** | `EffectiveAuthorityRequest`, `WorkArtifactVersionRef`, domain source locator refs |
| **NEW CAPABILITY REQUIRED** | `intergrax/contracts/context_view.py` — request/scope/entry/ref/result |
| **Explicit out of scope** | Default composer (MP-5E); source adapters (MP-5F); composition ports (MP-5D) |
| **Expected proof/evidence** | `test_context_view_contracts.py`; architecture gates; docs regression |
| **Next step** | **MP-5C — APPROVED / CLOSED** |

### MP-5C — Principal-scope visibility policy

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | **MP-5C — APPROVED / CLOSED** |
| **Purpose** | Fail-closed principal visibility eligibility (`ContextViewVisibilityPolicy` + `ContextViewPolicyDecision`) |
| **NEW CAPABILITY REQUIRED** | `intergrax/contracts/context_view_visibility_policy.py`; default `intergrax/collaborative_work/context_view_visibility.py` |
| **REUSED EXISTING CAPABILITY** | MP-1 `CollaborativeWorkAuthorityResolver` |
| **Explicit out of scope** | Composition ports (MP-5D); retrieval; source adapters |
| **Expected proof/evidence** | `test_context_view_visibility_policy.py`; architecture gates; docs regression |
| **Next step** | **MP-5D — APPROVED / CLOSED** |

### MP-5D — Source composition ports

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | **MP-5D — APPROVED / CLOSED** |
| **Purpose** | **source composition ports** — typed per-domain source ports for reference-first candidates after `ContextViewPolicyDecision` |
| **NEW CAPABILITY REQUIRED** | `intergrax/contracts/context_view_source_ports.py` |
| **Explicit out of scope** | Default composer (MP-5E); source adapters (MP-5F); retrieval; storage |
| **Expected proof/evidence** | `test_context_view_source_ports.py`; architecture gates; docs regression |
| **Next step** | **MP-5F — NEXT** |

---

## MP-6 - Collaborative Activity + provenance / evidence linkage

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | **MP-6 — IN PROGRESS**; **MP-6A — CLOSED / RECERTIFIED** (**MP-6A-C1**); **MP-6B — CLOSED / RECERTIFIED** (**MP-6B-C1** / **MP-6B-C1-R1**); **MP-6C — CLOSED / RECERTIFIED**; **MP-6D — CLOSED / RECERTIFIED**; **MP-6E — CLOSED / RECERTIFIED**; **MP-6F — NEXT** |
| **Purpose** | Collaborative activity stream linked to provenance and evidence. |
| **Owning domain** | **COLLABORATIVE_WORK** — **MP-6 ownership — FROZEN** ([ADR-MP-007](../../technical/adr/entries/2026-09-18/ADR-MP-007.md)) |
| **Dependencies** | MP-2, MP-3, MP-4, MP-5 recommended |
| **Exact scope** | Typed collaborative activity records + reference-only provenance; no authority transfer |
| **REUSED EXISTING CAPABILITY** | `ExecutionProvenanceRef`, observability run/step IDs, `GovernanceEvidenceRef`, `ProofReceipt` IDs, Decision/Approval IDs |
| **NEW CAPABILITY REQUIRED** | `intergrax/contracts/collaborative_activity.py`; MP-6B+ runtime |
| **Explicit out of scope** | Activity feeds/UI (MP-9); persistence (MP-6D until scheduled); repository inference |
| **Architecture/ADR gate** | **CLOSED / RECERTIFIED** (MP-6A-C1 / ADR-MP-007) |
| **User-visible outcome** | Auditable collaborative activity tied to evidence |
| **Expected proof/evidence** | `test_mp6a_*`, `test_mp6a_c1_*`; MP-6G qualification (future) |

### MP-6A — ownership / contract architecture gate

| Field | Value |
|-------|-------|
| **Status** | **MP-6A — CLOSED / RECERTIFIED** |
| **Proof** | ADR-MP-007; `test_mp6a_collaborative_activity_architecture_gates.py`; `test_mp6a_documentation_regression_gates.py`; `test_mp6a_c1_identity_extensibility_ordering_gates.py`; `test_mp6a_c1_r1_append_ownership_gates.py` |
| **Next step** | **MP-6F — NEXT** |

### MP-6A-C1 — identity / extensibility / timeline semantics

| Field | Value |
|-------|-------|
| **Status** | **CLOSED** (subject to independent audit) |
| **Scope** | Tenant/workspace idempotency; `CollaborativeActivityTypeId` / `CollaborativeActivitySourceId`; opaque pagination cursor vs event time |

### MP-6A-C1-R1 — atomic append ownership

| Field | Value |
|-------|-------|
| **Status** | **CLOSED** (subject to independent audit) |
| **Scope** | Append-store atomic idempotency + per-workspace `append_position` / `recorded_at` materialization; publication is append input only (no sequencing fields) |

### MP-6B — core contract hardening

| Field | Value |
|-------|-------|
| **Status** | **CLOSED / RECERTIFIED** (subject to independent audit) |
| **Proof** | `test_mp6b_collaborative_activity_contracts.py` |

### MP-6B-C1 — policy-resolved append intent

| Field | Value |
|-------|-------|
| **Status** | **CLOSED / RECERTIFIED** (subject to independent audit) |
| **Proof** | `test_mp6b_c1_policy_resolved_append_intent_boundary.py` |

### MP-6B-C1-R1 — AppendIntent pluginability proof

| Field | Value |
|-------|-------|
| **Status** | **CLOSED** (subject to independent audit) |
| **Scope** | Pluginability proof corrected to verify the current AppendIntent contract |

| Slice | Purpose | Status |
|-------|---------|--------|
| MP-6A | Ownership, contracts architecture, ADR, docs sync | **CLOSED / RECERTIFIED** |
| MP-6A-C1 | Identity, extensibility, timeline semantics | **CLOSED** |
| MP-6A-C1-R1 | Atomic append position / materialization ownership | **CLOSED** |
| MP-6B | Core activity/provenance contracts (runtime hardening) | **CLOSED / RECERTIFIED** |
| MP-6B-C1 | Policy-resolved append intent boundary | **CLOSED / RECERTIFIED** |
| MP-6B-C1-R1 | AppendIntent append-store proof | **CLOSED** |
| MP-6C | Publication / ingestion boundary | **CLOSED** |
| MP-6D | Persistence / store | **CLOSED / RECERTIFIED** ([`MP-6D-Q1_POSTGRESQL_PROVIDER_QUALIFICATION.md`](../../maintainers/qualification/MP-6D-Q1_POSTGRESQL_PROVIDER_QUALIFICATION.md)) |
| MP-6E | Scoped read / query | **CLOSED / RECERTIFIED** |
| MP-6F | Source integrations | **CLOSED** (subject to independent audit) |
| MP-6G | E2E / isolation / idempotency | **PLANNED** |
| MP-6H | Final enterprise certification | **PLANNED** |

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
| **Architecture/ADR gate** | Platform ownership and non-migration of current LKW Workspace are accepted; ADR-MP-008 completed |
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
