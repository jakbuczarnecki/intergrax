<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Multiplayer AI — Multi-layer Feature Architecture

**Status:** **MP-0** — canonical architecture and implementation roadmap (documentation only)
**Feature plan (1:1):** [`../plan/MULTIPLAYER_AI.md`](../plan/MULTIPLAYER_AI.md)
**Primary anchor domain (provisional):** `OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`
**Related domains (provisional):** `PLATFORM_FOUNDATION`, `UNIFIED_EXECUTION_RUNTIME`, `ORCHESTRATION`, `UNIFIED_CONTEXT_LIFECYCLE`, `CONTEXT_ENGINEERING`, `MEMORY`, `RAG`, `RELIABILITY_FAILURE_AND_HITL`, `NEXUS_EXECUTION_FLOW`, `OBSERVABILITY`, `PROOF_RECEIPTS`, `INTEGRATIONS`, `AGENT_CONTRACTS_AND_ASSEMBLY`, `APPLICATION_HOSTING`
**Current active task:** **MP-0**
**Next task after MP-0 acceptance:** **MP-1** — bounded ownership check, then domain plan synchronization

---

## Cursor read scope (token budget)

**Do not read this entire file in one session.**

- **Default:** §Purpose, §Strategic position, §Roadmap summary, §Capability classification, §Anti-substitution rules.
- **MP-0 / roadmap review:** §MP-0 through §MP-9 phase summaries + §Likely domain ownership.
- **Plan / task selection:** [`../plan/MULTIPLAYER_AI.md`](../plan/MULTIPLAYER_AI.md) read-scope block + active `MP-*` section only.
- **Satellites:** none required for MP-0. Create only when a future phase needs a cross-domain register.

---

## Purpose

Multiplayer AI is a **cross-layer Intergrax platform capability** for governed, multi-principal collaboration: shared work, durable collaborative outputs, explicit decisions and approvals, principal-scoped context views, activity and provenance, and interoperability with external agents — without collapsing those primitives into a single product channel, conversation transport, or application-local feature.

Multiplayer AI is **not**:

- a Slack or conversation-channel product vertical,
- LKW conversation-context ownership,
- Token Optimization or UCL relabeled as multiplayer,
- Hybrid Ask or any single application workflow,
- Nexus HITL itself (HITL remains the execution pause/resume mechanism that MP-4 may bridge to).

**LKW** (Local Knowledge Workspace) is the **first reference consumer** of platform Multiplayer primitives. LKW does **not** own Principal, WorkItem, WorkArtifact, Decision, ContextView, Activity, or AgentDirectory contracts.

```text
Tier-0/Tier-1 platform Multiplayer primitives
  → public contracts + runtime enforcement
  → Tier-2 agents (consume, do not own)
  → Tier-3 applications (LKW first reference consumer; others later)
```

Forbidden dependency direction:

```text
applications/local_workspace_application → defines platform Multiplayer ABI
intergrax/runtime/* → applications/local_workspace_application  (for multiplayer ownership)
```

---

## Strategic position

| Rule | Meaning |
|------|---------|
| **Platform primitive rule** | Each MP phase introduces or adopts a reusable collaborative primitive owned at the platform boundary unless explicitly classified as product-only UX. |
| **Reuse rule** | Adjacent existing capabilities may be **reused** where proven; they must not be **substituted** for a missing Multiplayer primitive. |
| **HITL rule** | Collaborative Decision / Approval semantics are distinct from Nexus HITL. MP-4 bridges to HITL only where execution must pause and resume. |
| **Channel adapter rule** | Slack, conversation channels, and notification surfaces are **adapters** that may surface Multiplayer primitives; they do not own Shared Work, WorkArtifact, or Decision. |
| **Context rule** | Principal-scoped `ContextView` composes existing UCL, Context Engineering, Memory, Knowledge, and Token Optimization mechanisms; it does not replace them or inherit their plan IDs. |
| **LKW rule** | LKW adopts platform primitives in MP-7; prior LKW rows remain application capabilities until explicitly integrated. |
| **Ownership rule** | Domain architecture/plan pairs remain authoritative. This feature doc **coordinates** them; it does not override domain canon. Before each `MP-1+` implementation phase: bounded ownership check → synchronize owning domain architecture/plan with concrete MP-owned rows. |

---

## Roadmap summary

```text
MP-0 → MP-1 → MP-2 → MP-3 → MP-4 → MP-5 → MP-6 → MP-7 → MP-8 → MP-9
```

| Phase | Summary |
|-------|---------|
| **MP-0** | Canonical architecture and implementation roadmap (this document + plan hub). No domain plan edits. |
| **MP-1** | Principal, WorkspaceMembership, Delegation / effective authority |
| **MP-2** | Shared Work: WorkItem, Assignment, lifecycle and concurrency |
| **MP-3** | WorkArtifact and WorkArtifactVersion: durable collaborative outputs, versioning and provenance |
| **MP-4** | Decision / DecisionResponse or Approval semantics + bridge to existing Nexus HITL where execution must pause |
| **MP-5** | Principal-scoped ContextView using existing UCL, Context Engineering, Memory and Knowledge |
| **MP-6** | Collaborative Activity + provenance / evidence linkage |
| **MP-7** | LKW reference-product adoption of platform Multiplayer primitives |
| **MP-8** | AgentDirectory / external-agent interoperability, ExternalWork reuse and future A2A adapter boundary |
| **MP-9** | Advanced collaborative UX, notifications/subscriptions and optional realtime/generative UI only when justified |

---

## Capability classification

Classification applies per phase. Until MP-1+ ownership is confirmed, rows below are **architectural intent**, not implementation truth.

### REUSED EXISTING CAPABILITY (may integrate; not MP phase owners)

| Existing capability | Role relative to Multiplayer | Must not substitute for |
|---------------------|------------------------------|-------------------------|
| `LKW-CONVERSATION-CONTEXT-*` | LKW conversation/context capability; may later integrate where relevant | MP-1 Principal / Membership / Delegation |
| `CONVERSATION-CHANNEL-1` (`ConversationChannelIntegrationContract`) | External near-real-time human ↔ application channel adapter | MP-1 foundation |
| Slack shared-conversation / Slack vertical integration rows | Product/channel adapter; may surface multiplayer UX | MP-2 Shared Work ownership |
| `LKW-HYBRID-ASK-*` | Application workflow; may later create or consume WorkArtifacts | MP-3 WorkArtifact primitive |
| Slack approval/decision surfacing rows | Channel presentation of decisions | MP-4 Decision primitive |
| Nexus HITL (`RELIABILITY_FAILURE_AND_HITL`, `NEXUS_EXECUTION_FLOW`) | Execution pause/resume when policy or human gate requires it | MP-4 Decision semantics alone |
| UCL (`UNIFIED_CONTEXT_LIFECYCLE`) | Durable/ephemeral context lifecycle, artifact coordination | MP-5 ContextView primitive |
| Context Engineering (`CONTEXT_ENGINEERING`) | Budget authority, assembly, provenance of model-facing context | MP-5 principal scope model |
| Memory (`MEMORY`) | Durable conversation/session ledger | MP-5 view composition store |
| RAG / Knowledge (`RAG`, `KNOWLEDGE_SOURCE_INTEGRATIONS`) | Knowledge retrieval and source boundaries | MP-5 membership-aware view policy |
| Token Optimization (`TOKEN_OPTIMIZATION`, `TOKEN-10E-*`) | Token economy and compaction executors | MP-5 implementation identity |
| ExternalWork / `ExternalWorkIntegration` | Governed external work host lifecycle | MP-8 AgentDirectory and A2A adapter boundary |
| `notification_channel` integrations | Outbound notification delivery | MP-9 subscriptions (optional) |

### NEW CAPABILITY REQUIRED (platform Multiplayer primitives)

| Primitive | MP phase | Notes |
|-----------|----------|-------|
| Principal (collaborative identity) | MP-1 | Distinct from generic request-context principal fields where product semantics require membership |
| WorkspaceMembership | MP-1 | Who belongs to a collaborative workspace and in what role |
| Delegation / effective authority | MP-1 | Who may act for whom, with what effective permissions |
| WorkItem | MP-2 | Addressable unit of shared work |
| Assignment | MP-2 | Binding of principals/agents to WorkItem with lifecycle |
| Shared Work lifecycle and concurrency rules | MP-2 | Platform-owned, not channel-owned |
| WorkArtifact | MP-3 | Durable collaborative output identity |
| WorkArtifactVersion | MP-3 | Versioning, lineage, provenance |
| Decision / DecisionResponse / Approval semantics | MP-4 | Collaborative decision primitive; bridge to HITL for execution pause |
| ContextView (principal-scoped) | MP-5 | Composes UCL/CE/Memory/Knowledge; does not relabel TOKEN/UCL rows |
| Activity (collaborative) | MP-6 | Observable collaborative events with evidence linkage |
| AgentDirectory | MP-8 | Registry/discovery for internal and external agents |
| External-agent interoperability boundary | MP-8 | Reuse ExternalWork; future A2A adapter at integration boundary |
| Collaborative UX / subscriptions / optional realtime | MP-9 | Product and hosting surfaces; only when justified |

---

## Anti-substitution rules (canonical)

The following mappings are **explicitly forbidden** in roadmap, architecture, and plan text:

1. `LKW-CONVERSATION-CONTEXT-*` is **not** the MP-1 architectural anchor.
2. `CONVERSATION-CHANNEL-1` is **not** the foundation of Principal / Membership / Delegation.
3. Slack shared-conversation rows are **not** MP-2; Slack is a channel adapter, not Shared Work owner.
4. `LKW-HYBRID-ASK-*` is **not** MP-3; Hybrid Ask may use WorkArtifacts later, but WorkArtifact is a platform primitive.
5. Slack vertical rows are **not** MP-4; Slack may surface Decisions; Decision is platform-owned; HITL remains execution pause/resume.
6. `TOKEN-10E-*` is **not** Multiplayer MP-5 implementation; Multiplayer may reuse UCL/Token Optimization work.
7. Do **not** force-map existing adjacent rows to MP phases merely because concepts are related.
8. Existing capabilities → `REUSED EXISTING CAPABILITY`. Missing primitives → `NEW CAPABILITY REQUIRED`. Never substitute adjacent rows for missing primitives.

---

## Phase architecture summaries

### MP-0 — Canonical architecture and implementation roadmap

**Scope:** Feature architecture/plan pair, README index entry, capability classification, anti-substitution rules, provisional domain ownership map.

**Out of scope:** Domain plan edits, satellites (unless genuinely required), code, tests, MP-1+ implementation.

**Acceptance:** Reviewable MP-0 docs; roadmap MP-0…MP-9 preserved; no incorrect row substitution; ownership marked provisional where unproven.

---

### MP-1 — Principal, WorkspaceMembership and Delegation / effective authority

**Intent:** Establish who collaborates, in which workspace, with what roles, and what effective authority applies when one principal acts for another.

**Likely owning domains:** `PLATFORM_FOUNDATION`, `APPLICATION_HOSTING`, `UNIFIED_EXECUTION_RUNTIME` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`**

**Reused (not owners):** request-context principal propagation where already present; LKW principal documentation in application layer.

**New required:** collaborative Principal model, WorkspaceMembership, Delegation / effective authority contracts and enforcement hooks.

---

### MP-2 — Shared Work: WorkItem, Assignment, lifecycle and concurrency

**Intent:** Platform-owned shared work units with explicit assignment, lifecycle states, and concurrency semantics independent of delivery channel.

**Likely owning domains:** `ORCHESTRATION`, `UNIFIED_EXECUTION_RUNTIME`, `BACKGROUND_TASKS` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`**

**Reused (not owners):** Nexus task/session concepts where they remain execution-runtime concerns; application workflows.

**New required:** WorkItem, Assignment, shared-work lifecycle and concurrency invariants.

---

### MP-3 — WorkArtifact and WorkArtifactVersion

**Intent:** Durable collaborative outputs with explicit versioning and provenance, reusable across applications and agents.

**Likely owning domains:** `UNIFIED_CONTEXT_LIFECYCLE`, `PROOF_RECEIPTS`, `MEMORY` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`**

**Reused (not owners):** UCL artifact lifecycle patterns; evidence/receipt models; application shadow artifacts.

**New required:** WorkArtifact and WorkArtifactVersion as platform collaborative primitives (distinct from single-application artifact names).

---

### MP-4 — Decision / DecisionResponse or Approval semantics + HITL bridge

**Intent:** Collaborative decision and approval semantics for multi-principal workflows. Where execution must pause until a human or policy gate responds, bridge to existing Nexus HITL pause/resume — without conflating Decision records with HITL machinery.

**Likely owning domains:** `RELIABILITY_FAILURE_AND_HITL`, `NEXUS_EXECUTION_FLOW`, `UNIFIED_EXECUTION_RUNTIME` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`**

**Reused (not owners):** Nexus HITL, policy evaluation surfaces, channel presentation of approval UX.

**New required:** Decision / DecisionResponse (or Approval) collaborative primitive and explicit HITL bridge contract.

---

### MP-5 — Principal-scoped ContextView

**Intent:** A principal-scoped view over what context is visible and composable for a collaborative actor, built on existing UCL, Context Engineering, Memory, and Knowledge — without relabeling TOKEN-10E or UCL implementation rows as Multiplayer.

**Likely owning domains:** `UNIFIED_CONTEXT_LIFECYCLE`, `CONTEXT_ENGINEERING`, `MEMORY`, `RAG` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`**

**Reused (not owners):** UCL lifecycle, CE assembly, Memory ledger, Knowledge/RAG boundaries, Token Optimization executors.

**New required:** ContextView contract, principal-scope policy, composition rules across reused subsystems.

---

### MP-6 — Collaborative Activity + provenance / evidence linkage

**Intent:** Observable collaborative activity stream linked to provenance and evidence for audit, debugging, and governance.

**Likely owning domains:** `OBSERVABILITY`, `PROOF_RECEIPTS`, `UNIFIED_EXECUTION_RUNTIME` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`**

**Reused (not owners):** traces, receipts, attempt ledger, existing provenance fields on knowledge and context artifacts.

**New required:** Activity model with collaborative semantics and stable linkage to evidence.

---

### MP-7 — LKW reference-product adoption

**Intent:** Adopt platform Multiplayer primitives in LKW as the first reference consumer; migrate or integrate application-local patterns only through explicit MP-owned integration rows.

**Likely owning domains:** Tier-3 LKW application plans (consumer); platform primitives remain in Tier-0/Tier-1 owning domains — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`**

**Reused (not owners):** all prior LKW conversation, Ask, and channel capabilities until explicitly integrated.

**New required:** LKW integration contract per adopted primitive; no ownership transfer to LKW.

---

### MP-8 — AgentDirectory / external-agent interoperability

**Intent:** Discover and bind internal and external agents; reuse governed ExternalWork host patterns; place future A2A (or similar) adapters at the integration boundary without transport in core contracts.

**Likely owning domains:** `AGENT_CONTRACTS_AND_ASSEMBLY`, `INTEGRATIONS`, `UNIFIED_EXECUTION_RUNTIME` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`**

**Reused (not owners):** `ExternalWorkIntegration`, governed external work orchestration, agent assembly contracts.

**New required:** AgentDirectory, interoperability policy, adapter boundary for external agents.

---

### MP-9 — Advanced collaborative UX, notifications, subscriptions, optional realtime

**Intent:** Product-facing collaboration UX, notification/subscription surfaces, and optional realtime or generative UI **only when justified** by MP-1…MP-8 primitives — not as a substitute for them.

**Likely owning domains:** `APPLICATION_HOSTING`, `INTEGRATIONS` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`**

**Reused (not owners):** `notification_channel`, `conversation_channel`, hosting/runtime presentation layers.

**New required:** subscription model tied to Activity/WorkItem/Decision events where not covered by generic notifications.

---

## Cross-cutting invariants

| ID | Invariant |
|----|-----------|
| **MP-INV-01** | Multiplayer primitives are platform-owned unless explicitly classified as Tier-3 product-only UX in MP-9. |
| **MP-INV-02** | No channel adapter (Slack, conversation channel, notification channel) owns WorkItem, WorkArtifact, or Decision semantics. |
| **MP-INV-03** | HITL pauses execution; Decision records collaborative intent and outcomes; bridge is explicit, not implicit. |
| **MP-INV-04** | ContextView composes UCL/CE/Memory/Knowledge; TOKEN and UCL plan rows keep their native IDs. |
| **MP-INV-05** | LKW is a reference consumer; it does not define the Multiplayer public ABI. |
| **MP-INV-06** | Before MP-1+ code: bounded ownership check → domain architecture/plan sync → then implementation. |
| **MP-INV-07** | `REUSED EXISTING CAPABILITY` ≠ `NEW CAPABILITY REQUIRED`; substitution is forbidden. |

---

## Integration boundaries (conceptual)

```text
Principal + Membership + Delegation (MP-1)
  → scopes Shared Work (MP-2) and ContextView (MP-5)

WorkItem + Assignment (MP-2)
  → may produce WorkArtifact versions (MP-3)
  → may require Decision / Approval (MP-4)
  → emits Activity + evidence (MP-6)

Decision (MP-4)
  → may trigger Nexus HITL pause/resume
  → does not replace HITL contracts

ContextView (MP-5)
  → UCL + CE + Memory + Knowledge + Token Optimization (reuse)

AgentDirectory (MP-8)
  → ExternalWork reuse
  → future A2A adapter at INTEGRATIONS boundary

LKW (MP-7)
  → consumes MP-1…MP-6 (+ MP-8/9 as needed)
```

---

## Authoring and sync procedure

```text
MP-0 docs (this file + plan hub + README)
  → MP-1+ bounded ownership check per phase
  → update affected docs/project/architecture/<DOMAIN>.md
  → add concrete rows to docs/project/maintainers/plans/<DOMAIN>.md
  → implement smallest domain-owned slice
  → gate + journal
```

**MP-0 explicitly does not edit domain plans.**
