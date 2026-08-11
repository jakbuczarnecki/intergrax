<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Multiplayer AI — Multi-layer Feature Plan

**Status:** **MP-0** — canonical architecture and implementation roadmap (documentation only; not started implementation)
**Feature architecture (1:1):** [`../architecture/MULTIPLAYER_AI.md`](../architecture/MULTIPLAYER_AI.md)
**Primary anchor domain (provisional):** `OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`
**Related domains (provisional):** `PLATFORM_FOUNDATION`, `UNIFIED_EXECUTION_RUNTIME`, `ORCHESTRATION`, `UNIFIED_CONTEXT_LIFECYCLE`, `CONTEXT_ENGINEERING`, `MEMORY`, `RAG`, `RELIABILITY_FAILURE_AND_HITL`, `NEXUS_EXECUTION_FLOW`, `OBSERVABILITY`, `PROOF_RECEIPTS`, `INTEGRATIONS`, `AGENT_CONTRACTS_AND_ASSEMBLY`, `APPLICATION_HOSTING`
**Current active task:** **MP-0**
**Next task after MP-0 acceptance:** **MP-1** — bounded ownership check, then domain plan synchronization

---

## Cursor read scope (token budget)

Do not read the whole repository.

Default read scope for Multiplayer AI work:

1. [`../architecture/MULTIPLAYER_AI.md`](../architecture/MULTIPLAYER_AI.md) — read-scope block + active `MP-*` phase summary only.
2. This file — read-scope block + **active MP-* section only**.
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

**MP-0 performs steps 0 only** (feature hub pair + README). No domain plan edits.

### Capability labeling (mandatory in future MP rows)

| Label | Use when |
|-------|----------|
| **REUSED EXISTING CAPABILITY** | An existing domain or application row is integrated as-is or extended without becoming the MP primitive owner |
| **NEW CAPABILITY REQUIRED** | A platform Multiplayer primitive must be introduced; adjacent rows cannot substitute |

### Anti-substitution (plan-level)

- `LKW-CONVERSATION-CONTEXT-*` ≠ MP-1 anchor
- `CONVERSATION-CHANNEL-1` ≠ MP-1 foundation
- Slack shared-conversation ≠ MP-2
- `LKW-HYBRID-ASK-*` ≠ MP-3
- Slack vertical rows ≠ MP-4
- `TOKEN-10E-*` ≠ MP-5 implementation identity
- Do not force-map adjacent rows to MP phases

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

## MP-0 — Canonical architecture and implementation roadmap

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | IN PROGRESS (documentation) |
| **Purpose** | Establish canonical Multiplayer AI architecture, MP-0…MP-9 roadmap, capability classification, anti-substitution rules, and provisional domain ownership map. |
| **Owning domain plan** | Feature plan + feature architecture hub only — **no domain plan edits in MP-0** |
| **Dependencies** | None |
| **Exact scope** | `docs/project/capabilities/architecture/MULTIPLAYER_AI.md`, `docs/project/capabilities/plan/MULTIPLAYER_AI.md`, `docs/project/capabilities/README.md` index row |
| **Explicit out of scope** | Satellites (unless genuinely required), domain architecture/plan edits, code, tests, scripts, MP-1+ implementation |
| **Acceptance criteria** | Feature pair exists 1:1; README lists Multiplayer AI; roadmap MP-0…MP-9 matches canonical sequence; twelve anti-substitution corrections preserved; `REUSED EXISTING CAPABILITY` vs `NEW CAPABILITY REQUIRED` tables present; provisional ownership uses `OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION` where not proven; no incorrect mapping of LKW/Slack/TOKEN/UCL rows to MP phases |
| **User-visible outcome** | Reviewable Multiplayer roadmap; no runtime change |

---

## MP-1 — Principal, WorkspaceMembership and Delegation / effective authority

| Field | Value |
|-------|-------|
| **Priority** | P0 (after MP-0) |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Platform collaborative identity, workspace membership, and delegation with effective authority. |
| **Likely owning domain plans** | `PLATFORM_FOUNDATION.md`, `APPLICATION_HOSTING.md`, `UNIFIED_EXECUTION_RUNTIME.md` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-0 accepted |
| **REUSED EXISTING CAPABILITY** | Request-context principal propagation; LKW application principal docs (consumer reference only) |
| **NEW CAPABILITY REQUIRED** | Principal (collaborative), WorkspaceMembership, Delegation / effective authority |
| **Explicit out of scope** | Using `LKW-CONVERSATION-CONTEXT-*` or `CONVERSATION-CHANNEL-1` as MP-1 anchor |
| **Pre-implementation gate** | Bounded ownership check → domain architecture/plan sync with MP-1 rows |
| **User-visible outcome** | Governed multi-principal identity and authority model |

---

## MP-2 — Shared Work: WorkItem, Assignment, lifecycle and concurrency

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Platform-owned shared work primitives with lifecycle and concurrency semantics. |
| **Likely owning domain plans** | `ORCHESTRATION.md`, `UNIFIED_EXECUTION_RUNTIME.md`, `BACKGROUND_TASKS.md` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-1 accepted |
| **REUSED EXISTING CAPABILITY** | Execution-runtime task/session concepts where they remain runtime-internal |
| **NEW CAPABILITY REQUIRED** | WorkItem, Assignment, shared-work lifecycle and concurrency |
| **Explicit out of scope** | Slack shared-conversation or any channel adapter as Shared Work owner |
| **Pre-implementation gate** | Bounded ownership check → domain architecture/plan sync with MP-2 rows |
| **User-visible outcome** | Addressable shared work units assignable to principals and agents |

---

## MP-3 — WorkArtifact and WorkArtifactVersion

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Durable collaborative outputs with versioning and provenance. |
| **Likely owning domain plans** | `UNIFIED_CONTEXT_LIFECYCLE.md`, `PROOF_RECEIPTS.md`, `MEMORY.md` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-2 accepted (may overlap MP-1 for authority on artifacts) |
| **REUSED EXISTING CAPABILITY** | UCL artifact lifecycle patterns; receipt/provenance models |
| **NEW CAPABILITY REQUIRED** | WorkArtifact, WorkArtifactVersion |
| **Explicit out of scope** | `LKW-HYBRID-ASK-*` as WorkArtifact owner |
| **Pre-implementation gate** | Bounded ownership check → domain architecture/plan sync with MP-3 rows |
| **User-visible outcome** | Versioned collaborative artifacts with lineage |

---

## MP-4 — Decision / DecisionResponse or Approval semantics + HITL bridge

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Collaborative decision and approval semantics with explicit bridge to Nexus HITL when execution must pause. |
| **Likely owning domain plans** | `RELIABILITY_FAILURE_AND_HITL.md`, `NEXUS_EXECUTION_FLOW.md`, `UNIFIED_EXECUTION_RUNTIME.md` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-1 accepted; MP-2 recommended |
| **REUSED EXISTING CAPABILITY** | Nexus HITL pause/resume; policy evaluation hooks |
| **NEW CAPABILITY REQUIRED** | Decision / DecisionResponse (or Approval) collaborative primitive; HITL bridge contract |
| **Explicit out of scope** | Slack vertical rows as Decision owner; conflating Decision records with HITL machinery |
| **Pre-implementation gate** | Bounded ownership check → domain architecture/plan sync with MP-4 rows |
| **User-visible outcome** | Explicit collaborative approvals that can pause and resume governed execution |

---

## MP-5 — Principal-scoped ContextView

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Principal-scoped context view composing UCL, Context Engineering, Memory, and Knowledge. |
| **Likely owning domain plans** | `UNIFIED_CONTEXT_LIFECYCLE.md`, `CONTEXT_ENGINEERING.md`, `MEMORY.md`, `RAG.md` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-1 accepted |
| **REUSED EXISTING CAPABILITY** | UCL, Context Engineering, Memory, RAG/Knowledge, Token Optimization (`TOKEN-10E-*` etc.) |
| **NEW CAPABILITY REQUIRED** | ContextView contract and principal-scope composition policy |
| **Explicit out of scope** | Relabeling `TOKEN-10E-*` or UCL rows as MP-5 implementation |
| **Pre-implementation gate** | Bounded ownership check → domain architecture/plan sync with MP-5 rows |
| **User-visible outcome** | Membership-aware context visibile to each collaborative principal |

---

## MP-6 — Collaborative Activity + provenance / evidence linkage

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Collaborative activity stream linked to provenance and evidence. |
| **Likely owning domain plans** | `OBSERVABILITY.md`, `PROOF_RECEIPTS.md`, `UNIFIED_EXECUTION_RUNTIME.md` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-2, MP-3, MP-4 recommended |
| **REUSED EXISTING CAPABILITY** | Traces, receipts, attempt ledger, existing provenance fields |
| **NEW CAPABILITY REQUIRED** | Activity model with collaborative semantics and evidence linkage |
| **Pre-implementation gate** | Bounded ownership check → domain architecture/plan sync with MP-6 rows |
| **User-visible outcome** | Auditable collaborative activity tied to evidence |

---

## MP-7 — LKW reference-product adoption

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Adopt platform Multiplayer primitives in LKW as first reference consumer. |
| **Likely owning plans** | Tier-3 LKW application implementation plan (consumer); platform primitives in Tier-0/Tier-1 domain plans — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-1…MP-6 platform primitives accepted for the adopted subset |
| **REUSED EXISTING CAPABILITY** | Prior LKW conversation, Ask, channel capabilities until explicitly integrated |
| **NEW CAPABILITY REQUIRED** | LKW integration rows per adopted primitive (consumer-side only) |
| **Explicit out of scope** | Transferring platform primitive ownership to LKW |
| **Pre-implementation gate** | Bounded ownership check → LKW plan sync; no substitution of LKW-local rows for missing primitives |
| **User-visible outcome** | LKW demonstrates end-to-end Multiplayer on platform contracts |

---

## MP-8 — AgentDirectory / external-agent interoperability

| Field | Value |
|-------|-------|
| **Priority** | P3 |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Agent discovery/registry, ExternalWork reuse, future A2A adapter at integration boundary. |
| **Likely owning domain plans** | `AGENT_CONTRACTS_AND_ASSEMBLY.md`, `INTEGRATIONS.md`, `UNIFIED_EXECUTION_RUNTIME.md` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-1, MP-2 recommended |
| **REUSED EXISTING CAPABILITY** | `ExternalWorkIntegration`, governed external work host lifecycle |
| **NEW CAPABILITY REQUIRED** | AgentDirectory, interoperability policy, external-agent adapter boundary |
| **Explicit out of scope** | Transport types (HTTP/A2A/REST) in core Multiplayer contracts |
| **Pre-implementation gate** | Bounded ownership check → domain architecture/plan sync with MP-8 rows |
| **User-visible outcome** | Governed participation of internal and external agents in shared work |

---

## MP-9 — Advanced collaborative UX, notifications, subscriptions, optional realtime

| Field | Value |
|-------|-------|
| **Priority** | P3 |
| **Status** | PLANNED / NOT STARTED |
| **Purpose** | Product-facing collaboration UX, notifications/subscriptions, optional realtime or generative UI when justified. |
| **Likely owning domain plans** | `APPLICATION_HOSTING.md`, `INTEGRATIONS.md` — **`OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION`** |
| **Dependencies** | MP-1…MP-8 as needed per UX slice |
| **REUSED EXISTING CAPABILITY** | `notification_channel`, `conversation_channel`, hosting presentation |
| **NEW CAPABILITY REQUIRED** | Subscription model tied to Activity/WorkItem/Decision where generic notifications are insufficient |
| **Explicit out of scope** | Using UX/realtime as substitute for missing MP-1…MP-6 primitives |
| **Pre-implementation gate** | Justification review + bounded ownership check → domain plan sync |
| **User-visible outcome** | Optional rich collaboration surfaces without owning core primitives |

---

## Domain plan row template (MP-1+ only)

When adding rows to an owning domain plan after ownership confirmation:

```text
MP-<n>-<slice> — <title>
  Classification: NEW CAPABILITY REQUIRED | REUSED EXISTING CAPABILITY
  Feature coordination: docs/project/capabilities/plan/MULTIPLAYER_AI.md §MP-<n>
  Owning domain: <DOMAIN>
  Dependencies: <prior MP or domain rows>
  Out of scope: <explicit anti-substitution items>
  Acceptance: <testable criteria>
```

**Not used in MP-0.**
