# Local Workspace Application — Implementation Plan

**Status:** Product-first MVP roadmap (2026-07-27)  
**Governing product rule:** [`PRODUCT_FIRST_MVP.md`](../../../docs/plan/PRODUCT_FIRST_MVP.md)  
**Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)  
**Ask Workspace discovery:** [`ASK_WORKSPACE_DISCOVERY.md`](ASK_WORKSPACE_DISCOVERY.md)  
**Slack MVP discovery:** [`SLACK_MVP_DISCOVERY.md`](SLACK_MVP_DISCOVERY.md)  
**Knowledge Intake discovery:** [`KNOWLEDGE_INTAKE_DISCOVERY.md`](KNOWLEDGE_INTAKE_DISCOVERY.md)  
**External verification:** [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md)  
**Historical full plan:** [`archive/IMPLEMENTATION_PLAN_2026-07-22.md`](archive/IMPLEMENTATION_PLAN_2026-07-22.md)

```text
Current product level: Backend Product Alpha
Current milestone: LKW MVP
Current roadmap stage: Workspace Contents — Knowledge Intake source expansion
Last accepted: LKW-WORKSPACE-CONTENTS-1B-4-1 + corrections
Current implementation: LKW-WORKSPACE-CONTENTS-1B-4-2 — IMPLEMENTED / CORRECTION REQUIRED
Current review gate: audit LKW-WORKSPACE-CONTENTS-1B-4-2-C2
Next after acceptance: LKW-WORKSPACE-CONTENTS-1B-5 — explicit Web URL intake
Following source expansion: 1B-6 VENDOR-KNOWLEDGE integration
Then: 1C shared synchronization and completion lifecycle → 1D provenance inspection → 1E safe removal
```

---

## 1. Document role and source of truth

This file is the canonical source of truth for:

- the active LKW execution order;
- the current task and review gate;
- the Workspace Contents source-expansion roadmap;
- the integration boundary between LKW and VENDOR-KNOWLEDGE;
- the planned synchronization, inspection and removal lifecycle;
- the post-MVP direction toward LKW 1.0.

The archived plan preserves the earlier full product brief, historical implementation gates, proof-portability notes and completed milestone detail. Historical descriptions do not override the current task order in this file.

### Governing rule

```text
Deliver the smallest real product experience that demonstrates user value.
Use implementation of that product to discover and improve Intergrax.
Do not build the platform first and hope a useful product appears later.
```

Every implementation slice is preceded by a bounded architecture review. Cursor receives the accepted architecture and a precise implementation scope, not an open-ended platform audit.

---

## 2. Product direction

LKW is a **private-by-default, tenant-scoped and deployment-neutral knowledge workspace**. It lets a user attach controlled knowledge sources, process them durably, ask natural-language questions and receive grounded answers with inspectable provenance.

“Local” means user-controlled deployment and configuration, including first-class self-hosted topology. It does not mean that every source must be a local folder or that all upstream data must physically live on one device.

The source portfolio grows through one LKW-owned knowledge lifecycle:

```text
managed files and channel attachments
→ preconfigured local folders
→ explicit Web URLs
→ organizational vendor systems through VENDOR-KNOWLEDGE
→ one Source / operation / queue / worker / indexing lifecycle
→ grounded Ask with provenance
```

Target vendor-backed knowledge includes, without making all providers one implementation task:

- Microsoft 365, Outlook and Teams knowledge;
- OneDrive and SharePoint files;
- Jira issues and Confluence pages;
- Databricks notebooks, datasets and related knowledge assets;
- Atlan catalog assets;
- Power BI reports, dashboards and semantic knowledge;
- e-mail, chats, documents, calendars, meeting material and future organizational sources.

Microsoft Teams in `1B-6` means Teams-hosted organizational knowledge. It does not automatically add Microsoft Teams as a second conversational frontend.

---

## 3. Current product state

### Completed and accepted intake foundations

| Task | Result | Status |
|---|---|---|
| `LKW-WORKSPACE-CONTENTS-1B-0` | Channel-neutral Knowledge Intake contract | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-1` | Durable Knowledge Input → Source → operation → queue/worker foundation | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-2` | Managed-file HTTP intake | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-3` | Slack attachment and multi-attachment adapter over managed-file intake | ACCEPTED |

### Current implementation and review gate

| Task | Result | Status |
|---|---|---|
| `LKW-WORKSPACE-CONTENTS-1B-4-1` | HTTP listing and acceptance of opaque preconfigured `LOCAL_FOLDER` Source Candidates | ACCEPTED (with corrections) |
| `LKW-WORKSPACE-CONTENTS-1B-4-1-C1` | Harden public safety, candidate identity and operation evidence | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-4-2` | Slack safe numbered Source Candidate selection | **IMPLEMENTED / CORRECTION REQUIRED** |

Current review gate: audit `LKW-WORKSPACE-CONTENTS-1B-4-2-C2` before advancing to `1B-5`.

---

## 4. Active Workspace Contents execution order

```text
LKW-WORKSPACE-CONTENTS-1B-4-2
→ audit and acceptance
→ LKW-WORKSPACE-CONTENTS-1B-5
→ LKW-WORKSPACE-CONTENTS-1B-6-0
→ LKW-WORKSPACE-CONTENTS-1B-6-1
→ LKW-WORKSPACE-CONTENTS-1B-6-2
→ LKW-WORKSPACE-CONTENTS-1B-6-3
→ LKW-WORKSPACE-CONTENTS-1C
→ LKW-WORKSPACE-CONTENTS-1D
→ LKW-WORKSPACE-CONTENTS-1E
```

| Task | User/product outcome | Status |
|---|---|---|
| `1B-4-2` | A Slack user can select a safe numbered preconfigured folder without seeing its path | IMPLEMENTED / CORRECTION REQUIRED |
| `1B-5` | A trusted client can attach an explicit Web URL through Knowledge Intake | PLANNED |
| `1B-6-0` | LKW and VENDOR-KNOWLEDGE have a frozen ownership and integration contract | PLANNED |
| `1B-6-1` | A user can discover safe opaque vendor connections and resources | PLANNED |
| `1B-6-2` | One real vendor resource is attached end to end through LKW Knowledge Intake | PLANNED |
| `1B-6-3` | Additional vendors do not require a new LKW ingestion architecture | PLANNED |
| `1C` | Local, URL and vendor Sources share synchronization, operation inspection and completion notification | PLANNED |
| `1D` | A user can inspect indexed documents and safe provenance | PLANNED |
| `1E` | A user can remove source-owned local knowledge without deleting upstream vendor data | PLANNED |

---

## 5. `LKW-WORKSPACE-CONTENTS-1B-4-2` — Slack Source Candidate selection

**One-sentence outcome:** An approved Slack user can list and select a numbered preconfigured folder for the active workspace while Slack receives only safe candidate metadata and opaque identity.

Expected flow:

```text
approved Slack DM
→ effective active workspace
→ public LKW Source Candidate list endpoint
→ safe numbered labels
→ user selects a number
→ Slack sends opaque candidate_id to the public acceptance endpoint
→ existing Knowledge Intake and KNOWLEDGE_INGESTION lifecycle
→ safe acknowledgement
```

Slack must never receive, persist or infer:

- the local path;
- allowlist or shadow roots;
- candidate fingerprint;
- provider locator or configuration path.

Slack remains a replaceable frontend and must not own folder discovery or indexing.

---

## 6. `LKW-WORKSPACE-CONTENTS-1B-5` — explicit Web URL intake

**One-sentence outcome:** A trusted client can attach an allowed Web URL as workspace knowledge through the same durable Knowledge Intake and operation lifecycle.

The slice must reuse:

- `KnowledgeInput` and the accepted intake boundary;
- durable Source ownership;
- idempotent operation creation;
- the existing queue, worker and recovery path;
- the existing document indexing and provenance model.

It must not create a second URL-specific queue, worker or document pipeline. URL access policy, redirects, private-network protection, size limits and safe error behavior must be frozen before implementation.

---

## 7. `LKW-WORKSPACE-CONTENTS-1B-6` — VENDOR-KNOWLEDGE integration

**One-sentence outcome:** LKW can attach organizational systems through one VENDOR-KNOWLEDGE boundary, normalize their resources into provider-neutral knowledge inputs and reuse the existing LKW Source, operation, queue, worker and indexing lifecycle without provider-specific LKW pipelines.

### 7.1 Ordering rule

`1B-6` is placed after explicit Web URL intake and before the shared `1C` lifecycle:

```text
1B-5 Web URL intake
→ 1B-6 VENDOR-KNOWLEDGE integration
→ 1C shared synchronization and completion lifecycle
→ 1D document inspection and provenance
→ 1E safe source-owned knowledge removal
```

This ensures synchronization, operation inspection, completion notification, provenance and removal are designed once for local folders, Web URLs and vendor-backed Sources.

### 7.2 Ownership boundary

```text
VENDOR-KNOWLEDGE owns:
provider authentication and credential handling
→ provider APIs and provider-specific errors
→ pagination, rate limits and token refresh
→ provider-specific resource discovery
→ incremental cursors, checkpoints and reconciliation semantics
→ canonical provider-neutral resource and knowledge-item projection

LKW owns:
tenant/workspace authorization and source attachment
→ KnowledgeInput and durable Source identity
→ idempotency and durable operation lifecycle
→ queue/worker dispatch and recovery
→ Documents, Chunks and Vectors
→ Ask Workspace and safe provenance projection
→ local retention and deletion semantics
```

VENDOR-KNOWLEDGE expands the provider catalog. It does not become a second knowledge workspace or a second LKW ingestion runtime.

### 7.3 Architectural invariants

LKW must not implement separate pipelines such as:

```text
Confluence ingestion pipeline
Teams ingestion pipeline
Power BI ingestion pipeline
Databricks ingestion pipeline
Jira ingestion pipeline
Atlan ingestion pipeline
```

Provider-specific differences end at the VENDOR-KNOWLEDGE boundary. Downstream LKW processing remains provider-neutral.

Public LKW contracts may expose only safe opaque fields such as:

```text
connection_id
resource_id
provider_id or vendor
resource_type
label
description
available
safe provenance
```

They must not expose:

- access or refresh tokens;
- client secrets;
- connection strings;
- private provider locators or endpoints;
- raw credentials;
- internal checkpoint or cursor material unless explicitly converted into a safe public status.

### 7.4 `1B-6-0` — integration contract and ownership gate

Freeze, from the real VENDOR-KNOWLEDGE implementation rather than assumptions:

- vendor connection identity;
- vendor resource identity;
- canonical knowledge item shape;
- credential ownership and access boundary;
- checkpoint/cursor ownership;
- LKW Source correlation and provenance;
- idempotency and configuration-version behavior;
- deletion and upstream-unavailability semantics;
- error normalization;
- the exact public capability LKW consumes.

Do not freeze a provider-specific LKW Source architecture. Exact names of new `KnowledgeInputKind` or Source representation remain an architecture decision for this gate.

### 7.5 `1B-6-1` — safe connection and resource discovery

Expose tenant-scoped, authorized and safely labelled vendor connections/resources. Discovery must support unavailable states without leaking credentials or technical locator details.

Representative resource families:

| Family | Examples |
|---|---|
| Files and documents | OneDrive, SharePoint |
| Wiki and technical knowledge | Confluence |
| Communication | Teams chats/channels, Outlook e-mail |
| Work management | Jira |
| Calendars and meetings | Outlook Calendar, Teams meeting material |
| Data and notebooks | Databricks |
| Data catalog | Atlan |
| Reports and analytics | Power BI |

### 7.6 `1B-6-2` — vendor resource Knowledge Intake

Attach one real vendor resource end to end:

```text
safe opaque vendor resource selection
→ public LKW Knowledge Intake capability
→ durable KnowledgeInput
→ durable Source
→ KNOWLEDGE_INGESTION operation
→ existing queue and worker
→ provider-neutral canonical items
→ existing document indexing
→ workspace Documents with safe provenance
```

This task proves one representative provider. It must not attempt to implement every provider listed in the roadmap.

### 7.7 `1B-6-3` — provider-neutral verification

Demonstrate that materially different provider families can reuse the same LKW contracts and lifecycle. The verification may use focused adapters/fakes where full production integrations are not yet justified, but it must prove that adding another vendor does not require:

- a new LKW queue or worker;
- a new operation model;
- a new Source ownership system;
- a new document indexing pipeline;
- provider credentials inside LKW public APIs;
- provider-specific business logic in Slack.

---

## 8. `LKW-WORKSPACE-CONTENTS-1C` — shared synchronization and completion lifecycle

**One-sentence outcome:** A user can synchronize and inspect asynchronous processing for local-folder, Web URL and vendor-backed Sources through one operation model and receive channel-neutral completion information.

The lifecycle must cover:

- manual synchronization;
- durable accepted, queued, processing, completed and failed states;
- reliable operation counters and safe errors;
- restart recovery;
- provider cursors/checkpoints through the VENDOR-KNOWLEDGE boundary;
- added, changed and removed upstream items according to frozen policy;
- channel-neutral completion events consumed by Slack or future surfaces.

Completion notification must not be implemented as a Slack-only ingestion mechanism.

---

## 9. `LKW-WORKSPACE-CONTENTS-1D` — document inspection and provenance

**One-sentence outcome:** A user can inspect which documents are indexed in a workspace, their Source ownership, status and safe origin without exposure of private paths, credentials or provider locators.

Safe provenance may include:

- Source label and safe source type;
- vendor/provider and resource type;
- original item type such as file, e-mail, chat message, thread, calendar event, wiki page, issue, dashboard, dataset, notebook or catalog asset;
- last synchronization time;
- safe document status and indexing metadata.

Every persisted Document remains owned by exactly one durable Source.

---

## 10. `LKW-WORKSPACE-CONTENTS-1E` — safe source-owned knowledge removal

**One-sentence outcome:** A user can detach a Source and remove its LKW-owned knowledge safely and idempotently without deleting original data in the upstream system.

Removal may include:

```text
Documents
→ Chunks
→ Vectors
→ local cached or managed objects where applicable
→ correlation state
→ LKW-owned synchronization/checkpoint state according to the 1B-6 contract
```

Removal must not automatically delete:

- Outlook e-mail or calendar events;
- Teams messages or meeting data;
- SharePoint or OneDrive files;
- Confluence pages or Jira issues;
- Power BI reports;
- Databricks data or notebooks;
- Atlan catalog assets;
- any other upstream vendor originals.

The operation means “detach from LKW and delete the local knowledge representation,” not “delete the source system record.”

---

## 11. Cross-cutting acceptance rules

Every source-expansion slice must preserve:

- tenant and workspace isolation;
- private-by-default behavior;
- opaque public identities;
- no private path or credential disclosure;
- idempotent acceptance and operation identity;
- one durable Source per attached logical origin according to its frozen identity contract;
- one existing Knowledge Intake runtime;
- one queue/worker lifecycle;
- shared document indexing and provenance;
- recovery without a parallel recovery table unless a separately accepted architecture gap requires it;
- Slack and future conversational surfaces as API/capability clients only.

Parallel work rules:

- work from the current repository state;
- preserve unrelated changes from other sessions;
- do not block solely because HEAD differs from an earlier instruction;
- do not use destructive Git commands;
- commit only the task’s own scope.

---

## 12. Post-MVP direction to LKW 1.0

LKW 1.0 aims to be installable, restart-safe, auditable, daily-usable and supportable for its declared source portfolio. Broad provider breadth is not automatically required for 1.0; the bounded `1B-6` contract, one end-to-end provider proof and provider-neutral verification establish the expansion path.

Future work remains product-pulled and may include:

- broader vendor coverage selected by user or commercial value;
- richer document reconciliation;
- workspace outputs and history;
- a local setup/diagnostics companion;
- another conversational frontend;
- organization membership and controlled sharing;
- production security, operations and release hardening.

A second conversational adapter and broad provider matrix do not delay the current Workspace Contents sequence unless a documented product blocker requires them.

---

## Appendix A — Current task summary

```text
CURRENT:
LKW-WORKSPACE-CONTENTS-1B-4-2
Slack safe numbered Source Candidate selection
IMPLEMENTED / CORRECTION REQUIRED

REVIEW GATE:
audit LKW-WORKSPACE-CONTENTS-1B-4-2-C2
→ ACCEPTED: advance to 1B-5
→ NEEDS_CORRECTION: issue only the next bounded 1B-4-2 correction

NEXT:
LKW-WORKSPACE-CONTENTS-1B-5
explicit Web URL intake

THEN:
1B-6-0 VENDOR-KNOWLEDGE contract
→ 1B-6-1 safe discovery
→ 1B-6-2 one end-to-end vendor proof
→ 1B-6-3 provider-neutral verification
→ 1C shared synchronization and completion lifecycle
→ 1D safe provenance inspection
→ 1E safe local knowledge removal

ARCHITECTURE:
VENDOR-KNOWLEDGE owns vendor-specific access, credentials and normalization.
LKW owns KnowledgeInput, Source, durable operations, queue/worker, Documents and indexed knowledge.
No duplicate provider-specific LKW pipelines.
```

## Appendix B — Status vocabulary

| Label | Meaning |
|---|---|
| `ACCEPTED` | Actual implementation was audited and accepted |
| `CURRENT` | Active bounded task |
| `CURSOR IN PROGRESS` | Implementation is being performed in the current Cursor task |
| `NEXT` | First task after the current review gate |
| `PLANNED` | Intentionally on the active execution path |
| `DEFERRED` | Outside active scope until justified by product need |
