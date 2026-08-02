# Local Workspace Application — Implementation Plan

**Status:** Product-first MVP roadmap (2026-07-31)  
**Governing product rule:** [`PRODUCT_FIRST_MVP.md`](../../../docs/plan/PRODUCT_FIRST_MVP.md)  
**Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)  
**Ask Workspace discovery:** [`ASK_WORKSPACE_DISCOVERY.md`](ASK_WORKSPACE_DISCOVERY.md)  
**Slack MVP discovery:** [`SLACK_MVP_DISCOVERY.md`](SLACK_MVP_DISCOVERY.md)  
**Conversation context architecture:** [`CONVERSATION_CONTEXT_ARCHITECTURE.md`](CONVERSATION_CONTEXT_ARCHITECTURE.md)
**Knowledge Intake discovery:** [`KNOWLEDGE_INTAKE_DISCOVERY.md`](KNOWLEDGE_INTAKE_DISCOVERY.md)  
**Hybrid knowledge access:** [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md)
**External verification:** [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md)  
**Historical full plan:** [`archive/IMPLEMENTATION_PLAN_2026-07-22.md`](archive/IMPLEMENTATION_PLAN_2026-07-22.md)

```text
Current product level: Backend Product Alpha
Current milestone: LKW MVP — Hybrid Knowledge Workspace
Last accepted implementation:
  LKW-MODEL-RUNTIME-1
  including C1–C4 corrections and evidence v2

Architecture:
  LKW-KNOWLEDGE-ACCESS-ARCHITECTURE-1 — ACCEPTED
  LKW-CONVERSATION-CONTEXT-ARCH-1 — READY_FOR_REVIEW

Next implementation:
  LKW-KNOWLEDGE-ACCESS-1 — NEXT

Platform next (vendor knowledge):
  SLACK-KNOWLEDGE-FOUNDATION-1 — DONE

Conversation context architecture prerequisite:
  LKW-CONVERSATION-CONTEXT-ARCH-1 — READY_FOR_REVIEW

Slack Knowledge vertical next (after architecture prerequisite):
  LKW-SLACK-CONNECTED-SOURCE-1
  → LKW-CONVERSATION-CONTEXT-1
  → LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1
  → SLACK-LIVE-CAPABILITY-1
  (join: above + LKW-HYBRID-ASK-1)
  → LKW-SLACK-KNOWLEDGE-PROOF-1

LKW-CONVERSATIONAL-INTERACTION-1A → planner core implemented sufficiently to continue the product roadmap
Final target: LKW-LIVE-PLATFORM-PROOF-1 → complete demonstrable Slack platform proof
```

---

## 1. Document role and source of truth

This file is the canonical source of truth for:

- the active LKW execution order;
- the current next implementation block;
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

LKW is a **private-by-default, tenant-scoped, deployment-neutral Hybrid Knowledge Workspace**. It lets a user attach controlled knowledge sources (indexed), authorize bounded live provider reads, process indexed knowledge durably, ask natural-language questions and receive grounded answers with inspectable provenance from indexed evidence, live evidence, or both.

Binding architecture: [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md).

“Local” means user-controlled deployment and configuration, including first-class self-hosted topology. It does not mean that every source must be a local folder or that all upstream data must physically live on one device.

The knowledge portfolio grows through one LKW-owned lifecycle for **indexed** knowledge and a separate, governed path for **live** access:

```text
managed files and channel attachments
→ preconfigured local folders
→ explicit Web URLs
→ organizational vendor systems (indexed and/or live)
→ Workspace Knowledge Configuration (Indexed Sources + Live Access Bindings + Query Policy)
→ grounded Hybrid Ask with unified provenance
```

Target vendor-backed knowledge includes, without making all providers one implementation task:

- Microsoft 365, Outlook and Teams knowledge;
- OneDrive and SharePoint files;
- Jira issues and Confluence pages;
- Databricks notebooks, datasets and related knowledge assets;
- Atlan catalog assets;
- Power BI reports, dashboards and semantic knowledge;
- e-mail, chats, documents, calendars, meeting material and future organizational sources.

Microsoft Teams in vendor-access blocks means Teams-hosted organizational knowledge. It does not automatically add Microsoft Teams as a second conversational frontend.

---

## 3. Active product roadmap (functional blocks)

Canonical execution order:

```text
COMPLETED / ACCEPTED:
LKW-WORKSPACE-CONTENTS-1B-5-2
→ END-TO-END WEB_URL KNOWLEDGE INTAKE
→ including accepted C1 and C2 corrections

LKW-MODEL-RUNTIME-1
→ OLLAMA / vLLM END-TO-END PORTABILITY
→ accepted with full canonical proof and evidence v2

NEXT:
LKW-KNOWLEDGE-ACCESS-1
→ WORKSPACE CONNECTIONS, INDEXED SOURCES AND LIVE ACCESS CONFIGURATION

LKW-HYBRID-ASK-1
→ RAG + LIVE KNOWLEDGE QUERY WITH UNIFIED PROVENANCE

LKW-CONVERSATIONAL-FRONTEND-1
→ NATURAL-LANGUAGE EXECUTION AND LIVE SLACK CUTOVER
    ├── internal: LKW-CONVERSATIONAL-INTERACTION-1B (resolver + executor)
    └── internal: LKW-CONVERSATIONAL-INTERACTION-1C (Slack mixed-message cutover)

LKW-VENDOR-ACCESS-COLLABORATION-1
→ MICROSOFT 365 + JIRA + CONFLUENCE INDEXED AND LIVE ACCESS

LKW-VENDOR-ACCESS-DATA-1
→ DATABRICKS + POWER BI + ATLAN READ-ONLY ACCESS

LKW-KNOWLEDGE-LIFECYCLE-1
→ SHARED SYNCHRONIZATION, FRESHNESS, PERMISSIONS AND REMOVAL
    ├── internal: 1C synchronization lifecycle
    ├── internal: 1D provenance inspection
    └── internal: 1E safe removal

LKW-LIVE-PLATFORM-PROOF-1
→ COMPLETE DEMONSTRABLE PLATFORM PROOF
```

### 3.1 Recently accepted

| Task | User/product outcome | Status |
|---|---|---|
| `1B-5-2` | A trusted client can attach an allowed public HTTPS URL to a workspace, after which LKW durably registers, securely captures, indexes and exposes the resulting knowledge through grounded Ask using the existing Knowledge Intake lifecycle | **ACCEPTED** |
| `LKW-KNOWLEDGE-ACCESS-ARCHITECTURE-1` | Hybrid Knowledge Workspace vocabulary, indexed/live/hybrid modes, security model and product roadmap frozen for review | **ACCEPTED** |
| `LKW-MODEL-RUNTIME-1` | The same LKW workspace, document and vector index pass generation, structured planning, validated tool calling, public HTTP Ask, citations and persisted runs on Ollama `qwen2.5:14b` and vLLM `Qwen/Qwen2.5-3B-Instruct` without reindexing | **ACCEPTED** |
| `LKW-CONVERSATION-CONTEXT-ARCH-1` | Provider-neutral Conversation Context Binding with observed-audience validation, binding identity, workspace resolution, thread memory isolation, shared capability boundary and deterministic guards — frozen for review | **READY_FOR_REVIEW** |

### 3.2 Next and planned product blocks

| Block | One-sentence outcome | Status |
|---|---|---|
| `LKW-MODEL-RUNTIME-1` | The same LKW workflows run on Ollama or vLLM through configuration, and both runtimes pass planner, tool-calling and grounded-Ask proof gates | **ACCEPTED** |
| `LKW-KNOWLEDGE-ACCESS-1` | A workspace can be configured with provider Connections, discoverable Remote Resources, Indexed Sources, Live Access Bindings and bounded Query Policies without exposing credentials | **NEXT** |
| `LKW-HYBRID-ASK-1` | One workspace question can combine indexed RAG evidence with authorized live provider evidence and return one grounded answer with unified provenance | **PLANNED** |
| `LKW-CONVERSATIONAL-FRONTEND-1` | A user can operate LKW naturally through Slack or another frontend while the planner, resolver and validated executor invoke real LKW capabilities | **PLANNED** |
| `LKW-VENDOR-ACCESS-COLLABORATION-1` | LKW supports indexed and controlled live knowledge access across Microsoft 365, Jira and Confluence through provider-neutral contracts | **PLANNED** |
| `LKW-VENDOR-ACCESS-DATA-1` | LKW provides governed read-only access to Databricks, Power BI and Atlan, allowing live analytical and metadata evidence to participate in Hybrid Ask | **PLANNED** |
| `LKW-KNOWLEDGE-LIFECYCLE-1` | Indexed and live workspace knowledge share coherent freshness, permission, operation, provenance and safe-removal semantics without deleting upstream data | **PLANNED** |
| `LKW-LIVE-PLATFORM-PROOF-1` | A live Slack demonstration shows files, Web URLs, indexed vendor knowledge, live vendor queries, unified citations and Ollama/vLLM portability in one LKW workspace | **PLANNED** |

### 3.2.1 Conversation context and Slack vertical (platform + LKW)

Provider-neutral personal/shared conversation context architecture precedes shared-channel runtime and Slack connected-source work. Canonical contract: [`CONVERSATION_CONTEXT_ARCHITECTURE.md`](CONVERSATION_CONTEXT_ARCHITECTURE.md).

| Task | Owner | User outcome | Status |
|---|---|---|---|
| `LKW-CONVERSATION-CONTEXT-ARCH-1` | LKW application (docs) | Provider-neutral Conversation Context Binding with observed-audience validation, binding identity, workspace resolution, thread memory isolation, shared capability boundary and deterministic guards — frozen for review | **READY_FOR_REVIEW** |
| `SLACK-KNOWLEDGE-THREE-MODE-ARCH-1` | Platform (docs) | Architecture frozen: one Slack integration reused across indexed RAG, durable materialization and live access; frontend and knowledge-source roles separated | **DONE** |
| `SLACK-KNOWLEDGE-FOUNDATION-1` | Platform | Platform can safely read and durably synchronize selected Slack conversations (bot token + bot-membership inventory for public/private/IM/MPIM) for any Intergrax application; no new Slack command or LKW feature implied yet | **DONE** |
| `LKW-SLACK-CONNECTED-SOURCE-1` | LKW application | User can attach an approved Slack conversation to an LKW workspace, synchronize it and ask questions about its indexed history | **PLANNED** (after architecture prerequisite) |
| `LKW-CONVERSATION-CONTEXT-1` | LKW application | Durable Conversation Context Bindings, workspace audience policy, memory partitioning and evidence guards | **PLANNED** |
| `LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1` | LKW application | Slack channel/private-channel mention handling over the generic LKW context layer | **PLANNED** |
| `SLACK-LIVE-CAPABILITY-1` | Platform | Authorized applications can read bounded current Slack information at request time without waiting for complete durable synchronization | **PLANNED** |
| `LKW-SLACK-KNOWLEDGE-PROOF-1` | LKW application | User asking through Slack receives one grounded answer combining indexed Slack history, authorized live Slack evidence and other workspace sources with strict audience isolation — requires Hybrid Ask | **PLANNED** |

**Required dependency (implementation tracks):**

```text
SLACK-KNOWLEDGE-FOUNDATION-1
→ LKW-CONVERSATION-CONTEXT-ARCH-1 (architecture prerequisite)
→ LKW-SLACK-CONNECTED-SOURCE-1          # independent from conversational activation
→ LKW-CONVERSATION-CONTEXT-1            # LKW-wide prerequisite for shared adapters
→ LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1
→ SLACK-LIVE-CAPABILITY-1
```

**Final proof join (all prerequisites):**

```text
LKW-SLACK-CONNECTED-SOURCE-1
+ LKW-CONVERSATION-CONTEXT-1
+ LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1
+ SLACK-LIVE-CAPABILITY-1
+ LKW-HYBRID-ASK-1
→ LKW-SLACK-KNOWLEDGE-PROOF-1
```

`LKW-CONVERSATION-CONTEXT-1` is a prerequisite/supporting block of the wider conversational frontend execution path — not a competing planner/executor. Final Slack proof cannot claim indexed + live combined evidence before Hybrid Ask exists. `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR` follows the complete Slack user vertical.

**Available today:** The user can operate LKW through Slack DM and ask about knowledge already present in the selected personal workspace (temporary in-memory selection). Durable Conversation Context Bindings, observed-audience validation, durable personal selection, shared-channel runtime, shared thread memory, shared capability enforcement, shared source eligibility, mention/thread-continuation runtime, Slack history indexing and live Slack Ask are **not** implemented.

### 3.3 Internal implementation slices (historical identity preserved)

| Slice | Placement | Status |
|---|---|---|
| `CONV-1A` | Planner contract under `LKW-CONVERSATIONAL-FRONTEND-1` | **ACCEPTED** (sufficient to continue) |
| `CONV-1B` | Resolver + executor under `LKW-CONVERSATIONAL-FRONTEND-1` | **PLANNED** |
| `CONV-1C` | Slack natural-language cutover under `LKW-CONVERSATIONAL-FRONTEND-1` | **PLANNED** |
| `1B-5-3` | Web URL ingestion, indexing and Ask proof | **MERGED INTO 1B-5-2** |
| `1B-6-0` | LKW / VENDOR-KNOWLEDGE ownership contract | **REPLANNED** → architecture in `LKW-KNOWLEDGE-ACCESS-ARCHITECTURE-1`; implementation in `LKW-KNOWLEDGE-ACCESS-1` |
| `1B-6-1` | Connection and Remote Resource discovery | **MAPPED INTO** `LKW-KNOWLEDGE-ACCESS-1` |
| `1B-6-2` | First real provider vertical slice | **MAPPED INTO** `LKW-HYBRID-ASK-1` |
| `1B-6-3` | Additional vendor packs | **MAPPED INTO** `LKW-VENDOR-ACCESS-COLLABORATION-1` and `LKW-VENDOR-ACCESS-DATA-1` |
| `1C` | Shared synchronization lifecycle | **MAPPED INTO** `LKW-KNOWLEDGE-LIFECYCLE-1` |
| `1D` | Provenance inspection | **MAPPED INTO** `LKW-KNOWLEDGE-LIFECYCLE-1` |
| `1E` | Safe removal | **MAPPED INTO** `LKW-KNOWLEDGE-LIFECYCLE-1` |

### 3.4 Old-to-new task mapping

| Previous task | New roadmap placement | Mapping |
|---|---|---|
| `1B-5-2` + former `1B-5-3` | Expanded end-to-end `1B-5-2` | **MERGED** |
| `CONV-1B` | Internal slice of `LKW-CONVERSATIONAL-FRONTEND-1` | **MAPPED INTO** |
| `CONV-1C` | Internal slice of `LKW-CONVERSATIONAL-FRONTEND-1` | **MAPPED INTO** |
| `1B-6-0` | `LKW-KNOWLEDGE-ACCESS-ARCHITECTURE-1` + `LKW-KNOWLEDGE-ACCESS-1` | **REPLANNED** |
| `1B-6-1` | Connection and Remote Resource discovery in `LKW-KNOWLEDGE-ACCESS-1` | **MAPPED INTO** |
| `1B-6-2` | First real provider vertical slice in `LKW-HYBRID-ASK-1` | **MAPPED INTO** |
| `1B-6-3` | Collaboration and Data connector packs | **MAPPED INTO** |
| `1C` | Internal lifecycle slice of `LKW-KNOWLEDGE-LIFECYCLE-1` | **MAPPED INTO** |
| `1D` | Internal provenance slice of `LKW-KNOWLEDGE-LIFECYCLE-1` | **MAPPED INTO** |
| `1E` | Internal removal slice of `LKW-KNOWLEDGE-LIFECYCLE-1` | **MAPPED INTO** |

---

## 4. Legacy Workspace Contents execution detail (accepted foundations)

The following table preserves accepted intake foundations and the accepted WEB_URL slice. It does not override the functional block order in §3.

### Completed and accepted intake foundations

| Task | Result | Status |
|---|---|---|
| `LKW-WORKSPACE-CONTENTS-1B-0` | Channel-neutral Knowledge Intake contract | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-1` | Durable Knowledge Input → Source → operation → queue/worker foundation | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-2` | Managed-file HTTP intake | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-3` | Slack attachment and multi-attachment adapter over managed-file intake | ACCEPTED |

### Accepted intake expansions

| Task | Result | Status |
|---|---|---|
| `LKW-WORKSPACE-CONTENTS-1B-4-1` | HTTP listing and acceptance of opaque preconfigured `LOCAL_FOLDER` Source Candidates | ACCEPTED (with corrections) |
| `LKW-WORKSPACE-CONTENTS-1B-4-1-C1` | Harden public safety, candidate identity and operation evidence | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-4-2` | Slack safe numbered Source Candidate selection | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-5-1` | Shared secure Web Content Capture contract and HTTPS backend | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-5-2` | End-to-end WEB_URL Knowledge Intake | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-5-2-C1` | Harden production index path and real RAG proof | ACCEPTED (correction to `1B-5-2`) |
| `LKW-WORKSPACE-CONTENTS-1B-5-2-C2` | Close retrieval evidence and regression review | ACCEPTED (correction to `1B-5-2`) |

`LKW-CONVERSATIONAL-INTERACTION-1A` — channel-neutral structured interaction plan contract and provider-neutral LLM planner (execution not wired; sufficient to continue).

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

### 6.1 Bounded decomposition (`1B-5`)

| Task | Outcome | Status |
|------|---------|--------|
| `1B-5-1` | Shared secure Web Content Capture contract and HTTPS backend | ACCEPTED |
| `1B-5-2` | End-to-end WEB_URL Knowledge Intake | **ACCEPTED** |
| `1B-5-2-C1` | Harden production index path and real RAG proof | **ACCEPTED** (correction to `1B-5-2`) |
| `1B-5-2-C2` | Close retrieval evidence and regression review | **ACCEPTED** (correction to `1B-5-2`) |
| `1B-5-3` | Web URL ingestion, indexing and Ask proof | **MERGED INTO 1B-5-2** |
| `1B-5-4` | Slack explicit Web URL intake | **SUPERSEDED** by `LKW-CONVERSATIONAL-INTERACTION-1C` |

**Why `1B-5-4` was superseded:** the target frontend will not use a separate strict URL command; one natural message can contain multiple source types and actions (attachments, URLs, local references, workspace targets). See [`CONVERSATIONAL_INTERACTION.md`](CONVERSATIONAL_INTERACTION.md).

Platform prerequisite: [`docs/architecture/WEB_CONTENT_CAPTURE.md`](../../../docs/architecture/WEB_CONTENT_CAPTURE.md). `1B-5-1` is accepted; `LKW-CONVERSATIONAL-INTERACTION-1A` (planner contract) precedes `1B-5-2` so URL intake is designed as a planner action, not another terminal command.

---

## 7. Product block definitions

Binding architecture: [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md). The sections below summarize one-sentence outcomes. Detailed `1B-6` / `1C`–`1E` material is preserved in §8–§10 as implementation reference mapped into the blocks above.

### 7.1 `LKW-MODEL-RUNTIME-1` — Ollama / vLLM end-to-end portability

**One-sentence outcome:** The same LKW workflows run on Ollama or vLLM through configuration, and both runtimes pass planner, tool-calling and grounded-Ask proof gates.

**Status:** **ACCEPTED**.

Accepted proof pair:

```text
Ollama qwen2.5:14b
vLLM Qwen/Qwen2.5-3B-Instruct
```

The proof covers basic generation, structured Conversation Interaction planning, deterministic plan validation, native tool calling, real `local.workspace.search` execution, public HTTP Ask, verified citations, persisted Ask runs and full shared-index invariance without reindexing. Canonical evidence: [`LKW_MODEL_RUNTIME_PORTABILITY.md`](evidence/LKW_MODEL_RUNTIME_PORTABILITY.md).

Provider switch may require application restart. Conversation LLM and embedding provider remain separate — switching chat runtime must not silently reindex. Runtime hot swapping and universal model compatibility are not claimed.

### 7.2 `LKW-KNOWLEDGE-ACCESS-1` — Workspace Knowledge Configuration

**One-sentence outcome:** A workspace can be configured with provider Connections, discoverable Remote Resources, Indexed Sources, Live Access Bindings and bounded Query Policies without exposing credentials. All user-managed product configuration that must survive restart is durable; tenant Connections are stored in a platform-owned durable Connection Catalog with `SecretsStore`-owned credentials and runtime registry rehydration.

**Status:** **NEXT** (`1B` is the next internal slice).

Expected capabilities: durable tenant Connection catalog with restart-safe rehydration; connection listing and safe inspection; remote-resource discovery; workspace binding; indexed vs live selection; capability allowlists; query policy; safe connection health.

**Internal decomposition:**

| Slice | Title | Status |
|-------|-------|--------|
| `1A` | Implementation contract freeze | **ACCEPTED** (including C3 mutation semantics; C4 persistence boundary) |
| `1B` | Provider-neutral durable workspace authorization foundation | **NEXT** |
| `1C-1` | Durable tenant Connection catalog and restart rehydration | **PLANNED** |
| `1C-2` | Safe Connection / Remote Resource discovery and typed capability catalog | **PLANNED** (depends on 1B, 1C-1) |
| `1D` | HTTP create/disable for bindings with server-derived metadata | **PLANNED** (depends on 1B, 1C-1, 1C-2) |
| `1E` | Query Policy and complete configuration projection | **PLANNED** |
| `1F` | One-connection indexed/live reuse proof (with restart reconstruction) | **PLANNED** |

**Acceptance gate:** Must prove that **one durable Connection** reconstructed after restart can support both Indexed Source and Live Access Binding without duplicating credentials or integrations.

#### 7.2.1 User-visible roadmap

**Available today:**

- create/select an LKW workspace;
- add already supported indexed inputs such as files and approved Web URLs;
- ask questions against indexed knowledge;
- use the current Slack/HTTP indexed workflow where already implemented.

**After `1B`:** backend-only durable workspace authorization and configuration state; no new connector-management UI yet.

**After `1C-1`:** an administrator can configure a connector once; the connector survives restart; LKW remembers that the Connection exists; secrets remain outside LKW. This may initially be backend/admin capability rather than Slack UI.

**After `1C-2` and `1D`:** users can list available Connections; inspect discoverable resources; select which resources are indexed; separately authorize bounded live access.

**Later frontend work:** natural-language configuration and use through Slack or another frontend.

Do not claim that connector-management actions are already implemented.

### 7.3 `LKW-HYBRID-ASK-1` — Hybrid Ask with unified provenance

**One-sentence outcome:** One workspace question can combine indexed RAG evidence with authorized live provider evidence and return one grounded answer with unified provenance.

**Status:** **PLANNED**.

Microsoft Graph remains an appropriate durable synchronization and exact-read connector proof because its low-level reads and several Vendor Knowledge adapters are implemented.

Jira or Confluence is the preferred first provider-neutral live search/read capability proof because the existing integrations already expose operational search and exact-read methods.

A Microsoft Graph live search capability requires a separate bounded contract and must not be simulated through delta, reconciliation or full inventory.

Initial live access is read-only.

**Acceptance gate:** Must prove that indexed evidence and live evidence can be combined; live evidence is **not** automatically persisted; each evidence item retains mode and provenance.

### 7.4 `LKW-CONVERSATIONAL-FRONTEND-1` — Natural-language execution and Slack cutover

**One-sentence outcome:** A user can operate LKW naturally through Slack or another frontend while the planner, resolver and validated executor invoke real LKW capabilities.

**Status:** **PLANNED**. Internal slices: `CONV-1B` (resolver + executor), `CONV-1C` (Slack mixed-message cutover). See [`CONVERSATIONAL_INTERACTION.md`](CONVERSATIONAL_INTERACTION.md).

### 7.5 `LKW-VENDOR-ACCESS-COLLABORATION-1`

**One-sentence outcome:** LKW supports indexed and controlled live knowledge access across Microsoft 365, Jira and Confluence through provider-neutral contracts.

**Status:** **PLANNED**. Scope: OneDrive/SharePoint files; mail; Teams-hosted organizational knowledge; Jira issue discovery/search/state and selected project sync; Confluence space/page discovery, search/read and selected space sync.

**Acceptance gate:** Must prove at least one provider vertical slice in both paths:

```text
same vendor integration
├── durable normalized delivery toward LKW/RAG
└── bounded live query/read toward Live Evidence
```

Jira or Confluence is the preferred first provider-neutral live search/read capability proof because the existing integrations already expose operational search and exact-read methods.

A Microsoft Graph live search capability requires a separate bounded contract and must not be simulated through delta, reconciliation or full inventory.

**Execution order:**

```text
1. freeze conversation context architecture (LKW-CONVERSATION-CONTEXT-ARCH-1);
2. complete Slack Knowledge vertical (implementation tracks through SLACK-LIVE-CAPABILITY-1; final proof joins LKW-HYBRID-ASK-1 at LKW-SLACK-KNOWLEDGE-PROOF-1);
3. complete Teams Chat and Calendar durable adapters (Calendar after Slack vertical);
4. perform the adapter-family and three-mode capability audit;
5. use Jira or Confluence for the first bounded live search/read capability proof;
6. add bounded Microsoft Graph live capabilities separately;
7. converge at Hybrid Ask.
```

The immediate current platform task in the Slack vertical is complete (`SLACK-KNOWLEDGE-FOUNDATION-1` **DONE**). Current architecture prerequisite is `LKW-CONVERSATION-CONTEXT-ARCH-1` (**READY_FOR_REVIEW**); next Slack-vertical implementation task is `LKW-SLACK-CONNECTED-SOURCE-1` after architecture review. Global LKW product next task remains `LKW-KNOWLEDGE-ACCESS-1`. Microsoft Graph Teams Chat adapter is **DONE**; Calendar adapter (`MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR`) is **PLANNED** after the complete Slack user vertical.

### 7.6 `LKW-VENDOR-ACCESS-DATA-1`

**One-sentence outcome:** LKW provides governed read-only access to Databricks, Power BI and Atlan, allowing live analytical and metadata evidence to participate in Hybrid Ask.

**Status:** **PLANNED**. All queries pass capability-specific validation and policy — no unrestricted user SQL/DAX.

**Acceptance gate:** Must state per resource whether it supports durable materialization, RAG indexing, live query, or a documented subset. Do not assume Power BI, Databricks and Atlan should all be copied into RAG.

### 7.7 `LKW-KNOWLEDGE-LIFECYCLE-1`

**One-sentence outcome:** Indexed and live workspace knowledge share coherent freshness, permission, operation, provenance and safe-removal semantics without deleting upstream data.

**Status:** **PLANNED**. Consolidates former `1C`, `1D`, `1E` slices (§8–§10).

### 7.8 `LKW-LIVE-PLATFORM-PROOF-1` — Complete demonstrable platform proof

**One-sentence outcome:** A live Slack demonstration shows files, Web URLs, indexed vendor knowledge, live vendor queries, unified citations and Ollama/vLLM portability in one LKW workspace.

**Status:** **PLANNED**.

Target scenario: start with Ollama → create/select workspace → upload files → add Web URL → configure MS365, Jira, Confluence, Databricks, Power BI, Atlan → Hybrid Ask with indexed + live evidence → restart with vLLM → repeat without changing LKW domain behavior. Public claims must distinguish real provider proof, controlled integration proof and deterministic fixture proof.

**Final platform proof requirement:**

```text
one durable vendor connection (reconstructed after restart)
→ one durable/indexed use
→ one live use
→ one hybrid answer
→ no duplicated vendor client
→ explicit provenance and freshness
→ no automatic persistence of live results
```

---

## 8. `LKW-WORKSPACE-CONTENTS-1B-6` — VENDOR-KNOWLEDGE integration (historical reference)

**Status:** **REPLANNED** — scope distributed across `LKW-KNOWLEDGE-ACCESS-1`, `LKW-HYBRID-ASK-1`, `LKW-VENDOR-ACCESS-COLLABORATION-1`, `LKW-VENDOR-ACCESS-DATA-1` and `LKW-KNOWLEDGE-LIFECYCLE-1`. Retained for ownership-boundary detail.

### 8.1 Ordering rule

`1B-6` is placed after explicit Web URL intake and before the shared `1C` lifecycle:

```text
1B-5 Web URL intake
→ 1B-6 VENDOR-KNOWLEDGE integration
→ 1C shared synchronization and completion lifecycle
→ 1D document inspection and provenance
→ 1E safe source-owned knowledge removal
```

This ensures synchronization, operation inspection, completion notification, provenance and removal are designed once for local folders, Web URLs and vendor-backed Sources.

### 8.2 Ownership boundary

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

### 8.3 Architectural invariants

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

### 8.4 `1B-6-0` — integration contract and ownership gate

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

### 8.5 `1B-6-1` — safe connection and resource discovery

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

### 8.6 `1B-6-2` — vendor resource Knowledge Intake

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

### 8.7 `1B-6-3` — provider-neutral verification

Demonstrate that materially different provider families can reuse the same LKW contracts and lifecycle. The verification may use focused adapters/fakes where full production integrations are not yet justified, but it must prove that adding another vendor does not require:

- a new LKW queue or worker;
- a new operation model;
- a new Source ownership system;
- a new document indexing pipeline;
- provider credentials inside LKW public APIs;
- provider-specific business logic in Slack.

---

## 9. `LKW-WORKSPACE-CONTENTS-1C` — shared synchronization and completion lifecycle

**Mapped into:** `LKW-KNOWLEDGE-LIFECYCLE-1` (**PLANNED**).

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

## 10. `LKW-WORKSPACE-CONTENTS-1D` — document inspection and provenance

**Mapped into:** `LKW-KNOWLEDGE-LIFECYCLE-1` (**PLANNED**).

**One-sentence outcome:** A user can inspect which documents are indexed in a workspace, their Source ownership, status and safe origin without exposure of private paths, credentials or provider locators.

Safe provenance may include:

- Source label and safe source type;
- vendor/provider and resource type;
- original item type such as file, e-mail, chat message, thread, calendar event, wiki page, issue, dashboard, dataset, notebook or catalog asset;
- last synchronization time;
- safe document status and indexing metadata.

Every persisted Document remains owned by exactly one durable Source.

---

## 11. `LKW-WORKSPACE-CONTENTS-1E` — safe source-owned knowledge removal

**Mapped into:** `LKW-KNOWLEDGE-LIFECYCLE-1` (**PLANNED**).

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

## 12. Cross-cutting acceptance rules

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

## 13. LKW-PF6 — Token Optimization product proof (planned)

**Status:** **LKW-PF6-0** Done / Closed (proof design only). **LKW-PF6-A**–**C** planned after universal platform proof.

Token Optimization is a platform capability under `intergrax/runtime/token_optimization`. LKW integrates through public contracts only — no duplication of router, pipeline, cache-aware gate, or proof harness.

| Phase | Depends on | Scope |
|-------|------------|-------|
| **LKW-PF6-A** | TOKEN-10G | Baseline measurement on product workflows |
| **LKW-PF6-B** | LKW-PF6-A, TOKEN-10D | Integrate public runtime contract |
| **LKW-PF6-C** | LKW-PF6-B | Baseline vs optimized product proof |

Canonical: [`docs/features/plan/TOKEN_OPTIMIZATION.md`](../../../docs/features/plan/TOKEN_OPTIMIZATION.md) · [`ARCHITECTURE.md`](ARCHITECTURE.md) (Token Optimization subsection) · [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md) §11.

---

## 14. Post-MVP direction to LKW 1.0

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
ARCHITECTURE:
LKW-KNOWLEDGE-ACCESS-ARCHITECTURE-1
HYBRID KNOWLEDGE ACCESS AND LIVE PLATFORM PROOF ROADMAP
→ ACCEPTED

LAST ACCEPTED IMPLEMENTATION:
LKW-MODEL-RUNTIME-1
OLLAMA / vLLM END-TO-END PORTABILITY
→ ACCEPTED (including C1–C4 corrections and evidence v2)

NEXT:
LKW-KNOWLEDGE-ACCESS-1
→ Connections, Remote Resources, Indexed Sources, Live Access Bindings and Query Policy

THEN (functional blocks):
LKW-HYBRID-ASK-1 → RAG + live with unified provenance
→ LKW-CONVERSATIONAL-FRONTEND-1 (CONV-1B + CONV-1C internal)
→ LKW-VENDOR-ACCESS-COLLABORATION-1
→ LKW-VENDOR-ACCESS-DATA-1
→ LKW-KNOWLEDGE-LIFECYCLE-1 (1C + 1D + 1E internal)
→ LKW-LIVE-PLATFORM-PROOF-1

FINAL TARGET:
Live Slack platform proof — indexed + live + Hybrid Ask + Ollama/vLLM portability

ARCHITECTURE DOC:
KNOWLEDGE_ACCESS_ARCHITECTURE.md — indexed/live/hybrid, Connections,
Live Access Bindings, Workspace Knowledge Configuration, Query Policy,
Knowledge Query Orchestrator, Evidence Items, MCP boundary, security.

Conversational planning (CONV-1A):
- LLM produces a strict semantic draft (`ConversationInteractionDraft`).
- The adapter automatically derives the draft JSON Schema from its Pydantic class.
- Application code deterministically compiles technical IDs and exact evidence spans (`interaction_plan_compiler`).
- The unchanged `ConversationInteractionPlan` remains the canonical plan contract.
- The existing deterministic request validator remains the final grounding boundary.
```

## Appendix B — Status vocabulary

| Label | Meaning |
|---|---|
| `ACCEPTED` | Implementation or architecture was independently audited and accepted |
| `NEXT` | First bounded implementation block to be started next |
| `CURRENT` | Task actively being implemented |
| `PLANNED` | Accepted roadmap block not yet started |
| `DOCUMENTED / READY_FOR_REVIEW` | Architecture or contract documented; implementation not claimed |
| `DEFERRED` | Intentionally outside the active execution path |
| `HISTORICAL` | Retained record that does not define current status |
