# Local Knowledge Workspace (LKW) — architecture

**Status:** Architecture baseline v2 (2026-06-07) — implementation-plan source of truth  
**Tier:** Tier-3 application (`local_workspace_application`)  
**Agents:** Tier-2 `local_indexer`, `local_search`, `local_synthesizer`  
**Canonical plan row:** [`docs/project/architecture/intergrax_runtime_architecture.md` §6.3a LKW.*](../../../architecture/intergrax_runtime_architecture.md#63a-business-backlog-register-consolidated)
**Derived plan:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — generated from this document; do not fork scope elsewhere  
**Public product-validation narrative:** [`docs/project/proofs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md`](../../../proofs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md)

---

## 0. How to use this document

This file is the **single product architecture** for LKW. From it you derive:

| Need | Read section |
|------|----------------|
| Product philosophy, boundaries | §3 · §4 |
| Deployment, storage, tenancy (canonical) | [Deployment, storage and tenancy model](.#deployment-storage-and-tenancy-model) |
| Platform capability audit / architecture stop gate | [Mandatory platform capability audit and architecture stop gate](.#mandatory-platform-capability-audit-and-architecture-stop-gate) · [`PRODUCT_FIRST_MVP.md`](../../../maintainers/plans/PRODUCT_FIRST_MVP.md#mandatory-platform-capability-audit-and-architecture-decision-gate) |
| Knowledge Intake / async ingestion (canonical) | [Channel-neutral Knowledge Intake and asynchronous ingestion](.#channel-neutral-knowledge-intake-and-asynchronous-ingestion) · [`KNOWLEDGE_INTAKE_DISCOVERY.md`](KNOWLEDGE_INTAKE_DISCOVERY.md) |
| Hybrid knowledge access (indexed + live) | [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md) |
| Hybrid Ask (unified evidence + live execution) | [`HYBRID_ASK_ARCHITECTURE.md`](HYBRID_ASK_ARCHITECTURE.md) |
| What is frontend vs backend | §4 |
| Solution + trust zones | §5 |
| Agent roster | §6 |
| Install / upgrade / uninstall | §7 |
| Integrations, tools, skills · LKW.4 background jobs | §8 |
| Runtime + Slack (optional) | §9 |
| Request flows | §10 |
| Implementation waves + acceptance | §15 |
| Env vars and paths on disk | §7.3 · §12 |

**Rule:** change architecture first, then update [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) and platform [`§6.3a`](../../../architecture/intergrax_runtime_architecture.md#63a-business-backlog-register-consolidated). One coherent diff per wave.

---

## 1. Strategic purpose

**Local Knowledge Workspace (LKW)** is the first **business product environment** on Intergrax after harness platform maturity. Its role is dual:

1. **Product:** Give a user a private-by-default, tenant-scoped, deployment-neutral **Hybrid Knowledge Workspace** — indexed RAG knowledge, controlled live access to external systems, natural-language frontends, unified evidence provenance, and structured outputs (reports, emails, estimates). Binding detail: [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md).
2. **Harness validation:** Exercise the Agent OS on a real, observable workload without external market APIs (unlike deferred K.1 Problem Radar / K.2 Vendor Discovery).

**What “Local” means in the product name:** the user controls deployment and configuration; full self-hosted / fully local topology remains first-class; LKW does not force a central SaaS. It does **not** mean that all data must always reside on a single user device, nor that remote storage, private enterprise hosting, hybrid topologies, or future controlled sharing are out of scope. Canonical detail: [Deployment, storage and tenancy model](.#deployment-storage-and-tenancy-model).

LKW validates: indexed RAG ingest/retrieve/index lifecycle, governed live knowledge access, document parsing, shadow workspace, multi-agent orchestration, memory, policy, trace, MCP/HTTP serving, provider-neutral model runtime wiring, and Tier-3 composition — while surfacing platform gaps early.

**Strategic frame:** [`docs/project/technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) — explicit product reprioritization after Appendix A sign-off.

---

## 2. Problem statement

Users store and access project knowledge across local folders, uploaded files, Web URLs and organizational systems (Microsoft 365, Jira, Confluence, Databricks, Power BI, Atlan and future providers). They need to:

| Need | Example |
|------|---------|
| **Find** | "Find documents about project X / settlement Y" |
| **Gather** | "Gather data from folders A and B about the cost estimate" |
| **Current state** | "What are today's Jira blockers for Project Orion?" |
| **Hybrid** | "Are we ready to deploy — using our plan, latest client mail and current KPI?" |
| **Synthesize** | „Przygotuj mail / sprawozdanie / kosztorys wg szablonu” |
| **Safety** | Do not delete or overwrite user original files; credentials never in chat |

LKW solves this with **indexed semantic retrieval**, **authorized live provider reads**, **Hybrid Ask with unified provenance**, and **isolated write artifacts**, orchestrated by Nexus. Indexed knowledge and live access are separate, composable capabilities — see [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md).

---

## 3. Product philosophy

### 3.1 What LKW is

LKW is a **private, governed, provider-neutral Hybrid Knowledge Workspace** whose product interface, domain model, API, Slack companion, Ask workflow, Workspace Knowledge Configuration and source lifecycle stay the same across topologies. A common first-class shape is a personal Agent OS instance under user control (often on the user's machine), but that shape is a deployment choice — not the definition of the product:

- **Always-on LKW host** owns product capabilities, policy, and orchestration wiring; persistent stores are selected by configuration / provider implementations.
- **Thin frontends** (tray, Cursor MCP, Slack, scripts) only invoke **capabilities / API** and show **results**.
- **Intergrax Nexus** is the only orchestrator — no ad-hoc agent loops in UI code.
- **Private by default** and **tenant-scoped** — privacy comes from the logical access model, not solely from “the database is local.”

### 3.2 What LKW is not

| Not this | Why |
|----------|-----|
| Slack bot that “is” the product | Slack is an **optional client**; product logic and storage stay behind LKW capabilities/API |
| Local-only product that forbids remote storage | Storage location is configuration; remote / private-cloud / hosted providers are in scope when wired |
| Public-by-default multi-tenant dump | New private workspaces are not automatically visible to others; sharing is future and controlled |
| RAG-only workspace | Indexed knowledge is one mode; live access and Hybrid Ask are first-class product capabilities ([`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md)) |
| Single monolithic “chat agent” | Bounded agents + graph pipeline; Conversation Interaction Planner and Knowledge Query Orchestrator have separate responsibilities |
| Unrestricted filesystem agent | Read allowlist + shadow-only writes for local-folder originals |
| Replacement for Nexus / Tier-0 | Composition and wiring only — reuse platform mechanisms |

### 3.3 Design principles (non-negotiable)

1. **Deployment-neutral product, multi-channel control** — domain/API/Slack do not branch on local/cloud/self-hosted/SaaS modes; providers are selected by host configuration; Slack/HTTP/MCP are equal task transports.
2. **Source originals stay read-only under product policy** — agents must not delete or overwrite user source material (local-folder files today; other source providers later under the same rule).
3. **Integration → Tool → Skill → Agent** — no vendor SDKs in Tier-2; Tier-3 wires profiles.
4. **Every surface → one Task** — same trace, policy, and agents regardless of UI.
5. **Slack optional** — product must work without Slack (HTTP/MCP); Slack is an enhancement client.
6. **Harness honesty** — gaps discovered during LKW feed back to Tier-0 plan, not Nexus forks.
7. **Authorization over physical isolation** — physical storage isolation may strengthen privacy, but authorization must not depend on physical isolation alone.

### 3.4 Primary vs optional user journeys

| Journey | Channel | Required wave |
|---------|---------|---------------|
| Developer at desk | MCP / HTTP | LKW.0–1 |
| Background index of folders | Daemon + watcher | LKW.7 |
| Quick search anytime | HTTP / tray | LKW.1 + LKW.8 |
| Remote command from phone | Slack slash | LKW.6b |
| Approve draft report | Slack HITL / HTTP | LKW.2 + notify |

---

## Deployment, storage and tenancy model

**Status:** architectural contract frozen by `LKW-STORAGE-TENANCY-CONTRACT-1` (`DOCUMENTED / READY_FOR_REVIEW`).

This section is the **canonical** definition of deployment neutrality, storage location, tenancy, and source boundaries. Implementation status of individual ports/providers is **not** implied here.

### Deployment-neutral product

LKW does **not** mean “storage always on the user's computer.”

LKW means a knowledge workspace whose:

- product interface,
- domain model,
- API,
- Slack companion,
- Ask workflow,
- source lifecycle,

remain the same regardless of deployment topology.

Allowed topologies (none of these is the domain-default architecture):

| Topology | Typical shape |
|----------|----------------|
| **Fully local** | LKW host local; Document Store local; Vector Store local; sources local |
| **Local application with cloud storage** | LKW host or connector local; Document Store remote; Vector Store remote; sources local or remote |
| **Fully hosted** | host in cloud; persistent stores in cloud; sources uploaded or connected remotely |
| **Private enterprise** | host and storage in organization infrastructure; same domain/product contract |
| **Hybrid** | components in different locations; no change to domain logic |

Docker Compose is the current **reference developer / self-hosted proof** deployment. It is **not** the definition of the product.

Domain, API, Slack companion, and user workflow must **not** have separate product paths such as “local mode”, “cloud mode”, “self-hosted mode”, or “SaaS mode.” Deployment difference is resolved by configuration and provider implementations behind stable ports.

### Storage location is configuration

```text
storage location = deployment configuration
not domain behavior
```

The domain must **not** ask:

```text
if local:
    ...
elif cloud:
    ...
```

Conceptual dependency:

```text
domain / application service
  → stable storage or connector port
  → provider selected by configuration
  → local or remote implementation
```

Conceptual capability boundaries (not all must exist in complete form today; do not treat planned interfaces as implemented):

- **Source Connector**
- **Document Store**
- **Vector Store**
- **Blob/Object Store** (optional; for managed originals / uploads)

Prefer independent capability boundaries. Avoid collapsing Document Store and Vector Store into one giant “StorageProvider” when the architecture already separates them.

### Four (plus optional fifth) locations

Always distinguish:

1. where the **LKW host** runs;
2. where the **source** data lives;
3. where **application / document state** (Document Store) is persisted;
4. where **vectors** (Vector Store) are persisted;
5. optionally, where **managed original files / blobs** are stored (Blob/Object Store).

Valid example:

```text
source: folder on user's computer
LKW connector/host: user's computer
Document Store: cloud Mongo-compatible provider
Vector Store: cloud Qdrant-compatible provider
```

Another valid example:

```text
source: remote object storage
LKW host: private server
Document Store: local/private database
Vector Store: local/private vector database
```

**Source location does not determine knowledge-storage location.**

### Diagram 1 — deployment-neutral flow

```text
Slack / Web / Mobile / Desktop / Teams / Telegram / MCP / HTTP
        |
        v
LKW product capabilities
        |
        +--> Knowledge Intake  (channel-neutral; durable Ingestion Operation)
        +--> Source Connector
        +--> Document Store
        +--> Vector Store
        +--> optional Blob Store
                 |
                 v
        providers selected by configuration
        local / remote / private cloud / hosted

Frontends collect channel-native input and invoke public capabilities.
They do not talk to Document Store, Vector Store, or Blob providers directly.
```

### Private by default

Every principal has access only to spaces for which they have an explicit grant/permission.

A newly created private workspace is **not** automatically visible to other principals.

Privacy must **not** rest only on “the database is local” or “each user has a separate deployment.”

```text
physical storage isolation may strengthen privacy,
but authorization must not depend on physical isolation
```

### Tenant-scoped and workspace-scoped invariant

Every durable record belonging to a workspace should carry a clear ownership scope.

**Required scope fields (invariant / implementation gate):**

- `tenant_id`
- `workspace_id`

Applies at least to workspace-owned state such as:

workspace; sources; document references; extracted document state; chunks; vectors; sync operations; Ask runs; future workspace-owned artifacts.

This is a **binding invariant for future implementation**. It does **not** claim that every existing record already satisfies the invariant without verification.

Operational rules:

- reads must be tenant-scoped;
- writes must be tenant-scoped;
- search must be tenant + workspace scoped;
- deletes must be tenant + workspace scoped;
- cross-tenant lookup remains fail-closed;
- missing tenant/workspace filters in a provider must **not** be compensated by post-fetch filtering when the provider can perform an isolated operation.

### Tenant is not permanently equal to user

Do **not** freeze `tenant_id == user_id` as the architecture.

- **Tenant** — ownership / administration space (personal tenant, organization tenant, private hosted tenant, enterprise tenant).
- **Principal** — actor performing the operation (user, bot, service, application agent).

The current stage may use a simplified tenant context. Target access model (capability direction, **not** implemented entities):

```text
principal
  → membership in tenant
  → workspace permission
```

Entity/code names are not frozen here.

### Diagram 2 — access and ownership

```text
Principal
   |
   v
Tenant membership / future grant   ← FUTURE / NOT IMPLEMENTED
   |
   v
Workspace
   |
   +--> Sources
   +--> Documents
   +--> Vectors
   +--> Ask runs
```

Notes:

- membership / grants are **future**;
- tenant / workspace scoping is already the isolation direction;
- physical storage location does **not** replace authorization.

### Future organization and sharing — FUTURE / NOT IMPLEMENTED

**Organizational workspace** — a future organization tenant may own a workspace available to multiple principals via membership and roles.

**Workspace sharing** — a future owner/admin may grant another principal, group, or application limited access.

Possible future access scopes (direction only): discover/list; read; ask; contribute sources; synchronize; manage; share/administer.

This is architectural direction, **not** current MVP scope. Do **not** treat ACL, role enums, membership repositories, invitations, share links, public workspaces, or cross-tenant grants as implemented.

### Source architecture contract

```text
source = durable logical origin of knowledge associated with a workspace
```

A Source may be:

- **connector-backed** (e.g. connected local folder, remote drive, object storage);
- **managed-upload-backed** (files or folder snapshots copied under LKW-managed storage policy);
- **web-resource-backed** (explicit URL intake under policy);
- **future application-feed-backed** (application-to-application knowledge feed).

A Source is **not** defined as “a local filesystem path.” **local-folder** is the first **implemented** source type / provider. It is **not** the domain definition of Source.

Illustrative future source types (examples of the architectural boundary — **not** committed roadmap deliverables):

local folder; uploaded file; uploaded folder snapshot; object storage; Google Drive; SharePoint; S3-compatible storage; remote repository; business system; application-to-application knowledge feed.

**Source locator is provider-specific** and must not leak into remote chat frontends. Do not freeze a universal filesystem `path` as the only source representation. Conceptually, connector-backed sources operate on `source_type` + provider-specific locator + options behind the LKW boundary (e.g. local-folder → filesystem path on the connector host; object-storage → bucket/key or URI; remote-drive → provider resource id; upload → managed blob id).

The existing **LOCAL_FOLDER** model is the first vertical slice. Later source-lifecycle work must not entrench “every source is a local path.” Slack and other remote chat adapters must not perform direct filesystem operations and must not accept raw local paths as product commands. Binding intake contract: [Channel-neutral Knowledge Intake and asynchronous ingestion](.#channel-neutral-knowledge-intake-and-asynchronous-ingestion) · [`KNOWLEDGE_INTAKE_DISCOVERY.md`](KNOWLEDGE_INTAKE_DISCOVERY.md).

### Original file storage

Separate conceptually:

| Concern | Store |
|---------|--------|
| External connected source | Source Connector / external system |
| Managed uploaded original | Blob/Object Store capability |
| Source, document and operation metadata | Document Store |
| Chunks / embeddings / search index | Vector Store |
| Temporary transfer data | Upload/session provider |
| Channel correlation | Conversation notification/correlation storage |

Document Store and Vector Store must **not** be automatically identified with storage of original uploaded files.

For managed uploads a Blob/Object Store provider may be required.

```text
architectural boundary defined
provider and product behavior not yet implemented
```

### Mandatory platform capability audit and architecture stop gate

LKW is an application-first platform proof.

Before LKW implements queueing, workers, events, persistence, uploads, notifications, connectors or provider lifecycle:

```text
→ audit existing Intergrax capabilities;
→ verify contract + implementation + provider + wiring + tests;
→ classify actual maturity;
→ reuse or improve existing platform code when appropriate;
→ keep LKW product semantics in the LKW domain;
→ stop for architecture decision when ownership or implementation state is unclear.
```

```text
Target platform architecture is not implementation evidence.
```

A documentation claim, roadmap item, interface name or target architecture is not evidence that a mechanism is implemented and usable.

```text
The platform capability audit is performed by the architecture/review workflow
before an implementation instruction is issued.

Cursor receives the accepted result as a narrow implementation contract.
```

```text
Cursor may verify only explicitly listed assumptions in the named scope.
It must not reopen architecture discovery during implementation.
```

Governing global rule: [`PRODUCT_FIRST_MVP.md` — Mandatory platform capability audit and architecture decision gate](../../../maintainers/plans/PRODUCT_FIRST_MVP.md#mandatory-platform-capability-audit-and-architecture-decision-gate).

Deployment-neutral rules in this document remain unchanged: local or hosted LKW; endpoint appropriate to deployment; storage selected through configuration; local, remote, cloud and hybrid providers; no local/cloud branches in domain logic.

#### Correct reuse

```text
LKW needs durable background execution
→ architecture/review audits TaskQueue / MessageBus / WorkerRuntime
→ reuse existing capability when sufficient
→ improve shared capability only for a verified missing requirement
→ LKW registers product-specific ingestion handler
```

#### Incorrect duplication — REJECTED

```text
LKW needs asynchronous work
→ create LkwPrivateQueue
→ create SlackUploadWorker
→ bypass platform task contracts
```

#### Correct domain ownership

```text
Knowledge Input and Ingestion Operation
→ LKW domain

task delivery and worker execution
→ Intergrax platform
```

Platform task state and product-domain operation state remain separate. Queue/task status must not automatically replace LKW Ingestion Operation state.

#### Token Optimization — platform contract, LKW product proof

Token Optimization is a **universal Tier-0/runtime platform capability**. LKW is a **later product client** — it must not own or duplicate Token Optimization mechanisms.

```text
LKW (Tier-3)
  → public Token Optimization contracts
  → intergrax/runtime/token_optimization
  → LLMAdapter / provider adapter
```

Forbidden: `intergrax/runtime/token_optimization` importing or special-casing LKW.

**Canonical ordering:**

```text
TOKEN-10A … TOKEN-10G → universal platform proof passes
TOKEN-10H             → checked-in proof and public wording
LKW-PF6-A             → LKW baseline measurement (product workflows)
LKW-PF6-B             → LKW integrates public runtime contract
LKW-PF6-C             → LKW baseline-vs-optimized product proof
```

| Phase | LKW supplies | LKW consumes (must not reimplement) |
|-------|--------------|-------------------------------------|
| **LKW-PF6-A** | Real workflows: search, evidence assembly, synthesis, tool exposure, conversational steps | Baseline measurement only |
| **LKW-PF6-B** | Product policy/profile, source classifications, evidence, tenant/run/step identity, explicit enablement | Stable prompt contract, router, cache-aware gate, pipeline, receipts, metrics |
| **LKW-PF6-C** | Product proof corpus and acceptance criteria | Same runtime path as universal proof |

**LKW-PF6-0** (proof design) is **Done / Closed** — does not close platform or product proof.

Canonical feature docs: [`docs/project/capabilities/architecture/TOKEN_OPTIMIZATION.md`](../../../capabilities/architecture/TOKEN_OPTIMIZATION.md) · [`docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md`](../../../capabilities/plan/TOKEN_OPTIMIZATION.md). Public claims: [`docs/project/capabilities/TOKEN_OPTIMIZATION_CLAIMS.md`](../../../capabilities/TOKEN_OPTIMIZATION_CLAIMS.md).

### Channel-neutral Knowledge Intake and asynchronous ingestion

**Status:** architectural contract frozen by `LKW-WORKSPACE-CONTENTS-1B-0` (`DOCUMENTED / READY_FOR_REVIEW`). Binding product detail: [`KNOWLEDGE_INTAKE_DISCOVERY.md`](KNOWLEDGE_INTAKE_DISCOVERY.md). Managed-file upload, Source Candidate intake and end-to-end `WEB_URL` indexed intake are **ACCEPTED** (`1B-5-2`, including C1 and C2); Slack natural-language URL execution remains `CONV-1C`.

### End-to-end `WEB_URL` Knowledge Intake (`1B-5-2`)

```text
trusted HTTP client
→ POST /v1/local_workspace/workspaces/{workspace_id}/knowledge/web-urls
→ tenant and workspace authorization
→ WebUrlAccessPolicy canonicalization + DNS/SSRF preflight
→ private WebUrlSourceLocator (not part of public Source projection)
→ durable KnowledgeInput (kind web_url)
→ durable WEB_RESOURCE Source (path="", recursive=false)
→ KNOWLEDGE_INGESTION operation
→ existing queue and worker
→ WebContentCapture
→ normalized UTF-8 staging text
→ WorkspaceDocumentIndexingService
→ Document / Chunks / Vectors
→ grounded Ask with safe provenance (safe_display_url only)
```

**Ownership split:**

| Layer | Owns |
|-------|------|
| `WebContentCapture` | URL parsing/canonicalization; scheme/port/host policy; DNS; private-network/SSRF blocking; redirect validation; pinned HTTPS transport; content-type/size limits; extraction; safe capture result and stable errors |
| LKW | Tenant/workspace authorization; idempotency; private locator persistence; `KnowledgeInput`; Source identity; operation lifecycle; queue/worker dispatch; document ownership; indexing; Ask provenance; local retention/removal |

The private canonical URL lives only in `WebUrlSourceLocator`. It is never stored on `WorkspaceSource`, in `submission_metadata`, queue payloads, or public API responses.

**Knowledge Intake** is a core LKW capability. LKW / Intergrax owns the complete knowledge intake and ingestion lifecycle: acceptance, durable operation state, queue dispatch, extraction/parsing/chunking/embedding, Document Store / Vector Store / managed-original persistence, retry classification, status events, idempotency, and tenant/workspace isolation.

**Frontends are replaceable.** Slack, web, mobile, desktop, Teams, Telegram, MCP, CLI and HTTP clients collect channel-native user input, invoke the same public LKW capabilities, and display accepted/progress/completion states. They contain no ingestion, RAG, storage or provider logic. Do not define channel-specific product APIs such as `/slack-upload`. The LKW core must not branch on `channel == "slack"`.

**Distinct concepts (must not be used as synonyms):**

| Concept | Meaning |
|---------|---------|
| **Knowledge Input** | Channel-neutral request or item submitted to introduce knowledge; captures submission intent; does **not** directly own persisted Documents; not automatically a durable resynchronizable Source |
| **Source** | Durable logical origin and ownership boundary for persisted knowledge in a workspace |
| **Document** | Persisted processed knowledge unit owned by exactly one durable Source |
| **Ingestion Operation** | Durable execution record that processes a Source created or resolved from a Knowledge Input, or synchronizes an existing Source; source of truth for execution state (`accepted` → `queued` → `processing` → `completed` \| `failed`) |
| **Intake Batch** | Logical grouping of multiple item-level Knowledge Inputs from one user action (not an aggregate Knowledge Input kind) |

**Binding ownership invariant:** every persisted Document belongs to exactly one durable Source. No persisted Document exists without Source ownership. Do **not** allow Knowledge Input → persisted Document without Source.

**Input-kind direction (contract vocabulary — not implementation claims):**

| Input kind | Managed original | Resynchronizable | Notes |
|------------|-----------------:|-----------------:|-------|
| `managed_file` | Yes | No (unless later replaced) | Single uploaded file → managed-upload-backed Source → Document |
| `uploaded_folder_snapshot` | Yes | No | Copied snapshot Source; **not** a live connector; many Documents |
| `source_candidate` | Provider-dependent | Usually yes | Opaque candidate id + safe label; resolve/create connector-backed Source |
| `web_url` | No (captured text indexed; private URL in locator only) | Policy-dependent | Explicit intake via `POST …/knowledge/web-urls`; Ask text with a URL must not auto-ingest |

**Submission grouping (not an input kind):** multi-file / multi-item submissions use **Intake Batch** → N item-level Knowledge Inputs → N Sources → N Ingestion Operations. Aggregate `managed_file_batch` Knowledge Input kind is **REJECTED**.

```text
uploaded folder
→ snapshot copied into managed storage
→ one snapshot-backed Source
→ many Documents
→ no live synchronization

connected local folder
→ connector remains attached to original location
→ future synchronization possible
→ remote chat sees only safe candidate identity and label
```

**Raw local paths in remote chat interfaces are prohibited** (e.g. `source add C:/...` or POSIX equivalents). Reasons include path disclosure, ambiguous host, no filesystem guarantee, deployment-neutrality violation, and unsafe FS surface. Only trusted local-capable surfaces may choose a local path and convert it behind the LKW boundary into a safe Source Candidate.

**Always operation-based:** every ingestion exposes the asynchronous operation contract. Do not split “small file = sync” vs “large file = async.” Upload/transfer and ingestion acceptance are separate phases for managed bytes.

**Queue / worker vs events:**

```text
Knowledge Input
→ resolve or create Source
→ durable Ingestion Operation (source of truth)
→ platform queue / message-bus capability
→ ingestion worker
→ parse → Documents (owned by Source) → chunks → embeddings → stores

Pub/sub or event bus = fan-out / notification only
(not the operation store)
```

**Multi-item cardinality:**

```text
Intake Batch
├── Knowledge Input A → Source A → Ingestion Operation A → Document(s) A
├── Knowledge Input B → Source B → Ingestion Operation B → Document(s) B
└── Knowledge Input C → Source C → Ingestion Operation C → Document(s) C
```

Reuse an existing Intergrax queue/message-bus/outbox capability when verified sufficient; do not invent an LKW-specific queue framework merely for this feature; classify a platform gap only after implementation audit. Do **not** claim the current platform queue or a production durable worker already satisfies this contract unless verified.

**Notification correlation:** LKW core never calls Slack (or other channels) directly. Completion emits a channel-neutral lifecycle event; a notification/correlation adapter resolves the destination (Slack thread, websocket, mobile push, Teams, …). Conversation Correlation must not become Source identity or ingestion domain behavior. Slack thread is not a queue, operation store, or retry mechanism.

**Lifecycle alignment:** Source ownership enables coherent contents lifecycle — intake (1B) → sources (1A) → ingestion / synchronization (1C) → documents (1D) → removal of source-owned knowledge (1E).

**Invariants (implementation gates):** every durable Knowledge Input, Source, Document, Ingestion Operation, batch relation and managed original is scoped by `tenant_id` + `workspace_id` (fail closed). Every persisted Document must carry Source ownership. Repeated delivery of channel events, uploads, queue messages or completion events must not create unintended duplicate sources/documents/embeddings.

### Hybrid Knowledge Access (indexed + live)

**Status:** architectural contract frozen by `LKW-KNOWLEDGE-ACCESS-ARCHITECTURE-1` (`ACCEPTED`). Binding detail: [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md).

LKW is not defined solely by RAG. The target product combines:

- **Indexed Sources** — durable workspace knowledge ingested through Knowledge Intake;
- **Live Access Bindings** — workspace-scoped, read-only, allowlisted query-time provider capabilities;
- **Hybrid Ask** — one grounded answer from normalized indexed and live **Evidence Items** with unified provenance.

**Current vs target Ask:** Today, `WorkspaceAskService` implements **indexed-only** Ask (`local.workspace.search` → `WorkspaceSearchHitV1` → `AskAnswerAssembler`). Hybrid Ask — combining indexed RAG with authorized read-only live provider evidence in one response — is **not implemented**. Target contract: [`HYBRID_ASK_ARCHITECTURE.md`](HYBRID_ASK_ARCHITECTURE.md) (**ACCEPTED / CLOSED**, task `LKW-HYBRID-ASK-ARCH-1`). Implementation block `LKW-HYBRID-ASK-1` is **IN_PROGRESS** (`1A` **READY_FOR_REVIEW**).

**Frozen architecture highlights** (detail in Hybrid Ask doc):

| Concept | Frozen decision |
|---------|-----------------|
| Descriptor catalog | `TenantLiveCapabilityCatalog` — metadata only; no provider invocation |
| Integration resolver | `TenantConnectionIntegrationResolverPort` → `KnowledgeConnectionRegistry`; not `TenantConnectionCapabilityReadService` |
| Executable handlers | `LiveCapabilityHandlerRegistry` keyed by `provider_id` + `integration_kind` + `capability_id` |
| Executor | `WorkspaceLiveCapabilityExecutor` — no direct provider branches |
| Transient evidence | `WorkspaceEvidenceV1` with `content` — in-memory during synthesis only |
| Durable evidence | `PersistedAskEvidenceV2` in `WorkspaceAskRunV2` — provenance only under `EPHEMERAL` |
| HTTP V1 | `POST/GET /v1/local_workspace/.../ask` — indexed-only; unchanged contract |
| HTTP V2 | `POST/GET /v2/local_workspace/.../ask` — `indexed_only`, `live_only`, `hybrid`; no `Accept`/`api_version` negotiation |
| Absent Query Policy | `indexed_only` → no error; `live_only`/`hybrid` → `query_policy_required` (409) |
| First live proof | Vendor Knowledge delivers first accepted `LiveCapabilityHandler`; not LKW-owned |

**Knowledge Intake** introduces or synchronizes durable indexed knowledge. **Live Access Binding** authorizes bounded query-time reads. Live provider results do not automatically become Documents.

Conversation/reasoning LLM (`LLMAdapter`, Ollama or vLLM via wiring) is separate from the embedding provider. Provider-neutral model runtime wiring is part of current architecture. Ollama/vLLM portability has an **accepted bounded proof** (`LKW-MODEL-RUNTIME`); see [`PROOFS.md`](../../../proofs/PROOFS.md). This does not imply runtime hot swapping, complete provider parity, production readiness, all-provider certification, automatic embedding changes, or no-restart switching.

### Application and adapter boundaries

**Slack is an optional frontend.** The Slack companion:

- does not decide local vs remote deployment;
- does not know a concrete database provider or Qdrant topology;
- does not write directly to Document Store or Vector Store;
- does not perform storage-specific branching;
- does not parse, chunk, embed, or run a second ingestion pipeline;
- does not become the source of truth for Ingestion Operation state;
- calls public LKW capabilities / API (including future Knowledge Intake).

**Separately, Slack may be attached as a provider-backed knowledge source** through platform Connections, approved Slack Remote Resources, Indexed Sources and/or Live Access Bindings. That path uses the same `SlackConversationChannelIntegration` foundation at the platform layer but has an independent authorization lifecycle.

```text
Slack frontend enabled  does not imply  Slack history ingestion or live access
Slack Indexed Source    does not imply  Slack Live Access Binding
```

LKW must not construct Slack SDK clients, read Slack API directly, store raw Slack tokens, implement provider paging, own provider cursors or implement Slack-specific synchronization. Application-local Slack clients and synchronization are prohibited.

A Slack command must behave the same regardless of where storage is located. Binding product detail for Slack: [`SLACK_MVP_DISCOVERY.md`](SLACK_MVP_DISCOVERY.md). Binding intake contract: [`KNOWLEDGE_INTAKE_DISCOVERY.md`](KNOWLEDGE_INTAKE_DISCOVERY.md). Binding knowledge access: [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md). Binding conversation context: [`CONVERSATION_CONTEXT_ARCHITECTURE.md`](CONVERSATION_CONTEXT_ARCHITECTURE.md). This section remains canonical for tenancy/storage and Knowledge Intake boundaries.

### Conversation Context Binding and memory partitions (LKW-owned)

**Conversation Context Binding** is an LKW application-domain durable product relationship — not a `slack_companion`, `SlackConversationChannelIntegration`, or `vendor_knowledge` concern.

It binds: one tenant + one conversational frontend connection + one external conversation (semantic identity: `tenant_id` + `conversation_connection_ref` + `opaque_conversation_ref`; at most one `ACTIVE` binding) + one audience mode (`PERSONAL` | `SHARED`) + workspace resolution policy (`FIXED_WORKSPACE` | `PERSONAL_SELECTION`) + one activation policy + thread context policy.

**Ingress:** provider adapters produce `ConversationIngressContext` with `observed_audience` (`PERSONAL` | `SHARED` | `UNKNOWN`). `binding.audience_mode` must match `ingress.observed_audience` before workspace resolution, memory lookup or Ask. `UNKNOWN` or mismatch fails closed.

**Persistence owner:** LKW application domain (future `LKW-CONVERSATION-CONTEXT-1`). Provider adapters map external conversation addresses into opaque refs and observed audience; they do not choose workspaces, merge memory partitions, or authorize evidence.

**Conversation-level state** (workspace selection, preferences) is separate from **thread-level memory** (keyed by `tenant_id` + `conversation_context_binding_id` + `canonical_thread_ref`). Partitions are separate for `PERSONAL` and `SHARED` contexts. `PERSONAL` context cannot be resolved from a `SHARED` conversation. No automatic copying between partitions or threads.

V1 shared conversations default to `READ_ONLY_ASK`; administrative binding mutations require an authorized admin path.

Canonical contract: [`CONVERSATION_CONTEXT_ARCHITECTURE.md`](CONVERSATION_CONTEXT_ARCHITECTURE.md).

**Domain / application services:**

- do not recognize “cloud mode” or “local mode”;
- operate on tenant / workspace context;
- receive provider implementations through wiring;
- a provider may be local or remote;
- one deployment may use different providers simultaneously.

---

## 4. Frontend vs backend boundaries

### 4.1 Layer map

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  FRONTEND (thin clients — no agent logic, no direct RAG)                 │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌─────────────────┐ │
│  │ LKW Tray    │ │ Cursor MCP  │ │ Slack client │ │ curl / scripts  │ │
│  │ (LKW.8)     │ │ (LKW.0)     │ │ (LKW.6b)     │ │ (LKW.0)         │ │
│  └──────┬──────┘ └──────┬──────┘ └──────┬───────┘ └────────┬────────┘ │
│         │               │               │                   │          │
│         └───────────────┴───────────────┴───────────────────┘          │
│                                    │ HTTP / MCP / interaction intake    │
└────────────────────────────────────┼────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  BACKEND — LKW Daemon (single product boundary on localhost)             │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Tier-3  local_workspace_application                                │  │
│  │  FastAPI Core · /health · /v1/local_workspace/* · /mcp           │  │
│  │  /v1/interactions/intake (LKW.6) · optional Socket Mode (LKW.6b) │  │
│  │  manifest · environment_profile · tool_wiring · factory          │  │
│  └───────────────────────────────┬───────────────────────────────────┘  │
│                                  ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Tier-1  Nexus Agent OS                                             │  │
│  │  NexusLoop · graph · HITL · trace · memory · policy                │  │
│  └───────────────────────────────┬───────────────────────────────────┘  │
│                                  ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Tier-2  local_indexer · local_search · local_synthesizer           │  │
│  └───────────────────────────────┬───────────────────────────────────┘  │
│                                  ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Tier-0  integrations · tools · skills · RAG · shadow workspace     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────┐  Knowledge Intake: durable op + queue/worker │
│  │ ingestion worker path  │  (required capability direction; topology    │
│  │ (+ optional LKW.7      │   not frozen; production durability not      │
│  │  file-watcher producer)│   claimed here)                              │
│  └──────────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LOCAL DATA PLANE (backend-owned paths — §7)                            │
│  ~/.local/share/intergrax/lkw/  or  %LOCALAPPDATA%\Intergrax\LKW\       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Responsibility matrix

| Concern | Frontend | Backend (LKW application host) |
|---------|----------|----------------------|
| User message / command | Collects channel-native text, attachments and references; only trusted local-capable surfaces may choose a local path | Validates and maps input into tenant/workspace-scoped Knowledge Intake / `Task` |
| Knowledge Intake | Maps channel artifacts to public LKW request; displays accepted/progress/completion | Owns upload acceptance, durable Ingestion Operation, queue/worker, parse/chunk/embed, stores |
| Capability routing | May suggest `capability` in JSON | Nexus selects agent / graph |
| RAG ingest / retrieve | **Never** | Agents + `rag.*` tools / Knowledge Intake worker path |
| LLM calls | **Never** (including for ingestion) | `RuntimeConfig` / agent pipeline |
| Connected local folder | Local-capable surface registers safe Source Candidate; remote chat selects opaque candidate id + safe label only | Connector-backed Source behind LKW boundary; remote chat must not accept raw local paths |
| File write (deliverables) | May open exported file | `workspace.*` shadow only |
| Auth to localhost | Optional API key in tray config | `LocalWorkspaceBackendSettings` |
| Slack tokens | **Never** stored in tray; never sent into LKW core with file bytes | Daemon config / env; Slack adapter uses integration credentials only inside the adapter |
| Trace / debug | May show run_id / operation_id link | Trace DB / durable operation record |

### 4.3 Frontend catalog (planned)

| Client | Technology | Talks to | Wave |
|--------|------------|----------|------|
| **HTTP API** | any HTTP client | `POST /v1/local_workspace/run` | LKW.0 Done |
| **Managed workspaces API** | any HTTP client | `/v1/local_workspace/workspaces` (create/list/get/delete), sources, sync, operations, search, ask | **LKW-PRODUCT-1 Done** + **LKW-WORKSPACE-MANAGEMENT-1** delete |
| **MCP** | Cursor, Claude Desktop | `http://127.0.0.1:8020/mcp` | LKW.0 Done |
| **LKW Tray** | Tauri/Electron or native | localhost HTTP + folder picker | LKW.8 |
| **Slack** | Slack App (Socket Mode) | intake via platform interaction stack on LKW host | LKW.6b |
| **CLI operator** | `intergrax.debug` | trace DB | platform Done |

### 4.4 Backend process model

| Process | Role | Required |
|---------|------|----------|
| **`lkw-host`** | Uvicorn + FastAPI + NexusLoop + MCP + optional Slack socket | **Yes** — one per user session |
| **Knowledge Intake / ingestion worker** | Durable queue/worker boundary for Knowledge Intake (parse → documents → chunks → embeddings → stores) | **Required capability direction** for Knowledge Intake — exact co-location vs separate process **not** frozen; do **not** claim a production durable worker already exists |
| **`lkw-indexer-worker` / file watcher** | Optional producer: filesystem watch → enqueue → ingest | LKW.7 optional — not the definition of Knowledge Intake |
| **External LLM API** | Inference only | Configurable (Ollama local or cloud) |

**Reference single-host rule (common self-hosted topology):** one `lkw-host` binds a configured listen address/port (often `127.0.0.1:8020`). Tray and MCP are clients, not second runtimes. Other host topologies remain valid under the deployment-neutral contract. Asynchronous ingestion may share the host process or run as a separate worker when implementation chooses; topology is configuration, not a second product mode.

---

## 5. Solution overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  User client (HTTP / MCP / future desktop shell)                        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ POST /v1/local_workspace/run
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Tier-3  local_workspace_application                                    │
│  manifest · environment_profile · tool_wiring · factory · MCP           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Tier-1  Nexus Agent OS                                                 │
│  Intake → Plan → Graph (index → search → synthesize) → Trace → Result   │
└───────┬─────────────────┬─────────────────┬─────────────────────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ local_indexer│  │ local_search │  │ local_synthesizer│
│ Tier-2       │  │ Tier-2       │  │ Tier-2           │
└──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
       │                 │                    │
       └─────────────────┴────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Tier-0  Platform — four-layer stack (canon §7.1.6–§7.1.8)              │
│  Integration → Tool → Skill → Agent                                     │
│  Docling · SQLite · vector store · rag.* · workspace.* · local.* skills │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Four-layer composition (Integration → Tool → Skill → Agent)

LKW follows the canonical Intergrax stack — agents never call vendor SDKs; Tier-3 selects backends and enables catalog surfaces:

```text
IntegrationProfile (Tier-3 host)
  ├── document_parser=docling     → DocumentParser backend
  ├── vector_store=inmemory|chroma → VectorStore for RAG index
  ├── relational_store=sqlite     → trace, session, task memory
  ├── rerank_provider=cohere_rerank → optional rerank on rag.retrieve
  └── observability_backend=otel  → traces (optional)

ToolProfile (Tier-3 host/tool_wiring.py)
  └── enabled tool_ids → ToolRegistry + ToolWiringContext

SkillProfile (Tier-3 environment)
  └── enabled_bundles: harness (LKW.0) · local (LKW.2 planned)
      → resolves skill_ids → allowed_tools + prompt refs on AgentContract

AgentContract (Tier-2)
  └── skill_ids[] + capabilities[] → UAEP steps invoke tools via ToolRuntime
```

**Rule:** Tier-3 **wires** integrations and tools; Tier-2 agents **declare** `skill_ids` on `AgentContract`; skills **compose** tool packs + prompts + policy fragments. See [`docs/project/architecture/SKILLS.md`](../../../architecture/SKILLS.md) · [`docs/project/architecture/TOOLS.md`](../../../architecture/TOOLS.md) · [`docs/project/architecture/INTEGRATIONS.md`](../../../architecture/INTEGRATIONS.md).

### 5.2 Trust zones (filesystem safety)

| Zone | Purpose | Mechanism | Mutations |
|------|---------|-----------|-----------|
| **Read zone** | User documents (allowlisted paths) | `rag.ingest_document`, `document.parse`; future `filesystem.*` read-only | **None** on user FS |
| **Artifact zone** | Reports, drafts, exports | `workspace.*` on **shadow workspace** | Only under `INTERGRAX_SHADOW_ROOT` |
| **Sandbox zone** | Risky experiments | `sandbox.exec` (opt-in per task) | Isolated under `INTERGRAX_SANDBOX_ROOT` |

**Rule:** LKW agents MUST NOT write to user home directories. All deliverables go to shadow workspace unless the user explicitly promotes an export path in a future Wave.

---

## 6. Agent roster and capabilities

| Agent | Module | Capability | Responsibility |
|-------|--------|------------|----------------|
| **LocalIndexerAgent** | `agents/local_indexer` | `local.workspace.index` | Discover paths (Wave 1: explicit), parse, chunk, embed, index via `rag.ingest_document` |
| **LocalSearchAgent** | `agents/local_search` | `local.workspace.search` | Semantic + metadata-filtered retrieval via `rag.retrieve`; rank and package evidence |
| **LocalSynthesizerAgent** | `agents/local_synthesizer` | `local.workspace.synthesize` | LLM synthesis from retrieved context; write artifacts to shadow workspace |

**Pipeline capability (graph-level):** `local.workspace.pipeline` — multi-step intent routing index → search → synthesize (Wave 2). Documented here; wired via Nexus `AgentGraph` / delegation like `research.pipeline`.

Agent architecture docs:

- [`docs/project/technical/agents/local_indexer/ARCHITECTURE.md`](../../agents/local_indexer/ARCHITECTURE.md)
- [`docs/project/technical/agents/local_search/ARCHITECTURE.md`](../../agents/local_search/ARCHITECTURE.md)
- [`docs/project/technical/agents/local_synthesizer/ARCHITECTURE.md`](../../agents/local_synthesizer/ARCHITECTURE.md)

---

## 7. Installation, lifecycle, and on-disk layout

### 7.1 Installation philosophy

LKW installs as a **user-level background service** plus optional tray frontend. No system-wide server required. Python/uv environment ships with the product bundle (or uses existing Intergrax dev tree for engineering builds).

**Target personas:**

| Persona | Install path |
|---------|--------------|
| Developer | `uv sync` + `uvicorn` from repo (today) |
| End user (future) | Installer → `%LOCALAPPDATA%/Intergrax/LKW` or `~/.local/share/intergrax/lkw` |

### 7.2 Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.12 + uv | Dev; packaged install may embed runtime |
| LLM endpoint | Ollama local (`INTERGRAX_OLLAMA_*`) or cloud API key |
| Disk space | Index + trace (plan ~1–5 GB for typical corpus) |
| OS permissions | Read access to user-selected folders; macOS Full Disk Access if needed |

### 7.3 On-disk layout (canonical paths)

Default root: **`$LKW_DATA_HOME`** (env) with fallbacks:

| OS | Default `LKW_DATA_HOME` |
|----|-------------------------|
| Linux | `~/.local/share/intergrax/lkw` |
| macOS | `~/Library/Application Support/Intergrax/LKW` |
| Windows | `%LOCALAPPDATA%/Intergrax/LKW` |

```text
$LKW_DATA_HOME/
├── config/
│   ├── .env                    # LOCAL_WORKSPACE_* secrets (gitignored)
│   ├── allowed_read_roots.json # folder allowlist (LKW.3+)
│   └── integration_profile.json  # optional Chroma override
├── data/
│   ├── chroma/                   # vector index (when chroma enabled)
│   ├── sqlite/
│   │   ├── intergrax_trace.db
│   │   ├── intergrax_session.db
│   │   └── intergrax_task_memory.db
│   └── shadow_workspaces/        # INTERGRAX_SHADOW_ROOT override
├── logs/
│   └── lkw-host.log
└── run/
    └── lkw-host.pid
```

**Engineering default (repo dev):** `build` under repository — override via env for product parity testing.

### 7.4 Install steps by OS (APP-HOST-7 — later operator/packaging targets)

**Ownership:** LKW declares product always-on requirements and adopts platform Application Hosting. Generic OS service integration, signal handling, restart supervision, and service-manager descriptors are **platform-owned** ([`APPLICATION_HOSTING`](../../../architecture/APPLICATION_HOSTING.md)). The examples below are **operator-facing targets** for post-APP-HOST-7 packaging — **not** LKW.6B initial proof requirements.

**LKW.6B initial proof** does not require service-manager installation or reboot survival unless APP-HOST-7 is completed. Initial acceptance covers: foreground hosted start, READY state, real LKW request, single-instance rejection, graceful stop, supervisor restart, new instance identity, real request after restart.

```text
LKW application
  → LKW-specific HostedApplicationProfile
  → platform HostedApplicationEngine
  → platform supervisor / OS adapters when applicable (APP-HOST-7)
```

#### Windows

```text
1. Installer copies bundle → %LOCALAPPDATA%\Intergrax\LKW\
2. APP-HOST-7: platform Windows hosting adapter registers always-on service (LKW does not own generic service framework)
3. Optional: tray app in Startup folder → localhost:8020
4. First-run wizard (LKW.8): pick folders → writes allowed_read_roots.json
```

#### Linux (systemd user unit)

```ini
# ~/.config/systemd/user/lkw-host.service
[Unit]
Description=Intergrax Local Knowledge Workspace
After=network.target

[Service]
ExecStart=%h/.local/share/intergrax/lkw/bin/lkw-host
Restart=on-failure
Environment=LKW_DATA_HOME=%h/.local/share/intergrax/lkw

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now lkw-host
```

#### macOS (LaunchAgent)

```xml
<!-- ~/Library/LaunchAgents/com.intergrax.lkw.plist -->
Label: com.intergrax.lkw
ProgramArguments: ~/.local/share/intergrax/lkw/bin/lkw-host
RunAtLoad: true
KeepAlive: true
```

Grant **Full Disk Access** if indexing outside home directory.

### 7.5 Upgrade and uninstall

| Action | Behaviour |
|--------|-----------|
| **Upgrade** | Stop service → replace `bin` + Python env → migrate sqlite/chroma if schema version bumps → start |
| **Uninstall** | Stop service → remove unit/plist → delete `$LKW_DATA_HOME` (user prompt: keep index?) |
| **Config only reset** | Delete `config/.env`; keep `data/chroma` |

### 7.6 Health and readiness

| Check | Endpoint / command |
|-------|-------------------|
| Process up | `GET http:/127.0.0.1:8020/health` |
| Agents registered | `GET /v1/local_workspace/agents` |
| Index ready | `rag.check_index_status` via MCP or debug task |
| Integration health | host bootstrap probes at startup (log on failure) |

---

## 8. Integrations, tools, and skills

### 8.1 Integrations (`IntegrationProfile`)

**Baseline preset:** `IntegrationProfile.legal_product()` — RAG + document parsing without mandatory web search (unlike `research_product()`).

| `IntegrationCategory` slot | Slug (default) | Role in LKW | Wired via |
|--------------------------|----------------|-------------|-----------|
| `relational_store` | `sqlite` | Trace DB, session state, task memory persistence | `wire_application_environment` → `memory_wiring` |
| `vector_store` | `inmemory` | RAG chunk index (dev); replace with `chroma` for durable local index | `rag_runtime_bridge` → `ToolWiringContext.vectorstore_manager` |
| `document_parser` | `docling` | PDF/DOCX/XLSX parsing inside `rag.ingest_document` / `document.parse` | `CatalogDocumentParser` — infra slot, not auto-exposed as agent tool |
| `rerank_provider` | `cohere_rerank` | Optional rerank after hybrid retrieval in `rag.retrieve` | `RetrievalService` / `RagProfile` |
| `observability_backend` | `otel` (optional) | Export traces when OTLP enabled on environment profile | `host/environment_profile.py` |
| `object_storage` | `filesystem` (Wave 4) | Export shadow artifacts / checkpoint blobs | `storage.*` tools when enabled |
| `message_bus` | message_bus provider slug (LKW.4) | Platform background ingest jobs | `message_bus.*` when a message bus provider is configured |

**Override:** `INTERGRAX_INTEGRATION_PROFILE_JSON` — e.g. swap `vector_store` to `chroma` for persistent local index.

**Explicitly excluded in baseline:** `search_provider` (web), `collaboration` (mail APIs) — LKW is local-first.

Authoring: [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md` Appendix K](../../guides/AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane) · catalog: [`docs/project/architecture/INTEGRATIONS.md`](../../../architecture/INTEGRATIONS.md).

### 8.2 Tools (`ToolProfile` + `host/tool_wiring.py`)

Tier-3 enables tools; agents invoke them through `BoundToolGateway` / `ctx.invoke_tool()` — never direct integration imports.

#### Host-wide tool allowlist (`_LKW_BASE_TOOL_IDS`)

| tool_id | Bundle | Composes (integration slot) | LKW role |
|---------|--------|----------------------------|----------|
| `rag.ingest_document` | `rag` | `vector_store` + `document_parser` + embedding managers | Index local files |
| `rag.retrieve` | `rag` | `vector_store` + optional `rerank_provider` | Semantic search |
| `rag.list_collections` | `rag` | `vector_store` | Index diagnostics |
| `rag.list_documents` | `rag` | `vectorstore_manager` | Paginated index inventory |
| `rag.get_document` | `rag` | `vectorstore_manager` | Fetch indexed chunk by id |
| `rag.check_index_status` | `rag` | `vectorstore_manager` | Index readiness probe |
| `document.parse` | `document` | `document_parser` | Ad-hoc parse without full ingest |
| `document.parse_preview` | `document` | `document_parser` | Bounded parse preview (no ingest) |
| `workspace.read_file` | `workspace` | runtime `ShadowWorkspace` | Read shadow artifacts |
| `workspace.write_file` | `workspace` | runtime `ShadowWorkspace` | Write drafts/reports |
| `workspace.list_files` | `workspace` | runtime `ShadowWorkspace` | List artifacts |
| `workspace.snapshot` | `workspace` | runtime `ShadowWorkspace` | Point-in-time snapshot |
| `workspace.delete_file` | `workspace` | runtime `ShadowWorkspace` | Remove draft revisions in shadow only |
| `workspace.search` | `workspace` | runtime `ShadowWorkspace` | Grep across shadow artifacts |
| `memory.read` / `memory.write` / `memory.list_keys` | `memory` | `relational_store` + task memory | Session working state |
| `cache.get` / `cache.set` | `cache` | optional KV backend | Dedup parse/embedding keys |

**Env-gated (settings):** `LOCAL_WORKSPACE_ENABLE_RAG` → `rag.retrieve`; `LOCAL_WORKSPACE_ENABLE_RAG_INGEST` → `rag.ingest_document`.

**Filesystem browse (T6 / LKW.3 Done):** when `INTERGRAX_ALLOWED_READ_ROOTS` or `allowed_read_roots` is set, host auto-enables `filesystem.list`, `filesystem.glob`, `filesystem.read_text`, `filesystem.stat`.

**Explicitly disabled:** `websearch.*`, `openai.file_search.*` — external retrieval out of scope for LKW baseline.

Catalog reference: [`docs/project/architecture/TOOLS.md`](../../../architecture/TOOLS.md) · wiring: [`host/tool_wiring.py`](host/tool_wiring.py).

### 8.3 Skills (`SkillProfile` + `AgentContract.skill_ids`)

Skills are **composable packs** (tools + prompt instruction ids + optional policy fragments). The LLM does not call skills directly — Nexus resolves `skill_ids` into `allowed_tools` at register time.

#### Enabled today (LKW.0)

| Bundle | `skill_bundles` | Purpose |
|--------|-----------------|---------|
| `harness` | `["harness"]` on `ApplicationEnvironmentProfile` | Platform smoke packs (`harness.tool_smoke`, `harness.trace_read`, …) — harness validation only |

Environment: [`manifest.py`](manifest.py) · [`host/environment_profile.py`](host/environment_profile.py).

#### Planned domain bundle (LKW.2) — `intergrax/skills/providers/local`

| `skill_id` | Agent | `tool_ids` | `prompt_instruction_ids` |
|------------|-------|------------|----------------------------|
| `local.workspace.index` | `local_indexer` | `rag.ingest_document`, `document.parse`, `rag.list_collections` | `local.workspace.index.system` |
| `local.workspace.search` | `local_search` | `rag.retrieve`, `rag.list_collections`, `cache.get`, `cache.set` | `local.workspace.search.system` |
| `local.workspace.synthesize` | `local_synthesizer` | `workspace.read_file`, `workspace.write_file`, `workspace.list_files`, `workspace.search`, `memory.read` | `local.workspace.synthesize.system` |
| `local.workspace.pipeline` | graph intent (all three) | union of above (via `requires_skills`) | orchestration prompt refs |

**Agent wiring (LKW.2):** each `AgentContract` gains `skill_ids=[...]`; register via `registry.register(agent, skill_registry=..., tool_registry=...)`. Until then, agents use scaffold `skills=[]` and rely on host `ToolProfile` only.

Skill authoring: [`docs/project/architecture/SKILLS.md`](../../../architecture/SKILLS.md) · Appendix J in [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md`](../../guides/AGENT_CREATION_GUIDE.md#appendix-j--tools--skills-control-plane).

### 8.4 Per-agent Integration / Tool / Skill matrix

| Agent | Integrations consumed (indirect) | Primary tools | Skill (LKW.2) |
|-------|----------------------------------|---------------|---------------|
| **LocalIndexerAgent** | `document_parser`, `vector_store`, embedding managers | `rag.ingest_document`, `document.parse`, `rag.list_collections` | `local.workspace.index` |
| **LocalSearchAgent** | `vector_store`, `rerank_provider` | `rag.retrieve`, `cache.*`, `memory.*` | `local.workspace.search` |
| **LocalSynthesizerAgent** | runtime shadow workspace (not integration slug) | `workspace.*`, `memory.read` | `local.workspace.synthesize` |

### 8.5 Runtime wiring path (Tier-3 → Tier-1 → Tier-2)

```text
wire_application_environment(manifest, environment, settings)
  ├── bootstrap_application_integration_catalog()
  ├── probe_integration_profile_health()
  ├── resolve_rag_stack_for_environment()     # ContextProfile.enable_rag=true
  ├── build_application_tool_wiring()           # ToolProfile → ToolRegistry
  ├── build_application_skill_wiring()        # SkillProfile → SkillRegistry
  └── ApplicationBuildContext                 # passed to agent factories

build_application_registry(manifest, build_context, builders)
  └── register agents; resolve skill_ids → allowed_tools

NexusLoop → AgentEngine → UAEP ctx.invoke_tool(ToolRequest(tool_name="rag.retrieve", ...))
```

### 8.6 Environment profile summary

- `ApplicationEnvironmentProfile.product_defaults(profile_id="local_workspace.product")`
- `skill_bundles=["harness"]` (LKW.0); extend with `"local"` at LKW.2
- `integration_profile=IntegrationProfile.legal_product()`
- `ContextProfile(enable_rag=True, enable_websearch=False)`
- `with_harness_memory()` — STM/LTM hooks for long sessions
- OTLP optional on `observability_profile` + `IntegrationProfile` OTEL slot

See [`host/environment_profile.py`](host/environment_profile.py).

### 8.7 LKW.4 — Background jobs via platform MessageBus

LKW.4 is a **platform message-bus / background-jobs proof track**, not an LKW-owned queue implementation. LKW must **not** implement an application-specific queue, a new queue system, or provider-specific SDK wiring. **LKW is the proof workload; platform owns queue infrastructure.**

**Platform proof pattern** (same as observability):

```text
Application/domain job
  → platform TaskQueue / MessageBus contract
  → provider-neutral message_bus.* tools
  → provider integration
  → LKW background ingest proof workload
```

#### Ownership boundaries

| Layer | Owns |
|-------|------|
| **Platform** | `TaskQueue` / `MessageBus` contract (`intergrax/queueing/contracts/task_queue.py`, `intergrax/integrations/contracts/message_bus.py`); `MessageBusIntegrationContract` (`intergrax/runtime/integrations/categories/messaging.py`); provider integrations; provider-neutral `message_bus.*` tools (`message_bus.enqueue`, `message_bus.get_status`, `message_bus.get_result`, `message_bus.list_tasks`, …); lifecycle / status / result abstraction |
| **LKW (Tier-3)** | `LkwBackgroundIngestJob` (`background_ingest/contracts.py`); `task_name` (`lkw.background_ingest.v1`); payload schema; idempotency key convention; handler mapping; proof workload and reviewer runbook |
| **Agents (Tier-2)** | Tool/skill invocation only — **no** provider SDK imports; **no** Kafka / RabbitMQ / Celery imports |
| **Providers** | Backend implementation behind the common contract (examples only — LKW.4 does not require all): `kafka`, `rabbitmq`, `celery`, `redpanda`, `sqs`, `service_bus`, `pubsub`, `nats`, `pulsar`, `confluent`, `temporal` |

#### Platform background task model dependency

LKW.4 is aligned with the platform background task architecture in [`docs/project/architecture/BACKGROUND_TASKS.md`](../../../architecture/BACKGROUND_TASKS.md). LKW background ingest is one concrete **TaskDefinition** in that model — not a separate queue design.

**LKW.4E must use the target concepts:**

- `TaskRequest` enqueue envelope
- `TaskDefinition` / handler mapping (`lkw.background_ingest.v1` → `handle_background_ingest_task_request`)
- `WorkerRuntime` with a **real local MessageBus provider** in the proof stack (BG-TASKS-7)
- Pull status/result via `message_bus.get_status` / `get_result`
- Lifecycle events and trace correlation (target model; proof may start minimal)

LKW.4E is a **platform proof through LKW**. It must demonstrate production-like platform behavior: real `message_bus.*` tools, a real local broker/provider in the proof stack, and asynchronous worker execution. Mocks, fake queues, in-memory-only bypasses, and unit-test-only handler invocation are **not** sufficient for platform proof. See [`docs/project/maintainers/plans/BACKGROUND_TASKS.md`](../../../maintainers/plans/BACKGROUND_TASKS.md) and public reviewer Step 8 in [`docs/project/proofs/LKW_PLATFORM_PROOF.md`](../../../proofs/LKW_PLATFORM_PROOF.md).

#### Intended background ingest flow

Triggers (file watcher, scheduler, or explicit user background action) build a domain job and enqueue through the platform surface — **without** duplicating queue logic in LKW:

```text
File watcher / scheduler / user background action
  → build LkwBackgroundIngestJob
  → encode_background_ingest_job()
  → background_ingest_payload_base64()
  → message_bus.enqueue
  → TaskRequest(
       tenant_id,
       run_id,
       task_name="lkw.background_ingest.v1",
       payload=<json bytes>,
       idempotency_key=<stable key>
     )
  → MessageBus / TaskQueue
  → provider adapter
  → worker handler
  → decode_background_ingest_job()
  → execute local.workspace.index through platform execution path
  → TaskResult / status via message_bus.get_status / get_result / list_tasks
```

**Execution rules:**

- `local.workspace.index` remains the indexing capability — the worker runs the **existing** capability path and must **not** duplicate `LocalIndexerAgent` logic inline.
- Payload carries paths and scope only (`LkwBackgroundIngestJob`); no raw document content in the job envelope.
- Idempotency is platform-backed via `TaskRequest.idempotency_key` and LKW's stable key convention.

**Compact request-flow diagram:**

```text
Background action
  → LkwBackgroundIngestJob
  → message_bus.enqueue
  → TaskQueue / MessageBus
  → provider
  → worker handler
  → local.workspace.index (platform execution path)
  → message_bus.get_status / get_result
```

#### LKW.4 vs LKW.7

| Wave | Proves / adds |
|------|----------------|
| **LKW.4** | Domain job payload; enqueue via platform `message_bus.*`; inspect lifecycle through provider-neutral tools; handler executes index without changing agent logic; live proof via search/index evidence |
| **LKW.7** (later) | File watcher; incremental index trigger policy; directory change detection; batching/debounce; recurring filesystem-driven enqueue |

File watcher and incremental index are **LKW.7**, not LKW.4. OS daemon and interaction intake remain **LKW.6**. Slack notify remains optional later (**LKW.6b**), not LKW.4 core.

#### LKW.4 vs provider portability

LKW.4 starts with **one real local message bus provider** in the proof stack (for example RabbitMQ in Docker). Provider portability proof can happen later. Provider-specific SDKs stay behind platform provider integrations — LKW.4 does **not** implement every listed backend. Mocks and in-memory-only queue bypasses do **not** satisfy LKW.4E platform proof.

#### Message bus tool exposure (LKW.4B guardrail — implemented)

When a `message_bus` integration is configured on the host integration profile, `message_bus.*` tools **may** be exposed to the relevant host/tool profile. When `message_bus` is **not** configured, `message_bus.*` tools remain **disabled** for LKW. Shared application wiring (`apply_resolved_integration_tool_guardrails` in `intergrax/applications/_shared/integration_tool_profile.py`) enforces the resolved `ToolWiringContext.message_bus` guardrail; LKW host (`host/tool_wiring.py`) consumes that helper — **LKW.4B closed** · **LKW.4B-PROP-1 closed**.

Code references: [`background_ingest/contracts.py`](../../../../../applications/local_workspace_application/background_ingest/contracts.py) · [`background_ingest/enqueue.py`](../../../../../applications/local_workspace_application/background_ingest/enqueue.py) (LKW.4C enqueue helper) · [`background_ingest/handler.py`](../../../../../applications/local_workspace_application/background_ingest/handler.py) (LKW.4D worker handler contract) · platform [`BACKGROUND_TASKS.md`](../../../architecture/BACKGROUND_TASKS.md) · [`INTEGRATIONS.md`](../../../architecture/INTEGRATIONS.md) · [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §6.

---

## 9. Local OS runtime and interaction model

This section describes the **common self-hosted / workstation reference topology** (Tier-3 host + Nexus), often run in the background on a user-controlled machine (Windows, Linux, macOS). It is **not** a claim that LKW is local-only; storage and host location remain configuration — see [Deployment, storage and tenancy model](.#deployment-storage-and-tenancy-model). The user can submit work **at any time**; agents are spawned by Nexus on demand. Chat apps (Slack, Teams) are **interaction surfaces / clients** — not the runtime — they deliver commands and receive summaries.

### 9.1 Design principle: deployment-neutral product, multi-channel control

| Layer | Where it runs (reference local topology) | What it holds |
|-------|------------------------------------------|---------------|
| **Execution** | User-controlled host (often localhost) | Nexus, agents, product capabilities, source connectors as wired |
| **Persistent knowledge stores** | Provider-selected (local or remote) | Document Store, Vector Store, optional Blob/Object Store |
| **Interaction** | Slack / Teams / HTTP / MCP / tray (optional) | Commands, status, HITL prompts, short answers |
| **External messaging / inference** | Optional (Slack API, LLM API) | Messaging + inference — not a substitute for authorization |

**Privacy default:** private by default and tenant-scoped. Physical co-location of stores with the host may strengthen privacy but does not replace authorization. Slack receives **commands and condensed answers**, not full file dumps.

### 9.2 Background process topology

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  User workstation (Windows / Linux / macOS)                              │
│                                                                          │
│  ┌─────────────────────┐    ┌──────────────────────┐                   │
│  │ LKW Host (always-on) │    │ Indexer sidecar (opt)│                   │
│  │ local_workspace_app  │    │ file watcher +       │                   │
│  │ :8020 localhost      │    │ background ingest    │                   │
│  │                      │    │ enqueue (LKW.7)      │                   │
│  └──────────┬──────────┘    └──────────┬───────────┘                   │
│             │                          │                                 │
│             ▼                          ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ Local data plane                                             │        │
│  │ Chroma/SQLite index · shadow_workspaces/ · trace DB          │        │
│  └─────────────────────────────────────────────────────────────┘        │
│             ▲                                                            │
│             │ POST /v1/local_workspace/run                               │
│             │ POST /v1/interactions/intake  (Slack / Teams / JSON)       │
│             │ MCP /v1/...                                                │
│  ┌──────────┴──────────┐   ┌─────────────┐   ┌──────────────────┐     │
│  │ Tray / CLI (LKW.8)  │   │ Cursor MCP  │   │ Slack / Teams    │     │
│  │ (optional UI)       │   │ (local)     │   │ (remote surface) │     │
│  └─────────────────────┘   └─────────────┘   └────────┬─────────┘     │
└──────────────────────────────────────────────────────────┼───────────────┘
                                                           │
                                              Slack Socket Mode or HTTPS tunnel
                                              (outbound from daemon — no public IP required)
```

#### OS service packaging (APP-HOST-7 — later operator/packaging targets)

**Platform ownership:** generic always-on hosting, lifecycle state machine, readiness aggregation, instance lock, signal handling, restart loop, and OS adapters (`systemd`, `launchd`, Windows Service) are owned by [`APPLICATION_HOSTING`](../../../architecture/APPLICATION_HOSTING.md). LKW is the first adopter and proof — it supplies an LKW-specific `HostedApplicationProfile`, hooks, and components only.

**LKW.6B initial proof** does not require the OS adapters below unless APP-HOST-7 is completed. Keep these as later operator/packaging targets.

| OS | Platform adapter target (APP-HOST-7) | Notes |
|----|--------------------------------------|-------|
| **Windows** | Platform Windows hosting adapter | User-session for file access; avoid SYSTEM account for home-folder indexing |
| **Linux** | Platform `systemd` user-unit integration | `After=network.target`; restart on failure via platform supervisor |
| **macOS** | Platform `launchd` LaunchAgent integration | Full Disk Access may be required for user folders |

Host entrypoint (today): `uvicorn local_workspace_application.host.main:app`. **LKW.6B** adopts platform hosting around this factory; LKW does not implement generic daemon engine or OS hosting mechanics in the application tree.

#### Always-on responsibilities

1. **Listen** for user tasks (HTTP, MCP, interaction intake).
2. **Maintain** local RAG index (platform message-bus background ingest — LKW.4; filesystem triggers — LKW.7).
3. **Run** Nexus graph on demand (search now, synthesize on request).
4. **Notify** on completion / HITL (`notification_channel=slack` on long-running tasks).
5. **Persist** checkpoints for pause/resume ([`docs/project/architecture/intergrax_runtime_architecture.md` Appendix F.4](../../../architecture/intergrax_runtime_architecture.md)).

### 9.3 Interaction surfaces (how the user talks to LKW)

| Surface | Status | Endpoint / mechanism | Best for |
|---------|--------|----------------------|----------|
| **Local HTTP** | Scaffold **Done** | `POST /v1/local_workspace/run` | Scripts, tray, local integrations |
| **Local MCP** | Scaffold **Done** | `/mcp` on same host | Cursor / IDE at desk |
| **Interaction intake** | Platform **Done**; LKW host **LKW.6A Done** | `POST /v1/interactions/intake` | Slack slash commands, Teams, lab JSON |
| **Slack outbound** | Platform **Done** | `INTERGRAX_SLACK_WEBHOOK_URL`, HITL templates | Alerts, approvals, result snippets |
| **Debug CLI** | Platform **Done** | `python -m intergrax.debug` | Operators |
| **Tray / native UI** | **Deferred LKW.8** | Calls localhost HTTP/MCP | Folder picker, status icon |

**Rule:** every surface normalizes to a Nexus `Task` — same agents, same policy, same trace. **LKW.6A** unifies `/v1/local_workspace/run` and `/v1/interactions/intake` through one `LocalWorkspaceTaskExecutor` before `NexusLoop`. See [`applications/USAGE.md` §4b](../USAGE.md) · canon §18.

### 9.3a LKW.6A — unified application execution boundary (closed)

Platform interaction intake exists and LKW host wiring exists; **LKW.6A** unifies execution and application-level readiness semantics (temporary until **LKW.6B** adopts platform Application Hosting).

```text
POST /v1/local_workspace/run
POST /v1/interactions/intake
(future tray / Slack / OS sources)
  → platform adapter (interaction only) / HTTP request model (/run)
  → Task
  → LocalWorkspaceTaskExecutor.prepare()  [capability policy + LKW defaults + reliability + orchestration ACP]
  → LocalWorkspaceTaskExecutor.execute_prepared()
  → NexusLoop.handle_task
  → TaskResult
```

| Concern | Owner | Notes |
|---------|-------|-------|
| Transport normalization | Platform `InteractionAdapter` / HTTP schemas | No LKW interaction models |
| Application execution prep | `LocalWorkspaceTaskExecutor` | Allowlisted capabilities: `local.workspace.search` / `.index` / `.synthesize` (+ graph triggers) |
| Reliability enrichment | Shared `build_reliability_task_enricher` | Applied once per execution |
| Application readiness (temporary) | `LocalWorkspaceHostLifecycle` (LKW.6A) | `STARTING` → `READY` → `STOPPING` → `STOPPED`; `FAILED` on startup errors — **not** canonical platform hosting; awaits LKW.6B migration to `HostedApplicationEngine` |
| Liveness | `GET /health` | Unchanged: `{"status":"ok"}` |
| Readiness | `GET /v1/local_workspace/readiness` | Requires `READY` + executor available + required components healthy |
| Work rejection | Both execution surfaces | HTTP 503 `lkw_host_not_ready` / `lkw_host_stopping` when not accepting work |
| Background extension point | Documented only | `execute=false` returns prepared `Task`; future background routing via platform message bus (LKW.4) |

**LKW.6A does not include:** platform Application Hosting adoption (**LKW.6B**), Socket Mode (**LKW.6b**), file watcher (**LKW.7**), or OS interaction adapters (**LKW.6C**).

### 9.3b LKW.6C — OS interaction adapters (Windows + Linux Docker)

**Status: Closed** for native Windows shared-runner certification
profile `windows_native_runtime` and for Linux Docker runtime
certification profile `linux_docker_runtime` (PROOF-PORTABILITY-1D):

- Windows Application Hosting Proof and Windows Optional OS Interaction
  Proof are live-certified on native Windows through the current shared
  Python client / OS proof runner.
- Linux Application Hosting Proof and Linux Optional OS Interaction
  Proof are live-certified in Linux Docker runtime.
- Full multi-phase Core Platform Proof is not re-certified by either
  profile in this refresh.

Native Linux host deployment and macOS remain not live-certified.

PROOF-PORTABILITY-1C adds shared cross-platform client/proof plumbing;
PROOF-PORTABILITY-1D records:

- `docs/project/maintainers/public-adoption/evidence/LKW_WINDOWS_NATIVE_CERTIFICATION.json`
- `docs/project/maintainers/public-adoption/evidence/LKW_LINUX_DOCKER_CERTIFICATION.json`

PROOF-PORTABILITY-1D-MATRIX consolidates the current certification state
into the authoritative cross-platform matrix (no new live proofs; macOS
and native Linux remain not live-certified):

- `docs/project/maintainers/public-adoption/LKW_PLATFORM_CERTIFICATION_MATRIX.md`
- `docs/project/maintainers/public-adoption/evidence/LKW_PLATFORM_CERTIFICATION_MATRIX.json`

LKW owns thin OS wrappers that launch one shared Python interaction
client. The shared client serializes the supported `lab_json` payload
and posts it to the existing platform interaction intake. No new
platform interaction channel is introduced.

```text
Windows PowerShell / Linux SH / macOS SH
  → invoke-lkw-interaction.py
  → POST /v1/interactions/intake?execute=true
  → LabJsonInteractionAdapter (lab_json / channel = lab)
  → InteractionIntakeService
  → LocalWorkspaceTaskExecutor
  → NexusLoop
  → real LKW capability execution
```

| Concern | Owner | Notes |
|---------|-------|-------|
| Shared Python client | `scripts/invoke-lkw-interaction.py` | Payload, HTTP, normalized JSON result |
| Windows wrapper | `scripts/invoke-lkw-interaction.ps1` | Adapter identity `lkw.windows_powershell`; source `windows_powershell` |
| Linux wrapper | `scripts/invoke-lkw-interaction-linux.sh` | Adapter identity `lkw.linux_shell`; source `linux_shell` |
| macOS wrapper | `scripts/invoke-lkw-interaction-macos.sh` | Adapter identity `lkw.macos_shell`; source `macos_shell` |
| Shared proof runner | `scripts/run-lkw-os-interaction-proof.py` | OS-family selection, evidence, ProofReceipt |
| Platform channel | `LabJsonInteractionAdapter` | `interaction_channel` remains `lab` |
| Task / enrichment / Nexus | Existing LKW host + platform | No Task, agent, or RAG logic in OS wrappers |
| Hosting / instance lock / signals | Platform Application Hosting | No generic OS hosting behavior in LKW |
| Windows Service | APP-HOST-7 | Not LKW.6C |
| Slack Socket Mode | LKW.6b (optional) | Not LKW.6C |
| File watcher | LKW.7 | Not LKW.6C |
| Tray | LKW.8 | Not LKW.6C |

Enable intake with existing settings: `LOCAL_WORKSPACE_INCLUDE_INTERACTIONS=true`, `LOCAL_WORKSPACE_INTERACTION_SURFACE=lab_json`, `LOCAL_WORKSPACE_INTERACTION_EXECUTE_DEFAULT=true`.

Reviewer commands:
`run-lkw-windows-interaction-proof.bat`,
`run-lkw-linux-interaction-proof.sh`,
`run-lkw-macos-interaction-proof.sh`
— see [`LKW_PLATFORM_PROOF.md`](../../../proofs/LKW_PLATFORM_PROOF.md) optional OS interaction section.

### 9.3c Cross-platform Core Platform Proof entrypoints

```text
Windows BAT ─┐
Linux SH ────┼→ shared Python core-proof runner
macOS SH ────┘
```

OS launchers are transport-only entrypoints.

Proof orchestration and acceptance live in Python.

OS-specific interaction adapters use the same shared-Python pattern:

```text
Windows BAT / PowerShell ─┐
Linux SH ─────────────────┼→ shared Python interaction client / proof runner
macOS SH ─────────────────┘
```

Shared interaction client: `scripts/invoke-lkw-interaction.py`.
Shared interaction proof runner: `scripts/run-lkw-os-interaction-proof.py`.

Shared core runner: `scripts/run-lkw-core-platform-proof.py`. Thin launchers:
`run-lkw-core-platform-proof-windows.bat`,
`run-lkw-core-platform-proof-linux.sh`,
`run-lkw-core-platform-proof-macos.sh`.

### 9.4 Slack as optional interaction channel

Slack is **supported and professional** as an **optional** channel — not the product core. Use existing Intergrax **interaction + notification** integrations (`slack` slug). Execution remains on the **local LKW application host** (always-available backend on localhost).

**Decision record:** Primary UX = localhost (HTTP/MCP/tray). Slack = remote/mobile/team + HITL. Product must pass acceptance tests **without** Slack configured.

#### Reference flow (slash command)

```text
User in Slack:  /lkw search dokumenty o projekcie Alpha
       │
       ▼
Slack Events API  ──►  (A) Socket Mode client in LKW host process   [preferred: no inbound port]
                    or (B) HTTPS tunnel → localhost:8020/v1/interactions/intake
       │
       ▼
InteractionIntakeService  +  SlackInteractionAdapter
       │  verify signature · parse slash payload · map text → Task
       ▼
LocalWorkspaceTaskExecutor  →  NexusLoop.handle_task(capability=local.workspace.search, message=...)
       │
       ▼
LocalSearchAgent → rag.retrieve (local Chroma index)
       │
       ▼
Reply to Slack (response_url / chat.postMessage) — citations + short summary only
```

**Platform primitives to reuse (no Nexus fork):**

| Primitive | Module / doc | LKW use |
|-----------|--------------|---------|
| `InteractionIntakeService` | `runtime/interactions/intake_service.py` | Inbound Slack → `Task` |
| `SlackInteractionAdapter` | `integrations/providers/notification_channel/slack` | Channel id `slack` |
| `wire_interaction_intake_service` | `applications/_shared/interaction_wiring.py` | Enable on `local_workspace_application` factory |
| `TaskLongRunningOptions.notify_channel="slack"` | plan Appendix F.4 | HITL + long ingest jobs |
| Organization worker runbook | plan §H.6 | Prior art for slash → Nexus → resume |

**Example intake (lab-equivalent, today):**

```bash
curl -s -X POST "http://127.0.0.1:8020/v1/interactions/intake?execute=true&tenant=U1" \
  -H "Content-Type: application/json" \
  -d '{"command":"/lkw","text":"search projekt Alpha","user_id":"U1","team_id":"T1"}'
```

LKW.6A wires interaction intake through the shared executor (see §9.3a). Enable with `LOCAL_WORKSPACE_INCLUDE_INTERACTIONS=true` and `LOCAL_WORKSPACE_INTERACTION_SURFACE=lab_json` for Slack-free proof.

#### Slack connectivity modes

| Mode | Pros | Cons | LKW recommendation |
|------|------|------|---------------------|
| **Socket Mode** | No public URL; daemon initiates outbound WebSocket | Requires Slack app + bot token in local config | **Default for desktop daemon** |
| **HTTPS tunnel** (ngrok, Cloudflare Tunnel) | Quick dev | Extra dependency; URL rotation | Dev / demo only |
| **Slack notifications only** | Simple webhook | User cannot command from Slack | Phase 1 fallback — local MCP/HTTP for commands, Slack for HITL |

#### Slack command mapping (convention)

| User text (after `/lkw`) | `Task.context.capability` | Agent |
|--------------------------|----------------------------|-------|
| `index <path>` | `local.workspace.index` | `local_indexer` |
| `search <query>` | `local.workspace.search` | `local_search` |
| `draft email/|report/|estimate …` | `local.workspace.synthesize` | `local_synthesizer` |
| free text (default) | `local.workspace.pipeline` | graph (LKW.2) |

Mapped `tenant_id` (and principal / user identifiers) from Slack identity feed Intergrax task scope for memory and index partitions. **Tenant is not permanently equal to user** — see [Deployment, storage and tenancy model](.#deployment-storage-and-tenancy-model).

#### What must NOT go through Slack

- Raw file uploads containing full document corpora (use local index instead).
- Shadow workspace binary artifacts (link to local export path or summary).
- Unredacted secrets from parsed files.

### 9.5 Task timing: foreground vs background

| Pattern | Trigger | Nexus behaviour |
|---------|---------|-----------------|
| **Interactive** | User message (Slack, HTTP, MCP) | Sync or async run; reply when `COMPLETED` or `WAITING_FOR_HUMAN` |
| **Background index** | File watcher / cron / explicit enqueue (LKW.4 + LKW.7) | `message_bus.enqueue` → platform queue lifecycle → worker runs `local.workspace.index`; optional Slack notify on batch complete (LKW.6b) |
| **Long-running synthesize** | Large report | `TaskLongRunningOptions` + checkpoint; user resumes via Slack `approve` / HTTP |

User can **always** submit a new interactive task while background indexing runs — platform message-bus/task-queue idempotency prevents duplicate ingests (LKW.4 payload key + LKW.7 watcher policy).

### 9.6 Integration profile extension for Slack (LKW.6b)

Extend `IntegrationProfile` on LKW host (in addition to `legal_product()` RAG slots):

```text
notification_channel = slack    # HITL + completion alerts
interaction_surface  = slack    # inbound slash / events (via intake router)
```

Env (mirror legal/lab): `LOCAL_WORKSPACE_INCLUDE_INTERACTIONS=true`, `LOCAL_WORKSPACE_INTERACTION_SURFACE=slack`, Slack signing secret + bot token for Socket Mode.

---

## 10. Request and data flows

### 10.1 Index flow

```text
Task(capability=local.workspace.index, metadata={source_paths: [...]})
  → LocalIndexerAgent UAEP steps
  → invoke rag.ingest_document per path
  → ParserPipeline + chunk + embed + vector store
  → StepOutput(metadata: {num_chunks, collection_id, parser_trace})
```

### 10.2 Search flow

```text
Task(capability=local.workspace.search, message="find documents about project X")
  → LocalSearchAgent
  → rag.retrieve(query, metadata filters)
  → Package evidence chunks + citations (path, page, chunk_id)
```

### 10.3 Synthesize flow

```text
Task(capability=local.workspace.synthesize, metadata={template: "email"|"report"|...})
  → LocalSynthesizerAgent
  → LLM with retrieved context (from graph handoff or prior step)
  → workspace.write_file("draft.md", content)
  → metadata: {shadow_workspace_id, artifact_paths}
```

### 10.4 Pipeline flow (Wave 2)

```text
Task(capability=local.workspace.pipeline, intent=local_workspace_full)
  → Nexus graph: DELEGATES_TO indexer? → search → synthesizer
  → SharedTaskContext carries evidence + artifact refs
```

---

## 11. Tier-3 composition map

| File | Role |
|------|------|
| [`manifest.py`](manifest.py) | Roster, capabilities, `LOCAL_WORKSPACE_APPLICATION_MANIFEST` |
| [`host/environment_profile.py`](host/environment_profile.py) | RAG-on, websearch-off product profile |
| [`host/tool_wiring.py`](host/tool_wiring.py) | LKW tool allowlist |
| [`host/settings.py`](host/settings.py) | `LOCAL_WORKSPACE_*` env, RAG flags |
| [`host/wiring.py`](host/wiring.py) | Registry + `wire_application_environment` |
| [`host/factory.py`](host/factory.py) | FastAPI Core + MCP |
| [`serving/fastapi_router.py`](serving/fastapi_router.py) | `/run`, `/agents` |
| [`mcp/server.py`](mcp/server.py) | FastMCP mount |

**No agent logic in Tier-3** — only wiring. Domain steps live in `agents/*/steps`.

Authoring guide: [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md`](../../guides/AGENT_CREATION_GUIDE.md) · Tier-3: [`applications/USAGE.md`](../USAGE.md).

---

## 12. Configuration

### 12.1 Environment variables (`.env.example`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `LOCAL_WORKSPACE_BACKEND_PORT` | `8020` | HTTP port |
| `LOCAL_WORKSPACE_DEFAULT_AGENT_ID` | `local_search` | Default roster agent |
| `LOCAL_WORKSPACE_ENABLE_RAG` | `true` | Enable `rag.retrieve` |
| `LOCAL_WORKSPACE_ENABLE_RAG_INGEST` | `true` | Enable `rag.ingest_document` |
| `INTERGRAX_SHADOW_ROOT` | `build/shadow_workspaces` | Artifact isolation root |
| `LKW_DATA_HOME` | OS default (§7.3) | Product data root |
| `INTERGRAX_ALLOWED_READ_ROOTS` | user config | Comma-separated read allowlist |
| `LOCAL_WORKSPACE_INCLUDE_INTERACTIONS` | `false` | Enable `/v1/interactions/intake` (LKW.6) |
| `LOCAL_WORKSPACE_INTERACTION_SURFACE` | `auto` | `slack` \| `teams` \| `lab` |

### 12.2 Task metadata conventions (Wave 1+)

```python
Task(
    metadata={
        "shadow_workspace": True,
        "source_paths": ["D:/Docs/project_a/report.pdf"],
        "collection_id": "user_u1_workspace",
        "synthesis_template": "email",
    }
)
```

---

## 13. Security and governance

- **Read-only user FS** in Waves 1–2 (ingest reads; no writes)
- **Shadow workspace** mandatory for synthesizer outputs
- **HITL** optional for sensitive exports (`REQUEST_HUMAN`) — [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md` Appendix A](../../guides/AGENT_CREATION_GUIDE.md#appendix-a--human-in-the-loop)
- **Cost governance:** `CostProfile` on environment; embedding batch limits per ingest job
- **Trace:** all tool calls via Nexus trace DB — debug with `intergrax.debug` CLI

---

## 14. Observability and verification

```bash
# Host smoke
uv run pytest applications/local_workspace_application/tests -q

# Agent smoke
uv run pytest agents/local_indexer/tests agents/local_search/tests agents/local_synthesizer/tests -q

# Run host
uv run uvicorn local_workspace_application.host.main:app --host 127.0.0.1 --port 8020
```

Deploy triad: `docker`, `BUILD_AND_DEPLOY.md` — gate `test_application_deploy_triad.py`.

---

## 15. Implementation plan derivation (canonical)

Each row is one implementable **wave**. Copy to [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) when scheduling work. **Depends** = prior waves. **Acceptance** = objective done criteria.

### 15.1 Wave summary

| ID | Wave | Title | Layer | Depends | Status |
|----|------|-------|-------|---------|--------|
| **LKW.0** | 0 | Scaffold + architecture v2 | Tier-2/3 docs | — | **Done** |
| **LKW.1** | 1 | Domain UAEP: ingest + search | Tier-2 agents | LKW.0 | Planned |
| **LKW.2** | 2 | Graph pipeline + local skills | Tier-1 graph + Tier-0 skills | LKW.1 | Planned |
| **LKW.3** | 3 | Filesystem browse + allowlist | Tier-0 tools + Tier-3 policy | LKW.0 | **Done** (T6) |
| **LKW.4** | 4 | Platform message-bus background ingest proof | Tier-0 message_bus + Tier-3 proof workload | LKW.1 | Planned |
| **LKW.5** | 5 | Chroma persistent index + `LKW_DATA_HOME` | Tier-3 config | LKW.1 | Planned |
| **LKW.6** | 6 | OS daemon packaging + interaction intake | Tier-3 host | LKW.1 | **Closed** (LKW.6A/6B/6C) |
| **LKW.6b** | 6b | Slack Socket Mode (optional) | Tier-3 + slack integration | LKW.6 | Planned / optional |
| **LKW.7** | 7 | File watcher + incremental index | Tier-3 sidecar + enqueue path | LKW.4, LKW.5 | **Closed** (LKW.7A/7B1/7B2A/7B2B Done; LKW.7B Closed; LKW.7C1 Done; LKW.7C2 Done; LKW.7C Closed) |
| **LKW.8** | 8 | Tray frontend (thin client) | Frontend | LKW.6 | Deferred |
| **LKW-PRODUCT-1** | P1 | Managed workspaces + folder sources | Tier-3 product API + DocumentStore state | LKW.1, LKW.3 | **Done** |
| **LKW-PRODUCT-1-HARDENING** | P1 | Durable sync + structured search evidence | MessageBus DocumentStoreTaskQueue + TaskResult search_summary | LKW-PRODUCT-1 | **Done** |

#### LKW-PRODUCT-1 — Managed workspaces and folder sources (Done)

First complete product scenario:

```text
create workspace → attach local folder → validate filesystem policy → sync
  → ingest/index real files → persist source/document state → search in workspace
```

| Owns (LKW) | Reuses (platform) |
|------------|-------------------|
| `Workspace` / `WorkspaceSource` / `WorkspaceOperation` / `WorkspaceDocumentReference` | `DocumentStore` persistence boundary |
| Public HTTP routes under `/v1/local_workspace/workspaces*` and `/operations/{id}` | Filesystem allowlist (`INTERGRAX_ALLOWED_READ_ROOTS`) |
| Sync operation lifecycle + idempotency (content hash) | `local.workspace.index` / `local.workspace.search` via task executor |
| Tenant + workspace isolation (fail-closed 404) | RAG ingest/retrieve metadata filters |

#### LKW-PRODUCT-1-HARDENING — Durable sync and structured search evidence (Done)

Removes temporary PRODUCT-1 workarounds:

```text
HTTP sync → persist WorkspaceOperation(queued) → MessageBus enqueue
  → DocumentStoreTaskQueue / co-located worker → ManagedWorkspaceSyncService
  → local.workspace.index → operation completed|failed
```

| Concern | Contract |
|---------|----------|
| Sync execution | Platform `message_bus.enqueue` + durable `DocumentStoreTaskQueue` (not `asyncio.create_task`) |
| Concurrent sync | `409 Conflict` when another sync for the same tenant/workspace/source is `queued` or `running` |
| Restart | Queued messages survive DocumentStore; interrupted `running` messages/operations fail closed |
| Search evidence | Complete typed `TaskResult.execution_result.structured_data["search_summary"]`; router verifies provenance only — no filesystem snippet reconstruction |

Live runner: `scripts/run-lkw-managed-workspace-live-proof.py` · proof kind `managed_workspace_folder_sync`.

### 15.2 Wave detail (tasks + acceptance)

#### LKW.1 — Domain UAEP: ingest + search

| Task | Owner module | Deliverable |
|------|--------------|-------------|
| LKW.1.1 | `agents/local_indexer/steps` | Ingest pipeline: validate paths → `rag.ingest_document` loop |
| LKW.1.2 | `agents/local_search/steps` | Search pipeline: `rag.retrieve` → evidence package |
| LKW.1.3 | `agents/local_synthesizer/steps` | Stub synthesize → `workspace.write_file` in shadow |
| LKW.1.4 | tests | Acceptance: ingest fixture PDF → search returns citation |

**Acceptance:** `POST /run` with `source_paths` + `local.workspace.search` returns grounded answer; shadow artifact on synthesize; pytest green.

**Frontend:** HTTP/MCP only. **Backend:** all logic in agents.

---

#### LKW.2 — Graph pipeline + skills

| Task | Owner module | Deliverable |
|------|--------------|-------------|
| LKW.2.1 | `intergrax/skills/providers/local` | Skill manifests `local.workspace.*` |
| LKW.2.2 | `agents/*/contract.py` | `skill_ids` on each agent |
| LKW.2.3 | `host/environment_profile.py` | `skill_bundles=["harness","local"]` |
| LKW.2.4 | `manifest` / graph_spec | `local.workspace.pipeline` graph |

**Acceptance:** Single `POST /run` with pipeline capability runs index→search→synthesize without manual capability selection.

---

#### LKW.5 — Persistent index + data home

| Task | Owner module | Deliverable |
|------|--------------|-------------|
| LKW.5.1 | `host/settings.py` | `LKW_DATA_HOME` resolution (§7.3) |
| LKW.5.2 | env / profile | Chroma under `data/chroma` |
| LKW.5.3 | `BUILD_AND_DEPLOY.md` | Document paths per OS |

**Acceptance:** Restart host → prior index still retrievable.

---

#### LKW.6 — OS daemon + interaction intake (backend productization)

**Platform ownership:** generic always-on hosting is owned by [`APPLICATION_HOSTING`](../../../architecture/APPLICATION_HOSTING.md); LKW is the first adopter and proof ([`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md) §34). LKW.6A delivered the application execution boundary; LKW.6B adopts platform hosting — it does not implement `HostedApplicationEngine`, supervisors, or generic OS adapters in the application tree.

| Task | Owner module | Deliverable |
|------|--------------|-------------|
| LKW.6.1 | `scripts/lkw-host.*` | Start/stop wrapper for uvicorn (dev/operator convenience — not generic hosting engine) |
| LKW.6.2 | `hosting` (LKW profile) + platform adoption | LKW-specific `HostedApplicationProfile` / hooks; platform OS adapter integration via APP-HOST-7 (§7.4 targets) |
| LKW.6.3 | `host/factory.py` | `wire_interaction_intake_service` + router |
| LKW.6.4 | `host/settings.py` | `LOCAL_WORKSPACE_INCLUDE_INTERACTIONS` |

**Acceptance (LKW.6B initial proof):** foreground hosted start → READY → real LKW request → single-instance rejection → graceful stop → supervisor restart → new instance identity → real request after restart. No Slack required. Service-manager installation and reboot survival are **APP-HOST-7** targets, not LKW.6B initial proof.

**Frontend:** none new. **Backend:** host only.

**ORCH-MAINT-02 — CFG-14 hybrid daemon enablement (operator runbook):**

1. Copy `.env.example` → `.env` in `applications/local_workspace_application`.
2. Set `LOCAL_WORKSPACE_INCLUDE_SCHEDULER=true`, `LOCAL_WORKSPACE_INCLUDE_INTERACTIONS=true`, `LOCAL_WORKSPACE_INCLUDE_TASK_CONTROL=true`.
3. Optional background-jobs path: `LOCAL_WORKSPACE_INCLUDE_QUEUE_WORKER=true` when a message_bus provider is configured (see ORCH-MAINT-01 lab scaffold default).
4. Start host: `uv run uvicorn local_workspace_application.host.main:app --port 8090`.
5. Verify: `GET /health` → 200; `POST /v1/local_workspace/run` with `echo.basic` completes; scheduler poll logs when `INTERGRAX_SCHEDULER_POLL_SECONDS` set.

**Platform audit (2026-06-09):** CFG-14 hybrid daemon E2E remains **deferred** (Band 3 / §6.3). Harness reference for task control + scheduler: `poc_template_application`, `legal_application`, `research_application` with `INCLUDE_TASK_CONTROL` — see [`docs/project/architecture/ORCHESTRATION.md`](../../../architecture/ORCHESTRATION.md) §59.2 · Phase **H-APP-WIRING.4**.

---

#### LKW.6b — Slack optional channel

| Task | Owner module | Deliverable |
|------|--------------|-------------|
| LKW.6b.1 | `host/slack_socket.py` (new) | Socket Mode client → local intake |
| LKW.6b.2 | mapping | Slash `/lkw` → capability table (§9.4) |
| LKW.6b.3 | profile | `notification_channel=slack` for HITL |

**Acceptance:** `/lkw search foo` in Slack returns summary; **LKW.1 acceptance still passes with Slack disabled.**

---

#### LKW.7 — File watcher + incremental index

**Status:** **Closed** — LKW.7A **Done**; LKW.7B **Closed**; LKW.7B1 **Done**; LKW.7B2 **Closed**; LKW.7B2A **Done**; LKW.7B2B **Done**; LKW.7C **Closed**; LKW.7C1 **Done**; LKW.7C2 **Done**.

| ID | Scope | Status |
|----|-------|--------|
| **LKW.7A** | Incremental file-change contract and idempotent batches | **Done** |
| **LKW.7B** | Watcher runtime + sidecar process | **Closed** |
| **LKW.7B1** | Runtime state machine, bounded debounce, existing enqueue boundary | **Done** |
| **LKW.7B2** | Cross-platform sidecar process, settings, checkpoint, graceful shutdown | **Closed** |
| **LKW.7B2A** | Durable checkpoint and restart recovery | **Done** |
| **LKW.7B2B** | Sidecar settings, process loop, signals and automatic checkpoint lifecycle | **Done** |
| **LKW.7C** | Persistent-index live proof and ProofReceipt | **Closed** |
| **LKW.7C1** | Watcher-triggered persistent search E2E workload | **Done** |
| **LKW.7C2** | ProofReceipt recording, reviewer guide and final LKW.7 closeout | **Done** |

**LKW.7A flow (contract only):**

```text
allowed roots
  → metadata snapshot (path + size_bytes + modified_time_ns)
  → snapshot diff
  → IncrementalFileChangeBatch
  → change_token
  → LkwBackgroundIngestJob(change_token=...)
```

**LKW.7B1 flow (runtime state machine — no OS process yet):**

```text
snapshot
  → diff
  → pending final state per canonical path
  → quiet debounce or maximum wait
  → IncrementalFileChangeBatch
  → LkwBackgroundIngestJob
  → enqueue_background_ingest_job
  → platform message bus
```

**LKW.7B2A flow (durable checkpoint):**

```text
runtime baseline + pending final changes
  → versioned FileWatcherCheckpoint
  → deterministic JSON
  → atomic replace under data home
  → process restart
  → fail-closed load
  → runtime restore
  → first poll detects downtime changes
  → existing LKW.7B1 debounce / enqueue flow
```

**LKW.7B2B flow (foreground sidecar process):**

```text
environment settings
  → existing Kafka message bus
  → watcher runtime
  → checkpoint restore or fresh baseline
  → immediate poll
  → checkpoint after every completed cycle
  → bounded sleep
  → signal-driven shutdown
  → final checkpoint
```

**LKW.7C1 live path (watcher-triggered E2E workload):**

```text
file created after watcher baseline
  → watcher metadata diff
  → deterministic background-ingest job
  → Kafka
  → background worker
  → LocalIndexerAgent
  → rag.ingest_document
  → persistent Qdrant
  → LocalSearchAgent
  → exact source_ref match
```

**LKW.7C1 restart path:**

```text
restart watcher + worker + backend + Qdrant
  → checkpoint restore
  → unchanged source does not enqueue again
  → persistent search still returns source
```

**LKW.7C2 final path (ProofReceipt closeout):**

```text
filesystem create
  → watcher
  → Kafka
  → worker
  → persistent Qdrant
  → search
  → non-destructive restart
  → search
  → verified ProofReceipt
  → MongoDB DocumentStore
```

LKW.7C2 records the live workload evidence through platform `ProofReceiptStore` → `DocumentStore` → MongoDB. Reviewer path: [`LKW_7_FILE_WATCHER_VERIFICATION.md`](LKW_7_FILE_WATCHER_VERIFICATION.md) and [`LKW_PLATFORM_PROOF.md`](../../../proofs/LKW_PLATFORM_PROOF.md) Steps 12–13.

| Concern | Notes |
|---------|-------|
| Version identity | Metadata-based only (`path` + `size_bytes` + `modified_time_ns`); not content hashing |
| `change_token` | Deterministic identity of final actionable `source_snapshots` in one batch |
| Initial files | Baseline only — not emitted as `created` and not enqueued at start |
| Pending state | Bounded by changed path count; last change wins per canonical path |
| Debounce | Quiet period on `last_change_at`, plus bounded `max_batch_wait_seconds` |
| Deletions | Deletion-only batches do not enqueue; not automatically removed from the index |
| Enqueue failure | Pending changes retained, checkpointed, and retried; deterministic batch/job identity |
| Checkpoint | Durable baseline + final pending `FileChange` values; no file content |
| Monotonic timestamps | Never persisted; restored pending work starts a new debounce window |
| Missing checkpoint | Valid fresh-start — initialize baseline and persist before first poll |
| Invalid checkpoint | Fail closed — not silently treated as missing |
| Identity mismatch | Fail closed when tenant/workspace/collection/roots disagree |
| Watcher roots | Reuse `INTERGRAX_ALLOWED_READ_ROOTS` / `allowed_read_roots` (no separate roots setting) |
| Data home | Relative `data_home` resolved to an absolute process path before checkpoint construction |
| Snapshot failure | Retried after poll interval without mutating or saving runtime state |
| Checkpoint failure | Stops the sidecar immediately (no further poll/sleep) |
| Signals | Platform `PortableForegroundSignalAdapter` owns SIGINT / SIGTERM / SIGBREAK |
| Process entrypoint | `python -m local_workspace_application.file_watcher` |
| Live proof | LKW.7 Closed — watcher E2E + verified MongoDB-backed ProofReceipt |
| Content | No raw file content enters the job or checkpoint |

**Acceptance (full LKW.7):** Drop file in watched folder → indexed within N minutes without user command → verified `ProofReceipt` in MongoDB. **Met.**

---

#### LKW.8 — Tray frontend (thin client)

| Task | Owner module | Deliverable |
|------|--------------|-------------|
| LKW.8.1 | `clients/lkw-tray` (new repo folder or app) | Status icon + search box |
| LKW.8.2 | | Folder picker → `allowed_read_roots.json` |
| LKW.8.3 | | Calls only `localhost:8020` API |

**Acceptance:** No Python agent code in tray; uninstall tray does not remove index.

---

### 15.3 End-to-end scenarios (validation scripts)

| # | Scenario | Channels | Waves required |
|---|----------|----------|----------------|
| E1 | First install → pick folders → index | Tray + HTTP | LKW.5, LKW.6, LKW.8 |
| E2 | "Find documents about X" at desk | MCP | LKW.1 |
| E3 | Full report draft | HTTP pipeline | LKW.2 |
| E4 | New file auto-indexed | background | LKW.7 |
| E5 | Search from phone | Slack | LKW.6b (optional) |

---

## 16. Known platform gaps (honest audit)

| Gap | Impact | Mitigation (Wave) |
|-----|--------|-------------------|
| No file watcher | No auto re-index | LKW.7 file watcher + enqueue path |
| In-memory vector store default | Index lost on restart | Chroma + `INTEGRATION_PROFILE_JSON` |
| Windows path / OneDrive edge cases | Parser failures | Test matrix in LKW.1 acceptance |
| Qdrant/Chroma lack `list_document_ids` | `rag.list_documents` empty/unsupported on some backends | Use InMemory for dev; extend provider bindings in follow-up |

These gaps are **expected** — LKW exists to discover and close them without Nexus forks.

---

## 17. References

| Topic | Document |
|-------|----------|
| Agent workflow | [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md`](../../guides/AGENT_CREATION_GUIDE.md) |
| Integration catalog | [`docs/project/architecture/INTEGRATIONS.md`](../../../architecture/INTEGRATIONS.md) |
| Tools catalog | [`docs/project/architecture/TOOLS.md`](../../../architecture/TOOLS.md) |
| Skill Library | [`docs/project/architecture/SKILLS.md`](../../../architecture/SKILLS.md) |
| Tools & skills control plane | [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md` Appendix J](../../guides/AGENT_CREATION_GUIDE.md#appendix-j--tools--skills-control-plane) |
| RAG control plane | [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md` Appendix K](../../guides/AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane) |
| Shadow workspace | [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md` Appendix B](../../guides/AGENT_CREATION_GUIDE.md#appendix-b--shadow-workspace-and-sandbox) |
| Multi-agent graphs | [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md` Appendix C](../../guides/AGENT_CREATION_GUIDE.md#appendix-c--multi-agent-graphs) |
| Nexus execution flow | [`docs/project/architecture/NEXUS_EXECUTION_FLOW.md`](../../../architecture/NEXUS_EXECUTION_FLOW.md) |
| Implementation plan | [`docs/project/architecture/intergrax_runtime_architecture.md`](../../../architecture/intergrax_runtime_architecture.md) |
| Quickstart | [`README.md`](../../../../../applications/local_workspace_application/README.md) · [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md) |

---

## 18. Runtime recovery (APP-EVOL-5)

| Scenario | Host action |
|----------|-------------|
| Host restart | `resume_scheduler` via `ReliabilityProfile.recovery_contract` |
| Task interrupted | `resume` with checkpoint + idempotency store |
| Graph node failure | `retry_node` via Nexus orchestration retries |
| Corrupt checkpoint | `replay_from_snapshot` using `environment_snapshot.v1` |

- **Checkpoint store:** SQLite task checkpoints (see `.env.example` / `BUILD_AND_DEPLOY.md`)
- **Scheduler:** `long_running_scheduler_enabled` for async and HITL paths
- **In-flight tasks on deploy:** drain via checkpoint + `resume_token`; do not abort without operator ack

## Application dependency project

Canonical packaging: [docs/project/architecture/APPLICATION_DEPENDENCY_MODEL.md](../../../architecture/APPLICATION_DEPENDENCY_MODEL.md).

```bash
uv sync --project applications/local_workspace_application
uv run --project applications/local_workspace_application python -m local_workspace_application.host.main
```

## 19. Exact connected-source materialization purge

`KnowledgeMaterializationPurgeService` is the internal, provider-neutral boundary
for the next Indexed Source detach task. Its scope is the exact tuple
`tenant_id/workspace_id/source_id/indexed_source_binding_id/knowledge_source_binding_ref`;
local-file and web ownership are never eligible.

Purge first persists its intent and then performs a CAS invalidation of the
publication fence. The fence becomes detached and disabled while retaining the
committed publication head, so Search/Ask fails closed before any local
document, chunk, embedding, candidate, manifest, receipt, pointer or sequence
record is removed. Sync permits and stale commits cannot cross that CAS.

Deletion is bounded by a 1–500 record page and a durable, authenticated cursor
through the phases `DOCUMENT_REFERENCES`, `RECOVERY_RECORDS`, `MANIFESTS`,
`DELIVERY_RECORDS`, `PUBLICATION_CHAIN` and `COMPLETION_PROOF`. Hard per-invocation
budgets are: ≤`page_size` document references, ≤`page_size` recovery records,
≤`page_size` manifest entries, ≤`page_size` delivery records, one publication
chain node, and one bounded completion-proof check set. No phase may loop all
pages or all manifest entries inside one `start_or_resume()` call.

The document-ownership index is the only document enumerator: each canonical
reference is ownership-validated, its exact vector/chunk/embedding materialization
is deleted first, then the canonical reference and derived index row are removed.
Missing canonical references are safe orphan-index evidence; index/reference
fingerprint or five-field ownership mismatches fail closed.

Complete-ownership recovery records (`_WorkspaceDocumentIndexReceipt`,
`ConnectedSourceSyncEnqueueIntent`, `ConnectedSourceOperationDeliveryAccounting`)
are dual-written into an exact recovery-ownership index keyed by the five-part
ownership scope plus `record_kind`. Enumeration uses the authenticated
DocumentStore ownership prefix only; tenant-wide recovery partitions are not
scanned for complete-ownership purge work. Before deletion the purge reloads the
canonical row, requires `COMPLETE_OWNERSHIP`, revalidates the five ownership
fields and the SHA-256 canonical fingerprint, then deletes canonical first and
the index row second. A crash after canonical deletion leaves safe orphan-index
evidence. Index/canonical mismatches fail closed. Missing canonical rows are
orphan evidence. Duplicate index repair from a known complete-ownership
canonical is idempotent. The index does not override the canonical record.

The exact recovery index does not discover historical unindexed legacy rows.
`LEGACY_MIGRATION_REQUIRED`, `LEGACY_NON_CONNECTED`, local-file and web records
are not indexed. Full historical migration is not implemented here; purge
completion applies only after an explicit durable migration gate confirms that no
relevant legacy recovery records remain for the exact five-part ownership scope.
Missing or `REQUIRED` gates fail closed as `BLOCKED_LEGACY_MIGRATION`; corrupt or
scope-mismatched gates fail as `BLOCKED_CORRUPT_STATE`. Empty recovery indexes
alone never prove historical absence. New connected-source bindings created under
the ownership-complete schema generation (`ownership_complete_schema.v1`) persist
the gate as `CLEARED` at first create (no predecessor), because no pre-contract
recovery rows can exist for that binding. Reactivation and historical scopes remain
`REQUIRED` until a future deterministic migration clears them. New
complete-ownership writes are indexed; records created before this contract are
not retroactively covered.

Delivery receipts keep a canonical row key of
`workspace:source:delivery_id`. Purge cleanup and completion therefore enumerate
receipts only through a derived exact-ownership index keyed by the five-part
binding scope, so a receipt belonging to binding B cannot observe or block
completion for binding A.

Manifest deletion uses a durable per-entry cursor bound to the purge ID:
`document_store_cursor` (manifest page continuation), `current_manifest_row_key`,
`current_manifest_id`, `current_manifest_fingerprint`, `current_delivery_id` and
`manifest_entry_offset`. One invocation loads at most one current manifest,
validates immutable identity (manifest ID, fingerprint, delivery ID, sequence and
ownership) on every resume, processes `[offset : offset + page_size]` entries,
and persists the next offset. Receipt, sequence assignment, delivery-index and
immutable-manifest cleanup run only after all entries are processed. A missing
or mismatched current manifest before entry completion is `BLOCKED_CORRUPT_STATE`
and never silently advances. Crash/restart reprocesses idempotently without
skipping entries; concurrent workers rely on purge-state CAS so the cursor never
moves backward and a losing worker reloads.

Manifest entries, prepared delivery indexes, remote candidates, active pointers,
receipts and sequence assignments are otherwise removed by deterministic scope
keys; the constant-size sequence head is removed last.

The publication chain is traversed from its exact head; each head is CAS-advanced
to its predecessor before the old immutable node is removed. A restart repeats
only idempotent exact deletions. Before `COMPLETED`, bounded empty-page checks
prove the document ownership index, recovery ownership indexes for all three
record kinds, manifests (with no partial entry cursor remaining), delivery
records, active pointers, sequence head and publication head are empty.
Counters remain non-authoritative. A legacy migration blocker or corruption never
reports completion. The detached fence remains as bounded purge/lifecycle
evidence, with no publication permit or committed head. Provider data is not
mutated. The service is intentionally not an HTTP or lifecycle-detach endpoint
and is the prerequisite for the final publication-fence closeout.

## 20. Purge enumeration prerequisites

DocumentStore traversal uses `DocumentQueryPageV1`. The cursor is an opaque,
bounded, checksum-validated token containing the partition, row-key prefix,
last returned row key and cursor schema. It is reusable by another process only
for the same query shape; traversal promises deterministic ordering over an
unchanged, snapshot-like dataset, not transactional snapshot isolation.

Connected-source index receipts, delivery accounting and enqueue recovery
intents carry complete binding identity on new writes and are dual-written into
the exact recovery-ownership index. Historical records without that identity
remain explicitly `LEGACY_MIGRATION_REQUIRED` and are never associated with a
purge by inference. Local-file and web records remain legacy records. The index
covers only new complete-ownership writes; it does not retroactively cover
records created before this contract.

Connected-source document references are indexed one-per-row by the exact
five-part ownership scope. The canonical reference is written first and the
derived ownership row second. A missing second write is repairable from a
known canonical reference; an index/reference fingerprint or ownership
mismatch fails closed. Paginated ownership lookup validates the canonical
reference before returning it and reports missing canonical rows as orphan
index evidence, without scanning the workspace.
