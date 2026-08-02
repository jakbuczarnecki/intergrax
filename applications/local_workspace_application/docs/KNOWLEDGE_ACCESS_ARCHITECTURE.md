# Hybrid Knowledge Access — LKW Architecture

**Status:** `ACCEPTED`
**Task:** `LKW-KNOWLEDGE-ACCESS-ARCHITECTURE-1`  
**Classification:** docs-only architecture and product contract  
**Top-level architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)  
**Implementation plan:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)  
**Knowledge Intake:** [`KNOWLEDGE_INTAKE_DISCOVERY.md`](KNOWLEDGE_INTAKE_DISCOVERY.md)  
**Conversational interaction:** [`CONVERSATIONAL_INTERACTION.md`](CONVERSATIONAL_INTERACTION.md)

---

## 1. One-sentence outcome

LKW is a **private, governed and provider-neutral Hybrid Knowledge Workspace** that combines indexed RAG knowledge, controlled live access to external systems, interchangeable Ollama/vLLM conversation runtimes, natural-language frontends and unified evidence provenance in one coherent product roadmap.

---

## 2. Product definition

### 2.1 What LKW is (target)

LKW lets a user:

1. upload files and folder snapshots;
2. connect local folders;
3. attach explicit Web URLs;
4. connect external systems (Microsoft 365, OneDrive, SharePoint, Outlook, Teams-hosted knowledge, Google Workspace — Drive, Docs, Sheets, Calendar and related surfaces when implemented, Jira, Confluence, Databricks, Power BI, Atlan, future native API providers, future curated MCP providers);
5. decide which connected resources are **indexed into RAG**;
6. decide which resources may be **queried live**;
7. ask one natural-language question through Slack or another frontend;
8. receive one answer assembled from indexed knowledge, live provider results, or both;
9. inspect where each piece of evidence came from;
10. run the same LKW application with **Ollama or vLLM** through configuration.

### 2.2 Status honesty

| Category | Examples |
|----------|----------|
| **Implemented today** | Managed-file upload; Source Candidate intake; end-to-end `WEB_URL` Knowledge Intake (**ACCEPTED**); HTTP Ask Workspace with indexed RAG; Slack thin client for Ask, workspace ops and source inspection; Conversation Interaction Planner contract (`CONV-1A`); Slack connected source discovery/create/sync with indexed Search/Ask proof (`LKW-SLACK-CONNECTED-SOURCE-1` **DONE**) |
| **Architecturally available in Intergrax** | `vendor_knowledge` connection resolution; integration/tool execution; RAG ingest/retrieve; `LLMAdapter` provider neutrality; embedding providers separate from conversation LLM; policy and trace; Slack three-mode knowledge architecture frozen (`SLACK-KNOWLEDGE-THREE-MODE-ARCH-1`) |
| **Planned for LKW** | Workspace Knowledge Configuration; Live Access Bindings; Hybrid Ask; Knowledge Query Orchestrator; model-runtime portability proof; vendor collaboration and data connector packs (including Google Workspace after Slack vertical — `GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1` **READY_FOR_REVIEW**, runtime **PLANNED**); Conversation Context Bindings and audience isolation (`LKW-CONVERSATION-CONTEXT-ARCH-1` **READY_FOR_REVIEW**, `LKW-CONVERSATION-CONTEXT-1` **PLANNED**); Slack knowledge proof (`LKW-SLACK-KNOWLEDGE-PROOF-1`); Google Workspace LKW proof (`LKW-GOOGLE-WORKSPACE-PROOF-1` **PLANNED**); live platform proof |
| **Future / not committed** | Write-capable provider actions; unrestricted SQL/DAX/JQL; runtime hot swapping; automatic persistence of live results; MCP as domain model |

Target architecture is **not** evidence of implementation. Public proof claims require checked-in evidence.

---

## 3. Three knowledge-access modes

### 3.1 Indexed Knowledge

Knowledge already processed into LKW-owned stores.

**Examples:** uploaded files; folder snapshots; local folders; Web URLs; synchronized SharePoint files; synchronized Confluence pages; synchronized Jira issues; synchronized Slack channel or conversation history (`PLANNED` — not implemented); synchronized Google Docs, Sheets, Calendar and Drive files (`PLANNED` — `LKW-GOOGLE-WORKSPACE-PROOF-1`).

**Benefits:** cross-source semantic retrieval; lower provider API usage; consistent chunking and embeddings; offline or temporarily disconnected usage; stable workspace-owned retrieval.

**Limitations:** data may be stale; synchronization is required; copying may be unsuitable for some data.

**Flow:**

```text
Connection or direct input
→ Remote Resource or managed resource
→ LKW Indexed Source
→ ingestion / synchronization
→ Document
→ Chunks
→ Embeddings
→ Vector Store
→ RAG retrieval
```

### 3.2 Live Knowledge Access

Read-only access executed when the user asks a question.

**Examples:** current Outlook messages; current Jira issue state; current Power BI metric; current Databricks job status; approved Databricks SQL result; current Atlan lineage; current Confluence page state; bounded current Slack message or thread reads (`PLANNED` — not implemented).

**Benefits:** current information; no requirement to copy every dataset; appropriate for dynamic or large systems.

**Limitations:** depends on provider availability; slower; subject to provider permissions, rate limits and cost; requires bounded tool execution.

Live results **do not automatically become** durable Documents.

### 3.3 Hybrid Knowledge (target default experience)

```text
indexed RAG evidence
+
authorized live evidence
+
one normalized evidence set
+
one grounded answer
+
unified provenance
```

**Representative example:**

```text
Question:
“Are we ready to deploy Project Orion?”

Indexed evidence:
- deployment plan
- meeting notes
- architecture documents

Jira live:
- current blockers

Microsoft 365 live:
- latest client messages

Power BI live:
- current readiness KPI

Result:
- one answer
- explicit risks
- citations
- visible distinction between indexed and live evidence
```

### 3.4 Platform provider foundation vs LKW product modes

LKW product modes remain:

```text
indexed
live
hybrid
```

The reusable Intergrax provider foundation has three **consumption modes**:

```text
indexed RAG
durable materialization without RAG
live access
```

LKW uses:

```text
durable materialization
→ LKW Knowledge Intake
→ indexed RAG
```

Other Intergrax applications may use durable materialization without creating a vector index. LKW is **not** a generic data-replication or ETL product.

### 3.5 Shared provider foundation diagram

```text
one Connection
→ one vendor integration
→ shared provider primitives
   ├── durable sync/materialization
   │      ├── application database/store
   │      └── LKW Knowledge Intake → RAG
   └── live capability execution → ephemeral evidence
```

**Slack foundation (`PLANNED` — architecture frozen):**

```text
Slack Connection
→ SlackConversationChannelIntegration
→ shared typed Slack read primitives
   ├── Slack Vendor Knowledge Adapter → durable sync → optional LKW RAG
   └── Slack Live Capability Adapter → ephemeral evidence
```

Slack frontend transport uses the same integration foundation but does not grant history ingestion or live access by itself.

Hybrid Ask combines indexed and live evidence at the application level. It does not create a fourth vendor integration.

---

## 4. Canonical vocabulary

### 4.1 Connection

A tenant-owned configured relationship with an external system.

**Examples:** company Microsoft 365 tenant; engineering Jira instance; internal Confluence instance; analytics Databricks workspace; finance Power BI tenant; data-governance Atlan instance; approved Slack workspace installation (`PLANNED` for knowledge reads); approved MCP server.

#### 4.1.1 Durable `TenantConnection` (platform-owned)

A tenant Connection is a **durable platform entity** — not an LKW workspace record and not an in-memory registry entry. The conceptual model is `TenantConnection` (**to be implemented in `LKW-KNOWLEDGE-ACCESS-1C-1`**; not yet present as a Python model).

**Durable identity:** `(tenant_id, connection_ref)` — `connection_ref` is opaque and unique within one tenant.

**Minimum durable fields:**

```text
connection_ref
tenant_id
provider_id
integration_kind
safe_display_name
administrative_status      # ACTIVE | DISABLED | REVOKED
credential_ref             # opaque SecretsStore reference only
validated_secret_free_config
configuration_version
created_at
updated_at
connected_principal_ref    # optional, when justified
```

**Administrative status** (`ACTIVE`, `DISABLED`, `REVOKED`) is the durable lifecycle. It must not be conflated with **runtime health** (`available`, `degraded`, `unavailable`), which is recomputed by resolution or health checks and is not the authoritative durable lifecycle.

`validated_secret_free_config` may hold private, non-secret configuration required to construct or validate the provider integration (tenant or organization identifier, provider account reference, approved base endpoint, region, non-secret scopes, provider feature flags). It must not contain credentials and must not automatically be exposed through public LKW APIs.

`credential_ref` points to `SecretsStore`. The Connection record does not contain the secret.

**Current repository gap:** durable `TenantConnection` persistence does not exist yet. Today the repository has only opaque `connection_ref` on bindings, an **instance-local** `KnowledgeConnectionRegistry` (runtime projection / cache — not durable catalog, not administrative source of truth), and application `IntegrationProfile` bootstrap (application-level composition — not a tenant Connection database).

A Connection must **not** expose credentials to the LLM, Slack, another frontend, workspace configuration responses, prompt context or provenance records. A workspace does not own raw credentials. LKW persists only a reference to the tenant Connection plus safe cached presentation data where already approved.

### 4.2 Remote Resource

A provider-owned resource discoverable through a Connection.

**Examples:** SharePoint site; OneDrive drive; mailbox; Jira project; Confluence space; Databricks catalog; Databricks SQL warehouse; Power BI workspace; Power BI semantic model; Atlan catalog scope; MCP resource or approved tool collection; approved Slack channel or conversation (`PLANNED` — not implemented).

A Remote Resource is **not** automatically an LKW Source.

### 4.3 Indexed Source

A durable workspace-owned Source whose content is ingested or synchronized into LKW knowledge stores.

**Examples:** uploaded PDF; uploaded XLSX; Slack attachment (managed-file intake — implemented); synchronized Slack channel or conversation (`PLANNED`); local folder; explicit Web URL; SharePoint site synchronized into LKW; Confluence space synchronized into LKW; selected Jira project synchronized into LKW.

Every persisted Document must remain owned by exactly one durable Source.

An **Indexed Source** authorizes durable ingestion/materialization into LKW. It does **not** automatically grant live access capabilities.

### 4.4 Live Access Binding

A workspace-scoped authorization that permits selected read-only capabilities against a Connection or Remote Resource at question time.

**Example:**

```text
workspace: Project Orion
connection: company-ms365
resource: project mailbox
allowed capabilities:
- ms365.mail.search
- ms365.mail.read
mode: read-only
```

**Slack example (`PLANNED` — not implemented):**

```text
workspace: Project Orion
connection: company-slack
resource: #project-orion channel
allowed capabilities:
- slack.conversation.read_bounded
- slack.thread.read_bounded
mode: read-only
```

Indexed permission and live-access authorization are separate grants. A Slack Live Access Binding does not imply durable synchronization or RAG indexing.

A Live Access Binding:

- is not automatically a Source;
- does not automatically copy provider data into RAG;
- does not contain credentials;
- does not expose every provider capability;
- must define an allowlisted, read-only capability surface;
- is tenant- and workspace-scoped;
- does **not** automatically grant durable ingestion rights.

A resource may have:

```text
Indexed Source only
Live Access Binding only
both
neither
```

Neither binding automatically grants the other.

### 4.5 Workspace Knowledge Configuration

The aggregate configuration that determines what knowledge one workspace can use.

**Diagram A — knowledge configuration:**

```text
Workspace
├── Indexed Sources
├── Live Access Bindings
├── Query Policy
├── Allowed Capabilities
└── Model Runtime Profile
```

Conceptual structure:

```text
WorkspaceKnowledgeConfiguration
├── Indexed Sources
├── Live Access Bindings
├── Allowed Read Capabilities
├── Query Policy
├── Model Runtime Profile reference
├── Evidence / retention policy
└── limits and safety controls
```

Exact persistence model, revision-head CAS publication protocol, mutation reservation, semantic no-op idempotency and API schemas are frozen in [`KNOWLEDGE_ACCESS_IMPLEMENTATION_CONTRACT.md`](KNOWLEDGE_ACCESS_IMPLEMENTATION_CONTRACT.md) (`LKW-KNOWLEDGE-ACCESS-1A-C3`).

### 4.6 Query Policy

Conceptual modes:

| Mode | Meaning |
|------|---------|
| `indexed_only` | Use only persisted workspace knowledge and RAG retrieval |
| `live_only` | Use only currently authorized live provider capabilities |
| `hybrid` | Use both indexed and live evidence when requested or required by the plan |
| `automatic` | Query orchestration may choose indexed, live or hybrid according to question, workspace configuration, policy and available capabilities — **not** unrestricted agent autonomy |

Possible policy controls (target vocabulary — not all implemented):

```text
prefer_indexed_evidence
allow_live_fallback
allowed_connections
allowed_capabilities
max_live_calls
max_total_query_duration
max_result_items
max_result_bytes
retention mode
freshness requirements
```

**V1 implementation subset:** [`KNOWLEDGE_ACCESS_IMPLEMENTATION_CONTRACT.md`](KNOWLEDGE_ACCESS_IMPLEMENTATION_CONTRACT.md) freezes `indexed_only` and `live_only` modes only; `prefer_indexed_evidence` and `allow_live_fallback` are deferred to future hybrid/automatic modes.

### 4.7 Model Runtime Profile

Product-level runtime selection containing at least:

```text
provider          # ollama | vllm
endpoint
model
context window
structured-output capability
tool-calling capability
timeout
qualification status
health status
```

**Persistence clarification:**

- Constructed `LLMAdapter` objects and provider clients are **runtime-only**.
- Deployment-wide default runtime (for example `INTERGRAX_LLM_PROVIDER`) may remain **deployment configuration**.
- A future user-selectable workspace runtime profile must be represented by a **durable profile or durable profile reference** — not by a constructed adapter object in DocumentStore.
- `LKW-MODEL-RUNTIME-1` (**ACCEPTED**) proves Ollama/vLLM portability; it does **not** imply that a multi-profile runtime catalog already exists.
- This architecture task does not implement or schedule runtime-profile administration unless an existing canonical task already owns it. Do not expand `LKW-KNOWLEDGE-ACCESS-1B` with runtime-profile persistence.

LKW receives a ready `LLMAdapter` through application wiring. The LKW domain must **not** contain provider branches such as `if provider == "ollama": … elif provider == "vllm": …`.

**Conversation LLM and embedding provider are separate concerns.** Switching from Ollama to vLLM must not silently change the embedding model, vector dimensions, rebuild collections, invalidate indexed data or change LKW product contracts.

### 4.8 Evidence Item

One provider-neutral evidence concept for both RAG and live results.

Conceptual fields:

```text
evidence_type: indexed | live
tenant_id
workspace_id
provider_id
connection_ref
source_id
remote_resource_id
remote_item_id
safe_display_name
safe_locator
retrieved_at
remote_updated_at
content_hash
tool_invocation_id
excerpt or structured result
```

Not every field is required for every evidence type. Exact Pydantic models are **not** frozen here.

### 4.9 Configuration persistence boundary

**Canonical principle (frozen):**

```text
All user-managed product configuration that must survive process or deployment
restart is durable.

Raw secrets remain in SecretsStore.

Constructed clients, registries and current health observations remain runtime
state.

Deployment bootstrap and infrastructure topology remain deployment
configuration unless a separately accepted administration-plane task moves
them into durable product configuration.
```

LKW configuration is **not** stored in one monolithic database. Four persistence boundaries apply:

#### 4.9.1 Durable Database / DocumentStore state

Durable database / DocumentStore state is one persistence boundary. It contains two separate ownership categories:

##### 4.9.1.1 Durable platform and tenant configuration

Durable platform and tenant configuration includes:

```text
Tenant Connections
KnowledgeSourceBindings
safe provider and source configuration
administrative connection lifecycle
configuration versions
tenant ownership
opaque credential references
```

**Owner:** shared platform integration / connection foundation.

**Not owner:** LKW workspace domain, Slack, `KnowledgeConnectionRegistry`, `IntegrationProfile`.

##### 4.9.1.2 Durable LKW workspace configuration

Durable LKW workspace configuration includes:

```text
Workspace
WorkspaceConnectionAttachment
WorkspaceIndexedSourceBinding
WorkspaceLiveAccessBinding
WorkspaceQueryPolicy
WorkspaceKnowledgeConfigurationHead
WorkspaceKnowledgeMutationRecord
WorkspaceSource
WorkspaceDocumentReference
KnowledgeInput
operations
Ask runs
configuration revisions
idempotency and recovery state
```

The C3 revision, publication, idempotency and recovery contract in [`KNOWLEDGE_ACCESS_IMPLEMENTATION_CONTRACT.md`](KNOWLEDGE_ACCESS_IMPLEMENTATION_CONTRACT.md) remains authoritative.

#### 4.9.2 SecretsStore state

`SecretsStore` owns OAuth access tokens, OAuth refresh tokens, client secrets, API keys, passwords, certificates and private keys, and other credential material.

Database records may contain only an opaque `credential_ref`. No raw secret may appear in `TenantConnection`, `KnowledgeSourceBinding` public projection, LKW workspace records, logs, traces, Slack, public API responses or provenance.

#### 4.9.3 Runtime-only state

Runtime state includes constructed provider clients, constructed integration objects, `KnowledgeConnectionRegistry` entries, adapter registry entries, in-flight requests, leases held in process memory, current health checks, ephemeral Remote Resource discovery results and ephemeral live evidence.

Runtime state is reconstructed from durable configuration and `SecretsStore`. It is not the source of truth.

#### 4.9.4 Deployment configuration

Deployment configuration includes DocumentStore endpoint, VectorStore endpoint, `SecretsStore` implementation, message bus endpoint, object storage endpoint, default application profile, default runtime provider, container topology, ports and bootstrap flags.

These may remain in environment variables, deployment manifests, configuration files and application bootstrap. They are not automatically tenant or workspace product configuration.

### 4.10 Startup and restart reconstruction

**Target restart flow:**

```text
application starts
→ load deployment/application bootstrap
→ open durable Tenant Connection Catalog
→ list enabled tenant Connections
→ load safe Connection configuration
→ resolve credential_ref through SecretsStore
→ construct exactly one integration instance per active Connection
→ register integration in KnowledgeConnectionRegistry
→ expose safe Connection projection through TenantConnectionPort
→ LKW workspace bindings continue to resolve through connection_ref
```

**Required properties:**

- Connections survive process restart.
- Workspace attachments survive process restart.
- Indexed and Live Access authorization survives process restart.
- Query Policy survives process restart.
- No connector must be manually reconstructed after every restart.
- A missing or invalid secret does not delete the Connection.
- A failed reconstruction produces a safe unavailable/degraded projection.
- A disabled Connection is not reconstructed as active.
- The same `connection_ref` resolves to one runtime integration instance.
- Workspace configuration remains readable when a Connection is temporarily unavailable.

Do not claim automatic token refresh or provider-specific authentication behavior beyond existing integration contracts.

### 4.11 Remote Resource persistence

`RemoteResourceDescriptorV1` is **ephemeral discovery output by default**. Discovering a resource does not make it durable product configuration.

A resource becomes durable only after an explicit operation creates a `KnowledgeSourceBinding`, `WorkspaceIndexedSourceBinding`, `WorkspaceLiveAccessBinding`, or a separately approved `RemoteResourceSnapshot`. Do not add automatic provider inventory mirroring.

### 4.12 Explicitly rejected configuration designs

The following are **rejected**:

- storing raw tokens in the LKW database;
- storing integration/client Python objects in DocumentStore;
- treating `KnowledgeConnectionRegistry` as durable state;
- using `IntegrationProfile` as a multi-tenant Connection database;
- copying full Connection configuration into each workspace;
- copying credentials into `WorkspaceConnectionAttachment`;
- automatically persisting all discovered Remote Resources;
- recreating Connections manually after every restart;
- creating separate provider clients for indexed and live use;
- moving deployment infrastructure topology into workspace state.

Requirements:

- safe provenance;
- source or provider identity;
- freshness information for live evidence;
- no credentials;
- no secret-bearing URLs;
- no unsafe provider payloads;
- traceability to the operation or tool invocation.

---

## 5. Provider resource: indexed, live, or both

```text
provider resource
├── may be attached as an Indexed Source
├── may be attached through a Live Access Binding
└── may support both
```

| Resource | Indexed | Live |
|----------|---------|------|
| Confluence space | synchronized into RAG | searched live |
| Power BI semantic model | metadata may optionally be indexed | usually queried live |
| SharePoint site | documents may be synchronized | current metadata may be read live |
| Slack channel or conversation (`PLANNED`) | history may be synchronized into RAG | bounded current reads when authorized |

Do not assume every provider resource must be copied into RAG. Do not assume every indexed Source automatically permits live access.

One `SlackConversationChannelIntegration` serves both Slack frontend transport and platform provider reads, but application bindings and authorization lifecycles remain independent.

---

## 6. Query execution architecture

**Diagram B — Hybrid Ask:**

```text
Question
├── RAG retrieval
├── live provider calls
└── normalized evidence
        ↓
grounded synthesis
        ↓
answer + citations
```

**Target flow:**

```text
Slack / Web / Mobile / MCP / HTTP
        ↓
Conversation Interaction Planner
        ↓
typed user-intent plan
        ↓
deterministic reference resolver
        ↓
tenant/workspace authorization
        ↓
validated capability executor
        ↓
workspace.ask
        ↓
Workspace Knowledge Configuration
        ↓
Knowledge Query Orchestrator
        ↓
Evidence Plan
        ├── indexed RAG retrieval
        ├── Microsoft 365 live reads
        ├── Jira live reads
        ├── Confluence live reads
        ├── Slack live reads (PLANNED)
        ├── Databricks live reads
        ├── Power BI live reads
        └── Atlan live reads
        ↓
deterministic policy validation
        ↓
bounded read-only execution
        ↓
normalized evidence
        ↓
grounded synthesis
        ↓
answer + citations + provenance
        ↓
original frontend
```

### 6.1 Conversation Interaction Planner

Responsible for interpreting user **product intent**:

```text
create workspace
select workspace
add source
connect resource
ask question
inspect operation
remove source
```

Must **not** understand or generate Graph API requests, JQL, DAX, SQL, provider credentials or arbitrary MCP calls.

### 6.2 Knowledge Query Orchestrator

Responsible for obtaining evidence for a concrete, authorized question.

Input conceptually includes:

```text
tenant_id
principal_id
workspace_id
question
Workspace Knowledge Configuration
authorized capability catalog
```

Decides whether the evidence plan requires RAG, live capabilities, both or clarification. The model may propose an evidence or tool plan, but the **deterministic runtime must validate every executable call**.

---

## 7. Controlled tool execution

**Rejected architecture:**

```text
LLM → arbitrary URL → arbitrary API → arbitrary parameters → execute
```

**Accepted flow:**

```text
Workspace Knowledge Configuration
→ authorized capability catalog
→ model proposes bounded tool call
→ deterministic schema validation
→ tenant/workspace authorization
→ policy check
→ integration/tool execution
→ safe result normalization
→ evidence
```

**Example capability IDs** (target vocabulary unless verified implemented):

```text
rag.search
ms365.mail.search
ms365.mail.read
ms365.drive.search
ms365.drive.read
jira.issues.search
jira.issue.read
confluence.pages.search
confluence.page.read
databricks.catalog.search
databricks.sql.query
databricks.jobs.read
powerbi.semantic.query
powerbi.metadata.read
atlan.assets.search
atlan.lineage.read
```

### 7.1 Read-only first

The first live-access milestone is **strictly read-only**.

**Explicit non-goals for the initial live proof:**

```text
send email
modify Jira
create Jira issue
edit Confluence
run or cancel Databricks jobs
write Databricks tables
modify Power BI
modify Atlan
execute arbitrary MCP side effects
```

Future write capabilities require separate policy, confirmation, authorization, audit, compensation and risk classification.

---

## 8. Integration, API and MCP boundary

**Diagram C — provider boundary:**

```text
LKW
→ provider-neutral capability
→ Integration / Tool
→ native API | SDK | MCP
→ external system
```

Different technical providers may implement the same product capability boundary:

```text
native vendor API adapter
vendor SDK adapter
OpenAPI/HTTP adapter
MCP client adapter
```

MCP is **not** the LKW domain model. Do not define MCP Source, MCP Workspace or MCP Query Mode.

Instead:

```text
approved MCP server
→ registered Connection
→ curated capability descriptors
→ workspace Live Access Binding
→ validated tool execution
```

An MCP server must not automatically expose every tool to every workspace. Requirements: explicit server registration; credential isolation; capability discovery followed by approval; safe schemas; read-only allowlisting for the initial milestone; per-workspace binding; timeout and result limits; audit receipts.

---

## 9. Vendor Knowledge relationship

The existing `vendor_knowledge` runtime remains useful for:

```text
connection resolution
scope inspection
resource enumeration
content fetch
permission fetch
incremental synchronization
reconciliation
provider-neutral knowledge projection
```

**LKW owns:**

```text
tenant/workspace authorization
workspace connection attachment
Source ownership
Live Access Binding
query policy
Knowledge Intake
ingestion operations
Documents / Chunks / Vectors
Ask orchestration
evidence provenance
frontend behavior
```

Do not make `vendor_knowledge` import or depend on LKW. Do not duplicate provider integrations inside LKW.

### 9.1 No-duplication invariant

LKW must **never** instantiate a vendor client for live access when the same Connection already resolves an Intergrax integration.

Both indexed and live modes resolve through the shared integration/connection foundation:

```text
one Connection
→ one vendor integration
→ shared provider read primitives
   ├── durable path (Vendor Knowledge adapters / sync)
   └── live path (Live Capability adapter / executor)
```

### 9.2 Live result promotion semantics

```text
Live result
→ ephemeral evidence by default

Live result
→ explicit promote/materialize operation
→ durable Source or application record
```

Promotion must use a reviewed application lifecycle. It is **not** a flag on the live API call.

---

## 10. Natural-language frontend architecture

**Diagram D — frontend flow:**

```text
Slack / HTTP / MCP / Web
→ planner
→ resolver
→ authorization
→ executor
→ LKW capabilities
→ result
```

Slack remains a thin, replaceable frontend. Target natural-language examples:

```text
Create a workspace for Project Orion.

Add these files and this website.

Connect the engineering Jira project and the Orion SharePoint site.

Use the latest Jira blockers and messages from the client to tell me
whether we are ready to deploy.
```

Slack must **not:** own knowledge configuration; store provider credentials; instantiate vendor clients; call Jira, Microsoft Graph, Power BI or Databricks directly; own RAG; own tool selection; own operation state; become required for LKW operation.

---

## 10.1 Conversational audience boundaries and evidence scope

Conversation Context Binding, Indexed Source Binding and Live Access Binding are **independent grants**. Canonical contract: [`CONVERSATION_CONTEXT_ARCHITECTURE.md`](CONVERSATION_CONTEXT_ARCHITECTURE.md).

**Primary invariant:**

```text
The audience of the outbound answer determines the maximum knowledge scope.
```

**Ingress:** `binding.audience_mode` must match `ingress.observed_audience` before workspace resolution, memory lookup or Ask. `UNKNOWN` fails closed.

**Shared source eligibility** (default-deny): sources and Live Access Bindings carry `PERSONAL_ONLY` | `SHARED_ALLOWED`. Shared evidence requires `SHARED` workspace audience **and** `SHARED_ALLOWED` eligibility. Existing sources are not silently promoted.

For a **SHARED** conversation, indexed and live evidence must satisfy the bound shared workspace and `SHARED_ALLOWED` eligibility. Caller private permissions, personal workspace selection, personal memory and private connector grants must **never** expand the evidence boundary.

**Before model invocation:** validate active unique binding, audience match, principal rules, activation policy, workspace resolution, thread partition identity, and that every evidence/memory item matches tenant + workspace + audience eligibility — including thread memory, live tool results and planner context, not only citations.

**Before outbound delivery:** validate response conversation/thread, citation workspace membership, audience unchanged, and absence of personal-memory or personal-workspace evidence. Guard failure suppresses the outbound answer.

V1 shared conversations default to `READ_ONLY_ASK`; ordinary shared messages must not mutate bindings, connections, sources or approvals.

Hybrid Ask and the Knowledge Query Orchestrator must reject mixed personal/shared evidence deterministically — not through prompt instructions alone.

Connecting a Slack conversation as an Indexed Source does not activate the bot in that channel. Activating the bot in a channel does not automatically index channel history.

## 11. Security and governance

### 11.1 Credential isolation

Credentials remain in integration/provider credential storage. Only opaque references (`credential_ref`, `connection_ref`) may appear in safe product state. Credentials must never enter LLM prompts, tool schemas, Slack payloads, Ask citations, workspace Source metadata, trace messages or public API responses.

### 11.2 Workspace-scoped authorization

```text
principal
→ tenant access
→ workspace permission
→ Live Access Binding
→ allowed capability
→ allowed Remote Resource
```

Unknown or unauthorized references fail closed.

### 11.3 Bounded execution

Target controls (not all fully implemented): maximum live calls per question; timeout per call; timeout for whole Ask; provider result count and byte limits; pagination limit; concurrency limit; rate-limit handling; retry classification; cancellation; safe partial-answer policy.

### 11.4 Safe receipts

Each live execution should produce a policy-safe receipt with: `tenant_id`, `workspace_id`, `principal_id`, `connection_ref`, `capability_id`, safe resource identity, `started_at`, `completed_at`, `status`, result count, freshness timestamp, `trace_id`. Receipts must not contain secrets, credentials, access tokens, raw authorization headers, unsafe URLs or full unbounded provider payloads.

### 11.5 Retention

Live results must not automatically become durable workspace knowledge. Policy directions: `ephemeral`; `receipt_only`; `cache_with_ttl`; `promote_to_indexed_source` through explicit workflow. The first proof should default to ephemeral evidence plus safe receipt unless a capability explicitly uses the existing ingestion lifecycle.

---

## 12. Model runtime portability

**Task:** `LKW-MODEL-RUNTIME-1` — Ollama / vLLM end-to-end portability (**NEXT**).

**One-sentence outcome:** The same LKW product workflows run through either Ollama or vLLM selected by configuration, with no product-domain branching and with both runtimes qualified for structured planning, tool calling and grounded Ask.

Planned proof covers: basic generation; structured output; Conversation Interaction Plan generation; tool calling; grounded synthesis; health check; configuration switch; same product contracts.

Provider selection may require application restart in the initial proof. A model visible on Ollama or vLLM is not sufficient — it must pass a bounded LKW qualification matrix. This is a focused LKW product proof, not the deferred broad five-model benchmark.

Existing configuration (not LKW portability proof): `INTERGRAX_LLM_PROVIDER` (`ollama` default; `vllm` optional), `INTERGRAX_LLM_MODEL`, commented vLLM base URL variables in `.env.example`.

---

## 13. Architecture invariants

1. LKW is deployment-neutral.
2. Slack is optional and replaceable.
3. The LKW backend owns product logic.
4. Frontends do not access stores or vendor SDKs directly.
5. Every durable Document belongs to exactly one Source.
6. Live results do not automatically become durable Documents.
7. Credentials remain outside prompts and workspace state.
8. Every durable workspace record remains tenant- and workspace-scoped.
9. Provider resources require explicit workspace binding.
10. The model proposes; deterministic policy validates and executes.
11. The initial live-access milestone is read-only.
12. Native APIs and MCP may implement the same provider-neutral capability boundary.
13. MCP does not automatically expose all tools.
14. Ollama and vLLM are selected through provider-neutral wiring.
15. Conversation LLM and embeddings remain separate concerns.
16. LKW reuses Intergrax integrations, tools, runtime, queueing, RAG and policy mechanisms.
17. LKW must not create vendor-specific pipelines when provider-neutral platform boundaries already exist.
18. Target architecture is not evidence of implementation.
19. Public proof claims require checked-in evidence.
20. Application-first development remains the governing strategy.

---

## 14. References

| Document | Role |
|----------|------|
| [`KNOWLEDGE_ACCESS_IMPLEMENTATION_CONTRACT.md`](KNOWLEDGE_ACCESS_IMPLEMENTATION_CONTRACT.md) | Frozen implementation contract (`LKW-KNOWLEDGE-ACCESS-1A-C3` mutation semantics; `1A-C4` persistence boundary) |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Top-level LKW product architecture |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Canonical execution order |
| [`KNOWLEDGE_INTAKE_DISCOVERY.md`](KNOWLEDGE_INTAKE_DISCOVERY.md) | Indexed knowledge intake contract |
| [`CONVERSATIONAL_INTERACTION.md`](CONVERSATIONAL_INTERACTION.md) | Planner vs executor |
| [`SLACK_MVP_DISCOVERY.md`](SLACK_MVP_DISCOVERY.md) | Slack thin-client contract |
| [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md) | Runtime configuration |
| [`docs/public-adoption/LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) | Public proof honesty |
