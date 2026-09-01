# Knowledge Source Integrations

**Intergrax Knowledge Source Integrations** defines how external vendor data enters the platform for knowledge use - through existing category-correct integrations, shared provider read primitives, and three separate consumption modes with distinct lifecycle, policy, and persistence semantics.

## Why it matters

One vendor integration should not be duplicated separately for RAG indexing, durable materialization, and live access. Without a shared provider foundation, each product path reimplements clients, credentials, pagination, and error handling - and lifecycle semantics blur between durable sync, indexed retrieval, and ephemeral reads.

Intergrax reuses **one provider/category integration** and **one set of typed provider read primitives** while keeping **indexed RAG**, **durable materialization**, and **live access** as separate modes. Hybrid Ask combines indexed and live evidence at the application level; it is not a fourth provider integration.

> [!NOTE]
> **Maturity boundary:** The canonical integration and Vendor Knowledge boundary in this hub is **defined and binding**. That is **not** universal provider coverage, complete live external validation, or production qualification for every vendor surface. Support remains **provider-specific** and must be declared explicitly per provider and source kind. See [Current reality / maturity boundary](#current-reality--maturity-boundary) and vendor sections below.

**Primary audience:** CTOs, software architects, principal engineers, and AI platform engineers evaluating how Intergrax connects external knowledge without duplicating integrations - after the platform overview in the root README.

**Related canon:** [`INTEGRATIONS.md`](INTEGRATIONS.md) · LKW intake discovery: [`KNOWLEDGE_INTAKE_DISCOVERY.md`](../../../applications/local_workspace_application/docs/KNOWLEDGE_INTAKE_DISCOVERY.md)

## Current reality / maturity boundary

- **Canonical architecture boundary is defined** - existing integration categories, Vendor Knowledge Facade, sync/materialization runtime, and live capability paths are specified in this document.
- **Existing provider/category integrations remain authoritative** - Jira, Confluence, Microsoft Graph, Slack, Google Workspace, Databricks, and peers keep one public integration each; knowledge use reuses them rather than introducing parallel `knowledge_source` integrations.
- **Three consumption modes are frozen direction** - indexed RAG, durable materialization, and live access retain separate lifecycles; hybrid access is application-level composition only.
- **Support is provider-specific** - capability matrices, read primitives, adapters, and proofs vary by vendor; a gap in one mode does not imply support in another.
- **Complete provider coverage and complete live external validation are not implied** - vendor sections document what is implemented, planned, or absent; manifest or shell presence is not operational proof.
- **Architecture definition ≠ universal implementation / production qualification** - binding decisions govern design; rollout, LKW Connected Sources, and external validation follow separate plan and proof tracks.

## At a glance

| Concern | Current rule |
| -------- | -------- |
| **Provider integration** | One category-correct public integration per provider/category - reuse client, transport, and credential resolution |
| **Indexed RAG** | Durable sync/materialization → shared parsing/chunking/embedding pipeline → vector store → scoped retrieval |
| **Durable materialization** | Durable sync/materialization → approved DocumentStore / DB / object storage - embeddings optional |
| **Live access** | Authorized typed capability → provider read at request time → ephemeral evidence; no automatic durable persistence |
| **Credentials** | Referenced through Connection/credential handles - never embedded in bindings or config records |
| **Synchronization** | Shared sync/materialization runtime - checkpoints, leases, replay, reconciliation - not per-vendor reimplementation |
| **ACL** | Enforced before content reaches the model - prompt instructions are not authorization |
| **Application boundary** | Applications use Vendor Knowledge Facade (durable) and validated live capability paths - not direct vendor SDK/API calls |
| **Vendor Knowledge Facade** | Platform service above integrations - not an integration category |
| **MCP** | Complementary live/action surface - does not replace durable synchronization |
| **Go deeper** | [Binding decisions](#2-binding-architectural-decisions) · [Layered architecture](#3-layered-architecture) · [Vendor direction](#13-vendor-direction) · [plan](../maintainers/plans/KNOWLEDGE_SOURCE_INTEGRATIONS.md) |

## Core mental model

**Frozen architectural principle.** One existing category-correct **Vendor Integration** remains the single owner of provider communication. Shared **provider read primitives** are designed independently of how a caller will later persist, index, or ephemerally use the result.

```text
MODE 1 - INDEXED RAG
provider data
→ durable synchronization/materialization
→ parser/chunker/embeddings
→ vector store
→ RAG retrieval

MODE 2 - DURABLE MATERIALIZATION
provider data
→ durable synchronization/materialization
→ DocumentStore / relational DB / NoSQL / object storage /
  application database / analytics store
→ no requirement to create embeddings or a RAG index

MODE 3 - LIVE ACCESS
user question or application request
→ authorized typed capability
→ provider API at request time
→ ephemeral normalized evidence/result
→ no automatic durable persistence
```

Binding rules:

1. One provider/category integration remains authoritative.
2. One client, transport and credential resolution path is reused.
3. Provider read primitives are designed independently of persistence mode.
4. RAG, durable materialization and live access may use the same provider primitives.
5. The three modes do not share one lifecycle.
6. Live results are ephemeral unless an explicit promotion or materialization workflow is invoked.
7. Durable materialization does not imply embeddings or vector indexing.
8. RAG indexing must use the shared downstream ingestion pipeline.
9. No application may create a second vendor client merely because it needs live access.
10. No sync adapter may become a generic free-form query interface.
11. No live capability may bypass source/resource authorization.
12. Capability support must be declared explicitly per provider and source kind.

### Canonical architecture diagram

```text
Connection / credential reference
                |
                v
existing category-correct vendor integration
                |
                v
shared typed provider read primitives
        /               |                \
       /                |                 \
      v                 v                  v
inventory/change     exact/search       exact/query
reads                reads              reads
      |                 |                  |
      +-----------------+------------------+
                        |
          provider-safe normalized boundary
              /                         \
             /                           \
            v                             v
durable knowledge path               live capability path
            |                             |
            v                             v
Vendor Knowledge Adapter          Live Capability Adapter
            |                     / Validated Executor
            v                             |
Sync / Materialization Runtime            v
            |                      ephemeral Live Evidence
            +------------------+
            |                  |
            v                  v
DocumentStore / DB        LKW Knowledge Intake
/object storage                 |
                                v
                       Documents / Chunks /
                       Embeddings / Vector Store
```

RAG is **one consumer** of durable materialization - not the definition of all durable vendor data.

### Binding terminology

| Term | Meaning |
|---|---|
| **Vendor Integration** | One existing category-correct public integration (`JiraIssueTrackerIntegration`, `ConfluenceWikiKnowledgeIntegration`, `Ms365GraphCollaborationSuiteIntegration`, …). Single owner of provider communication. |
| **Provider read primitive** | Typed, bounded operation exposed by the integration or an approved provider-specific read facet sharing the same client and credentials. Examples: list inventory page, read change/delta page, search bounded records, read exact item, read content, read attachments, read permissions. Must not know whether its result will be indexed, saved or used ephemerally. |
| **Indexed RAG** | Durable provider knowledge processed into documents, chunks, embeddings and a retrieval index. |
| **Durable materialization** | Durable normalized provider data in an approved platform or application store without requiring RAG. Broader than RAG. |
| **Live access** | Bounded read-only provider operation at request time. Ephemeral by default; must not automatically create Source, Document, Chunk, Embedding, vector record or durable provider replica. |
| **Hybrid access** | Application-level combination of indexed evidence and live evidence. Does not create a second vendor integration. |

### Three-mode reuse matrix

| Concern | Indexed RAG | Durable materialization | Live access |
|---|---:|---:|---:|
| Existing vendor integration | reused | reused | reused |
| Provider client and transport | reused | reused | reused |
| Credential resolution | reused | reused | reused |
| Typed remote references | reused | reused | reused |
| Provider response validation | reused | reused | reused |
| Exact item reads | reused where applicable | reused where applicable | reused where applicable |
| Search/query primitives | optional | optional | usually required |
| Cursor/checkpoint | required for sync | required for sync | not a durable checkpoint |
| Sync lease and replay | required | required | not used |
| Durable sink | required | required | forbidden by default |
| Parsing/chunking/embedding | required for indexable content | optional | not performed |
| Ephemeral evidence | not the primary result | not the primary result | required |
| Per-question call limits | not applicable to sync in the same form | not applicable in the same form | required |
| Retention policy | durable | durable or TTL | ephemeral by default |

### Provider-delivery rule

Every future provider task must answer:

```text
Which provider primitives are shared?
Which modes can reuse them now?
Which modes remain unsupported?
Is a missing feature a provider primitive gap,
a durable-adapter gap,
or a live-capability gap?
```

A provider task must never claim all three modes merely because one exact-read method exists. Support must be explicit and evidenced.

---

## 2. Binding architectural decisions

| Decision | Classification | Binding statement |
|---|---|---|
| Existing integration categories remain authoritative | `FROZEN` | Jira remains an issue-tracker integration, Confluence remains a wiki-knowledge integration, Microsoft Graph remains a collaboration-suite integration, and Databricks remains a relational-store integration unless a separately justified domain category is introduced. |
| No generic `knowledge_source` integration category | `REJECTED` | Knowledge ingestion is a cross-category application use case, not the primary domain identity of every vendor integration. |
| No duplicate public integration for knowledge use | `REJECTED` | Do not create `JiraKnowledgeSourceIntegration`, `ConfluenceKnowledgeSourceIntegration`, `SlackKnowledgeIntegration`, `SlackRagIntegration`, `SlackDatabaseIntegration`, `SlackLiveIntegration`, `GoogleDriveIntegration`, `GoogleDocsIntegration`, `GoogleSheetsIntegration`, `GoogleCalendarIntegration`, `GoogleSlidesIntegration`, `GmailKnowledgeIntegration`, `GoogleChatKnowledgeIntegration`, `GoogleWorkspaceKnowledgeIntegration`, or equivalent parallel public integrations beside existing provider/category integrations. |
| Slack remains one `conversation_channel` integration | `FROZEN` | `SlackConversationChannelIntegration` is the only public Slack integration for conversational runtime, shared typed Slack knowledge reads, durable materialization, indexed RAG and bounded live access. Reuse the existing client, transport and credential resolution. Do not create an LKW-owned Slack vendor client. |
| Slack dual role is independent | `FROZEN` | Slack-as-frontend (LKW companion transport) and Slack-as-knowledge-source (Connection → Remote Resource → bindings) are separate roles. Enabling the Slack chatbot does not authorize indexing or live Slack history access. Conversation transport events do not automatically become durable knowledge. |
| Google Workspace remains one `collaboration_suite` integration | `FROZEN` | `GoogleWorkspaceCollaborationSuiteIntegration` is the only public Google Workspace integration for collaboration operations, shared typed Google knowledge reads, durable materialization, indexed RAG and bounded live access. Reuse one credential-resolution boundary and one provider client/transport family. Do not create parallel public integrations per Drive, Docs, Sheets, Calendar, Slides, Mail or Chat surface. |
| Google provider integration ≠ Vendor Knowledge Adapter ≠ Live Capability ≠ LKW Connected Source | `FROZEN` | Google Workspace knowledge use follows the same separation as other vendors: provider integration owns transport; thin Vendor Knowledge adapters map canonical contracts; Live Capability adapters own ephemeral reads; LKW Connected Source owns workspace binding and indexing - without duplicating Google clients or credentials. |
| Vendor integration remains low-level | `FROZEN` | It owns provider transport, auth handoff, vendor request/response mapping, provider errors and category operations. It does not know LKW, workspaces, RAG or product workflows. |
| Unified knowledge behavior is exposed by platform boundaries | `FROZEN DIRECTION` | Durable knowledge behavior uses Vendor Knowledge Facade; live knowledge behavior uses the validated live capability boundary. Both resolve the same existing integration through separate adapter paths. |
| Facade is not an integration category | `FROZEN` | It is a platform service/facade and may use a registry of source adapters. It is not registered as another vendor integration. |
| Existing integrations may expose additional provider methods | `FROZEN DIRECTION` | Delta reads, pagination, attachments, permissions or inventory methods may be added to the correct existing integration or to a private/provider-specific read facet behind it. |
| Application does not call vendor methods directly | `FROZEN` | LKW and other knowledge-consuming applications do not call vendor APIs, vendor SDKs or provider-specific integration methods directly. Durable knowledge operations use: Vendor Knowledge Facade → Vendor Knowledge Adapter → existing vendor integration. Live knowledge operations use: Validated Capability Executor → Live Capability Adapter → existing vendor integration. Both paths must resolve the same Connection and existing provider/category integration. |
| One shared synchronization runtime | `FROZEN DIRECTION` | Checkpoints, leases, retry, reconciliation, durable item state and replay semantics are common platform/application mechanisms, not independently reimplemented by each vendor. |
| One shared ingestion pipeline | `FROZEN` | Parsing, structured normalization, chunking, embeddings, Document Store and Vector Store remain shared downstream capabilities. |
| Stable remote identity is separate from revision | `FROZEN` | Vendor item identity must not be derived from a content hash. Version, ETag, content hash and ACL hash represent change state. |
| ACL must be enforceable before model access | `FROZEN` | Prompt instructions are not authorization. Retrieval must not expose knowledge that the requesting principal is not allowed to access. |
| Secrets are referenced, never embedded | `FROZEN` | Source bindings and facade requests may carry only opaque connection or credential references. |
| MCP is complementary | `FROZEN DIRECTION` | MCP may expose live tools or approved actions, but it does not replace durable synchronization and application-owned knowledge state. |

---

## 3. Layered architecture

The platform layer model distinguishes:

```text
provider integration layer
shared provider read primitives
durable knowledge adapter/facade
sync and materialization runtime
live capability adapter/executor
application-owned consumption
```

Live query execution must **not** be forced through `Sync Coordinator`. Database materialization must **not** be forced through LKW. LKW is **not** a dependency of the platform layer.

### Two sibling application-facing paths

**DURABLE PATH**

```text
Vendor Knowledge Adapter
→ Vendor Knowledge Facade
→ Sync / Materialization Runtime
→ injected durable sink
→ optional RAG ingestion
```

**LIVE PATH**

```text
Live Capability Adapter
→ Capability Registry
→ Validated Capability Executor
→ normalized ephemeral evidence/result
```

The exact future Python contracts remain deferred, but the ownership boundary is frozen. `VendorKnowledgeFacade` is **not** an already implemented generic live-query service; it currently covers the durable synchronization/materialization path.

### 3.1 Layer 1 - platform integration base

The existing platform integration base remains unchanged in purpose.

It provides common integration identity and lifecycle concepts such as:

```text
provider_id
integration_kind
integration_id
config
enabled state
capabilities
health
security posture
```

This layer answers:

> What integration is this, which category does it belong to, how is it configured, and what platform-level capabilities does it declare?

It does not define an application knowledge model.

### 3.2 Layer 2 - category contracts

Category contracts describe the domain role of an integration.

Examples:

```text
CollaborationSuiteIntegrationContract
IssueTrackerIntegrationContract
WikiKnowledgeIntegrationContract
RelationalStoreIntegrationContract
ObjectStorageIntegrationContract
```

The category must remain semantically meaningful.

Examples:

- Jira is primarily an issue tracker;
- Confluence is primarily a wiki/knowledge system;
- Microsoft Graph is a collaboration-suite integration;
- Databricks is a relational/data platform integration;
- Power BI may require a future business-intelligence category;
- Atlan may require a future data-catalog or data-governance category.

A new domain category may be introduced only when the vendor capability does not fit an existing category. It must not be introduced merely because data will later be indexed by LKW.

### 3.3 Layer 3 - concrete vendor integration

Each provider/category pair has one public integration entrypoint.

Examples:

```text
JiraIssueTrackerIntegration
ConfluenceWikiKnowledgeIntegration
Ms365GraphCollaborationSuiteIntegration
```

A concrete vendor integration owns:

- provider API client or injected transport;
- provider-specific authentication handoff;
- provider request construction;
- provider response mapping;
- provider-native query behavior;
- provider rate-limit, timeout, authorization and availability errors;
- provider/category operations;
- optional provider-specific read methods needed for inventory or change feeds;
- safe provider health information.

A concrete vendor integration must not own:

- LKW `Workspace` or `WorkspaceSource` records;
- LKW `KnowledgeInput` or ingestion operations;
- product document identity;
- durable application checkpoints;
- synchronization scheduling;
- parsing, chunking or embeddings;
- vector-store writes;
- prompts, agents or artifact generation;
- Slack commands or frontend delivery;
- cross-provider normalization policy.

### 3.4 Layer 4 - shared provider read primitives

Typed, bounded operations exposed by the vendor integration or an approved provider-specific read facet sharing the same client and credentials.

Examples:

```text
list inventory page
read change/delta page
search bounded records
read exact item
read exact item version
read content
read attachments
read permissions
read bounded time window
read bounded query result
```

A provider read primitive must not know whether its result will later be indexed into RAG, saved into a database or used as ephemeral live evidence.

### 3.5 Layer 5 - vendor knowledge adapters

The facade may use small adapters that translate category/provider operations into the common knowledge model.

Conceptual examples:

```text
JiraKnowledgeAdapter
ConfluenceKnowledgeAdapter
Ms365DriveKnowledgeAdapter
Ms365MailKnowledgeAdapter
DatabricksCatalogKnowledgeAdapter
PowerBiMetadataKnowledgeAdapter
AtlanCatalogKnowledgeAdapter
```

These are not public integration entries and are not new integration categories.

An adapter:

- receives an already resolved existing integration;
- calls its public category methods or an approved provider-specific read facet;
- maps provider records into facade models;
- maps provider pagination or delta tokens into opaque continuation data;
- maps provider deletions and access loss into normalized change events;
- maps provider permissions into normalized ACL information;
- declares source-kind-specific capabilities.

An adapter must not:

- instantiate a second vendor client when the integration already owns one;
- duplicate authentication configuration;
- register as a parallel provider integration;
- write to LKW repositories;
- execute parsing, chunking or embeddings;
- own committed checkpoints or synchronization schedules.

### 3.6 Layer 6 - Vendor Knowledge Facade

The Vendor Knowledge Facade is the application-facing, vendor-neutral service boundary.

It answers:

> How can an application discover, bind, inspect and read knowledge from any supported integration without knowing whether the source is Jira, Confluence, Microsoft Graph, Databricks, Power BI or Atlan?

The facade owns:

- resolution of the correct existing integration;
- resolution of the correct source-kind adapter;
- validation of tenant, connection and source binding consistency;
- safe source discovery and candidate projection;
- unified capability reporting;
- unified source inspection;
- normalized item, content, provenance and ACL models;
- unified error classification;
- hiding provider-specific client and transport details from applications;
- delegation to the shared synchronization runtime;
- returning application-safe operation and status results.

The facade does not own:

- vendor HTTP or SDK implementation details;
- LKW-specific persistence models;
- document parsing or embeddings;
- Slack command syntax;
- generated emails, analyses, offers or reports;
- business decisions about which sources a product should connect.

### 3.7 Layer 7 - shared synchronization and materialization runtime

Durable synchronization semantics are shared above vendor integrations and below consuming applications.

The synchronization runtime owns:

- source leases and concurrency control;
- full and incremental synchronization orchestration;
- durable committed checkpoints;
- page-level replay safety;
- retries and backoff;
- reconciliation after webhook loss or expired cursors;
- durable remote item state;
- stable application document identity derivation;
- content, metadata and ACL revision comparison;
- rename and move handling;
- deletion and access-revocation handling;
- progress, counts and error state;
- downstream submission to the shared ingestion pipeline.

The integration or adapter returns a proposed continuation/checkpoint. The synchronization runtime commits it only after the corresponding page has been processed durably.

Required semantics:

```text
at-least-once delivery
+
idempotent processing
+
checkpoint commit after durable page completion
```

For indexed connected sources, page publication has one visibility authority:

```text
prepare receipt
→ stage hidden documents
→ write immutable delivery manifest
→ CAS the lifecycle/permit fence with the committed manifest descriptor
→ complete the receipt and rebuild derived projections
```

The fence CAS is the publication linearization point. Immutable manifests and
per-delivery commit records preserve bounded publication history, so retrieval
resolves the highest committed sequence for each remote item. Active pointers
are only accelerators; a failed pointer write cannot hide a committed document.
Prepared manifests and completed receipts without a fence descriptor remain
invisible, and lifecycle disable/detach conflicts with an unexpired permit
through the same authority record.

### 3.7.1 Provider-neutral Indexed Source eligibility

Discovery, live capability support and durable materialization are separate
decisions. The Vendor Knowledge
`IndexedSourceEligibilityResolverV1` is the authoritative read-only boundary
for the durable decision: it revalidates the tenant-owned active Connection,
the exact Remote Resource and discovery snapshot, then requires a complete
provider-neutral materialization registration and an available synchronization
handler for the exact `(provider_id, integration_kind, source_kind)` key.

The resolver returns a bounded immutable eligibility proof and an opaque
canonical binding plan. The proof is not an authorization grant, does not
persist a binding or start synchronization, and must be revalidated by the
consuming application during attach. Applications must not infer Indexed
Source support from provider names, source kinds, discovery-provider presence,
live capability IDs or adapter class names. The next consumer is
`LKW-INDEXED-SOURCE-LIFECYCLE-1`.

### 3.8 Layer 8 - live capability adapter and executor

The live capability path is a **sibling** of the durable path, not a branch of the sync runtime.

The live capability layer owns:

- typed live capability contracts per provider and source kind;
- capability registry and explicit support declaration;
- validated read-only executor with per-question timeout, call count, result count and byte limits;
- normalized ephemeral Live Evidence and execution receipts;
- authorization enforcement before content reaches the model.

The live capability layer does **not** own:

- sync cursors, checkpoints, leases or replay;
- durable sink writes by default;
- parsing, chunking or embeddings;
- a second vendor client when the integration already owns one.

### 3.9 Layer 9 - consuming application

LKW is the first platform proof consuming both:

- the durable Vendor Knowledge Facade boundary;
- the governed live capability boundary.

The live capability runtime is planned, not implemented.

LKW owns:

- binding a workspace source to a facade source binding;
- deciding which connected sources belong to a workspace;
- initiating source synchronization;
- presenting source and operation state through Slack/HTTP/MCP surfaces;
- receiving normalized knowledge into its existing Knowledge Intake and ingestion lifecycle;
- Document ownership within the workspace;
- retrieval and artifact-generation product behavior.

LKW must not:

- import Jira, Confluence, Microsoft Graph, Power BI, Atlan or Databricks SDKs;
- call provider APIs directly;
- implement provider-specific pagination;
- store provider tokens in workspace records;
- implement a separate RAG pipeline for each source;
- contain branches such as `if provider == "jira"` in its core ingestion services.

---

## 4. Target dependency direction

```text
LKW / knowledge-consuming application
        |
        +----------------------------+
        |                            |
        v                            v
Vendor Knowledge Facade      Validated Capability Executor
(durable path)               (live path)
        |                            |
        v                            v
Vendor Knowledge Adapter     Live Capability Adapter
        |                            |
        +-------------+--------------+
                      v
      existing category-correct vendor integration
                      |
                      v
          provider API / SDK / transport
```

The reverse dependency is forbidden:

```text
vendor integration -> facade -> LKW
```

The existing integration remains reusable by unrelated platform tools, agents or applications that do not consume knowledge through the LKW/Vendor Knowledge product boundary. That reuse must not imply that LKW may bypass its durable or live authorization boundary.

---

## 5. Existing integration methods and additional read needs

Existing operational protocols already expose normalized domain actions such as:

- mail and calendar operations through `CollaborationSuite`;
- issue lookup/search/mutation through `IssueTracker`;
- page lookup/search through `WikiKnowledge`.

These methods remain valid and must not be replaced by the facade.

The facade may require additional lower-level read abilities not currently present, for example:

```text
list all pages in a bounded scope
read the next provider page
read changes after a provider cursor
read attachments
read an item's current version
read an item's permission set
read deletion or revocation markers
read source inventory
```

Such abilities should be added in one of two forms, selected per provider/category:

### Pattern A - extension of the existing provider integration

Use when the method is a natural part of the existing category contract and is broadly meaningful for providers in that category.

Example direction:

```text
WikiKnowledge
├── get_page
├── search_pages
└── list_pages / read_page_version
```

### Pattern B - provider-specific private read facet

Use when the behavior is provider-specific or not appropriate for every implementation in the category.

Example direction:

```text
Ms365GraphCollaborationSuiteIntegration
└── private/approved DriveDeltaReader

JiraIssueTrackerIntegration
└── private/approved JiraIssueChangeReader
```

The adapter may use this facet through an explicit typed port. The facet reuses the integration's client, connection and transport; it is not a second integration.

The selection between Pattern A and Pattern B must be made in a scoped vendor task. Do not pollute every base category contract with methods that most providers cannot implement.

---

## 6. Facade contract semantics

The exact Python names are deferred, but the facade must support the following conceptual operations.

### 6.1 Source discovery

```text
list_source_candidates(tenant, connection, source_kind, filter)
inspect_source_candidate(candidate_id)
```

A candidate contains only safe information:

```text
candidate_id
provider_id
integration_kind
source_kind
safe label
description
capabilities
availability
```

It must not expose:

- tokens or secrets;
- raw authorization headers;
- signed temporary URLs;
- unsafe local paths;
- unrestricted provider query payloads.

### 6.2 Source binding validation

```text
validate_source_binding(connection_ref, source_selection)
```

Validation covers:

- tenant consistency;
- integration/provider identity;
- required grants;
- source existence or allowed temporary unavailability;
- bounded filters;
- source-kind support;
- ACL support required by the selected policy;
- absence of secrets in durable configuration.

### 6.3 Unified read and synchronization

Conceptual facade methods:

```text
capabilities(binding)
inspect(binding)
start_sync(binding, previous_checkpoint)
read_status(operation_id)
cancel_sync(operation_id)      # deferred until the platform has cancellation semantics
```

The facade may internally expose lower-level methods to the sync runtime:

```text
read_change_page(binding, provider_cursor, limit)
read_item_content(binding, item_ref)
read_item_acl(binding, item_ref)
```

These lower-level methods are not intended as LKW-specific APIs.

---

## 7. Canonical facade models

The following models belong to the facade/synchronization boundary, not to the vendor integration category taxonomy.

### 7.1 Knowledge connection reference

Represents a **durable tenant Connection record / reference** - not an in-memory registry entry or application bootstrap profile.

Minimum semantics (conceptual `TenantConnection` - **to be implemented in `LKW-KNOWLEDGE-ACCESS-1C-1`**):

```text
connection_ref          # durable identity component; opaque within tenant
tenant_id
provider_id
integration_kind
credential_ref          # opaque SecretsStore reference only
connected_principal_ref # optional
safe_display_name
administrative_status   # ACTIVE | DISABLED | REVOKED
validated_secret_free_config
configuration_version
```

The durable Connection catalog is **platform-owned**. Raw secrets remain in `SecretsStore`. `connection_ref` remains the correlation identity across bindings, workspace attachments and runtime resolution. The instance-local `KnowledgeConnectionRegistry` is reconstructed from durable state at startup - it is runtime projection only, not the administrative source of truth.

The connection record contains references and safe metadata, never secret values. Knowledge metadata/URL secret-safe detection uses the canonical engine with a Knowledge-owned policy: `credential_ref` is an allowed opaque reference; raw secret keys and credential-bearing URLs are rejected. This is not a secret manager. One provider integration is still reused across indexed RAG, durable materialization and live access (three consumption modes).

### 7.2 Source binding

Represents a selected remote source exposed through the facade.

Minimum semantics:

```text
binding_id
tenant_id
provider_id
integration_kind
connection_id
source_kind
remote_scope_ref
safe_display_name
validated configuration
ACL mode
sync mode
configuration version
```

`remote_scope_ref` may be provider-specific internally but must be validated and secret-free.

### 7.3 Knowledge item descriptor

Minimum semantics:

```text
remote_id
parent_remote_id
item_type
title
deep_link
mime_type
provider_version
etag
created_at
updated_at
author
content_mode
safe metadata
```

Binding identity plus `remote_id` defines stable origin. Content hash does not define identity.

### 7.4 Knowledge content

Supported normalized modes:

```text
BINARY
RICH_TEXT
STRUCTURED_RECORD
```

Examples:

- DOCX from SharePoint: `BINARY`;
- Confluence page: `RICH_TEXT`;
- Jira issue: `STRUCTURED_RECORD`;
- Power BI model metadata: `STRUCTURED_RECORD`;
- Databricks notebook: `RICH_TEXT` or `BINARY`, depending on the selected adapter contract.

The facade does not parse binaries into chunks. It returns a controlled stream/reference or stages content through an approved platform boundary for the downstream parser.

### 7.5 Change page

Minimum semantics:

```text
items
tombstones
provider_continuation
proposed_checkpoint
has_more
rate_limit_hint
warnings
```

The provider continuation may be opaque. The facade and adapter must not require LKW to understand vendor tokens.

### 7.6 ACL

Minimum semantics:

```text
visibility mode
allowed principals
denied principals
inheritance state
ACL version/hash
completeness
```

Normalized principals should distinguish provider identity and canonical tenant identity where mapping is available.

The facade must explicitly report incomplete or unsupported ACL information. It must not silently describe an unrestricted source as permission-preserving.

### 7.7 Tombstone

A tombstone identifies a previously visible stable remote item that is now:

- deleted;
- removed from the selected scope;
- inaccessible because authorization changed;
- hidden because source permissions changed.

Deletion and access loss may require different downstream policy, so the normalized reason must be retained when the provider exposes it.

---

## 8. Identity, revisions and update semantics

### 8.1 Stable identity

Application identity must be derived from stable origin, conceptually:

```text
tenant_id
+ binding_id
+ remote_id
```

The exact encoding is deferred.

Required behavior:

- rename with unchanged `remote_id` updates metadata, not identity;
- move with unchanged `remote_id` updates parent/scope metadata, not identity;
- content update changes revision state, not identity;
- ACL-only update changes ACL state without forcing re-embedding of unchanged content;
- deletion removes or deactivates source-owned knowledge according to product policy;
- reuse of the same provider identifier for a genuinely new entity must be handled through provider version/lifecycle evidence where available.

### 8.2 Revision dimensions

The sync runtime should track revision dimensions separately:

```text
provider version / ETag
content hash
metadata hash
ACL hash
```

This enables:

- content unchanged, metadata changed;
- content unchanged, ACL changed;
- content changed, ACL unchanged;
- rename/move without duplicate indexing;
- safe no-op handling.

---

## 9. Checkpoints, replay and reconciliation

The adapter or provider integration reads from a provider cursor and returns a proposed continuation/checkpoint.

The shared sync runtime performs:

```text
acquire source lease
read committed checkpoint
read one normalized change page
process all items and tombstones durably
persist item/revision state
commit proposed checkpoint
release or renew lease
```

A crash before checkpoint commit replays the page.

Therefore all downstream processing must be idempotent.

Checkpoint rules:

- checkpoint belongs to the source binding and tenant;
- checkpoint payload is opaque outside the adapter/facade boundary;
- checkpoint never contains credentials;
- checkpoint is not committed before durable page completion;
- expired/invalid checkpoint triggers an explicit reconciliation path;
- webhook notification is a hint, not the sole durable source of truth;
- periodic reconciliation is required when the provider cannot guarantee complete change delivery.

---

## 10. Security and tenancy

### 10.1 Tenant isolation

Every facade operation validates:

```text
request tenant
connection tenant
binding tenant
operation tenant
checkpoint tenant
```

Cross-tenant resolution fails closed.

### 10.2 Credentials

Allowed durable value:

```text
credential_ref
```

The durable tenant Connection catalog (`TenantConnection`) and `KnowledgeSourceBinding` public projections may carry only opaque `credential_ref`. The instance-local `KnowledgeConnectionRegistry` does not store credentials.

Forbidden in bindings, Connection records, checkpoints, events, logs, errors and LKW workspace records:

- access token;
- refresh token;
- API key;
- client secret;
- password;
- full authorization header;
- signed temporary content URL;
- unredacted credential payload.

### 10.3 ACL enforcement

Two initial policy modes may exist:

```text
PERSONAL_DELEGATED
ORGANIZATION_PRESERVED_ACL
```

`PERSONAL_DELEGATED` means knowledge is visible only within the connected user's authorized product boundary.

`ORGANIZATION_PRESERVED_ACL` requires source ACL acquisition, principal normalization, ACL refresh and retrieval-time filtering.

A broad app-only connection must not imply that every workspace user may see every indexed item.

The model must never receive content that authorization filtering has rejected.

---

## 11. Direct API/SDK and MCP

### Direct integration path

Existing vendor integrations and their transports remain the foundation for:

- source inventory;
- pagination;
- delta/change feeds;
- content retrieval;
- attachment retrieval;
- ACL reads;
- tombstones;
- reconciliation;
- deterministic bulk synchronization.

### MCP path

MCP may be used for:

- interactive live queries;
- tool/resource discovery;
- approved vendor actions;
- current context that should not be durably copied;
- user-driven actions with policy and HITL.

MCP is not the only persistence or synchronization mechanism.

Both paths may be exposed behind higher-level application capabilities, but their lifecycle and trust semantics remain distinct.

---

## 12. Relationship with LKW Knowledge Intake

The eventual integration point is not:

```text
LKW -> Jira SDK
LKW -> Confluence SDK
LKW -> Graph SDK
```

It is:

```text
LKW WorkspaceSource(CONNECTED_SOURCE)
        |
        v
Vendor Knowledge Facade source binding
        |
        v
shared synchronization runtime
        |
        v
normalized item/content/ACL
        |
        v
existing LKW Knowledge Intake / ingestion processor
        |
        v
shared parser or structured normalizer
        |
        v
Document -> Chunks -> Embeddings -> stores
```

The facade may produce one or many item-level ingestion submissions for a connected source. LKW remains owner of workspace Documents and product operation state; the facade remains owner of vendor-neutral access and synchronization orchestration.

The exact bridge to current `KnowledgeInput`, `WorkspaceSource`, `KnowledgeIngestionProcessor` and queue contracts is deferred until both work streams are ready for synchronization.

---

## 13. Vendor direction

### 13.1 Microsoft Graph

Keep `Ms365GraphCollaborationSuiteIntegration` as the category-correct public integration.

Add source-kind adapters above it, for example:

```text
SharePoint/OneDrive drive adapter
mail-folder adapter
Teams-channel adapter
calendar adapter
OneNote adapter
Planner adapter
SharePoint-list adapter
```

Where existing collaboration methods are insufficient, add approved Graph-specific read facets that reuse the same integration client/auth boundary.

Do not create `Ms365GraphKnowledgeSourceIntegration`.

### 13.2 Jira

Keep `JiraIssueTrackerIntegration`.

A Jira knowledge adapter may use:

- issue search/inventory;
- issue detail;
- comments;
- attachments;
- changelog/version data;
- project visibility;
- provider pagination/change query behavior.

Add missing read methods to the Jira integration or a typed Jira change-reader facet. Do not create a second Jira integration.

### 13.3 Confluence

Keep `ConfluenceWikiKnowledgeIntegration`.

A Confluence adapter maps pages, hierarchy, labels, versions, structured body, attachments and permissions into facade models.

Add missing page-list/change methods to the existing integration or a typed Confluence read facet. Do not create a second Confluence integration.

### 13.4 Databricks

Keep existing Databricks integration category identity for relational/data operations.

Knowledge adapters may expose selected:

- Unity Catalog metadata;
- lineage metadata;
- workspace files/notebooks;
- volume files;
- approved SQL snapshots;
- approved change-feed sources.

Do not treat the whole lakehouse as an unbounded document source.

### 13.5 Power BI

First audit whether Power BI fits an existing category. If not, introduce a justified domain category such as `business_intelligence`, with one public Power BI integration.

Above that integration, adapters may expose metadata, semantic model descriptions, approved query snapshots and controlled report exports.

Do not create a generic knowledge-source category solely for Power BI.

### 13.6 Atlan

First audit whether Atlan fits an existing category. If not, introduce a justified domain category such as `data_catalog` or `data_governance`, with one public Atlan integration.

Above that integration, an adapter may expose assets, glossary, ownership, certification, lineage, quality and governance metadata.

### 13.7 Slack - binding three-mode example (`SLACK-KNOWLEDGE-THREE-MODE-ARCH-1`)

**Classification:** `ARCHITECTURALLY FROZEN` - not implemented.

Slack remains one category-correct public `conversation_channel` integration. The existing `SlackConversationChannelIntegration`, its client, transport and credential resolution are reused for conversational runtime, shared typed Slack knowledge reads, durable materialization, indexed RAG and bounded live access. No application-specific or consumption-mode-specific Slack client or public integration may be introduced.

**Rejected duplicate names (do not create):**

```text
SlackKnowledgeIntegration
SlackRagIntegration
SlackDatabaseIntegration
SlackLiveIntegration
LkwSlackClient
```

**Canonical foundation:**

```text
Connection / credential reference
        |
        v
SlackConversationChannelIntegration
        |
        v
shared typed Slack provider read primitives
        |
        +---------------------------+
        |                           |
        v                           v
durable knowledge path          live capability path
        |                           |
        v                           v
Slack Vendor Knowledge          Slack Live Capability
Adapter                         Adapter / Executor
        |                           |
        v                           v
Vendor Knowledge Facade         ephemeral Live Evidence
        |
        v
Sync / Materialization Runtime
        |
        +------------------------------+
        |                              |
        v                              v
application database/store       LKW Knowledge Intake
                                       |
                                       v
                              Documents / Chunks /
                              Embeddings / RAG
```

**Binding rules:**

1. `SlackConversationChannelIntegration` remains the only public Slack integration for this provider/category.
2. The existing integration remains the owner of Slack SDK/client construction, tokens, transport, provider calls, provider errors and provider-specific validation.
3. Shared Slack read primitives must not know whether their output will be indexed, stored in a database or used as live evidence.
4. Provider-specific Slack read primitives belong to the same concrete integration and remain independent of persistence mode.
5. Vendor Knowledge owns durable provider-neutral projection and synchronization.
6. Live Capability owns bounded request-time access and ephemeral evidence.
7. LKW owns workspace binding, Indexed Sources, Live Access Bindings, Knowledge Intake, RAG, Ask and frontend behavior.
8. `vendor_knowledge` and the Slack provider integration must not import or depend on LKW.
9. Enabling the Slack chatbot does not automatically authorize indexing or querying Slack history.
10. Conversation transport events must not automatically become durable knowledge.
11. Durable Slack synchronization may feed any injected sink, not only LKW or RAG.
12. Live Slack results remain ephemeral unless an explicit promotion/materialization workflow is executed.
13. Indexed permission and live-access authorization are separate grants.
14. Slack-as-frontend and Slack-as-knowledge-source are independent roles even when they resolve the same provider integration foundation.
15. **Conversation Context Binding** (LKW application domain) controls where and under which audience the assistant may respond. Provider adapters supply `ConversationIngressContext` with `observed_audience`; `binding.audience_mode` must match before workspace resolution or Ask. At most one `ACTIVE` binding per semantic identity. Independent from Indexed Source Binding and Live Access Binding. Canonical contract: [`CONVERSATION_CONTEXT_ARCHITECTURE.md`](../../../applications/local_workspace_application/docs/CONVERSATION_CONTEXT_ARCHITECTURE.md).

**Independent grants (provider-neutral):**

```text
Conversation Context Binding  → where and for whom LKW responds
Indexed Source Binding        → durable knowledge ingestion
Live Access Binding           → request-time provider reads
```

None implies another. Enabling a bot in a channel does not index channel history. Indexing channel history does not enable bot responses. Bot responses do not imply live history reads.

**Implemented provider-specific read primitives (`SLACK-KNOWLEDGE-FOUNDATION-1` - DONE):**

```text
list_accessible_conversations_page   # bot-membership users.conversations (all supported kinds)
read_conversation_history_page       # root-window bounded history; bot token
read_thread_replies_page             # thread replies with root normalization; bot token
read_exact_message                   # bounded exact lookup with reply pagination; no root required on point page
read_file_info (safe inventory only)
```

**Credential model (same integration, same `AsyncWebClient`):** one `INTERGRAX_SLACK_BOT_TOKEN` (`xoxb-`) for conversational runtime and all knowledge reads. Inventory uses `users.conversations` with `types=public_channel,private_channel,im,mpim` for conversations where the bot is a member. Public/private channel reads require the bot to be added to the conversation with appropriate `channels:*` / `groups:*` read/history scopes.

**Slack Vendor Knowledge adapter (`slack_conversation`):** `IMPLEMENTED` - `tombstones=false`, `permissions=false`, `slack.conversation.scope.v2` root-window reconciliation (`root_oldest`/`root_latest`, strict ordering), structured schema `slack.conversation.message.knowledge.v1`, history/reply page maximum **15**. Root `message_ts` and reply `message_ts` must lie inside `[root_oldest, root_latest]`; `thread_broadcast` history records are not separately materialized. `full_inventory=true` is complete inventory inside the explicit root-window scope only; replies whose root lies outside the root window are not discovered.

**Not implemented:** LKW bridge, live capability, authoritative ACL, durable deletion feed, binary file download.

**Planned provider-specific read primitives (future live/search - not implemented):**

```text
read bounded search result where Slack and policy support it
read explicit durable deletion feed via Events API
```

Required Slack scopes for knowledge reads are documented per credential route in [`intergrax/integrations/providers/conversation_channel/slack/USAGE.md`](../../../intergrax/integrations/providers/conversation_channel/slack/USAGE.md); audit per installation against official Slack documentation and preserve least privilege.

**Three-mode reuse:**

| Mode | Slack direction |
|---|---|
| Indexed RAG | durable Slack synchronization → optional LKW Knowledge Intake → Documents → Chunks → Embeddings → RAG |
| Durable materialization without RAG | durable Slack synchronization → DocumentStore / DB / object storage / application repository - no LKW, embeddings or vector store required |
| Live access | authorized request → validated Slack live capability → bounded provider read → normalized ephemeral evidence - no automatic Source, Document, Chunk, Embedding, vector record, database replica or sync checkpoint |

**Platform versus LKW ownership:**

| Platform owns | LKW owns |
|---|---|
| `SlackConversationChannelIntegration`; Slack client/SDK and transport; credential references and token isolation; typed provider references; provider inventory/history/thread/exact-read primitives; provider response validation; provider error normalization; Slack Vendor Knowledge Adapter; Vendor Knowledge Facade integration; Sync / Materialization support; Slack Live Capability Adapter; validated live execution support; provider-neutral durable and live results | Workspace Knowledge Configuration; Slack Connection attachment to workspace; Remote Resource selection; Indexed Source; Live Access Binding; workspace and principal authorization; Knowledge Intake; Source / Document / Chunk / Vector ownership; RAG retrieval; Hybrid Ask; evidence provenance; Slack conversational commands and rendering; user-facing operation status |

LKW must not construct Slack SDK clients, read Slack API directly, store raw Slack tokens, implement provider paging, own provider cursors, implement Slack-specific synchronization or duplicate Slack response validation.

**LKW application status:** `LKW-SLACK-CONNECTED-SOURCE-1` is **IN_PROGRESS / CHANGES_REQUIRED** (`LKW-SLACK-CONNECTED-SOURCE-1-REVIEW-FIX-2` - **CHANGES_REQUIRED**; `REVIEW-FIX-3` not accepted; HTTP discovery/create/sync scaffold present; final crash-safe recovery and real indexed Search/Ask proof remain under correction). Next Slack-vertical LKW implementation task: `LKW-CONVERSATION-CONTEXT-1`. Not implemented: shared-channel Ask, Conversation Context Binding, mention activation, `SHARED_ALLOWED` administration, live Slack access, Hybrid Ask.

### 13.8 Google Workspace - binding three-mode example (`GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1`)

**Classification:** `READY_FOR_REVIEW` - architecture frozen; no Google knowledge runtime implemented.

Google Workspace remains one category-correct public `collaboration_suite` integration. The existing `GoogleWorkspaceCollaborationSuiteIntegration`, its client, transport and credential resolution are the single foundation for collaboration operations, shared typed Google knowledge reads, durable materialization, indexed RAG and bounded live access. No application-specific or consumption-mode-specific Google client or public integration may be introduced.

**Current implementation honesty (repository HEAD):**

```text
GoogleWorkspaceCollaborationSuiteIntegration public shell
google_workspace collaboration-suite manifest (BETA)
legacy CollaborationSuite client delegation
basic mail/calendar/directory-shaped public contract methods
provider registration/catalog structure
```

Not present today: production Google OAuth, Google API client construction, Drive inventory, Docs/Sheets/Slides content reads, Calendar knowledge synchronization, Gmail knowledge synchronization, Google Chat knowledge reads, Google Vendor Knowledge adapters, Google live capabilities, LKW Google Connected Sources, Google Search/Ask proof. Manifest status or shell existence is not proof of operational knowledge surfaces.

**Rejected duplicate names (do not create):**

```text
GoogleDriveIntegration
GoogleDocsIntegration
GoogleSheetsIntegration
GoogleCalendarIntegration
GoogleSlidesIntegration
GmailKnowledgeIntegration
GoogleChatKnowledgeIntegration
GoogleWorkspaceKnowledgeIntegration
```

**Canonical foundation:**

```text
Connection / credential reference
        |
        v
GoogleWorkspaceCollaborationSuiteIntegration
        |
        v
shared typed Google provider read primitives
(per source_kind: drive, docs, sheets, calendar, slides, mail, chat)
        |
        +---------------------------+
        |                           |
        v                           v
durable knowledge path          live capability path
        |                           |
        v                           v
Google Workspace Vendor         Google Workspace Live
Knowledge Adapter(s)            Capability Adapter(s)
        |                           |
        v                           v
Vendor Knowledge Facade         Validated Executor
```

**Provider identity (frozen):**

```text
provider_id: google_workspace
integration category: collaboration_suite
single public integration: GoogleWorkspaceCollaborationSuiteIntegration
```

**Approved source kinds (independent scope/cursor semantics each):**

```text
(google_workspace, collaboration_suite, drive)
(google_workspace, collaboration_suite, docs)
(google_workspace, collaboration_suite, sheets)
(google_workspace, collaboration_suite, calendar)
(google_workspace, collaboration_suite, slides)
(google_workspace, collaboration_suite, mail)
(google_workspace, collaboration_suite, chat)
```

Do not collapse every Google Workspace resource into one untyped generic file source. Do not create separate public integrations for these source kinds.

**Canonical durable resource ownership (frozen):**

Discovery surface does **not** determine durable `source_kind`. Drive may inventory any Drive-hosted resource; the platform derives the canonical binding kind from the authoritative Google resource type server-side. The frontend must not choose or override `source_kind`.

| Google resource class | MIME / resource class | Canonical durable `source_kind` |
|---|---|---|
| Google-native document | Google Docs | `docs` |
| Google-native spreadsheet | Google Sheets | `sheets` |
| Google-native presentation | Google Slides | `slides` |
| Ordinary uploaded/stored file | non-native binary or generic file | `drive` |
| Drive folder / My Drive / Shared Drive scope | folder or drive scope | `drive` |
| Google Calendar / calendar-event scope | calendar resource or event set | `calendar` |
| Gmail scope | mailbox / folder / message scope | `mail` |
| Google Chat space / conversation scope | chat space or conversation | `chat` |

**Drive discovery → canonical binding flow (frozen):**

```text
Drive inventory / discovery
→ inspect authoritative Google resource type
→ derive canonical target source_kind server-side
→ issue provider-neutral Remote Resource candidate
→ create only the canonical KnowledgeSourceBinding
```

**Stable Google Workspace resource identity (frozen):**

Conceptual identity namespace - independent from discovery surface:

```text
provider_id = google_workspace
connection_ref
canonical Google resource type
stable Google resource ID
```

Rules:

1. A rename does not change identity.
2. A move does not change identity where Google preserves the resource ID.
3. Drive discovery and Docs/Sheets/Slides exact reads must refer to the same underlying Google resource identity.
4. Export/download URL is never identity.
5. Revision, ETag, modified time and content hash are change state - not identity.
6. The same native Google file must not become unrelated `drive` and `docs`/`sheets`/`slides` durable objects.

Do not freeze an implementation-specific Python type in this architecture document.

**Overlapping-binding policy (frozen):**

Example overlap:

```text
selected native Google Doc
+
selected Drive folder containing that Google Doc
```

**First proof (narrow policy):**

```text
explicit selected resources only
broad Drive/folder synchronization deferred unless overlap semantics are proved
```

Selecting both a native Google Doc and a containing Drive folder in the first proof is out of scope unless overlap semantics are explicitly implemented. Broad folder scopes remain deferred.

**Future broad Drive scopes (required direction - not yet chosen):**

When broad Drive/folder synchronization is implemented, one of:

```text
Option A: reject an overlapping source binding in the same workspace

Option B: one canonical provider item ownership record deduplicates the resource
          across bindings while preserving provenance of every covering binding
```

Option B is **not** chosen unless the existing Vendor Knowledge and LKW ownership models can support it safely. Until that contract exists, broad overlapping scopes must fail closed or remain deferred. Do not claim duplicate prevention without an enforceable rule.

**Drive read surface versus Docs/Sheets/Slides content surfaces (frozen):**

| Surface | Owns |
|---|---|
| **Drive** | inventory; hierarchy; resource classification; folder/drive traversal; ordinary binary content; stable Drive-hosted resource metadata; change-feed/reconciliation primitives |
| **Docs** | typed native content extraction; native structure; exact native content reads; native revision interpretation |
| **Sheets** | typed native content extraction; native structure; exact native content reads; native revision interpretation |
| **Slides** | typed native content extraction; native structure; exact native content reads; native revision interpretation |

A Drive adapter may call a shared typed native-content primitive internally only when the durable source binding remains canonical and duplication is prevented. Do not create two independent durable copies merely because two provider APIs can read the same file.

Additional rules:

1. Drive may discover a native Docs, Sheets or Slides item; native content is read through the appropriate typed provider surface.
2. The same Google resource identity must not become unrelated duplicate provider objects.
3. Stable provider identity remains separate from content revision; a file rename must not change remote identity.
4. Provider-generated download URLs must not become durable identity.
5. Google-native resources must not be treated as ordinary binary files when a typed native content contract is required.
6. Uploaded PDF, DOCX, XLSX, PPTX and other binary files may follow the shared binary-content path when supported.
7. Exact API calls and export formats remain implementation-task decisions after an official API audit.

**Thin Vendor Knowledge adapters (planned - not implemented):**

```text
GoogleWorkspaceDriveKnowledgeAdapter
GoogleWorkspaceDocsKnowledgeAdapter
GoogleWorkspaceSheetsKnowledgeAdapter
GoogleWorkspaceCalendarKnowledgeAdapter
GoogleWorkspaceSlidesKnowledgeAdapter
GoogleWorkspaceMailKnowledgeAdapter
GoogleWorkspaceChatKnowledgeAdapter
```

Each adapter:

- receives the already resolved `GoogleWorkspaceCollaborationSuiteIntegration`;
- owns no Google credentials, constructs no independent Google client, owns no checkpoint repository, owns no application database, owns no LKW Source;
- declares only its own capabilities;
- maps provider records into canonical Vendor Knowledge contracts;
- uses the shared Vendor Knowledge synchronization coordinator;
- keeps independent source scope, cursor and deletion semantics;
- remains reusable by applications other than LKW.

**Three-mode support (per source kind - planned independently):**

| Mode | Google direction |
|---|---|
| Indexed RAG | durable Google synchronization → optional LKW Knowledge Intake → Documents → Chunks → Embeddings → RAG |
| Durable materialization without RAG | durable Google synchronization → DocumentStore / DB / object storage / application repository - no LKW, embeddings or vector store required |
| Live access | authorized request → validated Google live capability → bounded provider read → normalized ephemeral evidence - no automatic Source, Document, Chunk, Embedding, vector record, database replica or sync checkpoint |

Binding rules (same as §2.1): one provider integration is reused by all modes; durable and live access have separate adapters; live results are ephemeral by default; durable synchronization does not imply RAG; adding an Indexed Source does not create Live Access; adding Live Access does not index the resource; no second credential/client path for live access.

**Foundation task (`GOOGLE-WORKSPACE-KNOWLEDGE-FOUNDATION-1` - PLANNED):** typed Google Workspace integration configuration; credential-reference resolution; least-privilege credential modes; one shared provider client family; provider request execution boundary; pagination token normalization; provider error normalization; rate-limit and retry classification; stable provider resource references; safe timestamps and revisions; safe display labels; bounded request limits; capability declaration; no LKW imports; no RAG imports; no application workspace concepts.

**Foundation prerequisites (activation gates - not satisfied):**

```text
GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1 becomes ACCEPTED (currently READY_FOR_REVIEW)
canonical Tenant Connection / credential-reference boundary available
SecretsStore-owned credential persistence available
runtime integration rehydration/resolution boundary available
Vendor Knowledge binding, registry and synchronization contracts available
```

Canonical owners: durable tenant Connection Catalog and runtime integration rehydration/resolution are owned by `LKW-KNOWLEDGE-ACCESS-1` (and its platform prerequisites) - not by Google Foundation. Google Foundation must not introduce another tenant Connection catalog; must not introduce a Google-only credential database; must not put OAuth tokens into provider config records, `KnowledgeSourceBinding` or LKW state. If a required generic boundary remains unfinished, finish or reuse that boundary before production Google OAuth. Do not duplicate LKW application configuration inside the platform integration.

**Post-Slack implementation gate (exact):** Google Workspace runtime implementation starts only after `LKW-SLACK-KNOWLEDGE-PROOF-1` becomes **ACCEPTED** (join of `LKW-SLACK-CONNECTED-SOURCE-1`, `LKW-CONVERSATION-CONTEXT-1`, `LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1`, `SLACK-LIVE-CAPABILITY-1` and `LKW-HYBRID-ASK-1`). As of current HEAD: `LKW-SLACK-CONNECTED-SOURCE-1` is **IN_PROGRESS / CHANGES_REQUIRED**; `LKW-SLACK-KNOWLEDGE-PROOF-1` and remaining Slack-vertical tasks are **PLANNED** - Google implementation is not active.

Credential routes (conceptually separated): individual-user OAuth; organization/admin-approved Google Workspace access; service-account or delegated organizational access when justified. For the first testable proof, individual-user authorization is the preferred product route. Exact OAuth scopes and Google SDK signatures are **not** frozen here - implementation tasks must verify against current official Google documentation. Secrets remain owned by the existing Connection/SecretsStore boundary. No access token, refresh token, client secret or service-account private key may enter `KnowledgeSourceBinding`, `WorkspaceIndexedSourceBinding`, Remote Resource response, LKW Source, citation or provider cursor.

**LKW vertical (planned):**

```text
workspace Connection
→ Google Workspace Remote Resource discovery
→ selected Drive / Docs / Sheets / Calendar resource
→ tenant KnowledgeSourceBinding
→ WorkspaceIndexedSourceBinding
→ Vendor Knowledge synchronization
→ existing LKW materialization/indexing pipeline
→ Search → Ask → citations
```

`LKW-GOOGLE-WORKSPACE-CONNECTED-SOURCE-1` reuses the generic Connected Source implementation proved by Slack. Do not plan Google-specific LKW configuration aggregates, mutation engines, indexing pipelines, vector database access or Source tables. First Google LKW sources default to `PERSONAL_ONLY`; future `SHARED_ALLOWED` use remains governed by the accepted Conversation Context architecture.

**First proof (`LKW-GOOGLE-WORKSPACE-PROOF-1` - PLANNED):** user connects one Google account → selects approved Google resources → one Google Doc synchronized → one Google Sheet synchronized → one Google Calendar resource/event set synchronized → optionally one ordinary Drive file synchronized → LKW indexes selected resources → Search retrieves provider-derived evidence → Ask produces one grounded answer → citations identify the correct Google source and resource → no Google API call is made by Ask after durable synchronization. Proof demonstrates mixed source shapes: narrative document, structured spreadsheet, calendar/event data, ordinary stored file. User-oriented proof, not merely adapter unit tests.

**Proof-first execution gate (binding - vertically incremental):**

Each read surface and its adapter form one independently reviewable vertical step before the next surface begins. Proof-critical phase:

```text
GOOGLE-WORKSPACE-KNOWLEDGE-FOUNDATION-1

→ GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1A-DRIVE
→ GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1A-DRIVE
→ Drive contract/integration proof

→ GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1B-DOCS
→ GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1B-DOCS
→ Docs contract/integration proof

→ GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1C-SHEETS
→ GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1C-SHEETS
→ Sheets contract/integration proof

→ GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1D-CALENDAR
→ GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1D-CALENDAR
→ Calendar contract/integration proof

→ LKW-GOOGLE-WORKSPACE-CONNECTED-SOURCE-1
→ LKW-GOOGLE-WORKSPACE-PROOF-1
```

The final Google LKW proof may still combine Docs, Sheets, Calendar and an optional ordinary Drive file.

Family expansion after the proof:

```text
GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1E-SLIDES → GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1E-SLIDES
GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1F-MAIL → GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1F-MAIL
GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1G-CHAT → GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1G-CHAT
```

Global placement: Google Workspace runtime implementation starts only after `LKW-SLACK-KNOWLEDGE-PROOF-1` becomes **ACCEPTED** (complete Slack Knowledge vertical - currently **PLANNED**) → Google proof-critical path above → remaining Google family surfaces → `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR` and other lower-priority provider expansion.

**Product rationale:** Microsoft 365 proves enterprise-oriented collaboration and document access. Google Workspace lowers the entry barrier for individual testers, small teams and design partners who can authorize their own account. Supporting both proves that the LKW Connected Source architecture is provider-neutral rather than Microsoft-specific. The goal is not connector count - it is one convincing proof over different real-world source shapes and provider ecosystems. Google Workspace is the second strategic collaboration/document ecosystem, not an open-ended commitment to add every available SaaS provider.

**Platform versus LKW ownership:**

| Platform owns | LKW owns |
|---|---|
| `GoogleWorkspaceCollaborationSuiteIntegration`; Google client/transport family; credential references and token isolation; typed provider references; per-source-kind provider read primitives; provider response validation; provider error normalization; Google Vendor Knowledge adapters; Vendor Knowledge Facade integration; Sync / Materialization support; Google Live Capability adapters; validated live execution support; provider-neutral durable and live results | Workspace Knowledge Configuration; Google Connection attachment to workspace; Remote Resource selection; Indexed Source; Live Access Binding; workspace and principal authorization; Knowledge Intake; Source / Document / Chunk / Vector ownership; RAG retrieval; Hybrid Ask; evidence provenance; user-facing operation status |

LKW must not construct independent Google API clients, read Google APIs directly, store raw Google tokens, implement provider paging, own provider cursors, implement Google-specific synchronization or duplicate Google response validation.

**LKW application status:** `LKW-GOOGLE-WORKSPACE-CONNECTED-SOURCE-1` and `LKW-GOOGLE-WORKSPACE-PROOF-1` are **PLANNED**. No Google knowledge capability is implemented in LKW today.

---

## 14. Contract test requirements

### Existing vendor integration tests

Each integration continues to test its category and provider behavior:

- construction and disabled-by-default configuration;
- required client/transport behavior;
- provider request mapping;
- provider response normalization;
- provider error mapping;
- category methods;
- secret-safe public view;
- no network calls during import/registration.

### Knowledge adapter contract tests

Each adapter must test:

- source candidate mapping;
- bounded source validation;
- empty source;
- pagination;
- continuation replay;
- new item;
- content update;
- metadata-only update;
- ACL-only update;
- rename/move;
- deletion tombstone;
- access-revocation tombstone;
- rate-limit mapping;
- expired authorization;
- unavailable source;
- incomplete ACL signaling;
- deep links and provenance;
- no secret leakage.

### Facade/sync runtime tests

The common layer must test:

- integration and adapter resolution;
- unsupported provider/source kind;
- cross-tenant denial;
- checkpoint commit only after durable page completion;
- crash before commit and safe replay;
- lease/concurrency behavior;
- partial page failure;
- idempotent reprocessing;
- reconciliation after invalid checkpoint;
- ACL enforcement boundary;
- no provider-specific branching in LKW core;
- reuse of the shared ingestion pipeline.

---

## 15. Explicit non-goals

This architecture does not authorize:

- a new `IntegrationCategory.KNOWLEDGE_SOURCE`;
- duplicate public vendor integrations for knowledge ingestion;
- one multi-category vendor monster class;
- vendor SDK imports in LKW;
- vendor-specific RAG pipelines;
- credentials in source records;
- checkpoint ownership inside a provider client;
- direct vector-store writes from an adapter;
- direct LLM use inside integrations or adapters;
- unrestricted ingestion of entire mailboxes, tenants, BI estates or lakehouses;
- treating MCP as guaranteed durable synchronization;
- implementing all vendors in one task.

---

## Implementation / architecture history

### Maintainer note

**Internal architecture workflow status:** `CORRECTED / READY_FOR_REVIEW`
**Task:** `VENDOR-KNOWLEDGE-THREE-MODE-REUSE-ARCH-1 - three-mode provider reuse architecture`
**Classification:** docs-only architecture and contract boundary

### Correction summary

The previous version of this document proposed a new `knowledge_source` integration category and separate public integrations such as `jira:knowledge_source`, `confluence:knowledge_source`, or `ms365_graph:knowledge_source`.

That direction is rejected.

Intergrax already has the correct lower-level integration architecture:

```text
PlatformIntegrationContract
        |
        v
category-specific integration contract
        |
        v
one public integration implementation for one provider/category
        |
        v
provider-specific operational methods and client behavior
```

Examples already present in the platform:

```text
JiraIssueTrackerIntegration
ConfluenceWikiKnowledgeIntegration
Ms365GraphCollaborationSuiteIntegration
DatabricksRelationalStoreIntegration
SlackConversationChannelIntegration
GoogleWorkspaceCollaborationSuiteIntegration
```

These integrations must remain the single public provider/category entrypoints. They own vendor communication and implement the appropriate existing category contract. They must not be duplicated merely because an application wants to use their data as knowledge.

The missing capability belongs above the integration layer as **reusable provider foundations** consumed through **three separate modes**:

```text
indexed RAG
durable materialization
bounded real-time (live) access
```

Each mode retains its own lifecycle, policy and persistence semantics. Hybrid Ask combines indexed and live evidence at the application level; it is not a fourth provider integration.

One-sentence result:

> Every vendor integration and provider read primitive is designed once as a reusable foundation for indexed RAG, durable data materialization and bounded real-time access, while each consumption mode retains its own lifecycle, policy and persistence semantics.

---

## 16. Implementation sequence

### Step 0 - this correction

`VENDOR-KNOWLEDGE-THREE-MODE-REUSE-ARCH-1`

- freeze reusable provider foundations and three separate consumption lifecycles;
- document durable and live sibling paths;
- retain existing categories and public integrations;
- define the facade, adapter, sync-runtime and live-capability boundaries;
- identify the future LKW integration point.

`KNOWLEDGE-SOURCE-DISCOVERY-1` (prior)

- corrected the architectural direction;
- retained existing categories and public integrations;
- defined the facade, adapter and sync-runtime boundaries.

### Step 1 - facade contract discovery against code

`VENDOR-KNOWLEDGE-FACADE-AUDIT-1`

Perform a tightly scoped audit of:

- current platform integration resolution;
- current provider/category contract registry;
- operational protocols for collaboration, issue tracker and wiki knowledge;
- current Jira, Confluence and Microsoft Graph public integrations;
- reusable connection/secrets mechanisms;
- current queue/checkpoint/state mechanisms that may support the facade.

Output: exact reuse map and implementation file scope. No broad repository audit.

### Step 2 - minimal facade models and service port

`VENDOR-KNOWLEDGE-FACADE-1A`

Implement only:

- facade service contract;
- source binding and safe candidate models;
- canonical item/content/change/ACL models;
- adapter protocol and adapter registry;
- no concrete vendor adapter;
- no LKW changes.

### Step 3 - first adapter proof

`VENDOR-KNOWLEDGE-FACADE-1B`

Use one existing integration as the proof. Preferred choice is selected after the scoped audit, likely Jira or Confluence for structured content, or SharePoint/OneDrive for the first remote-file path.

The proof must reuse the existing public vendor integration and client.

### Step 4 - shared synchronization runtime

`VENDOR-KNOWLEDGE-SYNC-1C`

Add durable checkpoint, replay, item state, tombstone and reconciliation behavior above adapters.

### Step 5 - LKW bridge

`LKW-CONNECTED-SOURCE-BRIDGE-1D`

Only after both streams are stable:

- bind `CONNECTED_SOURCE` to a facade binding;
- route connected-source processing through the facade/sync runtime;
- submit normalized content to the existing LKW ingestion pipeline;
- expose safe source management and status through Slack/HTTP.

---

## 17. Acceptance criteria

This architecture is accepted when all statements below are unambiguous:

1. Existing integration categories remain the provider domain taxonomy.
2. Each provider/category retains one public integration entrypoint.
3. No generic knowledge-source integration category is introduced.
4. Vendor integrations remain independent of LKW and RAG.
5. The facade is a platform service above integrations, not a vendor integration.
6. Provider/source adapters normalize behavior without duplicating clients or auth.
7. LKW communicates with Vendor Knowledge Facade (durable) and the validated live capability boundary (live), not directly with vendor integrations.
8. Durable checkpoint, replay and reconciliation behavior is shared.
9. Parsing, chunking, embeddings and stores remain one shared pipeline.
10. Stable remote identity is distinct from content and ACL revisions.
11. ACL is enforced before content reaches the model.
12. Power BI and Atlan receive justified domain categories if existing categories are insufficient.
13. MCP remains complementary to the direct durable integration path.
14. The first implementation task begins with a scoped reuse audit, not with adding a new integration category.

---

## 18. Final architecture statement

```text
existing provider/category integration
        |
        v
shared provider read primitives
        |
        +----------------------------------+
        |                                  |
        v                                  v
durable knowledge path                live capability path
        |                                  |
        v                                  v
Vendor Knowledge Adapter           Live Capability Adapter
        |                                  |
        v                                  v
Vendor Knowledge Facade            Validated Executor
        |                                  |
        v                                  v
Sync / Materialization Runtime      ephemeral Live Evidence
        |
        v
injected durable sink
├── DocumentStore
├── relational / NoSQL database
├── object storage
├── application repository
└── optional LKW Knowledge Intake → RAG
```

> Vendor integrations provide the low-level provider capabilities and shared read primitives. The durable path normalizes and materializes provider data through Vendor Knowledge adapters and the sync runtime. The live path executes bounded read-only capabilities at request time. Applications consume either or both paths without duplicating vendor clients, SDKs or integration categories. Exactly three consumption modes apply: indexed RAG, durable materialization without RAG and bounded live access. Synchronization is a lifecycle mechanism of the durable modes, not a separate fourth consumption mode.

---

## 19. Vendor Knowledge contribution composition

`VendorKnowledgeProviderContribution` is the extension ABI for Vendor
Knowledge. Provider-owned builders publish adapters, source plugins,
connection factories and supported Live registrations into one
`VendorKnowledgeContributionCatalog`. Built-in contributions are deterministic;
optional external contributions use the explicit
`intergrax.vendor_knowledge.providers` entry-point group and are loaded only
when discovery is enabled.

The catalog is the single source for runtime registries. Application-owned
discovery and indexed materializer hooks are added through the typed host
extension context, then consumed generically by LKW composition. Provider
business cases are not repeated in the adapter, source-plugin, factory, Live
or materializer registry builders. External provider authoring guidance:
[`VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md`](../technical/guides/VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md)
(VK-EXT-4).
