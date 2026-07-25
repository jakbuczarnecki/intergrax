# Vendor Knowledge Facade and Integration Boundary

**Status:** `CORRECTED / READY_FOR_REVIEW`  
**Task:** `KNOWLEDGE-SOURCE-DISCOVERY-1 — architecture correction`  
**Classification:** docs-only architecture and contract boundary  
**Branch:** `development`  
**Integration canon:** [`INTEGRATIONS.md`](INTEGRATIONS.md)  
**LKW intake discovery:** [`../../applications/local_workspace_application/docs/KNOWLEDGE_INTAKE_DISCOVERY.md`](../../applications/local_workspace_application/docs/KNOWLEDGE_INTAKE_DISCOVERY.md)

---

## 1. Correction summary

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
```

These integrations must remain the single public provider/category entrypoints. They own vendor communication and implement the appropriate existing category contract. They must not be duplicated merely because an application wants to use their data as knowledge.

The missing capability belongs above the integration layer:

```text
existing vendor integrations
        |
        v
Vendor Knowledge Facade
        |
        v
shared synchronization and normalization runtime
        |
        v
LKW or another consuming application
```

One-sentence result:

> Vendor integrations remain low-level, category-correct provider implementations; a shared facade above them unifies source discovery, reading, change tracking, content, provenance and access information for applications such as LKW.

---

## 2. Binding architectural decisions

| Decision | Classification | Binding statement |
|---|---|---|
| Existing integration categories remain authoritative | `FROZEN` | Jira remains an issue-tracker integration, Confluence remains a wiki-knowledge integration, Microsoft Graph remains a collaboration-suite integration, and Databricks remains a relational-store integration unless a separately justified domain category is introduced. |
| No generic `knowledge_source` integration category | `REJECTED` | Knowledge ingestion is a cross-category application use case, not the primary domain identity of every vendor integration. |
| No duplicate public integration for knowledge use | `REJECTED` | Do not create `JiraKnowledgeSourceIntegration`, `ConfluenceKnowledgeSourceIntegration`, or equivalent parallel public integrations beside existing provider/category integrations. |
| Vendor integration remains low-level | `FROZEN` | It owns provider transport, auth handoff, vendor request/response mapping, provider errors and category operations. It does not know LKW, workspaces, RAG or product workflows. |
| Unified knowledge behavior is exposed by a facade | `FROZEN DIRECTION` | A shared platform service resolves existing integrations and exposes one vendor-neutral knowledge access boundary to consuming applications. |
| Facade is not an integration category | `FROZEN` | It is a platform service/facade and may use a registry of source adapters. It is not registered as another vendor integration. |
| Existing integrations may expose additional provider methods | `FROZEN DIRECTION` | Delta reads, pagination, attachments, permissions or inventory methods may be added to the correct existing integration or to a private/provider-specific read facet behind it. |
| Application does not call vendor methods directly | `FROZEN` | LKW communicates with the facade. The facade and its adapters resolve the correct vendor integration and normalize the result. |
| One shared synchronization runtime | `FROZEN DIRECTION` | Checkpoints, leases, retry, reconciliation, durable item state and replay semantics are common platform/application mechanisms, not independently reimplemented by each vendor. |
| One shared ingestion pipeline | `FROZEN` | Parsing, structured normalization, chunking, embeddings, Document Store and Vector Store remain shared downstream capabilities. |
| Stable remote identity is separate from revision | `FROZEN` | Vendor item identity must not be derived from a content hash. Version, ETag, content hash and ACL hash represent change state. |
| ACL must be enforceable before model access | `FROZEN` | Prompt instructions are not authorization. Retrieval must not expose knowledge that the requesting principal is not allowed to access. |
| Secrets are referenced, never embedded | `FROZEN` | Source bindings and facade requests may carry only opaque connection or credential references. |
| MCP is complementary | `FROZEN DIRECTION` | MCP may expose live tools or approved actions, but it does not replace durable synchronization and application-owned knowledge state. |

---

## 3. Layered architecture

### 3.1 Layer 1 — platform integration base

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

### 3.2 Layer 2 — category contracts

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

### 3.3 Layer 3 — concrete vendor integration

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

### 3.4 Layer 4 — vendor knowledge adapters

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

### 3.5 Layer 5 — Vendor Knowledge Facade

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

### 3.6 Layer 6 — shared synchronization runtime

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

### 3.7 Layer 7 — consuming application

LKW is the first platform proof and consumer of the facade.

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
LKW / other application
        |
        v
Vendor Knowledge Facade
        |
        +------------------------------+
        |                              |
        v                              v
Knowledge Sync Runtime        Source Discovery / Inspection
        |
        v
Knowledge Adapter Registry
        |
        v
source-kind adapter
        |
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

Vendor integrations must remain reusable by agents, applications and tools that never use LKW or knowledge ingestion.

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

### Pattern A — extension of the existing provider integration

Use when the method is a natural part of the existing category contract and is broadly meaningful for providers in that category.

Example direction:

```text
WikiKnowledge
├── get_page
├── search_pages
└── list_pages / read_page_version
```

### Pattern B — provider-specific private read facet

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

Represents a tenant-scoped reference to an existing integration connection.

Minimum semantics:

```text
connection_id
tenant_id
provider_id
integration_kind
credential_ref
connected_principal
safe_display_name
status
```

The connection record contains references and safe metadata, never secret values.

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

Forbidden in bindings, checkpoints, events, logs, errors and LKW workspace records:

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

## 16. Implementation sequence

### Step 0 — this correction

`KNOWLEDGE-SOURCE-DISCOVERY-1`

- correct the architectural direction;
- retain existing categories and public integrations;
- define the facade, adapter and sync-runtime boundaries;
- identify the future LKW integration point.

### Step 1 — facade contract discovery against code

`VENDOR-KNOWLEDGE-FACADE-AUDIT-1`

Perform a tightly scoped audit of:

- current platform integration resolution;
- current provider/category contract registry;
- operational protocols for collaboration, issue tracker and wiki knowledge;
- current Jira, Confluence and Microsoft Graph public integrations;
- reusable connection/secrets mechanisms;
- current queue/checkpoint/state mechanisms that may support the facade.

Output: exact reuse map and implementation file scope. No broad repository audit.

### Step 2 — minimal facade models and service port

`VENDOR-KNOWLEDGE-FACADE-1A`

Implement only:

- facade service contract;
- source binding and safe candidate models;
- canonical item/content/change/ACL models;
- adapter protocol and adapter registry;
- no concrete vendor adapter;
- no LKW changes.

### Step 3 — first adapter proof

`VENDOR-KNOWLEDGE-FACADE-1B`

Use one existing integration as the proof. Preferred choice is selected after the scoped audit, likely Jira or Confluence for structured content, or SharePoint/OneDrive for the first remote-file path.

The proof must reuse the existing public vendor integration and client.

### Step 4 — shared synchronization runtime

`VENDOR-KNOWLEDGE-SYNC-1C`

Add durable checkpoint, replay, item state, tombstone and reconciliation behavior above adapters.

### Step 5 — LKW bridge

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
7. LKW communicates with the facade, not directly with vendor integrations.
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
category-correct base contracts
        |
        v
single vendor integration implementation
        |
        v
small provider/source adapter
        |
        v
Vendor Knowledge Facade
        |
        v
shared synchronization runtime
        |
        v
LKW Knowledge Intake and shared RAG pipeline
```

> Vendor integrations provide the low-level provider capabilities. The facade accumulates and normalizes those capabilities into one application-facing knowledge boundary. LKW consumes that boundary without knowing vendor APIs and without creating parallel ingestion mechanisms.
