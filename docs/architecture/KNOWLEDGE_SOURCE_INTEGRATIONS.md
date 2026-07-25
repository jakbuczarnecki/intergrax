# Knowledge Source Integrations

**Status:** `DOCUMENTED / READY_FOR_REVIEW`  
**Task:** `KNOWLEDGE-SOURCE-DISCOVERY-1`  
**Classification:** docs-only architecture and integration contract discovery  
**Branch:** `development`  
**Integration canon:** [`INTEGRATIONS.md`](INTEGRATIONS.md)  
**LKW intake discovery:** [`../../applications/local_workspace_application/docs/KNOWLEDGE_INTAKE_DISCOVERY.md`](../../applications/local_workspace_application/docs/KNOWLEDGE_INTAKE_DISCOVERY.md)

---

## 1. Purpose

Intergrax needs one reusable platform boundary through which external enterprise systems can expose durable knowledge to applications without creating a separate ingestion, parsing, chunking, embedding, or retrieval pipeline for each vendor.

The first platform proof is Local Knowledge Workspace (LKW), but the contract defined here is not an LKW-specific API. It is a platform integration category intended for Microsoft Graph, Jira, Confluence, Power BI, Atlan, Databricks, and future enterprise sources.

Target product flow:

```text
Microsoft Graph / Jira / Confluence / Power BI / Atlan / Databricks / future vendor
                                  |
                                  v
                  Knowledge Source Integration
                                  |
             descriptors + content + ACL + changes
                                  |
                                  v
                   application-owned sync runtime
                                  |
                                  v
       shared parser / normalizer / chunk / embedding pipeline
                                  |
                                  v
                 Document Store + Vector Store
                                  |
                                  v
            Slack / HTTP / MCP / other product surface
```

One-sentence result:

> External vendors expose changes and content through one platform contract; applications decide how and when that knowledge becomes durable, searchable product data.

---

## 2. Cursor read scope and task boundary

This document is the complete read scope for `KNOWLEDGE-SOURCE-DISCOVERY-1`.

For the next implementation task, read only:

1. this document;
2. the category-contract section of [`INTEGRATIONS.md`](INTEGRATIONS.md);
3. the exact integration taxonomy and registry files named by the implementation task.

Do not perform a broad repository audit unless a concrete contract conflict is discovered.

This discovery task introduces no runtime behavior and makes no claim that any vendor knowledge source is implemented.

---

## 3. Decision summary

| Decision | Classification | Binding statement |
|---|---|---|
| Introduce a dedicated Knowledge Source integration category | `FROZEN DIRECTION` | Durable enterprise knowledge synchronization is distinct from action-oriented collaboration, issue-tracker, wiki, database, search, and object-storage contracts. |
| Vendor adapters are platform integrations, not LKW components | `FROZEN` | Vendor packages must not import or depend on `local_workspace_application`. |
| One provider may implement multiple categories | `FROZEN / EXISTING CANON` | A provider such as `ms365_graph` may have separate collaboration-suite and knowledge-source integration classes. Never create one multi-category monster integration. |
| Integration returns data; consumer owns durable processing | `FROZEN` | The integration reads remote changes, content, metadata, and ACL. It does not create product Documents, chunks, embeddings, prompts, or generated artifacts. |
| Stable remote identity is separate from content revision | `FROZEN` | `remote_id` identifies the durable vendor item. ETag, version, content hash, and ACL hash represent revisions. Content hash must not be the document identity. |
| Incremental synchronization uses consumer-committed checkpoints | `FROZEN` | The integration proposes the next checkpoint. The consuming runtime persists it only after the page has been processed durably. |
| Delivery semantics are at-least-once and idempotent | `FROZEN` | Replaying a change page must be safe. Exactly-once delivery is not assumed. |
| ACL is part of the knowledge contract | `FROZEN` | Source permissions must be available for access-controlled ingestion and retrieval. Prompt instructions are never an authorization mechanism. |
| Credentials are referenced, not embedded | `FROZEN` | Durable connection/source records contain only `credential_ref` or equivalent opaque references. Tokens and secrets must not appear in source configuration, events, or logs. |
| Direct API/SDK is the durable sync foundation | `FROZEN` | Inventory, delta synchronization, pagination, tombstones, and reconciliation use vendor APIs or SDKs through the integration. |
| MCP is an optional live interaction gateway | `FROZEN DIRECTION` | MCP may expose interactive resources, queries, and approved actions, but it is not the only durable synchronization mechanism. |
| A vendor-specific RAG pipeline | `REJECTED` | Providers must reuse shared parsing, normalization, chunking, embedding, storage, and retrieval capabilities. |
| Vendor SDK calls inside LKW services | `REJECTED` | Product/application code resolves an integration contract; it does not directly call Graph, Jira, Confluence, Power BI, Atlan, or Databricks SDKs. |
| Checkpoint commit before durable page completion | `REJECTED` | A crash must not permanently skip remote changes. |
| Tenant-wide ingestion without enforceable ACL | `REJECTED` | Broad app-only access cannot be treated as equivalent to user-authorized visibility. |
| Exact Python class, module, persistence, and route names | `DEFERRED` | Semantics are frozen here; exact implementation names are selected by scoped implementation tasks. |

---

## 4. Architectural boundary

### 4.1 Knowledge Source Integration owns

A vendor integration may own:

- vendor API transport and authentication handoff;
- provider-specific request construction;
- pagination and delta/change-feed traversal;
- conversion of provider responses into canonical descriptors;
- retrieval of binary, rich-text, or structured content;
- retrieval and normalization of source ACL information;
- provider rate-limit, timeout, authentication, and availability error mapping;
- safe provider health checks;
- provider capability declaration;
- validation of a vendor-specific scope;
- provider-specific cursor/checkpoint serialization and parsing;
- normalization of deletions, revocations, moves, renames, and version changes.

### 4.2 Knowledge Source Integration must not own

A vendor integration must not:

- create or mutate LKW `Workspace`, `WorkspaceSource`, `KnowledgeInput`, or ingestion operation records;
- create product Documents;
- select a Document Store or Vector Store;
- parse Office/PDF files into chunks;
- chunk or embed content;
- invoke an LLM;
- build prompts;
- decide product retrieval ranking;
- generate emails, reports, offers, contracts, analyses, or other artifacts;
- communicate with Slack, Teams chat, Telegram, or another conversation frontend;
- persist application checkpoints in an application repository;
- decide whether a user is allowed to see a retrieved result;
- become the source of truth for product operation status;
- contain application-specific workflow orchestration.

### 4.3 Consuming application/runtime owns

The consuming application or shared sync runtime owns:

- binding a platform connection and source scope to an application source;
- source leases and concurrency control;
- durable source and sync-run state;
- committed checkpoints;
- retries, backoff, scheduling, and reconciliation policy;
- item state and revision state;
- idempotency and deduplication;
- staging of remote binary content when required;
- invocation of the shared parser or structured-record normalizer;
- Document ownership and persistence;
- chunking, embedding, vector upsert, and obsolete-vector cleanup;
- ACL persistence and retrieval-time enforcement;
- product lifecycle events and frontend notifications;
- downstream artifact generation and approved vendor actions.

---

## 5. Relation to existing integration categories

Knowledge-source behavior is not a replacement for existing categories.

| Existing category | Primary responsibility | Knowledge-source relationship |
|---|---|---|
| `collaboration_suite` | Mail, calendar, directory, and collaboration actions | The same provider may separately expose durable knowledge synchronization. |
| `issue_tracker` | Issue lookup and issue mutations | A separate knowledge-source integration exposes durable, incremental issue knowledge. |
| `wiki_knowledge` | Wiki lookup APIs | A separate knowledge-source integration adds cursor, revision, ACL, and deletion semantics for ingestion. |
| `relational_store` | Query execution and relational access | A database provider may separately expose governed metadata or approved snapshots as knowledge. |
| `object_storage` | Binary object persistence | Used by a consumer for staging or managed originals; it is not a vendor knowledge contract. |
| `document_parser` | File parsing | Invoked downstream by the consumer; never reimplemented in a vendor adapter. |
| `search_provider` | Search execution | Search may help discovery but does not replace durable item/change semantics. |

Existing integration canon remains binding:

```text
provider_id identifies the vendor
integration_kind identifies one category
integration_id identifies provider + category
```

Example direction:

```text
ms365_graph:collaboration_suite
ms365_graph:knowledge_source

jira:issue_tracker
jira:knowledge_source

confluence:wiki_knowledge
confluence:knowledge_source

databricks:relational_store
databricks:knowledge_source
```

Each category has a separate public integration class. Private transport, auth, response models, and SDK clients may be reused within the provider package when dependency direction remains valid.

---

## 6. Canonical vocabulary

| Term | Meaning |
|---|---|
| **Knowledge Connection** | Tenant-scoped technical authorization for one provider account, organization, site, or service principal. References credentials but does not contain secrets. |
| **Knowledge Source Scope** | The remote boundary selected for synchronization, such as a drive folder, Jira project set, Confluence space, Power BI workspace, Atlan domain, or Databricks catalog. |
| **Knowledge Source Integration** | Category-specific vendor adapter implementing the contract defined by this architecture. |
| **Knowledge Item** | One stable remote entity that may become one or more application Documents. |
| **Knowledge Item Descriptor** | Provider-neutral identity, revision, type, title, locator, timing, and metadata for an item. |
| **Knowledge Item Content** | Binary, rich-text, or structured payload fetched for a descriptor. |
| **Knowledge Item ACL** | Normalized visibility and principal data associated with an item. |
| **Knowledge Change Page** | One bounded page of item changes and tombstones plus a proposed continuation/checkpoint. |
| **Checkpoint** | Opaque provider state required to resume incremental reading. Persisted by the consumer after durable completion. |
| **Tombstone** | A deletion, revocation, or no-longer-visible change referring to a stable remote item identity. |
| **Reconciliation** | Periodic broader scan used to detect missed webhooks, lost delta state, permission drift, or provider inconsistencies. |
| **Source Candidate** | Safe application-facing choice that resolves to a connection and scope without exposing credentials or unsafe locators. |

---

## 7. Knowledge Connection

A Knowledge Connection represents authorization and provider identity. It is separate from the source scope because one connection may authorize many independently configured sources.

Minimum semantic fields:

```text
connection_id
tenant_id
provider_id
auth_mode
credential_ref
connected_principal
safe_display_name
granted_scopes
status
expires_at
created_at
updated_at
```

### 7.1 Required invariants

- `connection_id` is opaque and unique within the platform tenancy model.
- `tenant_id` is mandatory and is checked on every resolution.
- `provider_id` identifies one catalog provider.
- `credential_ref` points to a secret-bearing system; it is not the secret.
- `connected_principal` is safe identity metadata, not an authorization token.
- `granted_scopes` may be displayed only in sanitized form.
- connection status changes are explicit: connected, degraded, expired, revoked, or error.
- connection records and source scopes cannot be resolved across tenants.

### 7.2 Forbidden durable fields

The following must not be present in a Knowledge Connection public view, source record, checkpoint, event, log, or error message:

- access token;
- refresh token;
- API key;
- client secret;
- password;
- full authorization header;
- signed temporary download URL;
- unredacted credential payload.

---

## 8. Knowledge Source Scope

A Knowledge Source Scope identifies what remote data is eligible for synchronization.

Minimum semantic fields:

```text
provider_id
source_kind
remote_scope_id
remote_scope_type
safe_display_name
filters
configuration_version
```

Provider-specific data belongs in a validated, bounded configuration object. It must not become an unstructured secret-bearing dictionary.

Examples:

| Provider | `source_kind` examples | Scope examples |
|---|---|---|
| Microsoft Graph | `drive`, `sharepoint_list`, `mail_folder`, `teams_channel`, `calendar`, `onenote`, `planner` | site, drive, folder, mailbox folder, team/channel, calendar |
| Jira | `issues` | site + projects + validated JQL/filter configuration |
| Confluence | `pages` | site + spaces + optional page roots/labels |
| Power BI | `metadata`, `semantic_model`, `approved_query_snapshot`, `report_export` | workspace, report, model, approved query definition |
| Atlan | `catalog_assets` | domain, asset types, glossary, certification, lineage filters |
| Databricks | `unity_catalog`, `workspace_tree`, `volume_files`, `approved_sql_snapshot`, `change_feed` | metastore/catalog/schema, folder, volume, approved query/table |

### 8.1 Scope safety

A scope must be validated before synchronization. Validation checks include:

- provider identity matches the selected integration;
- remote identifiers are well formed;
- connection has required grants;
- requested resource exists or is intentionally allowed to be temporarily unavailable;
- filters are bounded and safe;
- scope does not contain secrets;
- scope belongs to the same tenant as the connection;
- broad organization-wide selection requires explicit policy approval.

---

## 9. Capabilities

Every integration declares capabilities instead of forcing the consumer to infer behavior from provider names.

Minimum capability dimensions:

```text
supports_incremental_changes
supports_full_inventory
supports_content_fetch
supports_binary_content
supports_rich_text_content
supports_structured_content
supports_acl
supports_tombstones
supports_remote_versions
supports_webhooks
supports_reconciliation
supports_server_side_filtering
supports_item_deep_links
```

Capabilities describe supported behavior, not current connection authorization. A provider may support ACL globally while a specific connection lacks permission to read it.

The consumer must fail closed when a required capability is unavailable. It must not silently claim ACL preservation, delta safety, or deletion handling when the integration cannot provide it.

---

## 10. Knowledge Item Descriptor

The descriptor is the canonical identity and revision envelope returned by inventory or change reads.

Minimum semantic fields:

```text
remote_id
parent_remote_id
item_type
title
web_url
mime_type
version
etag
created_at
updated_at
author
content_mode
content_available
metadata
```

### 10.1 Identity invariant

`remote_id` is the stable provider identity for the remote entity within the connection and source scope.

Canonical application identity is derived from stable origin, for example:

```text
tenant_id + provider_id + connection_id + source identity + remote_id
```

The exact hashing/encoding algorithm is deferred to implementation, but the following are binding:

- content hash is not the stable document identity;
- rename does not create a new item when `remote_id` is unchanged;
- move does not create a new item when `remote_id` is unchanged;
- a content update changes revision state, not identity;
- an ACL-only update changes authorization state, not identity;
- provider IDs are treated as opaque strings unless the provider contract defines normalization.

### 10.2 Revision signals

The integration may expose several revision signals:

```text
version
etag
updated_at
content_hash when cheaply and reliably available
acl_version or acl_hash
metadata_version or metadata_hash
```

The consumer decides whether content must be fetched and reprocessed. Provider-specific ETag semantics must be documented by the concrete integration.

### 10.3 Metadata constraints

Descriptor metadata must be:

- JSON-compatible;
- size bounded;
- free of credentials and signed download URLs;
- stable enough for deterministic comparison;
- namespaced when vendor-specific;
- safe for logs only after normal platform sanitization.

---

## 11. Knowledge Item Content

Content has one explicit mode.

### 11.1 `BINARY`

Used for file-like content requiring a shared parser or controlled binary handling.

Examples:

- DOCX, XLSX, PPTX, PDF, image, text file from SharePoint or OneDrive;
- Jira or Confluence attachment;
- file from a Databricks Volume;
- exported Power BI report.

The integration may return a stream, bounded bytes, or a controlled temporary content handle. Temporary vendor URLs must not become durable product provenance.

### 11.2 `RICH_TEXT`

Used when the provider exposes semantic page or document content.

Examples:

- Confluence page body;
- OneNote page;
- notebook source;
- Teams thread represented with preserved message/thread structure.

Rich text should preserve meaningful structure such as headings, lists, tables, code blocks, links, authorship, and thread boundaries. Regex-based HTML stripping is not an acceptable production normalization strategy.

### 11.3 `STRUCTURED_RECORD`

Used for domain records whose fields are more important than a rendered file.

Examples:

- Jira issue with comments, status, assignee, labels, links, and changelog summary;
- Power BI semantic model metadata;
- Atlan catalog asset, owner, glossary, lineage, and certification;
- Databricks catalog metadata or approved query snapshot;
- Planner task or calendar event.

The integration returns a canonical structured payload plus provider-specific namespaced fields. The consumer owns conversion into an indexable application Document.

### 11.4 Content safety

Content fetch must support:

- bounded size policy;
- media-type declaration;
- safe filename when applicable;
- timeout and cancellation;
- rate-limit propagation;
- content-not-available result distinct from provider failure;
- content revision validation when the provider supports conditional fetches.

---

## 12. Knowledge Item ACL

ACL is a first-class part of the contract, not optional metadata hidden in content.

Minimum semantics:

```text
visibility_mode
allowed_principals
denied_principals
inherited
inheritance_source
acl_version_or_hash
complete
```

### 12.1 Principal normalization

Provider principals are normalized into stable typed identities, for example:

```text
provider:user:<remote-id>
provider:group:<remote-id>
provider:service-principal:<remote-id>
tenant:user:<canonical-id>
tenant:group:<canonical-id>
```

Exact mapping into canonical tenant identities may require the identity-provider integration and remains separate from the vendor adapter.

### 12.2 ACL completeness

`complete=false` means the integration could not prove a complete authorization view. The consumer must apply an explicit fail-closed policy. It must not silently treat missing ACL information as public visibility.

### 12.3 Personal delegated mode

A first product slice may use a delegated personal-source mode in which all ingested content is visible only to the connected user. This is valid only when explicitly modeled and enforced. It must not be presented as organization-wide ACL preservation.

### 12.4 Retrieval invariant

Authorization filtering occurs before source content is provided to an LLM or artifact generator.

```text
candidate retrieval
→ tenant/workspace filter
→ principal/ACL filter
→ ranking/context assembly
→ model
```

Prompt text must never be used as a substitute for access control.

---

## 13. Knowledge Change Page

The incremental read contract returns one bounded page.

Minimum semantics:

```text
items
deleted_items
continuation
proposed_checkpoint
has_more
provider_request_id
rate_limit_state
observed_at
```

### 13.1 Items

Each changed item includes a descriptor and a change classification when the provider can determine it:

```text
created
content_changed
metadata_changed
acl_changed
moved
renamed
unknown_changed
```

The consumer must tolerate `unknown_changed` and perform safe comparison/fetch behavior.

### 13.2 Tombstones

Deleted items contain at least:

```text
remote_id
deletion_kind
observed_at
last_known_parent_remote_id
last_known_title
```

Deletion kinds may include:

```text
deleted
revoked
out_of_scope
inaccessible
```

`revoked` and `inaccessible` are not automatically equivalent to permanent remote deletion. The consumer defines retention and recheck policy, but must immediately prevent unauthorized retrieval where access is no longer proven.

### 13.3 Continuation vs checkpoint

- **Continuation** advances within the current read sequence/page traversal.
- **Proposed checkpoint** represents provider state suitable for a later incremental run after durable completion.

A provider may use the same opaque token for both, but the contract keeps their semantics separate.

---

## 14. Checkpoint ownership and synchronization semantics

### 14.1 Ownership

The integration interprets and emits provider checkpoint data. The consumer persists the committed checkpoint.

The provider adapter must not update an application checkpoint repository internally.

### 14.2 Required processing sequence

```text
1. acquire source lease
2. load committed checkpoint
3. call read_changes(scope, checkpoint, limit)
4. process every returned item and tombstone
5. durably persist item/document/vector/ACL effects
6. durably persist sync-page outcome
7. commit proposed checkpoint
8. release or renew source lease
```

Checkpoint commit before step 5 or 6 is forbidden.

### 14.3 Delivery guarantee

The contract assumes:

```text
at-least-once page delivery
+
idempotent item processing
+
consumer-committed checkpoint
```

A crash after remote read but before checkpoint commit replays the page. Replay must not create duplicate logical Documents, duplicate active vectors, or incorrect ACL state.

### 14.4 Full sync and reconciliation

A provider may require:

- initial full inventory followed by incremental changes;
- periodic full or scoped reconciliation;
- webhook acceleration combined with scheduled delta reads;
- checkpoint reset when a vendor invalidates cursor state.

Webhooks are hints or accelerators unless the provider contract can prove complete durable delivery. They do not replace reconciliation by default.

### 14.5 Invalid checkpoint

When a provider rejects or expires a checkpoint, the integration returns a typed classification. The consumer chooses a policy such as:

- restart scoped inventory;
- mark source degraded and require operator action;
- reconcile from a supported time boundary;
- pause when restart could create unsafe or excessively broad reads.

The adapter must not silently restart from the beginning without communicating the state transition.

---

## 15. Item-state and revision semantics

The future consumer-side item state should distinguish at least:

```text
stable origin identity
last descriptor revision
last content revision/hash
last metadata revision/hash
last ACL revision/hash
last successful processing state
last seen sync run
current deletion/visibility state
```

Required behavior:

| Remote change | Expected consumer effect |
|---|---|
| New item | Fetch required content/ACL and create durable application representation. |
| Content changed | Reprocess content and replace obsolete chunks/vectors safely. |
| Metadata-only change | Update metadata; re-embed only when indexable text actually changes. |
| ACL-only change | Update authorization state without forcing content re-embedding. |
| Rename | Update title/locator while preserving stable identity. |
| Move | Update hierarchy/scope metadata while preserving identity when still in scope. |
| Deleted | Remove or tombstone application knowledge according to retention policy and ensure retrieval exclusion. |
| Access revoked | Immediately deny retrieval; deletion/retention policy may proceed separately. |
| Replayed unchanged item | No duplicate logical Document or active vectors. |

---

## 16. Error taxonomy

The contract needs typed, safe errors. Exact class names are deferred, but semantic categories are frozen.

```text
configuration_error
authentication_required
authentication_expired
authorization_denied
scope_not_found
scope_invalid
item_not_found
content_unavailable
checkpoint_invalid
rate_limited
temporary_provider_failure
permanent_provider_failure
timeout
payload_too_large
unsupported_content
acl_unavailable
protocol_error
```

Each error classification may carry safe fields such as:

```text
provider_id
operation
retryable
retry_after
provider_request_id
safe_detail
```

Errors must not expose tokens, credentials, signed URLs, raw vendor payloads containing secrets, or cross-tenant identifiers.

---

## 17. Security and tenancy invariants

The following are mandatory:

1. Every connection is tenant scoped.
2. Every source binding is tenant and application-workspace scoped.
3. Connection resolution verifies tenant equality.
4. Checkpoints are scoped to one connection and source identity.
5. Remote item identities cannot be used to read content through another tenant or connection.
6. Provider deep links are treated as presentation metadata, not authorization proof.
7. Secrets are resolved only at the integration construction/execution boundary.
8. Public integration views and health responses are secret free.
9. Logs and telemetry use safe provider/request identifiers only.
10. ACL enforcement occurs before model context construction.
11. App-only or service-principal access does not imply that all connected users may see all ingested data.
12. Missing or incomplete ACL information fails according to explicit policy, never implicit public visibility.

---

## 18. Direct API/SDK and MCP boundary

### 18.1 Direct API or SDK

Direct vendor APIs or SDKs are the primary mechanism for:

- initial inventory;
- delta/change feeds;
- deterministic pagination;
- checkpoint/cursor handling;
- item versioning;
- deletion/tombstone detection;
- ACL reads;
- controlled binary download;
- rate-limit handling;
- periodic reconciliation;
- repeatable automated synchronization.

### 18.2 MCP

MCP may be used for:

- live interactive queries;
- provider-exposed resources;
- user-driven tools;
- dynamic context that should not be copied durably;
- approved actions such as creating or updating vendor records;
- exploratory workflows where provider MCP permissions are preserved.

### 18.3 Separation rule

```text
Knowledge Source Integration
→ durable, repeatable synchronization contract

MCP gateway
→ interactive resources/tools/actions contract
```

Both may share a provider and authentication system, but they are not interchangeable. A preview or unavailable MCP server must not block durable synchronization when a stable vendor API exists.

---

## 19. Future LKW integration seam

This architecture intentionally does not modify LKW in the discovery task.

The current LKW direction already provides a channel-neutral intake lifecycle:

```text
Knowledge Input
→ resolve/create Source
→ durable Ingestion Operation
→ queue/worker
→ shared indexing pipeline
```

The future integration seam is:

```text
safe Source Candidate
→ application resolves Knowledge Connection + Knowledge Source Scope
→ creates/resolves CONNECTED_SOURCE
→ Connected Source ingestion/sync processor
→ resolves KnowledgeSource integration by provider/category
→ reads change page
→ stages or normalizes item content
→ invokes shared application indexing service
→ commits checkpoint after durable page completion
```

The vendor integration must not know whether the consumer is LKW, another Intergrax application, a batch service, or a future hosted product.

### 19.1 Future application-owned records

Exact models are deferred, but LKW or a shared platform service will likely require equivalents of:

```text
ConnectedSourceBinding
SourceCheckpoint
SourceItemState
SourceSyncRun
SourceSyncPageResult
```

These records belong outside vendor packages.

### 19.2 Slack boundary

Slack remains a replaceable frontend:

```text
Slack source connect/select/sync command
→ safe application capability
→ opaque candidate / connection / source identity
→ LKW operation
```

Slack must never receive or persist vendor access tokens, raw local paths, client secrets, or temporary signed download URLs.

---

## 20. Vendor implementation direction

The following maps are product direction, not implementation claims.

### 20.1 Microsoft Graph

Separate source kinds may include:

- SharePoint/OneDrive drive and folder content;
- SharePoint lists;
- Outlook mail folders;
- Teams channels and threads;
- calendars;
- OneNote;
- Planner.

The first proof should be SharePoint/OneDrive because it exercises stable remote identity, binary content, delta changes, deletion, move/rename, deep links, and ACL.

Word files remain binary documents parsed by the shared parser. Excel may initially use file parsing; structured workbook/table extraction is a later explicit source mode.

### 20.2 Jira

The knowledge source should support bounded project/JQL scope, cursor pagination, issues, selected comments, selected attachments, links, revisions, and visibility data.

Issue mutation remains in the issue-tracker/action contract. Durable issue ingestion belongs to the knowledge-source contract.

### 20.3 Confluence

The source should support spaces, optional page roots and labels, cursor pagination, versions, hierarchy, attachments, deep links, and permissions.

Page structure, tables, lists, code, and links must be preserved. Simple regex stripping is not an acceptable production content path.

### 20.4 Power BI

Power BI must be modeled through explicit source kinds rather than “all BI data”:

- workspace/report/semantic-model metadata;
- tables, columns, measures, relationships, owners, lineage, sensitivity metadata;
- approved query snapshots;
- optional report export as binary content.

The integration must not automatically execute arbitrary model-generated DAX during ingestion or copy entire data estates into vectors.

### 20.5 Atlan

Atlan is primarily an enterprise metadata and governance source:

- assets and descriptions;
- owners and domains;
- glossary terms;
- classifications and certifications;
- lineage;
- data quality context;
- linked assets.

Durable selected sync uses the knowledge-source contract. Interactive catalog exploration may additionally use MCP.

### 20.6 Databricks

Separate source kinds may include:

- Unity Catalog metadata;
- lineage;
- workspace notebooks/files/folders;
- Volume files;
- approved SQL snapshots;
- explicitly approved change-data-feed use cases.

The knowledge source must remain separate from the existing relational-store category. It must not embed an entire lakehouse by default.

---

## 21. Artifact generation boundary

Knowledge ingestion and artifact generation are separate stages:

```text
knowledge sources
→ authorized retrieval
→ evidence set
→ artifact generation workflow
→ review / approval
→ export or approved action
```

Possible artifacts include:

- email drafts;
- offers and proposals;
- reports;
- documentation;
- contract and policy analyses;
- trend analyses;
- project plans;
- meeting briefs;
- decision memoranda;
- scenarios and recommendations.

Vendor integrations provide evidence or execute separately governed actions. They do not generate product artifacts themselves.

---

## 22. Contract test expectations

Every concrete knowledge-source integration must eventually pass common contract tests covering supported capabilities.

Minimum scenarios:

1. initial full inventory;
2. empty scope;
3. deterministic pagination;
4. incremental resume from checkpoint;
5. replay after crash before checkpoint commit;
6. unchanged replay idempotency;
7. content update;
8. metadata-only update;
9. ACL-only update;
10. rename;
11. move;
12. tombstone/deletion;
13. access revocation;
14. rate limit with retry metadata;
15. timeout and temporary provider failure;
16. expired/revoked authentication;
17. invalid checkpoint;
18. partial page/item failure behavior;
19. unavailable source;
20. content unavailable independently of descriptor availability;
21. cross-tenant connection resolution denial;
22. cross-connection item read denial;
23. credentials absent from public views, logs, and errors;
24. safe deep links and citation provenance;
25. bounded content and metadata handling;
26. reconciliation after missed change notification.

Provider-specific tests add semantics such as Graph delta behavior, Jira JQL paging, Confluence page versions, Power BI metadata scope, Atlan lineage, or Databricks catalog traversal.

---

## 23. Implementation sequence

The implementation sequence remains one narrowly scoped task at a time.

### `KNOWLEDGE-SOURCE-CONTRACT-1`

Introduce the platform category, vendor-neutral models, contract, capability declarations, and contract-level unit tests. No concrete provider and no LKW changes.

### `KNOWLEDGE-CONNECTION-1`

Introduce tenant-scoped connection identity and secret references. No OAuth UI and no vendor-specific authorization flow unless separately scoped.

### `KNOWLEDGE-SYNC-RUNTIME-1`

Introduce consumer-side lease, page processing, checkpoint commit, retry, reconciliation, tombstone, and item-state abstractions. Ownership location must be decided before implementation; it must not be hidden inside vendor packages.

### `MSGRAPH-DRIVE-KNOWLEDGE-1`

Implement SharePoint/OneDrive as the first provider proof with injected transport, delta pagination, item descriptors, binary fetch, tombstones, and ACL behavior.

### `LKW-CONNECTED-SOURCE-1`

After synchronization with the parallel LKW intake work, add application binding, source candidate resolution, connected-source processing, and the first end-to-end Graph-to-LKW proof.

### Later provider tasks

```text
JIRA-KNOWLEDGE-SOURCE-1
CONFLUENCE-KNOWLEDGE-SOURCE-1
MSGRAPH-MAIL-KNOWLEDGE-1
MSGRAPH-TEAMS-KNOWLEDGE-1
POWER-BI-KNOWLEDGE-SOURCE-1
ATLAN-KNOWLEDGE-SOURCE-1
DATABRICKS-KNOWLEDGE-SOURCE-1
```

Each task adds one bounded capability slice and must not expand into a multi-vendor implementation.

---

## 24. Explicit non-goals

This discovery does not:

- implement `IntegrationCategory.KNOWLEDGE_SOURCE`;
- add Python contracts or Pydantic models;
- add provider packages;
- add Microsoft Graph, Jira, Confluence, Power BI, Atlan, or Databricks network clients;
- add OAuth or secret-store flows;
- add a checkpoint repository;
- add a sync scheduler or worker;
- change LKW models or services;
- change Slack behavior;
- change RAG, parsing, chunking, embedding, or vector storage;
- add database migrations;
- promise organization-wide ACL support in the first provider slice;
- claim MCP availability or production readiness for any provider;
- claim that the listed vendor sources are currently implemented.

---

## 25. Acceptance checklist

`KNOWLEDGE-SOURCE-DISCOVERY-1` is complete when review confirms that this document answers all of the following:

- [ ] What belongs to a vendor integration?
- [ ] What belongs to the consuming application/runtime?
- [ ] How are provider identity and integration category separated?
- [ ] How are connection and source scope separated?
- [ ] How is a remote item identified independently of content revision?
- [ ] How are binary, rich-text, and structured content represented?
- [ ] How are pagination and incremental checkpoints represented?
- [ ] Who commits the checkpoint and at what point?
- [ ] How are replay and idempotency handled?
- [ ] How are deletion, revocation, rename, and move distinguished?
- [ ] How is an ACL-only update handled without unnecessary re-embedding?
- [ ] Where are credentials stored and how are they referenced?
- [ ] How is tenant isolation enforced?
- [ ] How is retrieval authorization enforced before model context?
- [ ] How are direct API/SDK synchronization and MCP separated?
- [ ] Where will the future LKW connection occur?
- [ ] How is a vendor-specific RAG pipeline prevented?
- [ ] Which decisions remain explicitly deferred?

---

## 26. Final architectural statement

Intergrax will treat enterprise systems as category-specific Knowledge Source integrations that expose stable items, revisions, content, ACL, changes, and proposed checkpoints. Applications remain responsible for durable synchronization, product source ownership, shared content processing, authorization, retrieval, and artifact workflows.

This separation allows LKW and future applications to use Microsoft Graph, Jira, Confluence, Power BI, Atlan, Databricks, and additional vendors without coupling vendor packages to Slack, LKW, or a duplicate RAG stack.
