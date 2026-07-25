# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

# Knowledge Source Integrations

**Status:** `FROZEN DISCOVERY / READY_FOR_PLAN_EXECUTION`  
**Task:** `KNOWLEDGE-SOURCE-DISCOVERY-1`  
**Domain:** Integrations  
**Plan:** [`../plan/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../plan/KNOWLEDGE_SOURCE_INTEGRATIONS.md)  
**Integration canon:** [`INTEGRATIONS.md`](INTEGRATIONS.md)  
**LKW intake contract:** [`../../applications/local_workspace_application/docs/KNOWLEDGE_INTAKE_DISCOVERY.md`](../../applications/local_workspace_application/docs/KNOWLEDGE_INTAKE_DISCOVERY.md)

---

## 1. Decision summary

Intergrax will introduce a vendor-neutral **knowledge source integration** boundary for durable, incremental, permission-aware ingestion from external systems such as Microsoft 365, Jira, Confluence, Power BI, Atlan and Databricks.

The boundary is intentionally separate from:

- LKW product orchestration;
- document parsing, chunking and embeddings;
- vector and document persistence;
- conversational surfaces such as Slack or Teams;
- agent-invokable write actions;
- MCP tool invocation.

A vendor integration discovers remote changes, fetches remote content and permissions, and returns typed vendor-neutral records. A platform or product-owned synchronization coordinator decides when a checkpoint is durable and when content is ready for the existing ingestion pipeline.

```text
external vendor
→ KnowledgeSourceIntegrationContract
→ descriptors / content / ACL / proposed cursor
→ platform-owned sync coordinator
→ LKW Knowledge Intake / ingestion processor
→ parser / canonical renderer
→ documents / chunks / vectors
```

**Binding decision:** vendor integrations do not parse, chunk, embed, write vectors, construct prompts, call an LLM, or own product workflow state.

---

## 2. Why the existing categories are insufficient

Intergrax already has operational categories such as:

- `collaboration_suite`;
- `issue_tracker`;
- `wiki_knowledge`;
- `relational_store`;
- `object_storage`.

These categories expose interactive or backend operations such as listing mail, searching issues, reading a wiki page, executing SQL or fetching an object. They do not provide the complete semantics required by durable knowledge synchronization:

- cursor or delta checkpoint handling;
- multi-page change enumeration;
- stable remote item identity;
- content revision identity;
- tombstones and deletion reconciliation;
- ACL retrieval and refresh;
- retry-safe page replay;
- full reconciliation after missed events;
- durable provenance and source deep links.

The new category does not replace existing categories. The same provider may implement multiple category-specific contracts.

```text
ms365_graph:collaboration_suite  → mail/calendar actions
ms365_graph:knowledge_source     → durable Graph-backed knowledge sync

jira:issue_tracker               → create/search/comment/update actions
jira:knowledge_source            → durable issue knowledge sync

databricks:relational_store      → SQL execution
databricks:knowledge_source      → catalog/notebook/lineage knowledge sync
```

One provider may appear in several categories through separate integration classes. A multi-category vendor “monster integration” remains forbidden.

---

## 3. Architecture ownership boundary

### 3.1 Knowledge source integration owns

A knowledge source integration may:

- authenticate through an injected connection or transport;
- validate a vendor-specific source scope;
- enumerate a page of changed or current remote items;
- map vendor payloads into canonical descriptors;
- fetch binary, rich-text or structured content;
- fetch or map source permissions;
- expose vendor rate-limit and health state;
- translate protocol failures into platform integration failures;
- perform protocol-level retry when safe and observable;
- propose the next cursor or checkpoint after a successful page fetch.

### 3.2 Knowledge source integration must not own

It must not:

- import from `applications/` or `agents/`;
- create or update an LKW `WorkspaceSource`;
- create `KnowledgeInput` or `WorkspaceOperation` records;
- decide when a cursor is durably committed;
- parse files or vendor markup into RAG chunks;
- select embedding models or vector stores;
- write documents, chunks or vectors;
- invoke Slack, Teams or another frontend;
- create prompts or call an LLM;
- generate emails, offers, reports or other user artifacts;
- perform agent-visible side effects outside ToolRuntime;
- own product scheduling, queueing or recovery loops.

### 3.3 Platform/product sync coordinator owns

The future coordinator that consumes this contract owns:

- source lease and concurrency control;
- durable sync-run state;
- page processing and replay;
- checkpoint commit after durable item processing;
- content staging;
- document identity and revision replacement;
- parser and canonical renderer selection;
- document/chunk/vector persistence;
- tombstone reconciliation;
- ACL persistence and retrieval filtering;
- product-visible progress and lifecycle events.

---

## 4. Canonical vocabulary

| Term | Meaning |
|------|---------|
| **Knowledge Connection** | Tenant-scoped authorization and transport configuration for one external provider. It stores a safe credential reference, not raw secrets in public product models. |
| **Knowledge Source Scope** | Read-only description of the remote collection selected for synchronization, for example one SharePoint folder, Jira JQL scope or Confluence space. |
| **Knowledge Item Descriptor** | Stable identity and metadata for one remote item before content fetch. |
| **Knowledge Item Content** | Binary, rich-text or structured-record content returned by a provider integration. |
| **Knowledge Change Page** | One replay-safe page of changed/current items, tombstones and a proposed next cursor. |
| **Knowledge Tombstone** | Vendor-neutral deletion or access-loss marker for a stable remote item identity. |
| **Knowledge ACL** | Normalized visibility and principal information required for authorization-aware retrieval. |
| **Provider Cursor** | Opaque provider-specific continuation or delta state. The integration may propose it; the coordinator commits it. |
| **Full Reconciliation** | Periodic comparison against the current provider inventory to recover from lost events, expired cursors or provider inconsistencies. |
| **Source Candidate** | Safe product-facing option that can later resolve to a connector-backed LKW Source. It contains no credential or raw private locator. |

---

## 5. Canonical contract direction

Exact class and field names remain implementation-task decisions, but the semantic shape is frozen.

```python
class KnowledgeSourceIntegrationContract(PlatformIntegrationContract):
    def capabilities(self) -> KnowledgeSourceCapabilities:
        ...

    def validate_scope(
        self,
        scope: KnowledgeSourceScope,
    ) -> KnowledgeSourceScopeValidation:
        ...

    def read_changes(
        self,
        *,
        scope: KnowledgeSourceScope,
        cursor: str | None,
        limit: int,
    ) -> KnowledgeChangePage:
        ...

    def fetch_content(
        self,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeItemContent:
        ...

    def fetch_acl(
        self,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeItemAcl:
        ...
```

### 5.1 Required capabilities

The contract must be able to declare, without attempting network I/O:

- supported source kinds;
- full-list support;
- incremental or delta support;
- tombstone support;
- ACL support;
- binary content support;
- rich-text content support;
- structured-record support;
- deep-link support;
- delegated and/or application authentication;
- webhook/change-notification support;
- maximum or recommended page size;
- whether full reconciliation is required;
- whether a cursor can expire.

Capability declarations are honest metadata, not maturity claims.

### 5.2 Knowledge item descriptor

A descriptor must separate stable identity from revision identity.

```text
provider_id
source_kind
remote_id                 # stable provider identity
remote_parent_id
item_type
title
source_url
mime_type
version / etag
created_at
updated_at
author
content_mode
provider_metadata         # bounded, sanitized, non-secret
```

`remote_id` must remain stable across content edits and ordinary rename/move operations whenever the provider supplies such an identity.

### 5.3 Knowledge item content

Content mode is one of:

```text
BINARY
RICH_TEXT
STRUCTURED_RECORD
```

- **BINARY** is staged under platform policy and passed to the existing parser path.
- **RICH_TEXT** is normalized through a platform-owned structured text renderer.
- **STRUCTURED_RECORD** is rendered deterministically into an inspectable canonical representation before ingestion.

A provider must not implement provider-specific chunking or embedding.

### 5.4 Knowledge change page

A page contains:

```text
items
removed_items
next_page_cursor
proposed_checkpoint
has_more
rate_limit_state
provider_diagnostics
```

`proposed_checkpoint` is not durable merely because the provider returned it.

---

## 6. Checkpoint and replay semantics

The platform synchronization guarantee is:

```text
at-least-once page delivery
+
idempotent item processing
+
checkpoint commit after durable page completion
```

Required behavior:

1. load the last committed cursor;
2. request one bounded change page;
3. persist or process every descriptor and tombstone;
4. durably complete all item outcomes for that page;
5. commit the proposed checkpoint;
6. continue with the next page.

A failed or interrupted page must be replayable without duplicate durable documents or vectors.

The vendor integration must not persist checkpoints itself. It has no authority to decide whether downstream parsing, indexing, ACL persistence or deletion reconciliation succeeded.

A full reconciliation path remains mandatory for providers whose webhooks, delta cursors or search windows can lose history or expire.

---

## 7. Stable document identity direction

The integration layer supplies stable remote identity and revision metadata. The consuming product owns document identity.

Recommended product identity direction:

```text
document_id = hash(tenant_id | workspace_id | source_id | provider_id | remote_id)
revision_id = provider_version_or_etag_or_content_hash
```

A content edit must replace or supersede a previous revision of the same logical document. It must not automatically create an unrelated document identity.

A rename or move with the same provider `remote_id` is metadata change, not a new document.

A permission-only change should not require content re-embedding unless product storage makes that unavoidable.

---

## 8. ACL and authorization boundary

ACL preservation is required before an organization-wide connector may be described as safe.

### 8.1 Supported product modes

**Personal/delegated mode**

- connection belongs to one user identity;
- synchronized knowledge is visible only to that principal or private workspace;
- recommended first production slice.

**Organizational ACL mode**

- application identity may enumerate broader data;
- provider ACLs are normalized and persisted;
- retrieval filters authorization before evidence reaches an LLM;
- unresolved or unsupported permissions fail closed.

### 8.2 Canonical ACL direction

```text
visibility
allowed_users
allowed_groups
denied_users
denied_groups
inherited_from
sensitivity_labels
acl_version
acl_hash
```

The exact principal model remains a later identity/ACL implementation task. Raw provider ACL payloads must not become public integration diagnostics.

### 8.3 Forbidden design

```text
fetch all tenant data with application credentials
→ embed everything
→ rely on prompts or UI to hide unauthorized results
```

Authorization filtering must occur before model context construction.

---

## 9. Knowledge connection direction

Credentials must not be stored in LKW `WorkspaceSource`, `KnowledgeInput`, source candidate metadata or Slack messages.

The platform direction is a tenant-scoped connection record:

```text
connection_id
tenant_id
provider_id
auth_mode
credential_ref
connected_principal
granted_scopes
status
expires_at
safe_display_name
created_at
updated_at
```

`credential_ref` points to an approved secrets or credential mechanism. Public views expose only safe connection state.

Connections are reusable by one or more source scopes only when policy permits it.

---

## 10. Provider mapping

The first provider set is planned as follows.

| Provider | Existing category | Planned knowledge source kinds | First proof priority |
|----------|-------------------|--------------------------------|----------------------|
| `ms365_graph` | `collaboration_suite` | SharePoint/OneDrive drive scope, mail, calendar, Teams, OneNote, Planner, SharePoint lists | SharePoint/OneDrive folder delta |
| `jira` | `issue_tracker` | JQL issue scope, comments, attachments, links and change reconciliation | Selected project/JQL issues |
| `confluence` | `wiki_knowledge` | Space/page-root scope, structured page bodies, attachments and versions | Selected spaces/pages |
| `power_bi` | none | workspace metadata, reports, semantic models, lineage, curated query snapshots | metadata scanner |
| `atlan` | none | assets, glossary, owners, classifications, lineage and quality metadata | metadata/catalog scope |
| `databricks` | `relational_store` | Unity Catalog, workspace tree, notebooks, volumes, lineage and approved SQL snapshots | Unity Catalog metadata |

Provider-specific endpoints, authentication scopes and limitations belong to provider implementation tasks and operator documentation, not this generic contract.

---

## 11. MCP relationship

MCP and durable knowledge synchronization are complementary but different.

### Knowledge source integration

Use for:

- inventory and full sync;
- delta/cursor processing;
- tombstones;
- deterministic retry;
- durable provenance;
- ACL persistence;
- scheduled reconciliation.

### MCP gateway

Use for:

- interactive discovery;
- current point lookups;
- vendor tools and resources;
- agent-invokable actions through ToolRuntime;
- ecosystems where an official MCP server provides useful live capabilities.

```text
durable knowledge ingestion → KnowledgeSourceIntegrationContract
interactive tools/resources → MCP gateway
```

An MCP server may later be wrapped by a knowledge source adapter only if it can satisfy the same deterministic pagination, identity, ACL, deletion and checkpoint contract. MCP availability alone is not sufficient.

---

## 12. LKW integration seam

The current LKW intake architecture already freezes:

```text
Knowledge Input
→ resolve/create durable Source
→ durable Ingestion Operation
→ processor
→ Documents owned by Source
```

The future connector integration should use existing seams rather than introduce a second LKW pipeline.

### 12.1 Source selection seam

```text
safe Source Candidate
→ product-owned candidate resolver
→ connector-backed WorkspaceSource
→ ConnectedSourceBinding
```

The candidate exposed to Slack or another frontend contains a safe opaque identity and label only.

### 12.2 Ingestion processor seam

A future product-owned processor router may select:

```text
managed upload processor
uploaded snapshot processor
connected source processor
web resource processor
```

The connected source processor consumes `KnowledgeSourceIntegrationContract`; the provider integration does not import or instantiate the processor.

### 12.3 Parallel development ownership

Until synchronization of the two branches:

**Vendor/platform branch may modify:**

- this architecture and its plan;
- generic integration contracts and category registration;
- provider packages under the future `knowledge_source` category;
- provider-focused unit and contract tests.

**Vendor/platform branch must not modify:**

- `applications/local_workspace_application/**`;
- `agents/local_indexer/**`;
- `intergrax/rag/**`;
- LKW Knowledge Intake models, services, routes or Slack companion;
- LKW document indexing and vector persistence.

The later integration task rebases the vendor branch on the current LKW branch and implements only the explicitly identified product seams.

---

## 13. Error and retry model

Provider integrations normalize protocol failures into stable categories such as:

```text
configuration_error
authentication_required
authorization_denied
scope_not_found
cursor_expired
rate_limited
provider_unavailable
timeout
invalid_provider_response
item_not_found
content_too_large
acl_unavailable
```

Vendor-level retry is restricted to safe protocol concerns. It must not hide semantic failure or create an independent orchestration loop.

The consuming coordinator decides whether to:

- retry an item;
- replay a page;
- restart from full reconciliation;
- pause a source;
- require reauthorization;
- expose a safe product error.

---

## 14. Observability and safe diagnostics

Required safe diagnostics include:

- provider and integration identity;
- source kind;
- page size and item counts;
- cursor presence, never raw cursor value in public logs;
- elapsed time;
- retry/rate-limit class;
- stable sanitized error code;
- capability and health state.

Diagnostics must not expose:

- access or refresh tokens;
- authorization headers;
- raw provider payloads by default;
- private URLs when policy classifies them as sensitive;
- file bytes or document body;
- ACL member lists in public logs;
- raw cursor/delta tokens.

---

## 15. Explicit non-goals

This discovery does not implement:

- the `knowledge_source` enum/category;
- provider packages;
- OAuth browser flows;
- Slack source-management commands;
- LKW processor routing;
- checkpoint persistence;
- document revision replacement;
- ACL-aware retrieval;
- webhooks or scheduling;
- artifact generation;
- MCP gateway implementation;
- production certification for any vendor.

No existing provider may be described as a shipped knowledge source merely because it has an operational integration package.

---

## 16. Acceptance criteria for the discovery

`KNOWLEDGE-SOURCE-DISCOVERY-1` is complete when:

1. the ownership boundary between vendor integration and LKW ingestion is explicit;
2. the canonical vocabulary is frozen;
3. the contract supports full and incremental synchronization;
4. checkpoint commit authority belongs to the consuming coordinator;
5. stable remote identity is separate from content revision;
6. tombstone and reconciliation semantics are explicit;
7. ACL preservation and fail-closed retrieval are explicit;
8. credentials are separated from LKW Source and frontend state;
9. MCP is classified as complementary, not a replacement for durable sync;
10. provider priorities and the parallel-development boundary are recorded;
11. no LKW code is changed by this task;
12. the implementation plan identifies the first executable code task.

---

## 17. Frozen next step

The next task is:

```text
KNOWLEDGE-SOURCE-CONTRACT-1
```

It introduces only the vendor-neutral platform contract and category wiring, with no concrete provider and no LKW changes.

One-sentence result:

> Intergrax gains one typed integration boundary through which external vendors can expose replay-safe knowledge changes without owning product ingestion, parsing or vector persistence.
