# Vendor Knowledge Facade — Implementation Plan

**Status:** `ACTIVE`  
**Branch:** `development`  
**Architecture:** [`../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md)  
**Reuse audit:** [`../audit/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../audit/KNOWLEDGE_SOURCE_INTEGRATIONS.md)  
**LKW intake discovery:** [`../../applications/local_workspace_application/docs/KNOWLEDGE_INTAKE_DISCOVERY.md`](../../applications/local_workspace_application/docs/KNOWLEDGE_INTAKE_DISCOVERY.md)

---

## 1. Objective

Build one platform-level facade above existing category-specific vendor integrations so applications such as Local Knowledge Workspace can consume external enterprise knowledge through one stable, vendor-neutral boundary.

```text
LKW / another application
        |
        v
Vendor Knowledge Facade
        |
        v
source adapter
        |
        v
existing provider/category integration
        |
        v
vendor API
```

Existing integrations remain low-level and authoritative. The facade is not an integration category.

---

## 2. Current position

```text
DONE:     VENDOR-KNOWLEDGE-FACADE-ARCH-1
DONE:     VENDOR-KNOWLEDGE-FACADE-PLAN-1
DONE:     VENDOR-KNOWLEDGE-FACADE-AUDIT-1
DONE:     VENDOR-KNOWLEDGE-FACADE-CONTRACT-1
DONE:     VENDOR-KNOWLEDGE-FACADE-CORE-1
DONE:     VENDOR-KNOWLEDGE-CONNECTION-1
NEXT:     VENDOR-KNOWLEDGE-SYNC-1A
PLANNED:  VENDOR-KNOWLEDGE-SYNC-1B
PLANNED:  JIRA-KNOWLEDGE-ADAPTER-1
PLANNED:  CONFLUENCE-KNOWLEDGE-ADAPTER-1
PLANNED:  MSGRAPH-KNOWLEDGE-READ-SURFACE-1
PLANNED:  MSGRAPH-KNOWLEDGE-ADAPTERS-1
DEFERRED: LKW-CONNECTED-SOURCE-1
```

Current runtime state:

```text
Facade contracts implemented
Adapter registry implemented
IntegrationProfile resolver implemented
Connection-aware resolver implemented
Stateless facade core implemented
Tenant-scoped source bindings implemented
DocumentStore binding repository implemented
Synchronization coordinator not implemented
Vendor adapters not implemented
LKW connected-source bridge not implemented
```

---

## 3. Frozen rules

1. No `knowledge_source` integration category.
2. No duplicate public vendor integrations.
3. Existing provider/category integrations remain the only vendor entrypoints.
4. Vendor integrations own API transport, authentication handoff, provider mapping and category operations.
5. Vendor integrations do not import LKW, RAG or workspace code.
6. The facade is a platform service above integrations.
7. Source adapters are thin mappings over already resolved integration instances.
8. Adapters do not own clients, credentials, persistence or checkpoints.
9. LKW communicates with the facade, not vendor SDKs.
10. Reuse `IntegrationProfile` and the existing integration catalog for integration resolution.
11. Reuse `SecretsStore` for secret material; durable bindings contain opaque references only.
12. Reuse `DocumentStoreTaskQueue`, `DocumentStoreTaskWorker` and `TaskExecutionRegistry` for later asynchronous sync.
13. Reuse provider-neutral `DocumentStore` for later facade persistence.
14. Do not import `ManagedWorkspaceRepository` into platform facade code.
15. Stable remote identity is separate from version, ETag and content hash.
16. One shared parser/chunk/embedding/indexing pipeline remains downstream.
17. ACL must be enforceable before model access.
18. One existing provider/category integration may expose multiple knowledge `source_kind` values through separate thin adapters.
19. All work remains on branch `development`.

---

## 4. Reuse decisions

| Area | Decision |
|---|---|
| Integration resolution | Reuse `IntegrationProfile.resolve()` / `resolve_from_profile()` through an injected resolver port. |
| Integration catalog | Reuse unchanged. Do not create another vendor catalog. |
| Adapter resolution | Add one minimal source-adapter registry keyed by provider, category and source kind. |
| Multi-surface vendors | One provider/category integration may serve several source adapters, for example Microsoft Graph `drive`, `mail`, `calendar`, `teams_chat` and `teams_channel`. |
| Multiple connections | Add tenant-scoped facade bindings above `IntegrationProfile`; the profile itself remains application composition. |
| Secrets | Reuse `SecretsStore`; persist only `connection_ref` / `credential_ref`. |
| Durable work | Reuse DocumentStore-backed queue and worker. |
| Durable state | Add later facade-owned repositories over `DocumentStore`. |
| Errors | Reuse integration errors as causes; expose a safe normalized facade error envelope. |
| LKW repository/runtime | Use as a proven pattern and later convergence point, not as a platform dependency. |

---

## 5. Ownership during parallel work

### Vendor facade track

Owns:

- vendor-neutral contracts;
- integration resolver port;
- source adapter port and registry;
- facade core;
- connection/source binding boundary;
- platform-neutral synchronization coordinator;
- vendor-specific adapters over existing integrations;
- focused contract and unit tests.

### LKW ingest track

Owns:

- Knowledge Intake;
- managed uploads and snapshots;
- Object Storage and staging;
- application operations and workers;
- parser/chunk/embedding/indexing invocation;
- Source → Document ownership;
- Slack file-intake UX.

### Deferred convergence

Deferred until both tracks are stable:

- `WorkspaceSource(CONNECTED_SOURCE)` binding;
- `SOURCE_CANDIDATE` resolution;
- connected-source ingestion processor;
- Slack source-management UX;
- retrieval-time ACL integration.

---

## 6. Implementation roadmap

### Phase 0 — Architecture, plan and reuse audit

#### `VENDOR-KNOWLEDGE-FACADE-ARCH-1`

**Status:** `DONE`

Corrected the architecture:

- rejected a generic knowledge-source category;
- rejected duplicate vendor integrations;
- placed the facade above existing integrations.

#### `VENDOR-KNOWLEDGE-FACADE-PLAN-1`

**Status:** `DONE`

Established ordered phases and the convergence point with LKW.

#### `VENDOR-KNOWLEDGE-FACADE-AUDIT-1`

**Status:** `DONE`

Confirmed reuse of:

- IntegrationProfile/factory/catalog;
- SecretsStore;
- DocumentStore;
- DocumentStoreTaskQueue/Worker;
- TaskExecutionRegistry.

Confirmed gaps:

- vendor-neutral contracts;
- source-adapter registry;
- tenant-scoped connection/source bindings;
- normalized facade errors;
- later checkpoint/lease/item state;
- missing vendor read/change methods.

---

### Phase 1 — Facade contracts

#### `VENDOR-KNOWLEDGE-FACADE-CONTRACT-1`

**Status:** `DONE`

**Purpose:** Define the minimum stable vocabulary and ports without implementing runtime behavior.

**Allowed scope:**

```text
intergrax/runtime/vendor_knowledge/__init__.py
intergrax/runtime/vendor_knowledge/models.py
intergrax/runtime/vendor_knowledge/contracts.py
intergrax/runtime/vendor_knowledge/errors.py
tests/unit/runtime/vendor_knowledge/
```

**Deliverables:**

- tenant-aware source binding reference;
- source scope and capabilities;
- stable remote item identity;
- separate revision/version state;
- page and opaque cursor result;
- binary, rich-text and structured-record content envelope;
- provenance/deep-link data;
- ACL/permission envelope;
- normalized facade error;
- `VendorIntegrationResolver` protocol;
- `VendorKnowledgeAdapter` protocol;
- `VendorKnowledgeFacade` protocol.

**Out of scope:**

- new integration category;
- registry implementation;
- facade implementation;
- vendor code;
- secrets lookup implementation;
- persistence;
- queues/workers;
- checkpoints/retries/leases;
- LKW/RAG changes.

**Acceptance:**

- strict models with `extra="forbid"` or equivalent;
- mandatory tenant identity where state crosses boundaries;
- no secret-bearing fields;
- remote identity separated from revision/content hash;
- explicit content modes;
- deterministic validation;
- focused tests green.

---

### Phase 2 — Facade core and adapter registry

#### `VENDOR-KNOWLEDGE-FACADE-CORE-1`

**Status:** `DONE`

**Dependency:** Phase 1

Implement:

```text
integration resolver adapter over IntegrationProfile
source adapter registry
facade core
fake integration + fake adapter proof
```

Expected flow:

```text
request
→ validate tenant/binding reference
→ resolve existing integration
→ resolve source adapter
→ invoke adapter
→ normalize result/error
```

Acceptance:

- no provider `if/elif` chain;
- no duplicate integration construction;
- duplicate adapter registration rejected;
- unknown adapter fails deterministically;
- cross-tenant request fails closed;
- no network or persistence required for proof.

---

### Phase 3 — Connection and source binding

#### `VENDOR-KNOWLEDGE-CONNECTION-1`

**Status:** `DONE`

Add tenant-scoped binding semantics:

```text
binding_id
tenant_id
provider_id
integration_kind
source_kind
integration reference
connection_ref / credential_ref
validated scope
safe display metadata
status
configuration version
```

Rules:

- no raw tokens or secrets;
- multiple connections/scopes per tenant supported;
- binding resolves exactly one existing integration and one source adapter;
- one Microsoft 365 connection may expose several independently configured source bindings;
- revocation/expiry represented explicitly;
- broad scopes require explicit policy approval.

---

### Phase 4 — Shared synchronization coordinator

#### `VENDOR-KNOWLEDGE-SYNC-1A`

**Status:** `NEXT`

Implement platform-neutral orchestration with fake adapters and repository ports:

- source-level lease;
- checkpoint read;
- bounded page read;
- at-least-once replay;
- remote-item revision state;
- tombstone handling;
- checkpoint commit after durable page completion;
- retry classification;
- reconciliation entrypoint.

The coordinator outputs normalized items to a sink port. It does not parse, chunk, embed or write LKW documents.

#### `VENDOR-KNOWLEDGE-SYNC-1B`

**Status:** `PLANNED`

Wire the coordinator onto:

- `DocumentStoreTaskQueue`;
- `DocumentStoreTaskWorker`;
- `TaskExecutionRegistry`;
- facade-owned `DocumentStore` repositories.

Add retry/backoff only as a scoped extension of existing queue/runtime behavior.

---

### Phase 5 — Vendor proofs

#### `JIRA-KNOWLEDGE-ADAPTER-1`

**Status:** `PLANNED`

Content mode: `STRUCTURED_RECORD`.

Extend the existing Jira integration only where required, then map bounded issue data through a Jira source adapter.

#### `CONFLUENCE-KNOWLEDGE-ADAPTER-1`

**Status:** `PLANNED`

Content mode: `RICH_TEXT`.

Extend the existing Confluence integration only where required, then map pages, versions and visibility through a Confluence adapter.

#### `MSGRAPH-KNOWLEDGE-READ-SURFACE-1`

**Status:** `PLANNED`

Extend the single existing Microsoft Graph collaboration-suite integration/private client boundary with the low-level read behavior required by all approved Microsoft 365 knowledge surfaces.

Approved source kinds:

```text
drive
mail
calendar
teams_chat
teams_channel
```

Shared responsibilities:

- bounded inventory and pagination;
- delta/cursor support where Microsoft Graph provides it;
- stable object identity separated from revision;
- ETag/cTag or equivalent revision information;
- tombstones, deletions and revocations;
- attachment inventory and content retrieval;
- safe provider error and throttling mapping;
- permission and visibility reads where available;
- no LKW, RAG, parser, chunker or embedding imports.

Surface-specific low-level behavior:

- `drive`: SharePoint sites, document libraries, OneDrive drives/folders/files, delta, binary content and permissions;
- `mail`: Outlook folders, messages, conversation/thread metadata, bodies and attachments;
- `calendar`: calendars, events, organizers, attendees, recurrence and online-meeting metadata;
- `teams_chat`: one-to-one and group chats, messages, replies, edits, deletions, attachments and links;
- `teams_channel`: teams, channels, posts, threaded replies, mentions, edits, deletions and attachments.

This task must not create separate public Microsoft integrations for Drive, mail, calendar or Teams. The existing Microsoft Graph integration remains the single provider/category entrypoint.

#### `MSGRAPH-KNOWLEDGE-ADAPTERS-1`

**Status:** `PLANNED`

Add separate thin adapters over the same resolved Microsoft Graph integration:

```text
MsGraphDriveKnowledgeAdapter
MsGraphMailKnowledgeAdapter
MsGraphCalendarKnowledgeAdapter
MsGraphTeamsChatKnowledgeAdapter
MsGraphTeamsChannelKnowledgeAdapter
```

Registry keys:

```text
(ms365_graph, collaboration_suite, drive)
(ms365_graph, collaboration_suite, mail)
(ms365_graph, collaboration_suite, calendar)
(ms365_graph, collaboration_suite, teams_chat)
(ms365_graph, collaboration_suite, teams_channel)
```

Content mapping:

- `drive` → `BINARY` for files, with safe structured metadata for folders and inventory records;
- `mail` → `RICH_TEXT` or `STRUCTURED_RECORD`; attachments may produce separate `BINARY` items;
- `calendar` → `STRUCTURED_RECORD`;
- `teams_chat` → `RICH_TEXT` or `STRUCTURED_RECORD`; attachments may produce separate `BINARY` items;
- `teams_channel` → `RICH_TEXT` or `STRUCTURED_RECORD`; attachments may produce separate `BINARY` items.

Each adapter:

- declares only its own capabilities;
- maps provider records into the canonical facade models;
- receives the already resolved Microsoft Graph integration;
- owns no client, credentials, persistence, checkpoint or retry runtime;
- uses the shared synchronization coordinator;
- remains independent from LKW.

Recommended implementation/proof order inside the Microsoft scope:

```text
1. drive / SharePoint
2. mail
3. teams_channel
4. teams_chat
5. calendar
```

The task is grouped as one Microsoft Graph adapter family, but implementation and verification must preserve independent `source_kind`, scope, cursor and ACL semantics for every surface.

#### `DATABRICKS-KNOWLEDGE-ADAPTER-1`

**Status:** `DEFERRED`

First select one precise source kind: Unity Catalog metadata, workspace tree, volume files or an approved query snapshot.

---

### Phase 6 — LKW convergence

#### `LKW-CONNECTED-SOURCE-1`

**Status:** `DEFERRED`

Dependency:

- facade core stable;
- connection/source binding stable;
- synchronization coordinator stable;
- at least one vendor proof stable;
- LKW managed-file intake stable.

Target flow:

```text
WorkspaceSource(CONNECTED_SOURCE)
→ connected-source binding
→ facade sync coordinator
→ normalized item/content
→ existing LKW ingestion pipeline
→ Document Store + Vector Store
```

No duplicate parsing or embedding path is allowed.

---

### Phase 7 — Slack source management

#### `LKW-SLACK-CONNECTED-SOURCES-1`

**Status:** `DEFERRED`

Add safe source discovery, selection, sync request and status through Slack. Slack remains a replaceable frontend and never receives credentials or unsafe provider locators.

---

## 7. Immediate next action

Implement only:

```text
VENDOR-KNOWLEDGE-SYNC-1A
```

Do not start Jira, Microsoft Graph, Confluence, secrets resolution, LKW bridge or vendor adapters in the same task.
