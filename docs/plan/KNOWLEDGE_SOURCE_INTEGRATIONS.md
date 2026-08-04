# Vendor Knowledge Facade — Implementation Plan

**Status:** `ACTIVE`  
**Branch:** `development`  
**Architecture:** [`../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md)  
**Reuse audit:** [`../audit/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../audit/KNOWLEDGE_SOURCE_INTEGRATIONS.md)  
**LKW intake discovery:** [`../../applications/local_workspace_application/docs/KNOWLEDGE_INTAKE_DISCOVERY.md`](../../applications/local_workspace_application/docs/KNOWLEDGE_INTAKE_DISCOVERY.md)

---

## 1. Objective

Build one platform-level reusable provider foundation above existing category-specific vendor integrations so applications can consume external enterprise knowledge through **three separate consumption modes**:

```text
indexed RAG
durable materialization without RAG
bounded live access
```

Synchronization is a lifecycle mechanism of the durable modes, not a separate fourth consumption mode.

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

Existing integrations remain low-level and authoritative. Vendor Knowledge Facade and Sync Coordinator cover the **durable** path today. Live capability execution remains planned. The facade is not an integration category.

---

## 2. Current position

```text
DONE:     VENDOR-KNOWLEDGE-FACADE-ARCH-1
DONE:     VENDOR-KNOWLEDGE-FACADE-PLAN-1
DONE:     VENDOR-KNOWLEDGE-FACADE-AUDIT-1
DONE:     VENDOR-KNOWLEDGE-FACADE-CONTRACT-1
DONE:     VENDOR-KNOWLEDGE-FACADE-CORE-1
DONE:     VENDOR-KNOWLEDGE-CONNECTION-1
DONE:     VENDOR-KNOWLEDGE-SYNC-1A
DONE:     PLATFORM-DOCUMENT-STORE-CONDITIONAL-1
DONE:     VENDOR-KNOWLEDGE-SYNC-1B
DONE:     JIRA-KNOWLEDGE-ADAPTER-1
DONE:     CONFLUENCE-KNOWLEDGE-ADAPTER-1
DONE:     MSGRAPH-KNOWLEDGE-READ-SURFACE-1
DONE:     VENDOR-KNOWLEDGE-THREE-MODE-REUSE-ARCH-1
DONE:     SLACK-KNOWLEDGE-THREE-MODE-ARCH-1
DONE:     SLACK-KNOWLEDGE-FOUNDATION-1
ACCEPTED:
LKW-CONVERSATION-CONTEXT-ARCH-1
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-2
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A
  through REVIEW-FIX-3
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B
  through REVIEW-FIX-5-REVIEW-CORRECTION-1
READY_FOR_REVIEW:
GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1
MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1
CHANGES_REQUIRED:
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1
  correction under review
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-1
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-1
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-2
IN_PROGRESS:
LKW-SLACK-CONNECTED-SOURCE-1 — CHANGES_REQUIRED
MSGRAPH-KNOWLEDGE-ADAPTERS-1
  DONE:
  MSGRAPH-KNOWLEDGE-ADAPTERS-1A-DRIVE
  MSGRAPH-KNOWLEDGE-ADAPTERS-1B-MAIL
  MSGRAPH-KNOWLEDGE-ADAPTERS-1C-TEAMS-CHANNEL
  DONE:
  MSGRAPH-KNOWLEDGE-ADAPTERS-1D-0-TEAMS-CHAT-REFERENCE-BASED-PAGING-AND-EXACT-MESSAGE-READ
  DONE:
  MSGRAPH-KNOWLEDGE-ADAPTERS-1D-0-TEAMS-CHAT-REFERENCE-READS-REVIEW-FIX-1
  DONE:
  MSGRAPH-KNOWLEDGE-ADAPTERS-1D-TEAMS-CHAT
  DONE:
  MSGRAPH-KNOWLEDGE-ADAPTERS-1D-TEAMS-CHAT-REVIEW-FIX-1
CHANGES_REQUIRED:
MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR
  Calendar acceptance proof complete; awaiting external review
PLANNED:
LKW-CONVERSATION-CONTEXT-1
LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1
SLACK-LIVE-CAPABILITY-1
LKW-SLACK-KNOWLEDGE-PROOF-1
GOOGLE-WORKSPACE-KNOWLEDGE-FOUNDATION-1
GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1
GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1
LKW-GOOGLE-WORKSPACE-CONNECTED-SOURCE-1
LKW-GOOGLE-WORKSPACE-PROOF-1
DEFERRED: LKW-CONNECTED-SOURCE-1
```

`GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1` freezes the complete Google Workspace knowledge architecture and proof-first roadmap so one existing `GoogleWorkspaceCollaborationSuiteIntegration` can support Drive, Docs, Sheets, Calendar, Slides, Mail and Chat through shared provider primitives, separate Vendor Knowledge adapters and provider-neutral LKW consumption immediately after the complete Slack Knowledge vertical. Status: **READY_FOR_REVIEW**. Runtime tasks below remain **PLANNED** — no Google knowledge capability is implemented.

`VENDOR-KNOWLEDGE-THREE-MODE-REUSE-ARCH-1` is the architecture/plan correction that freezes reusable provider foundations and separate consumption lifecycles for indexed RAG, durable materialization and bounded live access. Live capability execution is **not** marked as implemented.

`SLACK-KNOWLEDGE-THREE-MODE-ARCH-1` freezes Slack as a reusable three-mode platform knowledge provider built on the existing `SlackConversationChannelIntegration`, distinguishes Slack-as-frontend from Slack-as-knowledge-source, and reprioritizes the roadmap so the complete Slack Knowledge vertical slice precedes Google Workspace knowledge work. `SLACK-KNOWLEDGE-FOUNDATION-1` platform typed reads, Vendor Knowledge adapter and durable sync proof are **DONE** (membership-correct inventory, root-window scope v2, hardened provider validation). `LKW-CONVERSATION-CONTEXT-ARCH-1` is **ACCEPTED** — provider-neutral Conversation Context Binding with observed-audience validation, binding identity, workspace resolution, thread memory isolation, shared capability boundary and deterministic guards in the LKW application domain. LKW conversation context implementation, shared-channel runtime and live capability remain **not** implemented.

**Execution order (frozen) — parallel tracks:**

```text
DONE:
SLACK-CONVERSATION-RUNTIME-1
LKW Slack frontend and Ask workflow foundations
VENDOR-KNOWLEDGE-THREE-MODE-REUSE-ARCH-1
JIRA-KNOWLEDGE-ADAPTER-1
CONFLUENCE-KNOWLEDGE-ADAPTER-1
Microsoft Graph Drive / Mail / Teams Channel / Teams Chat adapters
SLACK-KNOWLEDGE-THREE-MODE-ARCH-1

DONE:
SLACK-KNOWLEDGE-FOUNDATION-1
LKW-CONVERSATION-CONTEXT-ARCH-1 — ACCEPTED

IN_PROGRESS / CHANGES_REQUIRED:
LKW-SLACK-CONNECTED-SOURCE-1 (LKW-SLACK-CONNECTED-SOURCE-1-REVIEW-FIX-2 — CHANGES_REQUIRED; REVIEW-FIX-3 not accepted; final crash-safe recovery and real indexed Search/Ask proof remain under correction)

THEN (Slack / LKW track):
LKW-CONVERSATION-CONTEXT-1            # LKW-wide prerequisite for shared adapters
LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1
SLACK-LIVE-CAPABILITY-1

JOIN (final Slack proof):
LKW-SLACK-CONNECTED-SOURCE-1
+ LKW-CONVERSATION-CONTEXT-1
+ LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1
+ SLACK-LIVE-CAPABILITY-1
+ LKW-HYBRID-ASK-1
→ LKW-SLACK-KNOWLEDGE-PROOF-1

CURRENT VENDOR KNOWLEDGE / MICROSOFT GRAPH TRACK (independent of Google Workspace):
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1 — CHANGES_REQUIRED, correction under review
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-1 — CHANGES_REQUIRED
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-2 — ACCEPTED
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A — CHANGES_REQUIRED
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-1 — CHANGES_REQUIRED
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-2 — READY_FOR_REVIEW
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B — ACCEPTED through REVIEW-FIX-5-REVIEW-CORRECTION-1
→ MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1 — READY_FOR_REVIEW
→ Microsoft Graph adapter-family audit

INDEPENDENT GOOGLE WORKSPACE TRACK (after LKW-SLACK-KNOWLEDGE-PROOF-1 ACCEPTED):
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
→ GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1E–1G + matching adapters

Google Workspace does not gate reconciliation finalization or Microsoft Calendar acceptance.
Microsoft Calendar work does not gate the independent Google Workspace workstream.

remaining Hybrid Ask and provider packs
```

Microsoft Graph Mail low-level knowledge-read support is complete.

Mailbox folder paging, per-folder immutable-ID message delta, text message
content, normalized participants, attachment inventory and bounded ordinary
file-attachment content reads are implemented.

Removed delta entries remain folder-scoped and are not treated as proof of
global mailbox deletion.

Item attachments, reference-attachment downloads, MIME, raw internet headers
and recursive attached-message expansion are intentionally not implemented.

No Microsoft Vendor Knowledge adapter is exposed yet except Drive, Mail, Teams Channel, Teams Chat and Calendar.

Microsoft Graph Drive Vendor Knowledge adapter (`MSGRAPH-KNOWLEDGE-ADAPTERS-1A-DRIVE`)
is implemented.

Microsoft Graph Mail Vendor Knowledge adapter (`MSGRAPH-KNOWLEDGE-ADAPTERS-1B-MAIL`)
is implemented.

Microsoft Graph Teams Channel Vendor Knowledge adapter
(`MSGRAPH-KNOWLEDGE-ADAPTERS-1C-TEAMS-CHANNEL`) is implemented.

Microsoft Graph Teams Chat Vendor Knowledge adapter
(`MSGRAPH-KNOWLEDGE-ADAPTERS-1D-TEAMS-CHAT`) is implemented.

Microsoft Graph Calendar Vendor Knowledge adapter
(`MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR`) is **CHANGES_REQUIRED** pending external review of `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1`. Durable reconciliation finalization is accepted through Review Fix 5 correction; the Calendar proof covers safe non-primary missing-item finalization.

Drive capability matrix:

```text
source_kind: drive
scope: one known Microsoft Graph drive ID
full_inventory: yes
incremental_changes: yes
reconciliation: yes
binary_content: yes
structured_content: no
permissions: no
tombstones: yes
remote_versions: yes
```

Drive files map to `BINARY` content.

Folders, packages and unknown non-file records are metadata-only descriptors.

Graph `NEXT_PAGE` and `DELTA` continuations are wrapped in adapter-owned
opaque `KnowledgeCursor` values.

Drive permission capability remains false because the current low-level
permission projection explicitly does not prove a complete ACL or complete
inheritance graph.

The existing permission read surface is preserved for a future ACL-contract
task and is not represented as authoritative `KnowledgePermissions`.

Mail capability matrix (`MSGRAPH-KNOWLEDGE-ADAPTERS-1B-MAIL`):

```text
source_kind: mail
scope: one known mailbox user ID plus one known folder ID
full_inventory: yes
incremental_changes: yes
reconciliation: yes
content_fetch: yes
binary_content: no
rich_text_content: no
structured_content: yes
permissions: no
tombstones: yes
remote_versions: yes
```

The initial per-folder Graph delta sequence supplies reconciliation inventory.
The final `DELTA` continuation is the durable incremental checkpoint.

Mail content is `msgraph.mail.message.knowledge.v1` structured JSON. Participants
live in content, not descriptor metadata.

`REMOVED` delta entries mean removed from the synchronized folder view, not
global mailbox deletion.

Attachment presence (`has_attachments`) is preserved in descriptor metadata and
structured content. Attachment inventory and binary bytes remain deferred.
Permissions remain false.

Teams Channel capability matrix (`MSGRAPH-KNOWLEDGE-ADAPTERS-1C-TEAMS-CHANNEL`):

```text
source_kind: teams_channel
scope: one known team ID plus one known channel ID
full_inventory: yes
incremental_changes: no
reconciliation: yes
content_fetch: yes
binary_content: no
rich_text_content: no
structured_content: yes
permissions: no
tombstones: yes, explicit deletedDateTime only
remote_versions: yes
```

Reconciliation traverses root posts one at a time. Replies for the current root
are read immediately before advancing to the next root page. The final
reconciliation checkpoint is a complete adapter-owned cursor.

Exact message content is materialized as `msgraph.teams-channel.message.knowledge.v1`
structured JSON. Tombstones apply only when Graph explicitly returns
`deletedDateTime`; absence from a page is not treated as deletion.

Attachment inventory is included in structured content. Attachment URLs,
embedded card payloads and hosted-content bytes are excluded. Channel member
inventory is not an authoritative ACL projection. No Graph delta, webhook,
subscription or LKW changes are introduced.

Teams Chat capability matrix (`MSGRAPH-KNOWLEDGE-ADAPTERS-1D-TEAMS-CHAT`):

```text
source_kind: teams_chat
scope: one known mailbox-visible chat plus one immutable lastModifiedDateTime window
full_inventory: yes, inside the fixed source window
incremental_changes: no
reconciliation: yes
content_fetch: yes
binary_content: no
rich_text_content: no
structured_content: yes
permissions: no
tombstones: yes, explicit deletedDateTime only
remote_versions: yes
```

Structured schema: `msgraph.teams-chat.message.knowledge.v1`

Flat message collection: yes

Delta: not implemented

Absence-based deletion: not permitted

Attachment inventory: included

Attachment URLs: excluded

Embedded attachment payloads: excluded

Hosted-content bytes: excluded

Authoritative ACL: not implemented

LKW connected-source bridge: not implemented

Live capability layer: not implemented

Live search: not implemented

Calendar capability matrix (`MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR`):

```text
source_kind: calendar
scope: one known mailbox Calendar ID + immutable UTC time window
primary full inventory: yes
primary incremental changes: yes
non-primary full inventory: yes
non-primary incremental changes: no
reconciliation: yes
content: STRUCTURED_RECORD
safe attachment inventory: yes
binary attachment content: no
permissions: no
delta tombstones: primary only
remote versions: yes
```

Structured schema: `msgraph.calendar.event.knowledge.v1`

Primary calendar delta: yes

Non-primary snapshot reconciliation: yes

Attachment inventory: included (bounded)

Attachment bytes: excluded

Authoritative ACL: not implemented

Microsoft Graph Calendar low-level knowledge-read support is complete using
stable Graph v1.0 contracts.

Caller-visible user calendars can be enumerated.

The primary calendar supports full and incremental calendar-view delta
synchronization for an explicit fixed time window.

Every selected user calendar, including non-default and shared caller-visible
calendars, supports a complete paged calendar-view snapshot for an explicit
fixed time window.

A later Vendor Knowledge adapter can use delta for the primary calendar and
full-snapshot reconciliation for other calendars.

Event content, recurring occurrences and exceptions, participants, locations,
recurrence, attachment inventory and bounded ordinary file-attachment content
reads are implemented.

Removed delta entries apply only to the primary calendar view and are not
treated as proof of global event deletion.

No beta Graph endpoint, group calendar, recursive item attachment or reference-attachment download is implemented.

The Microsoft Graph Calendar Vendor Knowledge adapter is **CHANGES_REQUIRED** (`MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR`) pending external review of `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1`, which is **READY_FOR_REVIEW**. Non-primary missing-item detection is implemented and proven only after the final snapshot page. Calendar ACL is not implemented.

Microsoft Graph Teams Chat low-level knowledge-read support is complete using
stable Graph v1.0 contracts.

Caller-visible chats can be enumerated and complete member rosters can be read.

Each selected chat supports complete paged message snapshots for explicit
lastModifiedDateTime windows. Reference-based fixed-window snapshot paging and
exact revision-bound active message reads are implemented for stateless adapter
use. Stable Graph v1.0 chat-message delta is not used.

Deleted messages are recognized only when Graph explicitly returns
deletedDateTime. Absence from a window snapshot is not treated as proof of
global deletion.

Message bodies, senders, mentions, reactions, safe attachment references,
forwarded-message metadata and bounded Teams-hosted content reads are
implemented.

Chat messages are a flat collection. No synthetic chat-replies endpoint is
implemented.

The Microsoft Graph Teams Chat Vendor Knowledge adapter and its review correction
are implemented.

The Teams Chat live capability layer, provider-neutral live search and LKW
connected-source bridge are not implemented.

File attachment URLs are retained only as hidden provider references and are
not downloaded directly. A later Microsoft Vendor Knowledge adapter can resolve
supported SharePoint and OneDrive references through the existing Drive
surface.

No beta Graph endpoint, webhook,
subscription, rich-card semantic renderer or direct external attachment
download is implemented.

Microsoft Graph Teams Channel low-level knowledge-read support is complete
using stable Graph v1.0 contracts.

Caller-visible channels can be enumerated per team and complete effective
member rosters can be read through allMembers.

Each selected channel supports complete paged root-message inventory and
threaded reply reads for explicit root posts. Stable Graph v1.0 channel
message delta is not used.

Deleted root posts and replies are recognized only when Graph explicitly
returns deletedDateTime. Absence from a page is not treated as proof of
global deletion.

Message bodies, senders, mentions, reactions, safe attachment references,
forwarded-message metadata and bounded Teams-hosted content reads are
implemented.

Channel messages use the dedicated replies endpoint for threaded replies.
File attachment URLs are retained only as hidden provider references and are
not downloaded directly.

No beta Graph endpoint, Teams Channel webhook,
subscription, rich-card semantic renderer or direct external attachment
download is implemented.

Current runtime state:

```text
Facade contracts implemented
Adapter registry implemented
IntegrationProfile resolver implemented
Connection-aware resolver implemented
Stateless facade core implemented
Tenant-scoped source bindings implemented
DocumentStore binding repository implemented
Platform-neutral synchronization coordinator implemented
Conditional DocumentStore capability implemented
DocumentStore sync lease repository implemented
DocumentStore checkpoint repository implemented
DocumentStore remote-item state repository implemented
DocumentStoreTaskQueue/Worker wiring implemented
Delivery-ID continuation scheduling implemented
Interrupted-task recovery implemented
Bounded sync-handler retry/backoff implemented
Jira issues knowledge adapter implemented
Confluence pages knowledge adapter implemented
First real vendor facade/coordinator proof implemented
Jira Issues, Confluence Pages, Microsoft Graph Drive, Microsoft Graph Mail, Microsoft Graph Teams Channel and Microsoft Graph Teams Chat Vendor Knowledge adapters implemented.

Microsoft Graph Teams Chat Vendor Knowledge adapter is implemented.
Microsoft Graph Calendar Vendor Knowledge adapter code exists but remains
**CHANGES_REQUIRED** until shared reconciliation-finalization is implemented.
LKW connected-source bridge not implemented
```

Notes after `VENDOR-KNOWLEDGE-SYNC-1B`:

- generic delayed queue scheduling was **not** added;
- retry/backoff is scoped to the Vendor Knowledge sync handler only;
- the sink remains an injected port (`KnowledgeSyncSink`);
- LKW intake / connected-source bridge remains a separate later task.

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
9. LKW and other knowledge-consuming applications use Vendor Knowledge Facade for durable operations and the validated live capability boundary for live operations; they do not call vendor SDKs or provider-specific integration methods directly.
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
20. Every future vendor task description and `READY_FOR_REVIEW` report must include a **THREE-MODE REUSE ASSESSMENT** (documentation evidence; does not require implementing all three modes):

```text
THREE-MODE REUSE ASSESSMENT:
- shared integration:
- shared client/transport:
- shared provider references:
- shared exact-read primitives:
- indexed RAG readiness:
- durable materialization readiness:
- live exact-read readiness:
- live search/query readiness:
- provider primitive gaps:
- durable adapter gaps:
- live capability gaps:
- duplicate client/integration introduced: no
```

---

## 4. Current provider readiness matrix

Status-honest readiness across the three consumption modes. Architecture is not proof of implementation.

### Jira

```text
durable inventory/content foundation: implemented
RAG downstream bridge: not yet connected to LKW
live search/read primitives: existing integration operations available
provider-neutral live capability/executor: not implemented
```

### Confluence

```text
durable inventory/content foundation: implemented
RAG downstream bridge: not yet connected to LKW
live search/read primitives: existing integration operations available
provider-neutral live capability/executor: not implemented
```

### Microsoft Graph Drive

```text
durable adapter: implemented
exact live read foundation: available where an exact item is known
bounded provider-neutral live search: not implemented
LKW bridge: not implemented
```

### Microsoft Graph Mail

```text
durable adapter: implemented
exact live read foundation: available where an exact message is known
bounded provider-neutral live search: not implemented
LKW bridge: not implemented
```

### Microsoft Graph Teams Channel

```text
durable reconciliation adapter: implemented
exact message/thread read foundation: available
provider-neutral live search/discovery capability: not implemented
LKW bridge: not implemented
```

### Microsoft Graph Teams Chat

```text
durable reconciliation adapter: implemented
exact message read foundation: implemented
provider-neutral live search/discovery capability: not implemented
LKW bridge: not implemented
```

### Microsoft Graph Calendar

```text
low-level read foundation: implemented
Vendor Knowledge adapter: CHANGES_REQUIRED — acceptance proof ready for external review
non-primary missing-item detection: implemented; final-page proof ready for external review
provider-neutral live capability: not implemented
LKW bridge: not implemented
```

---

## 5. Reuse decisions

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

## 6. Ownership during parallel work

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

## 7. Implementation roadmap

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

**Status:** `DONE`

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

**Status:** `DONE`

**Prerequisites:**

```text
DONE:     VENDOR-KNOWLEDGE-SYNC-1A
DONE:     PLATFORM-DOCUMENT-STORE-CONDITIONAL-1
DONE:     VENDOR-KNOWLEDGE-SYNC-1B
NEXT:     JIRA-KNOWLEDGE-ADAPTER-1
```

Wired the coordinator onto:

- `DocumentStoreTaskQueue`;
- `DocumentStoreTaskWorker`;
- `TaskExecutionRegistry`;
- facade-owned `DocumentStore` repositories.

Conditional write requirements (from `PLATFORM-DOCUMENT-STORE-CONDITIONAL-1`):

- source lease requires atomic `put_if_absent` / CAS;
- checkpoint commit requires CAS;
- remote delivery marker requires conditional write;
- implementation must fail closed when the resolved `DocumentStore` does not satisfy `ConditionalDocumentStore`;
- do not emulate CAS with ordinary `get()` + `put()`.

Retry/backoff is a bounded Vendor Knowledge handler policy only — not a generic delayed queue scheduler. The sync sink remains an injected port; LKW intake remains separate.

#### Composition modes

```text
Platform owns:
- durable job schema;
- scheduler and canonical worker handler;
- checkpoint/lease/item-state repositories;
- optional standalone tenant-scoped runtime.

Applications own:
- application operation lifecycle;
- queue/worker composition;
- user-facing recovery and status.

LKW must use the application-composition adapter from sync_task
inside its existing runtime and must not start a second
VendorKnowledgeSyncRuntime.
```

#### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1`

**Status:** `CHANGES_REQUIRED` — correction under review

**Review fix:** `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-1` — `CHANGES_REQUIRED`

**Review fix 2:** `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-2` — `READY_FOR_REVIEW`

**Purpose:** Freeze provider-neutral durable reconciliation-finalization semantics so completed snapshot reconciliation can emit deterministic synthetic tombstones for items absent from the synchronized inventory without the checkpoint-failure / `restart=True` nondeterminism present on HEAD.

**Architecture:** [`VENDOR_KNOWLEDGE_RECONCILIATION_FINALIZATION.md`](VENDOR_KNOWLEDGE_RECONCILIATION_FINALIZATION.md)

**Frozen decisions (review-fix-2):**

- separate reconciliation-run state machine (`COLLECTING` → `PAGE_PREPARED` → `FINALIZING` → `COMPLETED`) with fail-closed `RECOVERY_REQUIRED` and controlled terminal `ABORTED`;
- single active slot keyed by `(tenant_id, binding_id)` with `binding_configuration_version` inside the run;
- `PAGE_PREPARED` intent before sink or remote-item state mutation;
- non-circular identity pipeline: `prepared_state_mutation_templates` → `prepared_state_mutations_fingerprint` → `prepared_batch_payload_fingerprint` (excluding `delivery_id`) → `delivery_id`;
- `applied_page_count` and `last_applied_delivery_id` as durable applied-page evidence; `effects_started := applied_page_count > 0`;
- exact private cursor objects in run state (`current_input_cursor`, `prepared_input_cursor`, `prepared_next_cursor`, `prepared_proposed_checkpoint`) with matching fingerprints;
- bounded prepared-intent payload bytes and state-mutation count checked before CAS;
- inspectable sink and item-state delivery receipts with receipt-driven retry (provider reread only when both receipts `ABSENT`);
- `ABORT_PRISTINE` requires `applied_page_count == 0` and pristine receipt proof; `COLLECTING` does not imply pristine;
- sink receipt `UNKNOWN` maps to `DEPENDENCY_UNAVAILABLE`, not retryable, → `RECOVERY_REQUIRED`;
- `FINALIZING` checkpoint idempotency including crash-after-checkpoint-commit recovery;
- operator-safe recovery interface (`RESUME_EXACT`, `FINALIZE_ALREADY_COMMITTED`, `ABORT_PRISTINE`, `REPAIR_REQUIRED`);
- bounded active candidate inventory with count and byte limits;
- incremental sync blocked while an active reconciliation run exists;
- reconciliation tombstones mean `absent_from_completed_synchronized_source_inventory` only.

**Next after acceptance:** `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A` (currently **PLANNED**, blocked pending architecture correction acceptance).

**Blocks:** `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR` (currently **CHANGES_REQUIRED** pending external review). Calendar missing-item detection is implemented and covered by the Calendar acceptance proof.

---

### Phase 5 — Vendor proofs

#### `JIRA-KNOWLEDGE-ADAPTER-1`

**Status:** `DONE`

Content mode: `STRUCTURED_RECORD`.

Extend the existing Jira integration only where required, then map bounded issue data through a Jira source adapter.

Capability matrix (Jira `issues`):

```text
source_kind: issues
scope: one Jira project
content: STRUCTURED_RECORD
full_inventory: yes
reconciliation: yes
incremental_changes: no
permissions: no
tombstones: no
remote_versions: yes
```

Deferred / not implemented for Jira issues adapter:

```text
comments deferred
attachments deferred
custom-field projection deferred
end-user ACL projection deferred
deletion/revocation projection deferred
incremental change-feed deferred
general IssueTracker.search_issues endpoint migration deferred
```

#### `CONFLUENCE-KNOWLEDGE-ADAPTER-1`

**Status:** `DONE`

Content mode: `RICH_TEXT`.

Extend the existing Confluence integration only where required, then map pages, versions and visibility through a Confluence adapter.

Capability matrix (Confluence `pages`):

```text
source_kind: pages
scope: one Confluence space ID
content: RICH_TEXT / Confluence Storage Format
full_inventory: yes
reconciliation: yes
incremental_changes: no
permissions: no
tombstones: no
remote_versions: yes
```

Deferred / not implemented for Confluence pages adapter:

```text
blog posts deferred
attachments deferred
comments deferred
labels deferred
custom content deferred
end-user ACL projection deferred
deletion/revocation projection deferred
incremental change-feed deferred
legacy WikiKnowledge search migration deferred
```

#### `MSGRAPH-KNOWLEDGE-READ-SURFACE-1`

**Status:** `DONE`
Drive metadata, delta, tombstones, bounded binary content and caller-visible
sharing-permission reads are implemented.
The permission response is explicitly not treated as a proven complete
end-user ACL.
Sharing URLs, share IDs and invitation email addresses are not retained.
Download URLs are never persisted.
File conversion is not implemented yet.
Microsoft Graph Drive low-level read support is complete.
Mailbox root and child folder paging is implemented.
Mail folder paging and per-folder message metadata delta are implemented.
Message IDs are requested in ImmutableId format on every delta request.
A removed delta entry means removed from the synchronized folder and is not
treated as proof of global mailbox deletion.
Mail text bodies, participants and attachments are implemented.
Microsoft Graph Calendar low-level read support is complete using stable
Graph v1.0 contracts: primary-calendar incremental delta, per-calendar full
snapshots, bounded event content, participants, locations, recurrence and
bounded file attachments.
Removed primary-calendar delta entries are not treated as proof of global
event deletion.

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

**Status:** `IN_PROGRESS`

`MSGRAPH-KNOWLEDGE-ADAPTERS-1A-DRIVE` is **DONE**.

`MSGRAPH-KNOWLEDGE-ADAPTERS-1B-MAIL` is **DONE**.

`MSGRAPH-KNOWLEDGE-ADAPTERS-1C-TEAMS-CHANNEL` is **DONE**.

`MSGRAPH-KNOWLEDGE-ADAPTERS-1D-0-TEAMS-CHAT-REFERENCE-BASED-PAGING-AND-EXACT-MESSAGE-READ` is **DONE**.

`MSGRAPH-KNOWLEDGE-ADAPTERS-1D-0-TEAMS-CHAT-REFERENCE-READS-REVIEW-FIX-1` is **DONE**.

`MSGRAPH-KNOWLEDGE-ADAPTERS-1D-TEAMS-CHAT` is **DONE**.

`MSGRAPH-KNOWLEDGE-ADAPTERS-1D-TEAMS-CHAT-REVIEW-FIX-1` is **DONE**.

`MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR` is **CHANGES_REQUIRED** pending external review of its completed acceptance proof (`MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1` is **READY_FOR_REVIEW**). Shared reconciliation finalization is accepted through Review Fix 5 correction.

**Next (Microsoft Graph Vendor Knowledge):** `MSGRAPH-KNOWLEDGE-ADAPTERS-1-FAMILY-CLOSEOUT` after external Calendar review (`LKW-SLACK-CONNECTED-SOURCE-1` remains **IN_PROGRESS / CHANGES_REQUIRED** on the LKW track; `LKW-CONVERSATION-CONTEXT-1` is not the next Microsoft Graph task).

**Vendor Knowledge / Microsoft Graph track (independent of Google Workspace):**

```text
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1 — CHANGES_REQUIRED, correction under review
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-1 — CHANGES_REQUIRED
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-2 — ACCEPTED
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A — CHANGES_REQUIRED
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-1 — CHANGES_REQUIRED
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-2 — READY_FOR_REVIEW
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B — ACCEPTED through REVIEW-FIX-5-REVIEW-CORRECTION-1
MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR — CHANGES_REQUIRED
MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1 — READY_FOR_REVIEW
→ Microsoft Graph adapter-family audit
```

Google Workspace does not gate reconciliation finalization or Microsoft Calendar acceptance. Microsoft Calendar work does not gate the independent Google Workspace workstream.

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

- `drive` → `BINARY` for files; metadata-only non-file records for folders and inventory records;
- `mail` → `STRUCTURED_RECORD`; attachment binary content deferred;
- `teams_channel` → `STRUCTURED_RECORD` only; safe attachment inventory included; attachment URLs, embedded payloads, hosted-content bytes and binary attachment materialization excluded;
- `teams_chat` → `STRUCTURED_RECORD` only; safe attachment inventory included; attachment URLs, embedded payloads, hosted-content bytes and binary attachment materialization excluded;
- `calendar` → `STRUCTURED_RECORD`; safe attachment inventory included; attachment bytes deferred;

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
5. calendar — CHANGES_REQUIRED (`MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR`; acceptance proof READY_FOR_REVIEW)
```

Google Workspace remains an independent workstream. Google Workspace does not gate reconciliation finalization or Microsoft Calendar acceptance. Microsoft Calendar work does not gate the independent Google Workspace workstream. After architecture correction acceptance, the next Vendor Knowledge task in the Microsoft Graph adapter family is `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A`.

The complete Slack Knowledge vertical slice (`SLACK-KNOWLEDGE-FOUNDATION-1` → `LKW-CONVERSATION-CONTEXT-ARCH-1` → implementation tracks through `SLACK-LIVE-CAPABILITY-1`; final proof joins `LKW-HYBRID-ASK-1` at `LKW-SLACK-KNOWLEDGE-PROOF-1`) precedes the Google Workspace proof-critical path.

The task is grouped as one Microsoft Graph adapter family, but implementation and verification must preserve independent `source_kind`, scope, cursor and ACL semantics for every surface.

Drive adapter capability matrix (`MSGRAPH-KNOWLEDGE-ADAPTERS-1A-DRIVE`):

```text
source_kind: drive
scope: one known Microsoft Graph drive ID
full_inventory: yes
incremental_changes: yes
reconciliation: yes
binary_content: yes
structured_content: no
permissions: no
tombstones: yes
remote_versions: yes
```

Drive files map to `BINARY` content.

Folders, packages and unknown non-file records are metadata-only descriptors.

Graph `NEXT_PAGE` and `DELTA` continuations are wrapped in adapter-owned
opaque `KnowledgeCursor` values.

Drive permission capability remains false because the current low-level
permission projection explicitly does not prove a complete ACL or complete
inheritance graph.

The existing permission read surface is preserved for a future ACL-contract
task and is not represented as authoritative `KnowledgePermissions`.

#### `DATABRICKS-KNOWLEDGE-ADAPTER-1`

**Status:** `DEFERRED`

First select one precise source kind: Unity Catalog metadata, workspace tree, volume files or an approved query snapshot.

---

### Phase 6 — Post-adapter roadmap (parallel branches)

After the Microsoft Graph adapter-family audit (`MSGRAPH-KNOWLEDGE-ADAPTERS-1`), platform work splits into durable and live branches converging at Hybrid Ask.

#### COMMON PROVIDER FOUNDATION

```text
- existing integrations
- connections
- remote resources
- typed provider references
- exact reads
- inventory/change reads
- safe validation
- normalized errors
- provider capability matrix
```

Planned tasks:

| Task | Purpose |
|---|---|
| `VENDOR-KNOWLEDGE-ADAPTER-FAMILY-AUDIT-1` | Audit adapter-family completeness and gap classification |
| `VENDOR-KNOWLEDGE-THREE-MODE-CAPABILITY-MATRIX-1` | Explicit per-provider, per-source-kind, per-mode capability matrix |

#### DURABLE BRANCH

```text
- source bindings
- Vendor Knowledge adapters
- Sync / Materialization Runtime
- generic durable sink contract
- application/database materialization
- LKW Connected Source bridge
- optional RAG ingestion
```

Planned tasks:

| Task | Purpose | LKW mapping |
|---|---|---|
| `VENDOR-MATERIALIZATION-SINK-CONTRACT-1` | Generic injected durable sink contract beyond DocumentStore proof | `LKW-KNOWLEDGE-LIFECYCLE-1` |
| `LKW-CONNECTED-SOURCE-1` | LKW bridge from Connected Source to facade sync runtime | `LKW-KNOWLEDGE-ACCESS-1`, `LKW-VENDOR-ACCESS-COLLABORATION-1` |

#### LIVE BRANCH

```text
- Live Access Bindings
- typed live capability contracts
- live capability registry
- validated read-only executor
- normalized Live Evidence
- execution receipts
- result/count/byte/time limits
- ephemeral retention default
```

Planned tasks:

| Task | Purpose | LKW mapping |
|---|---|---|
| `VENDOR-LIVE-CAPABILITY-CONTRACT-1` | Typed live capability contracts and registry | `LKW-KNOWLEDGE-ACCESS-1` |
| `VENDOR-LIVE-CAPABILITY-EXECUTOR-1` | Validated read-only executor with bounded limits | `LKW-HYBRID-ASK-1`, `LKW-VENDOR-ACCESS-COLLABORATION-1` |

#### CONVERGENCE

```text
- Knowledge Query Orchestrator
- indexed + live evidence normalization
- Hybrid Ask
- unified provenance
```

Maps to existing LKW product blocks — do not create duplicate LKW roadmap blocks:

```text
LKW-KNOWLEDGE-ACCESS-1
LKW-HYBRID-ASK-1
LKW-VENDOR-ACCESS-COLLABORATION-1
LKW-VENDOR-ACCESS-DATA-1
LKW-KNOWLEDGE-LIFECYCLE-1
```

---

### Phase 7 — LKW convergence

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

### Phase 8 — Slack Knowledge vertical (`SLACK-KNOWLEDGE-THREE-MODE-ARCH-1`)

**Classification:** architecture frozen; platform foundation **IMPLEMENTED**; LKW bridge and live capability **PLANNED**.

One existing `SlackConversationChannelIntegration` is reused across indexed RAG, durable materialization without RAG and bounded live access. LKW application tasks remain outside platform ownership.

#### `SLACK-KNOWLEDGE-FOUNDATION-1`

**Status:** `DONE` (platform)

**Classification:** `IMPLEMENTED` — platform foundation; not LKW bridge or live capability.

Delivered scope includes same-integration bot-token credential model on one `AsyncWebClient`, bot-membership inventory via single `users.conversations` stream, `conversation_kind`-explicit history/reply/exact reads, strict descriptor validation, `thread_broadcast` exclusion from history materialization, and durable sync retry proof for multi-page reply traversal followed by history resumption.

Target scope:

```text
existing Slack integration reuse
typed bounded Slack knowledge read surface
safe conversation inventory
history paging
thread/reply paging
exact message/version reads
edits and explicit deletion semantics
safe attachment inventory
provider-safe references and cursors
Slack Vendor Knowledge Adapter
capabilities
Facade registration
Sync Coordinator durable proof
injected database/store sink proof
three-mode reuse assessment
```

Do not freeze implementation signatures before the repository audit. Do not claim full ACL support unless an authoritative Slack authorization projection is proved.

**User-facing meaning after completion:** The platform can safely read and durably synchronize selected Slack conversations for any Intergrax application. No new Slack command or LKW feature is implied yet.

#### Planned Slack three-mode matrix (`PLANNED` — not implemented)

| Concern | Indexed RAG | Durable materialization | Live access |
|---|---:|---:|---:|
| `SlackConversationChannelIntegration` | reused | reused | reused |
| Slack client, transport, credentials | reused | reused | reused |
| Shared Slack read primitives | reused | reused | reused |
| Slack Vendor Knowledge Adapter | implemented | implemented | not used |
| Slack Live Capability Adapter | not used | not used | required |
| LKW Knowledge Intake / RAG | optional consumer | not required | not automatic |
| Ephemeral evidence | not primary | not primary | required |
| Automatic persistence of live results | no | n/a | forbidden by default |

#### `LKW-SLACK-CONNECTED-SOURCE-1`

**Status:** `IN_PROGRESS / CHANGES_REQUIRED` (LKW application — `LKW-SLACK-CONNECTED-SOURCE-1-REVIEW-FIX-2` **CHANGES_REQUIRED**; `REVIEW-FIX-3` not accepted; HTTP discovery/create/sync scaffold present; final crash-safe recovery and real indexed Search/Ask proof remain under correction)

Application-only use of the completed platform foundation:

```text
workspace Connection
→ approved Slack Remote Resource
→ Indexed Source
→ Vendor Knowledge synchronization
→ LKW Knowledge Intake
→ RAG
```

No new Slack client or provider adapter in LKW.

**User-facing meaning after completion:** The user can attach an approved Slack conversation to an LKW workspace, synchronize it and ask questions about its indexed history.

Connecting a Slack conversation as an Indexed Source does **not** activate the bot in that channel. Activating the bot in a channel does **not** automatically index channel history.

#### `LKW-CONVERSATION-CONTEXT-1`

**Status:** `PLANNED` (LKW application — LKW-wide, not platform Slack ownership)

Provider-neutral durable Conversation Context Bindings, observed-audience validation, workspace audience policy, conversation-level state versus thread-level memory, evidence guards and shared `READ_ONLY_ASK` capability boundary. Canonical architecture: [`CONVERSATION_CONTEXT_ARCHITECTURE.md`](../../applications/local_workspace_application/docs/CONVERSATION_CONTEXT_ARCHITECTURE.md).

#### `LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1`

**Status:** `PLANNED` (LKW application — first provider adapter over generic context layer)

Slack channel/private-channel mention handling (`MENTION_ONLY` activation) over `LKW-CONVERSATION-CONTEXT-1`. Slack-specific event terms (`app_mention`, `message.channels`, etc.) remain in the adapter — not in the LKW core contract.

#### `SLACK-LIVE-CAPABILITY-1`

**Status:** `PLANNED` (platform)

Platform live path using the same integration and shared read primitives:

```text
validated capability
→ bounded Slack read
→ normalized ephemeral evidence
```

No automatic durable persistence.

**User-facing meaning after completion:** Authorized applications can read bounded current Slack information at request time without waiting for a complete durable synchronization.

#### `LKW-SLACK-KNOWLEDGE-PROOF-1`

**Status:** `PLANNED` (LKW application)

**Prerequisites (join):** `LKW-SLACK-CONNECTED-SOURCE-1` + `LKW-CONVERSATION-CONTEXT-1` + `LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1` + `SLACK-LIVE-CAPABILITY-1` + `LKW-HYBRID-ASK-1`. Cannot claim indexed + live combined evidence before Hybrid Ask exists.

Required user proof:

```text
approved user selects a Slack conversation
→ binds it to an LKW workspace
→ synchronization and indexing complete
→ user asks through Slack
→ answer uses indexed Slack evidence
→ bounded live Slack evidence may be included when authorized
→ citations identify safe Slack message/thread provenance
```

**User-facing meaning after completion:** A user asking through Slack can receive one grounded answer combining Slack history, current authorized Slack evidence and other workspace sources — with strict personal/shared audience isolation per [`CONVERSATION_CONTEXT_ARCHITECTURE.md`](../../applications/local_workspace_application/docs/CONVERSATION_CONTEXT_ARCHITECTURE.md).

---

### Phase 9 — Slack source management (frontend)

#### `LKW-SLACK-CONNECTED-SOURCES-1`

**Status:** `DEFERRED`

Add safe source discovery, selection, sync request and status through Slack. Slack remains a replaceable frontend and never receives credentials or unsafe provider locators.

---

### Phase 10 — Google Workspace Knowledge vertical (`GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1`)

**Architecture:** [`../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md) §13.8. **Provider usage:** [`../../intergrax/integrations/providers/collaboration_suite/google_workspace/USAGE.md`](../../intergrax/integrations/providers/collaboration_suite/google_workspace/USAGE.md).

`GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1` is **READY_FOR_REVIEW**. All runtime tasks below are **PLANNED** — no Google knowledge read surface, Vendor Knowledge adapter, live capability or LKW Connected Source is implemented.

One existing `GoogleWorkspaceCollaborationSuiteIntegration` (`provider_id: google_workspace`, category: `collaboration_suite`) is reused across indexed RAG, durable materialization without RAG and bounded live access. Seven independently scoped source kinds: `drive`, `docs`, `sheets`, `calendar`, `slides`, `mail`, `chat`. Drive may discover all Drive-hosted resources; discovery does not determine durable `source_kind` — the platform derives canonical binding kind server-side. The frontend must not choose or override `source_kind`.

**Canonical durable resource ownership:**

| Google resource class | Canonical durable `source_kind` |
|---|---|
| Google-native document (Docs) | `docs` |
| Google-native spreadsheet (Sheets) | `sheets` |
| Google-native presentation (Slides) | `slides` |
| Ordinary uploaded/stored file | `drive` |
| Drive folder / My Drive / Shared Drive scope | `drive` |
| Google Calendar / calendar-event scope | `calendar` |
| Gmail scope | `mail` |
| Google Chat space / conversation scope | `chat` |

**Drive discovery flow:** Drive inventory → inspect authoritative Google resource type → derive canonical `source_kind` server-side → issue provider-neutral Remote Resource candidate → create only the canonical `KnowledgeSourceBinding`.

**Stable resource identity:** `provider_id = google_workspace` + `connection_ref` + canonical Google resource type + stable Google resource ID. Rename/move (where ID preserved) do not change identity. Export/download URL is never identity. The same native Google file must not become unrelated `drive` and `docs`/`sheets`/`slides` durable objects.

**Overlapping-binding policy (first proof):** explicit selected resources only; broad Drive/folder synchronization deferred. Future broad scopes require Option A (reject overlapping binding) or Option B (canonical deduplication record) — Option B not chosen until Vendor Knowledge and LKW ownership models support it safely; until then broad overlapping scopes fail closed or remain deferred.

**Product rationale:** Microsoft 365 proves enterprise-oriented collaboration and document access. Google Workspace lowers the entry barrier for individual testers, small teams and design partners who can authorize their own account. Supporting both proves that the LKW Connected Source architecture is provider-neutral rather than Microsoft-specific. The goal is not connector count — it is one convincing proof over different real-world source shapes and provider ecosystems. Google Workspace is the second strategic collaboration/document ecosystem, not an open-ended commitment to add every available SaaS provider.

#### `GOOGLE-WORKSPACE-KNOWLEDGE-FOUNDATION-1`

**Status:** `PLANNED`

**Prerequisites (activation gates — not satisfied):**

```text
GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1 becomes ACCEPTED (currently READY_FOR_REVIEW)
Google Workspace runtime implementation starts only after LKW-SLACK-KNOWLEDGE-PROOF-1 becomes ACCEPTED (complete Slack Knowledge vertical — currently PLANNED)
canonical Tenant Connection / credential-reference boundary available
SecretsStore-owned credential persistence available
runtime integration rehydration/resolution boundary available
Vendor Knowledge binding, registry and synchronization contracts available
```

Canonical owners: durable tenant Connection Catalog and runtime integration rehydration/resolution — `LKW-KNOWLEDGE-ACCESS-1`. Google Foundation must not introduce another tenant Connection catalog, a Google-only credential database, or OAuth tokens in provider config / bindings / LKW state.

Typed Google Workspace integration configuration; credential-reference resolution; least-privilege credential modes; one shared provider client family; provider request execution boundary; pagination token normalization; provider error normalization; rate-limit and retry classification; stable provider resource references; safe timestamps and revisions; safe display labels; bounded request limits; capability declaration; no LKW imports; no RAG imports; no application workspace concepts.

Credential routes (conceptually separated): individual-user OAuth (preferred for first proof); organization/admin-approved Google Workspace access; service-account or delegated organizational access when justified. Exact OAuth scopes and Google SDK signatures are verified in implementation against current official Google documentation — not frozen here. Secrets remain in Connection/SecretsStore; never in bindings, Remote Resources, LKW Sources, citations or cursors.

#### `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1`

**Status:** `PLANNED`

Extend the single existing `GoogleWorkspaceCollaborationSuiteIntegration` client boundary with typed read behavior per source kind. Substeps:

| Substep | Source kind | Status |
|---|---|---|
| `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1A-DRIVE` | `drive` | `PLANNED` |
| `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1B-DOCS` | `docs` | `PLANNED` |
| `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1C-SHEETS` | `sheets` | `PLANNED` |
| `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1D-CALENDAR` | `calendar` | `PLANNED` |
| `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1E-SLIDES` | `slides` | `PLANNED` (post-proof) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1F-MAIL` | `mail` | `PLANNED` (post-proof) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1G-CHAT` | `chat` | `PLANNED` (post-proof) |

Every read-surface task must define: resource scope; stable identity; revision/change identity; full inventory support; incremental/change-feed support; reconciliation support; exact read support; content mode; attachment behavior; permission capability; tombstone/deletion semantics; pagination/cursor behavior; bounded limits; safe provider errors. Do not claim a capability merely because Google exposes some related API method.

#### `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1`

**Status:** `PLANNED`

Each adapter substep pairs with its read surface and must be independently reviewed before the next surface begins:

| Substep | Source kind | Status |
|---|---|---|
| `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1A-DRIVE` | `drive` | `PLANNED` (proof-critical) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1B-DOCS` | `docs` | `PLANNED` (proof-critical) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1C-SHEETS` | `sheets` | `PLANNED` (proof-critical) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1D-CALENDAR` | `calendar` | `PLANNED` (proof-critical) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1E-SLIDES` | `slides` | `PLANNED` (post-proof) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1F-MAIL` | `mail` | `PLANNED` (post-proof) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1G-CHAT` | `chat` | `PLANNED` (post-proof) |

Thin adapters over the same resolved `GoogleWorkspaceCollaborationSuiteIntegration`:

```text
GoogleWorkspaceDriveKnowledgeAdapter      (1A — proof-critical)
GoogleWorkspaceDocsKnowledgeAdapter       (1B — proof-critical)
GoogleWorkspaceSheetsKnowledgeAdapter     (1C — proof-critical)
GoogleWorkspaceCalendarKnowledgeAdapter   (1D — proof-critical)
GoogleWorkspaceSlidesKnowledgeAdapter     (1E — post-proof)
GoogleWorkspaceMailKnowledgeAdapter       (1F — post-proof)
GoogleWorkspaceChatKnowledgeAdapter       (1G — post-proof)
```

Registry keys:

```text
(google_workspace, collaboration_suite, drive)
(google_workspace, collaboration_suite, docs)
(google_workspace, collaboration_suite, sheets)
(google_workspace, collaboration_suite, calendar)
(google_workspace, collaboration_suite, slides)
(google_workspace, collaboration_suite, mail)
(google_workspace, collaboration_suite, chat)
```

Each adapter receives the resolved integration; owns no client, credentials, persistence, checkpoint or retry runtime; uses the shared synchronization coordinator; remains independent from LKW.

#### `LKW-GOOGLE-WORKSPACE-CONNECTED-SOURCE-1`

**Status:** `PLANNED`

**Prerequisites:** proof-critical Google read surfaces and adapters (Drive, Docs, Sheets, Calendar).

Target flow:

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

Reuses the generic Connected Source implementation proved by Slack. No Google-specific LKW configuration aggregate, mutation engine, indexing pipeline, vector database access or Source table. First Google LKW sources default to `PERSONAL_ONLY`.

#### `LKW-GOOGLE-WORKSPACE-PROOF-1`

**Status:** `PLANNED`

**Prerequisites:** `LKW-GOOGLE-WORKSPACE-CONNECTED-SOURCE-1` + proof-critical adapters + LKW indexed Search/Ask path.

Required first proof:

```text
user connects one Google account
→ selects approved Google resources
→ one Google Doc synchronized
→ one Google Sheet synchronized
→ one Google Calendar resource/event set synchronized
→ optionally one ordinary Drive file synchronized
→ LKW indexes the selected resources
→ Search retrieves provider-derived evidence
→ Ask produces one grounded answer
→ citations identify the correct Google source and resource
→ no Google API call is made by Ask after durable synchronization
```

User-oriented proof demonstrating narrative document, structured spreadsheet, calendar/event data and ordinary stored file — not merely adapter unit tests.

**Proof-first gate (binding — vertically incremental):**

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
→ read surfaces 1E–1G + adapters 1E–1G
```

Google Workspace does not gate reconciliation finalization or Microsoft Calendar acceptance. Microsoft Calendar work does not gate the independent Google Workspace workstream.

Each read surface and its adapter must be independently reviewable before proceeding. The final Google LKW proof may combine Docs, Sheets, Calendar and an optional ordinary Drive file.
