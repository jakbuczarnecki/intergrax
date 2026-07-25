# Vendor Knowledge Facade — Implementation Plan

**Status:** `PLANNED / READY_FOR_REVIEW`  
**Task:** `VENDOR-KNOWLEDGE-FACADE-PLAN-1`  
**Branch:** `development`  
**Architecture (1:1):** [`../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md)  
**Integration canon:** [`../architecture/INTEGRATIONS.md`](../architecture/INTEGRATIONS.md)  
**Integration plan:** [`INTEGRATIONS.md`](INTEGRATIONS.md)  
**LKW intake discovery:** [`../../applications/local_workspace_application/docs/KNOWLEDGE_INTAKE_DISCOVERY.md`](../../applications/local_workspace_application/docs/KNOWLEDGE_INTAKE_DISCOVERY.md)

---

## 1. Objective

Build one platform-level facade above the existing category-specific vendor integrations so that applications such as Local Knowledge Workspace (LKW) can consume external enterprise knowledge through one stable, vendor-neutral boundary.

The facade must reuse the existing integration architecture:

```text
PlatformIntegrationContract
        |
        v
category-specific integration contract
        |
        v
single public provider/category integration
        |
        v
provider API / SDK / transport
```

The new platform layer is placed above it:

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
existing vendor integration
        |
        v
vendor API
```

One-sentence result:

> Existing vendor integrations remain low-level and category-correct; one shared facade normalizes their readable knowledge for synchronization and later ingestion by LKW or another application.

---

## 2. Current position

### Completed foundations

| Area | Status | Evidence / consequence |
|---|---|---|
| Platform integration base | `DONE` | `PlatformIntegrationContract`, integration identity, config, capabilities, health and security posture already exist. |
| Provider category contracts | `DONE` | Existing categories include collaboration suite, issue tracker, wiki knowledge, relational store, object storage and others. |
| Single public provider/category entrypoint | `DONE` | Runtime cutover keeps one public integration class per provider/category. |
| Jira integration | `AVAILABLE` | `JiraIssueTrackerIntegration` exposes issue-tracker behavior. |
| Confluence integration | `AVAILABLE` | `ConfluenceWikiKnowledgeIntegration` exposes wiki behavior. |
| Microsoft Graph integration | `AVAILABLE` | `Ms365GraphCollaborationSuiteIntegration` exposes current collaboration behavior. |
| Databricks integration | `AVAILABLE` | Existing relational-store integration is the low-level provider boundary. |
| LKW Knowledge Intake | `IN PARALLEL` | Managed-file intake, durable Sources, item-level operations and shared indexing pipeline are being developed separately. Baseline supplied for coordination: `6d28174a444ef26ccfaa0e3db6dc8475549b7602`. |
| Facade architecture correction | `DONE` | New `knowledge_source` integration category and duplicate public vendor integrations were rejected. |

### Current implementation state

```text
Architecture corrected
Implementation plan being established
No Vendor Knowledge Facade runtime code implemented
No adapter registry implemented
No synchronization runtime implemented
No vendor knowledge adapter implemented
No LKW connected-source bridge implemented
```

### Current marker

```text
CURRENT: VENDOR-KNOWLEDGE-FACADE-PLAN-1
NEXT:    VENDOR-KNOWLEDGE-FACADE-AUDIT-1
```

The next code task must not begin until the reuse audit has identified the exact existing resolver, registry, binding, secrets and queue mechanisms to reuse.

---

## 3. Frozen architectural rules

The following rules apply to every implementation phase.

1. **No `knowledge_source` integration category.** Knowledge ingestion is a cross-category use case, not the domain identity of every provider.
2. **No duplicate public vendor integration.** Do not create `JiraKnowledgeSourceIntegration`, `ConfluenceKnowledgeSourceIntegration`, `Ms365GraphKnowledgeSourceIntegration` or equivalent parallel entrypoints.
3. **Existing provider/category integration remains authoritative.** Vendor transport, auth handoff, provider errors and category operations remain there.
4. **Facade is a platform service, not a provider integration.** It resolves existing integrations and source adapters.
5. **Adapters are thin mapping components.** They do not own a second vendor client, auth flow, registry or persistence framework.
6. **LKW talks to the facade, not directly to vendor SDKs or provider-specific methods.**
7. **One shared synchronization runtime.** Checkpoints, leases, replay, reconciliation and item state must not be reimplemented independently by every vendor.
8. **One shared ingestion pipeline.** Parsing, structured normalization, chunking, embeddings, Document Store and Vector Store remain downstream shared capabilities.
9. **Vendor item identity is stable and separate from revision.** Content hash must not become the durable document identity.
10. **ACL is enforced before model access.** Prompt instructions are not authorization.
11. **Secrets are referenced, never embedded.** Durable records use opaque connection or credential references only.
12. **All work remains on branch `development`.**

---

## 4. Ownership boundaries during parallel work

### Vendor facade session owns

- facade architecture and implementation plan;
- reuse audit;
- vendor-neutral facade contracts;
- source adapter contract and adapter resolution;
- platform-neutral synchronization orchestration;
- vendor-specific adapters over existing integrations;
- contract and unit tests for these layers.

### LKW ingest session owns

- `KnowledgeIntakeService`;
- managed uploads and uploaded snapshots;
- LKW Object Storage and staging;
- LKW ingestion operations and workers;
- shared parser, chunking, embeddings and indexing invocation;
- LKW Source → Document ownership;
- Slack upload/intake UX.

### Deferred shared integration ownership

The following are intentionally deferred until both tracks are stable:

- binding an LKW `WorkspaceSource(CONNECTED_SOURCE)` to a facade source binding;
- `SOURCE_CANDIDATE` resolution through the facade;
- `ConnectedSourceKnowledgeIngestionProcessor` or equivalent bridge;
- Slack source connection, selection, sync and status UX;
- retrieval-time ACL integration in the LKW query path.

---

## 5. Implementation roadmap

## Phase 0 — Architecture and planning

### `VENDOR-KNOWLEDGE-FACADE-ARCH-1`

**Status:** `DONE`

**Purpose:** Correct the initial architecture and freeze the facade-above-integrations direction.

**Deliverable:**

- `docs/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`

**Accepted decisions:**

- no generic knowledge-source category;
- no duplicate provider integrations;
- existing integrations remain low-level;
- facade and adapters sit above them;
- LKW consumes the facade later.

### `VENDOR-KNOWLEDGE-FACADE-PLAN-1`

**Status:** `CURRENT / READY_FOR_REVIEW`

**Purpose:** Establish the implementation sequence, dependencies, acceptance gates and current position before code changes.

**Deliverable:**

- this implementation plan.

**Acceptance:**

- all phases are ordered;
- parallel-session ownership is explicit;
- current and next tasks are visible;
- no implementation task is ambiguously broad;
- the LKW convergence point is deferred and named.

---

## Phase 1 — Existing-platform reuse audit

### `VENDOR-KNOWLEDGE-FACADE-AUDIT-1`

**Status:** `NEXT`

**Type:** docs-only audit

**Purpose:** Determine exactly which existing platform mechanisms can host facade resolution and synchronization without creating duplicate infrastructure.

**Audit questions:**

1. How are existing provider/category integrations currently resolved from `IntegrationProfile`, bindings and registries?
2. Which resolver must the facade call instead of introducing a second integration registry?
3. Is there an existing generic service/facade registry pattern suitable for source adapters?
4. How should an opaque connection or credential reference resolve through `SecretsStore` and provider config?
5. Which existing queue, worker, task, lease, idempotency and retry mechanisms can be reused?
6. Which existing durable repository patterns can store source bindings, checkpoints and remote item state?
7. Which existing error taxonomy should normalize vendor authentication, rate-limit, timeout and unavailable states?
8. Which current Jira, Confluence, Microsoft Graph and Databricks methods are sufficient, and which low-level read methods are genuinely missing?
9. Where should the facade code live without violating Tier boundaries or importing LKW?
10. Which exact files may be modified by the first implementation slice?

**Required output:**

A concise audit matrix:

| Concern | Existing mechanism | Reuse decision | Proven gap | Later owner |
|---|---|---|---|---|
| integration resolution | ... | reuse / extend | ... | ... |
| adapter resolution | ... | reuse / minimal new | ... | ... |
| credentials | ... | reuse / extend | ... | ... |
| queue/worker | ... | reuse / extend | ... | ... |
| durable state | ... | reuse / extend | ... | ... |
| error mapping | ... | reuse / extend | ... | ... |

**Acceptance gates:**

- no code changes;
- no broad repository refactor proposal;
- every proposed new mechanism is backed by a concrete platform gap;
- exact next-task file scope is identified;
- exact public/private boundary is identified;
- the plan is updated after the audit.

---

## Phase 2 — Facade vocabulary and contracts

### `VENDOR-KNOWLEDGE-FACADE-CONTRACT-1`

**Status:** `PLANNED`

**Dependency:** `VENDOR-KNOWLEDGE-FACADE-AUDIT-1`

**Purpose:** Add the minimum vendor-neutral models and ports required by the facade, without vendor implementations, synchronization persistence or LKW changes.

**Semantic deliverables:**

- source binding/reference;
- source scope;
- source capabilities;
- remote item identity and revision;
- page/cursor result;
- binary, rich-text and structured content envelope;
- provenance/deep-link data;
- ACL/permission envelope;
- normalized facade error;
- source adapter port;
- facade port.

Exact Python names and module paths remain deferred until the audit.

**Must not include:**

- integration-category changes;
- concrete Jira, Confluence, Graph or Databricks code;
- adapter registry;
- checkpoint repository;
- queue worker;
- LKW imports;
- RAG calls.

**Acceptance:**

- models are strict and tenant-aware;
- no secret-bearing field exists;
- stable identity and revision are separate;
- content modes cover binary, rich text and structured records;
- focused contract tests pass.

---

## Phase 3 — Integration resolution, adapter registry and facade core

### `VENDOR-KNOWLEDGE-FACADE-CORE-1`

**Status:** `PLANNED`

**Dependency:** Phase 2

**Purpose:** Resolve an existing integration, select the correct adapter and return a vendor-neutral result through one facade.

**Expected flow:**

```text
facade request
→ validate tenant and source binding
→ resolve existing provider/category integration
→ resolve adapter by provider/category/source kind
→ invoke adapter
→ normalize result/error
→ return facade response
```

**Proof strategy:**

- fake existing integration;
- fake source adapter;
- deterministic adapter registry/resolver;
- no network calls;
- no persistence beyond configuration needed for the proof.

**Acceptance:**

- no `if provider == ...` chain in the facade;
- no duplicate integration construction;
- adapter does not own credentials or a second vendor client;
- cross-tenant resolution fails closed;
- unknown provider/source kind fails deterministically;
- focused unit and contract tests pass.

---

## Phase 4 — Connection and source binding boundary

### `VENDOR-KNOWLEDGE-CONNECTION-1`

**Status:** `PLANNED`

**Dependency:** Phase 3 and audit findings

**Purpose:** Represent the safe relationship between a tenant, an existing integration configuration, a credential/connection reference and a selected remote source scope.

**Required semantics:**

```text
binding_id
 tenant_id
 provider_id
 integration_kind
 source_kind
 integration reference
 opaque connection / credential reference
 validated remote scope
 safe display metadata
 status
 configuration version
```

**Security requirements:**

- no token, secret, password or signed URL in durable/public models;
- tenant-scoped resolution;
- safe public view;
- revocation/expiry state;
- broad source scopes require explicit policy approval.

**Acceptance:**

- binding resolves one existing integration and one adapter;
- secrets remain outside facade state;
- source scope is validated and bounded;
- cross-tenant binding reuse is impossible.

---

## Phase 5 — Shared synchronization runtime

### `VENDOR-KNOWLEDGE-SYNC-1A`

**Status:** `PLANNED`

**Dependency:** Phases 3–4

**Purpose:** Add platform-neutral synchronization orchestration over the facade, initially with fake adapters and a fake durable sink.

**Responsibilities:**

- source lease/concurrency control;
- checkpoint read;
- bounded page read;
- at-least-once replay;
- deterministic item identity;
- durable item/revision state;
- tombstone handling;
- retry classification and backoff handoff;
- checkpoint commit only after durable page completion;
- periodic reconciliation capability.

**Runtime boundary:**

The sync runtime must deliver normalized items to a sink/consumer port. It must not parse, chunk, embed or write directly to LKW stores.

**Acceptance scenarios:**

- initial full read;
- empty source;
- multi-page source;
- crash before checkpoint commit;
- safe page replay;
- unchanged item;
- content update;
- metadata-only update;
- ACL-only update;
- rename/move with stable remote identity;
- tombstone;
- token expiry;
- rate limit;
- partial item failure;
- cross-tenant denial.

### `VENDOR-KNOWLEDGE-SYNC-1B`

**Status:** `PLANNED`

**Purpose:** Add production-aligned queue/worker and durable repository wiring by reusing the exact mechanisms selected by the audit.

**Constraint:** No new queue framework or generic persistence framework may be introduced solely for the facade.

---

## Phase 6 — Three content-mode vendor proofs

The first three vendor proofs intentionally cover the three canonical content modes.

### `JIRA-KNOWLEDGE-ADAPTER-1`

**Status:** `PLANNED`

**Content mode:** `STRUCTURED_RECORD`

**Purpose:** Prove that an existing issue-tracker integration can feed structured project knowledge through the facade without a duplicate Jira integration.

**Scope:**

- one bounded project/JQL source kind;
- issue identity, revision, title, description, status and deep link;
- optional comments/attachments only when explicitly configured;
- pagination and safe replay;
- test transport/fake client first.

### `CONFLUENCE-KNOWLEDGE-ADAPTER-1`

**Status:** `PLANNED`

**Content mode:** `RICH_TEXT`

**Purpose:** Prove hierarchical wiki-page ingestion with structure-preserving body normalization, versions and deep links.

**Scope:**

- bounded spaces/page roots;
- cursor pagination;
- page identity/version;
- headings, lists, tables, code and links preserved;
- attachments represented explicitly;
- ACL capability declared honestly.

### `MSGRAPH-DRIVE-KNOWLEDGE-ADAPTER-1`

**Status:** `PLANNED`

**Content mode:** `BINARY`

**Purpose:** Prove SharePoint/OneDrive file synchronization over the existing Microsoft Graph integration boundary and the later shared LKW parser pipeline.

**Scope:**

- one site/drive/folder source kind;
- stable drive-item identity;
- delta pages;
- rename/move/delete;
- binary download through the existing integration/client;
- ETag/revision and deep link;
- permissions capability where available.

**Vendor-proof acceptance rule:**

Every adapter must extend or reuse the existing provider integration. It must not introduce a second auth configuration, second public integration, parallel vendor client or vendor-specific RAG path.

---

## Phase 7 — LKW convergence

### `LKW-CONNECTED-SOURCE-BRIDGE-1`

**Status:** `DEFERRED UNTIL BOTH TRACKS READY`

**Dependencies:**

- stable LKW Knowledge Intake and indexing path;
- stable Vendor Knowledge Facade contracts;
- stable sync runtime;
- at least one vendor adapter proof.

**Target flow:**

```text
LKW WorkspaceSource(CONNECTED_SOURCE)
→ connected source binding
→ Vendor Knowledge Sync runtime
→ facade
→ adapter
→ existing vendor integration
→ normalized item/content/ACL
→ LKW ingestion sink
→ existing parser or structured normalizer
→ chunking / embeddings / stores
```

**Expected LKW touchpoints:**

- `KnowledgeInputKind.SOURCE_CANDIDATE` for safe source selection;
- `WorkspaceSourceType.CONNECTED_SOURCE` for durable source ownership;
- existing operation/worker lifecycle;
- shared document indexing service;
- existing managed staging/Object Storage for binary files where appropriate.

**Acceptance:**

- LKW contains no vendor SDK calls;
- vendor facade contains no LKW imports;
- every persisted Document belongs to one LKW Source;
- remote identity remains stable across content updates;
- checkpoint commits only after LKW has durably accepted the page;
- ACL information is persisted for retrieval enforcement;
- local/managed-file ingestion regression remains green.

---

## Phase 8 — Slack source management

### `LKW-SLACK-CONNECTED-SOURCES-1`

**Status:** `DEFERRED`

**Purpose:** Let an authorized Slack user discover safe source candidates, connect them to a workspace, request synchronization and inspect status without exposing secrets or raw unsafe locators.

**Direction:**

```text
Slack
→ safe candidate selection
→ LKW public capability
→ facade connection/source binding
→ sync operation
→ channel-neutral status
→ Slack presentation
```

Slack remains a replaceable frontend and must not contain vendor, ingestion or storage logic.

---

## Phase 9 — Additional vendor coverage

### Planned after the first three proofs

- Microsoft Graph mail;
- Microsoft Graph Teams channels;
- Microsoft Graph calendar;
- Microsoft Graph OneNote;
- Microsoft Graph Planner;
- SharePoint lists;
- Power BI metadata and approved semantic/query snapshots;
- Atlan catalog, glossary, lineage and governance context;
- Databricks Unity Catalog, workspace tree, volumes and approved snapshots;
- additional vendors selected by real product demand.

Power BI and Atlan require a separate category-fit decision if the current taxonomy does not represent their primary domain. They must not be forced into a generic knowledge-source category.

---

## Phase 10 — Knowledge-powered product workflows

### Status: `FUTURE PRODUCT LAYER`

After trusted retrieval over connected and uploaded sources is proven, applications may build workflows such as:

- drafting emails and replies;
- generating offers and proposals;
- producing reports and documentation;
- contract and policy analysis;
- project and organizational analysis;
- trend and situation analysis;
- decision memos and scenarios;
- reviewed exports and approved vendor actions.

These workflows sit above retrieval and evidence assembly. They are not responsibilities of vendor integrations, adapters or the facade.

---

## 6. Phase gates

A phase may move to `DONE` only when:

1. its exact scope was implemented without opportunistic refactoring;
2. focused tests pass;
3. existing integration-category behavior remains backward-compatible;
4. no duplicate provider integration/client/registry was introduced;
5. tenant isolation and secret redaction are verified;
6. the implementation plan is updated with status, evidence and the next task;
7. an audit/review is completed before selecting the following task.

---

## 7. Required test strategy

### Contract tests

- strict model validation;
- tenant-aware identifiers;
- secret-bearing fields rejected or absent;
- stable item identity independent from content revision;
- capability declaration;
- normalized error taxonomy;
- safe public views.

### Facade tests

- correct existing integration resolution;
- correct adapter resolution;
- unsupported source kind;
- disabled/unconfigured integration;
- adapter/integration mismatch;
- cross-tenant denial;
- provider error normalization.

### Sync tests

- initial inventory;
- pagination;
- checkpoint resume;
- crash/replay;
- idempotent duplicate page;
- update/unchanged/ACL-only distinctions;
- delete/tombstone;
- reconciliation;
- retryable and terminal failures;
- no checkpoint advancement after partial durable failure.

### Vendor adapter tests

- injected fake transport/client;
- no live network requirement;
- provider response mapping;
- pagination/delta token mapping;
- permissions mapping;
- safe errors and log redaction;
- no LKW imports;
- no separate vendor client construction.

### LKW convergence tests

- connected source → normalized content → existing indexing pipeline;
- binary, rich-text and structured-record paths;
- stable document identity and revision update;
- source-owned deletion cleanup;
- ACL retrieval filter;
- managed-file/local-folder regression.

---

## 8. Explicit non-goals

Do not implement:

- a generic `knowledge_source` integration category;
- duplicate public vendor integrations;
- a second provider registry;
- a second vendor client per adapter;
- vendor-specific parser/chunk/embed pipelines;
- direct vendor SDK calls from LKW;
- direct Qdrant or Document Store access from adapters;
- arbitrary tenant-wide ingestion without enforceable ACL;
- checkpoint commit before durable page completion;
- artifact generation inside integrations or the facade;
- all vendors in one Cursor task.

---

## 9. Current task queue

| Order | Task | Type | Status |
|---:|---|---|---|
| 0 | `VENDOR-KNOWLEDGE-FACADE-ARCH-1` | Docs architecture correction | `DONE` |
| 1 | `VENDOR-KNOWLEDGE-FACADE-PLAN-1` | Docs implementation plan | `CURRENT / READY_FOR_REVIEW` |
| 2 | `VENDOR-KNOWLEDGE-FACADE-AUDIT-1` | Docs reuse audit | `NEXT` |
| 3 | `VENDOR-KNOWLEDGE-FACADE-CONTRACT-1` | Core contracts | `PLANNED` |
| 4 | `VENDOR-KNOWLEDGE-FACADE-CORE-1` | Resolution, adapters, facade proof | `PLANNED` |
| 5 | `VENDOR-KNOWLEDGE-CONNECTION-1` | Secure binding boundary | `PLANNED` |
| 6 | `VENDOR-KNOWLEDGE-SYNC-1A` | Sync semantics with fakes | `PLANNED` |
| 7 | `VENDOR-KNOWLEDGE-SYNC-1B` | Durable queue/repository wiring | `PLANNED` |
| 8 | `JIRA-KNOWLEDGE-ADAPTER-1` | Structured vendor proof | `PLANNED` |
| 9 | `CONFLUENCE-KNOWLEDGE-ADAPTER-1` | Rich-text vendor proof | `PLANNED` |
| 10 | `MSGRAPH-DRIVE-KNOWLEDGE-ADAPTER-1` | Binary vendor proof | `PLANNED` |
| 11 | `LKW-CONNECTED-SOURCE-BRIDGE-1` | LKW convergence | `DEFERRED` |
| 12 | `LKW-SLACK-CONNECTED-SOURCES-1` | Slack UX | `DEFERRED` |

---

## 10. Immediate next task

```text
VENDOR-KNOWLEDGE-FACADE-AUDIT-1
```

One-sentence summary:

> Audit the existing Intergrax integration resolution, registries, profiles, secrets, task execution and durable-state mechanisms so the facade can be implemented by extending proven platform capabilities rather than duplicating them.

The audit must finish by updating this plan with:

- exact reuse decisions;
- proven gaps;
- exact package/file placement;
- exact scope of `VENDOR-KNOWLEDGE-FACADE-CONTRACT-1`;
- any roadmap correction required before code begins.
