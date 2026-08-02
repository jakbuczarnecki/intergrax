# Workspace Knowledge Access - Implementation Contract

**Status:** `READY_FOR_REVIEW`
**Task:** `LKW-KNOWLEDGE-ACCESS-1A-C4 - DURABLE TENANT CONNECTION CATALOG AND CONFIGURATION PERSISTENCE BOUNDARY`
**Prior task:** `LKW-KNOWLEDGE-ACCESS-1A-C3` (commit `6fa2dffc6cecd2a7539dce01d4bf14c1db7d4a5d`)
**Classification:** docs-only architecture-to-implementation contract

**C1 correction (preserved):**

- tenant `KnowledgeSourceBinding` is authoritative for provider/resource identity;
- workspace records are authorization references only;
- aggregate revision is monotonic and CAS-protected via `WorkspaceKnowledgeConfigurationHead`;
- indexed detach is non-destructive (logical disable, indexed data preserved);
- live capabilities require typed `CapabilityEffectV1` metadata (suffix checks are defense-in-depth only).

**C2 correction:**

- document normalized to UTF-8 without BOM;
- one pending writer and one CAS publication point;
- immutable revisioned child rows;
- durable mutation/idempotency record;
- exact pre-commit rollback and post-commit recovery;
- binding status, not Source error status, controls detach;
- every mutation requires aggregate If-Match.

**C3 correction:**

- `RESERVED` mutations have no `target_revision` before writer-slot acquisition;
- semantic no-op success is durably stored as `EXISTING_RESULT`;
- publication recovery is proven from immutable revision rows, not one mutable last-committed pointer;
- staged Sources carry exact mutation ownership (`creation_mutation_id`, `visibility_revision`);
- rollback deletion always uses `delete_if_match`;
- `Idempotency-Key` HTTP header is mandatory for every configuration mutation (not request body).

**C4 correction:**

- all user-managed product configuration that must survive restart is durable;
- four persistence boundaries: durable Database / DocumentStore state (separate platform/tenant and LKW workspace ownership categories), `SecretsStore`, runtime-only state, deployment configuration;
- durable `TenantConnection` is platform-owned (**to be implemented in `LKW-KNOWLEDGE-ACCESS-1C-1`**);
- `KnowledgeConnectionRegistry` is instance-local runtime projection only;
- `IntegrationProfile` is application composition, not a tenant Connection catalog;
- restart rehydration from durable Connections is the target contract;
- `1C` decomposed into `1C-1` (durable catalog + rehydration) and `1C-2` (safe discovery + capability catalog);
- `1F` proof must use a durable Connection reconstructed after restart.

**Architecture:** [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md)
**Implementation plan:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
**Platform integration canon:** [`../../../docs/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../../docs/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md)

---

## 1. Task status and scope

### 1.1 Outcome

Freeze the smallest provider-neutral implementation contract so one existing tenant `connection_ref` can support both an **Indexed Source** and a **read-only Live Access Binding** in one LKW workspace without duplicated credentials, vendor clients or provider-specific LKW pipelines.

### 1.2 In scope

- Repository audit of real LKW, Vendor Knowledge, integration resolution and persistence foundations.
- Typed Pydantic contracts, persistence records, service boundaries and HTTP proposal for the first implementation sequence.
- Bounded decomposition of `LKW-KNOWLEDGE-ACCESS-1` into subtasks `1B`-`1F`.
- Security threat review and observable no-duplication proof design.
- One exact revisioned prepare/publish/recovery protocol for multi-record configuration mutations.

### 1.3 Out of scope

Hybrid Ask, Knowledge Query Orchestrator, live provider execution, Slack UI, new vendor adapters, MCP as domain model, provider-specific LKW tables, second credential stores, second connection registries, second Source systems, production code changes.

### 1.4 Repository note

`intergrax/applications/local_workspace/` does **not** exist. LKW product code lives under `applications/local_workspace_application/`. Tier-1 Vendor Knowledge lives under `intergrax/runtime/vendor_knowledge/`.

---

## 2. Inspected repository foundations

### 2.1 LKW workspace domain and persistence

| Concern | Verified owner |
|---------|----------------|
| Workspace identity | `Workspace.workspace_id` (`uuid.uuid4()` on create) - `applications/local_workspace_application/workspaces/models.py` |
| Tenant identity | `Workspace.tenant_id`; resolved via `resolve_tenant_id()` in `serving/workspace_routes.py` |
| Principal / authorization | Request context `get_request_context(request).tenant_id` preferred; workspace existence checked via `ManagedWorkspaceService.require_workspace()` -> **404** when unknown or cross-tenant |
| Workspace repository | `ManagedWorkspaceRepository` - `workspaces/repository.py` |
| Workspace service | `ManagedWorkspaceService` - `workspaces/service.py` |
| FastAPI routes | `mount_managed_workspace_routes()` - `serving/workspace_routes.py`, prefix `/v1/local_workspace` |
| Public response conventions | `serving/workspace_schemas.py` (`*ResponseV1`, `extra="forbid"` on requests) |
| Error normalization | HTTP `detail` string codes (`not_found`, `workspace_not_found`, domain `error_code` on operations); Ask uses `WorkspaceAskLookupError` -> 404 |
| Idempotency | Deterministic IDs via SHA-256 in `knowledge_intake.py` (`ki:`, `op:`, `src:` prefixes); managed-file and web-url intake have dedicated idempotency conflict types |
| Timestamps | UTC `datetime` on all durable models (`created_at`, `updated_at` where applicable) |
| Repository atomicity | Single-record `DocumentStore.put()`; no cross-entity transactions; compensating deletes in `delete_workspace()` |
| Application wiring | `host/integration_wiring.py`, `host/wiring.py`, `serving/workspace_routes.py` factory block |

**Partition convention (existing):**

```text
lkw.managed_workspace:{tenant_id}:{entity}
lkw.ask_run:{tenant_id}:ask_run
```

### 2.2 Source, Document and Knowledge Intake

| Concern | Verified behavior |
|---------|-------------------|
| `WorkspaceSource` | Durable per-workspace Source; types include `CONNECTED_SOURCE` (**implemented** for Slack via `LKW-SLACK-CONNECTED-SOURCE-1`) |
| `WorkspaceSourceStatus` | `registered`, `syncing`, `processing`, `ready`, `error` - indexing/processing health only; **no** `disabled` or `unavailable` |
| Source ID generation | `uuid.uuid4()` for local folder; deterministic `src:knowledge_input_source:{input_id}` for intake-derived sources |
| Tenant/workspace validation | `KnowledgeIntakeService.accept()` and resolvers check `repository.get_workspace()` |
| Source metadata storage | `ManagedWorkspaceRepository.put_source()` row key `{workspace_id}:{source_id}` |
| Provider correlation | **Not present** on `WorkspaceSource` today; `CONNECTED_SOURCE` path is deferred per `docs/plan/KNOWLEDGE_SOURCE_INTEGRATIONS.md` Section 6 |
| `KnowledgeInputKind` | `managed_file`, `uploaded_folder_snapshot`, `source_candidate`, `web_url` - **no new provider-specific kind required** for first connected-source milestone |
| Document ownership | `WorkspaceDocumentReference` - every indexed document references exactly one `source_id` |
| Operation state | `WorkspaceOperation` + `KnowledgeInput` linked by `operation_id` / `input_id` |

### 2.3 Vendor Knowledge foundation

| Symbol | Role | Reuse for LKW |
|--------|------|---------------|
| `KnowledgeSourceRef` | Tenant-scoped vendor-neutral source identity with optional `connection_ref` | **Direct** - build from tenant `KnowledgeSourceBinding` via `to_source_ref()` |
| `KnowledgeSourceScope` | `remote_scope_id`, `remote_scope_type`, `safe_display_name`, safe `parameters` | **Direct** - maps to Remote Resource scope |
| `KnowledgeScopeInfo` | Output of `inspect_scope` | **Direct** - discovery / inspect projection |
| `KnowledgeSourceBinding` | Tenant-scoped durable sync binding (`connection_ref`, optional `credential_ref`) | **Reuse at tenant layer** - not workspace-scoped; LKW must not duplicate |
| `KnowledgeConnectionRegistry` | Instance-local `(tenant_id, connection_ref) -> integration` | **Direct** - prevents second client when registered; **not** durable catalog; does not load secrets, create clients or persist Connection metadata |
| `ConnectionAwareVendorResolver` | Registry-first resolver with profile fallback | **Direct** |
| `IntegrationProfileVendorResolver` | Profile-only; **rejects** `connection_ref` | Fallback path only |
| `VendorKnowledgeFacadeService` | Durable/indexed read path | **Direct** for indexed sync; not for live |
| `KnowledgeAdapterRegistry` | `(provider_id, integration_kind, source_kind) -> adapter` | **Direct** |
| `DocumentStoreKnowledgeSourceBindingRepository` | Tenant binding persistence | **Do not duplicate** in LKW |

**Implemented adapters (verified):** Jira issues, Confluence pages, MS365 Graph drive/mail/teams_channel.

**Missing (gap):** durable tenant `TenantConnection` catalog (**owned by `LKW-KNOWLEDGE-ACCESS-1C-1`**), `list_source_candidates`, live capability executor, provider-neutral live capability IDs.

### 2.4 Integration and registry foundation

| Path | Role |
|------|------|
| `intergrax/integrations/registry/profile.py` - `IntegrationProfile` | Application composition; `resolve(IntegrationCategory)` returns constructed integration; **not** a tenant Connection database |
| `intergrax/integrations/contracts/secrets_store.py` - `SecretsStore` | Credential storage; secrets never in LKW state |
| `intergrax/tools/registry/runtime.py` - `ToolRegistry` | Tool execution registry (future live path, not LKW domain model) |

**Reusable resolution path (frozen):**

```text
connection_ref + tenant_id + provider_id + integration_kind
-> KnowledgeConnectionRegistry.resolve()   # when registered
-> OR IntegrationProfileVendorResolver     # profile fallback; connection_ref must be None
-> existing integration instance (single)
-> VendorKnowledgeAdapter (indexed) OR LiveCapabilityAdapter (future, platform)
```

LKW configuration stores only `connection_ref`. It never constructs vendor clients.

### 2.5 Representative provider descriptors

| Provider | `provider_id` | `integration_kind` | `source_kind` (examples) | Scope type | Implemented read |
|----------|---------------|--------------------|---------------------------|------------|------------------|
| Jira | `jira` | `issue_tracker` | `jira.issues` | `jira_project` | inventory, content, reconciliation |
| Confluence | `confluence` | `wiki_knowledge` | `confluence.pages` | `confluence_space` | inventory, rich_text content |
| MS365 Graph | `ms365_graph` | `collaboration_suite` | `msgraph.drive`, `msgraph.mail`, `msgraph.teams_channel` | `msgraph_drive`, etc. | delta/incremental reads |
| Google Workspace | `google_workspace` | `collaboration_suite` | `drive`, `docs`, `sheets`, `calendar`, `slides`, `mail`, `chat` | per-surface scope types | **PLANNED** — `GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1` **READY_FOR_REVIEW** |

Capabilities are declared per adapter via `KnowledgeAdapterCapabilities`. Live capability IDs are **planned** in architecture docs, **not implemented** as a registry.

### 2.6 Public API and security conventions (existing LKW)

| Rule | Verified behavior |
|------|-------------------|
| Tenant resolution | Auth context > `X-Tenant-Id` > body > `"default"` |
| Cross-tenant | **404** `not_found`, never 403 for workspace/resource existence |
| Validation errors | HTTP 400/422 with stable string `detail` |
| Pagination | List endpoints use repository `limit` (typically 500-2000); no cursor pagination on workspace lists today |
| Opaque IDs | `workspace_id`, `source_id`, `operation_id`, `run_id` |
| Secret redaction | `SourceSummaryResponseV1` omits raw `path` for list; web URL locators keep private URL `repr=False` |
| Response size | Managed file upload max from settings; Ask `limit` capped at 100 |

### 2.7 ConditionalDocumentStore (verified)

`intergrax/integrations/contracts/document_store.py` exposes `ConditionalDocumentStore` with:

- `put_if_absent`
- `replace_if_match`
- `delete_if_match`

`DocumentStore` does **not** provide multi-record transactions. Configuration mutations must use single-record conditional operations only.

---

## 3. Existing contracts to reuse

| Symbol / path | Purpose in knowledge access |
|---------------|----------------------------|
| `KnowledgeSourceRef` | Canonical vendor-neutral source identity for facade resolution |
| `KnowledgeSourceScope` | Remote resource scope half of identity |
| `KnowledgeScopeInfo` | Safe inspect/discovery projection |
| `KnowledgeConnectionRegistry` + `ConnectionAwareVendorResolver` | Single integration instance per `(tenant_id, connection_ref)` |
| `VendorKnowledgeFacadeService` | Indexed synchronization reads |
| `KnowledgeAdapterRegistry` | Adapter lookup |
| `KnowledgeSourceBinding` + `DocumentStoreKnowledgeSourceBindingRepository` | **Authoritative** tenant-level provider/resource/scope/connection configuration |
| `TenantKnowledgeSourceBindingPort` | Provider-neutral LKW lookup port for tenant bindings (to be implemented in 1B) |
| `ConditionalDocumentStore` | `put_if_absent`, `replace_if_match`, `delete_if_match` - required for configuration mutations |
| `ManagedWorkspaceRepository` | LKW durable state pattern |
| `WorkspaceSource` + `WorkspaceDocumentReference` | One Source owns Documents |
| `KnowledgeIntakeService` idempotency patterns | Deterministic IDs for configuration mutations |
| `resolve_tenant_id()` | Tenant boundary for new routes |
| `IntegrationProfile` | Application-level integration composition |
| `SecretsStore` | Credential ownership |

---

## 4. Existing contracts that must not be duplicated

| Symbol / path | Reason |
|---------------|--------|
| `KnowledgeConnectionRegistry` | Second registry would allow second client construction |
| `KnowledgeSourceBinding` provider identity fields in LKW records | Tenant binding is authoritative; LKW stores only `knowledge_source_binding_ref` authorization |
| `credential_ref` in LKW persistence | Credentials stay in tenant binding / `SecretsStore`; LKW never persists credentials |
| `IntegrationProfile` per workspace | Application composition, not workspace product state |
| `SecretsStore` / credential blobs | Credentials stay in integration foundation |
| `VendorKnowledgeFacadeService` inside LKW routes | LKW calls ports; facade stays Tier-1 |
| Provider-specific LKW models (`JiraWorkspaceSource`, etc.) | Rejected by architecture |
| Generic `POST /execute-provider` | Unbounded provider access |

---

## 5. Ownership matrix

### 5.1 Configuration persistence principle (frozen)

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

Do **not** claim that all LKW configuration lives in one database. Four persistence boundaries apply:

1. durable Database / DocumentStore state, containing separate platform/tenant and LKW workspace ownership categories;
2. `SecretsStore` credentials;
3. runtime-only state;
4. deployment configuration.

See [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md) §4.9.

### 5.2 Persistence and ownership matrix

| Concern | Owner | Verified / decision |
|---------|-------|---------------------|
| Raw credentials and tokens | Integration / `SecretsStore` | Confirmed - not in LKW models; only opaque `credential_ref` in durable records |
| Durable `TenantConnection` catalog | Platform connection foundation | **Gap — `LKW-KNOWLEDGE-ACCESS-1C-1`:** model, repository, service, administrative lifecycle, restart rehydration |
| Runtime integration registration | `KnowledgeConnectionRegistry` | Instance-local runtime projection / cache; **not** durable catalog; **not** administrative source of truth |
| Application integration bootstrap | `IntegrationProfile` | Application-level composition; **not** tenant Connection database, workspace configuration, multi-tenant connector catalog or credential record |
| Vendor API client | Existing integration instance | Confirmed via `IntegrationProfile` / connection registry after rehydration |
| Remote Resource discovery | Vendor Knowledge adapters + future list port | `inspect_scope` exists; `RemoteResourceDescriptorV1` ephemeral by default; list candidates in `1C-2` |
| Workspace connection attachment | **LKW** | New `WorkspaceConnectionAttachment` record — reference + safe cached label only |
| Tenant knowledge source binding | **Vendor Knowledge** (`KnowledgeSourceBinding`) | Authoritative provider/resource/scope/connection; durable; stores `connection_ref` and optional `credential_ref`; LKW references via `knowledge_source_binding_ref` |
| Indexed Source authorization | **LKW** | `WorkspaceIndexedSourceBinding` (authorization reference) + `WorkspaceSource(CONNECTED_SOURCE)` |
| Live Access Binding | **LKW** | New `WorkspaceLiveAccessBinding` |
| Query Policy | **LKW** | New `WorkspaceQueryPolicy` |
| Durable Source | **LKW** | `WorkspaceSource` + `WorkspaceDocumentReference` |
| Documents, chunks, vectors | **LKW** | Existing indexing pipeline |
| Live capability implementation | Shared integration/tool foundation (future) | Not in LKW domain |
| Live execution authorization | LKW + platform policy | LKW binding allowlist + executor gate |
| Live evidence normalization | Provider-neutral boundary (future executor) | Ephemeral by default |
| Slack presentation | Slack frontend only | Confirmed thin-client architecture |
| Deployment topology | Deployment configuration | Environment variables, manifests, bootstrap — not automatically tenant/workspace product configuration |

**Current gap (documented, sequenced):** Architecture describes a durable tenant `TenantConnection` record. Repository has only opaque `connection_ref` on bindings and an **instance-local** `KnowledgeConnectionRegistry`. **Owner:** `LKW-KNOWLEDGE-ACCESS-1C-1`. LKW never persists Connection metadata beyond cached safe labels on attachments.

---

## 6. Connection implementation decision

### 6.1 What is a Connection in the current codebase?

A **Connection** is not a standalone durable LKW entity today. It is:

1. An opaque **`connection_ref`** string carried on `KnowledgeSourceRef` / `KnowledgeSourceBinding`.
2. A runtime registration in **`KnowledgeConnectionRegistry`** mapping `(tenant_id, connection_ref)` to an already-constructed integration instance with matching `provider_id` and `integration_kind`. The registry stores constructed integration objects only; it does not load secrets, create clients or persist Connection metadata. It is **not** durable state.
3. Application **`IntegrationProfile`** bootstrap for deployment-level integration composition — **not** a tenant Connection database.
4. An architectural durable tenant `TenantConnection` record documented here and in platform canon — **not yet persisted** in code; **to be implemented in `LKW-KNOWLEDGE-ACCESS-1C-1`**.

### 6.1.1 Target durable `TenantConnection` (platform-owned — not implemented)

```text
connection_ref
tenant_id
provider_id
integration_kind
safe_display_name
administrative_status      # ACTIVE | DISABLED | REVOKED
credential_ref
validated_secret_free_config
configuration_version
created_at
updated_at
connected_principal_ref      # optional
```

**Identity:** `(tenant_id, connection_ref)`.

**Platform connection foundation owns (to be implemented in `LKW-KNOWLEDGE-ACCESS-1C-1`):** `TenantConnection` model, `TenantConnectionRepository` port, durable repository implementation, `TenantConnectionService`, administrative lifecycle, configuration-version concurrency, `credential_ref` association, safe public projection, restart reconstruction contract.

**`KnowledgeConnectionRegistry` owns:** instance-local mapping `(tenant_id, connection_ref) -> constructed integration instance`; runtime identity validation; runtime resolution. Documented as runtime projection / cache — not durable catalog, not administrative source of truth.

**`IntegrationProfile` owns:** application-level integration composition; bootstrap defaults; category-to-provider selection; construction of application infrastructure. Must not be reused as tenant Connection database, workspace configuration, multi-tenant connector catalog or credential record.

### 6.2 LKW representation (frozen)

LKW persists **only**:

- `connection_ref` (required, opaque, non-empty)
- optional cached `safe_display_label` on `WorkspaceConnectionAttachment`

LKW does **not** persist a Connection entity, `credential_ref`, tokens or integration configuration.

### 6.3 Frozen Connection projection fields (read-only API)

```python
class SafeConnectionSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connection_ref: str = Field(..., min_length=1, max_length=128)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    provider_id: str = Field(..., min_length=1, max_length=64)
    integration_kind: IntegrationCategory
    safe_display_label: str = Field(..., min_length=1, max_length=256)
    status: Literal["available", "degraded", "unavailable"]
    supported_source_kinds: tuple[str, ...] = ()
    read_only: Literal[True] = True  # first milestone
```

**Validation:** `connection_ref` must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`.
**Serialization:** never emit `credential_ref`, tokens or endpoint URLs with embedded credentials.

### 6.4 Identity, ownership and lifecycle

| Field | Source |
|-------|--------|
| `connection_ref` | Issued by platform connection administration (out of LKW) |
| `tenant_id` | Must match resolver tenant; cross-tenant ref -> fail closed |
| `provider_id` / `integration_kind` | From platform Connection metadata |
| Health / capability projection | Platform port + adapter registry; runtime health (`available`/`degraded`/`unavailable`) distinct from durable `administrative_status` |
| Delete semantics | Removing LKW attachment does not delete tenant Connection |
| Unavailable | `status=unavailable` -> discovery and binding mutations rejected; existing bindings -> `unavailable` state, no credential copy |

---

## 7. Remote Resource contract

### 7.1 Durability

| Contract | Durability |
|----------|------------|
| `RemoteResourceDescriptorV1` | **Ephemeral** discovery output |
| Optional future `RemoteResourceSnapshotV1` | **Not in 1B** - defer cached snapshots |

Remote Resource never auto-becomes Indexed Source, Live Access Binding or Document.

### 7.2 Model

```python
class RemoteResourceAvailabilityV1(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"


class RemoteResourceDescriptorV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connection_ref: str = Field(..., min_length=1, max_length=128)
    remote_resource_id: str = Field(..., min_length=1, max_length=256)
    provider_id: str = Field(..., min_length=1, max_length=64)
    integration_kind: IntegrationCategory
    source_kind: str = Field(..., min_length=1, max_length=64)
    resource_type: str = Field(..., min_length=1, max_length=64)
    safe_display_label: str = Field(..., min_length=1, max_length=256)
    safe_description: str = Field(default="", max_length=1024)
    availability: RemoteResourceAvailabilityV1
    supported_capability_ids: tuple[str, ...] = ()
    parent_scope_ref: str | None = Field(default=None, max_length=256)
    discovered_at: datetime
    snapshot_version: str = Field(..., min_length=1, max_length=64)
```

**Identity:** `(connection_ref, remote_resource_id, source_kind)` within tenant.
**Stable identity:** provider `remote_resource_id` + `source_kind`; renames update `safe_display_label` only.
**Pagination:** cursor `next_page_token: str | None`, `limit` 1-100, opaque token max 4096 chars.
**Permission loss:** `availability=permission_denied`; bindings transition to `unavailable`, execution fail closed.
**Duplicate discovery:** dedupe by identity; deterministic sort by `(connection_ref, remote_resource_id, source_kind)`.
**Unsafe metadata:** map to `KnowledgeSourceScope.parameters` rules - secret keys forbidden, URL credential embedding forbidden.

**Implementation mapping:** build from `KnowledgeScopeInfo` + adapter-specific list operations when added in `1C-2`.

---

## 8. Indexed Source contract

### 8.1 Decision

**Dedicated workspace authorization** `WorkspaceIndexedSourceBinding` that references one tenant `KnowledgeSourceBinding` and one durable `WorkspaceSource` with `source_type=CONNECTED_SOURCE`.

LKW must **not** duplicate provider identity, connection scope, credentials or remote scope configuration from the tenant binding. The tenant binding is the single authoritative provider-resource definition for durable/indexed access.

### 8.2 Tenant binding reference (frozen)

| Aspect | Frozen decision |
|--------|-----------------|
| Field name | `knowledge_source_binding_ref` |
| Field type | `str`, `min_length=1`, `max_length=128` |
| Format | Opaque binding ID matching `KnowledgeSourceBinding.binding_id` |
| Tenant ownership | Lookup via `TenantKnowledgeSourceBindingPort.get_binding(tenant_id, binding_id)`; cross-tenant -> safe `None` -> 404 |
| Not-found | 404 `knowledge_source_binding_not_found` |
| Status validation | Only `KnowledgeSourceBindingStatus.ACTIVE` permits create; `DISABLED`/`REVOKED`/`EXPIRED` -> 400 `knowledge_source_binding_unavailable` |
| Corrupt record | Port raises or returns invalid -> 400 `knowledge_source_binding_invalid` |
| Version relationship | LKW stores tenant binding ref only; tenant `configuration_version` is not copied into LKW records |
| Credential handling | LKW never persists `credential_ref`; resolved at sync time through tenant binding |

```python
class TenantKnowledgeSourceBindingPort(Protocol):
    def get_binding(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSourceBinding | None:
        ...
```

LKW domain services must not import a concrete Vendor Knowledge repository. Application wiring adapts the existing `KnowledgeSourceBindingRepository` / `KnowledgeSourceBindingService` behind this port.

### 8.3 Model (to be implemented in 1B)

Revisioned child records. Each immutable version includes `mutation_id`, `effective_revision` and domain fields.

```python
class IndexedSourceSyncModeV1(StrEnum):
    FULL = "full"
    INCREMENTAL = "incremental"


class WorkspaceIndexedSourceBindingStatusV1(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class WorkspaceIndexedSourceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    indexed_source_binding_id: str = Field(..., min_length=1, max_length=128)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)

    knowledge_source_binding_ref: str = Field(..., min_length=1, max_length=128)
    source_id: str = Field(..., min_length=1, max_length=128)

    sync_mode: IndexedSourceSyncModeV1 = IndexedSourceSyncModeV1.INCREMENTAL
    status: WorkspaceIndexedSourceBindingStatusV1 = WorkspaceIndexedSourceBindingStatusV1.ACTIVE

    mutation_id: str = Field(..., min_length=1, max_length=128)
    effective_revision: int = Field(..., ge=1)

    semantic_identity_hash: str = Field(..., min_length=64, max_length=64)

    created_at: datetime
    updated_at: datetime

    # Optional non-authoritative presentation snapshot only:
    cached_safe_display_label: str | None = Field(default=None, max_length=256)
```

**Forbidden fields on this record:** `credential_ref`, `provider_id`, `integration_kind`, `source_kind`, `connection_ref`, `remote_resource_id`, `resource_type`, remote scope configuration, provider parameters.

**Source relationship:** exactly one `WorkspaceSource` per binding; `WorkspaceSource.source_type = connected_source`, `path=""`, `recursive=false`.

**Authorization vs indexing health (frozen):**

- `WorkspaceSource.status` is indexing/processing health (`registered`, `syncing`, `processing`, `ready`, `error`).
- `WorkspaceIndexedSourceBinding.status` is workspace authorization.
- Logical detach changes binding status to `DISABLED` only. It does **not** set `WorkspaceSource.status` to `ERROR`. `1B` does **not** add `WorkspaceSourceStatus.DISABLED`.
- The Source record is **not** deleted on detach.

### 8.4 Semantic identity and idempotency (separate)

**Request identity** (idempotency replay via `WorkspaceKnowledgeMutationRecord`):

```text
(tenant_id, workspace_id, operation, sha256(normalized Idempotency-Key header))
```

- Same key + same normalized request -> replay stored result (200); no semantic re-evaluation required.
- Same key + different normalized request -> 409 `configuration_idempotency_conflict`.

The raw `Idempotency-Key` value is **not** included in `normalized_request_hash`. Request hash covers operation intent and normalized request content only.

**Semantic identity** (logical resource):

```text
(tenant_id, workspace_id, knowledge_source_binding_ref)
```

First milestone: one Indexed Source authorization per tenant binding per workspace (`sync_mode` excluded from semantic identity).

**Semantic no-op behavior (frozen):** second request with a different `Idempotency-Key` but the same semantic identity and an ACTIVE committed record that already satisfies the request -> **durable `EXISTING_RESULT` mutation** (200) with stable `indexed_source_binding_id`. No writer slot, no child row, no Source creation, no configuration revision increment. The no-op outcome is persisted in `WorkspaceKnowledgeMutationRecord` with `outcome=EXISTING_RESULT`, `target_revision=None`, `committed_revision=current head.committed_revision`.

**Disabled reactivation:** when a DISABLED committed record exists and the user requests activation -> create a new version of the same logical entity with status ACTIVE and a new committed revision. Do not create a second logical entity ID.

**Normalized comparison rules:**

- Trim opaque refs (`knowledge_source_binding_ref`, `connection_ref`).
- Canonical enum serialization for `sync_mode`.
- No frontend-controlled display fields in identity.
- `semantic_identity_hash = sha256(canonical_semantic_identity_json)` hex.

**Binding ID generation:** `indexed_source_binding_id = "idx:" + sha256(tenant_id, workspace_id, knowledge_source_binding_ref)[:32]` - derived from semantic identity, not idempotency key.

**Source ID generation:** `source_id = "src:connected:" + sha256(tenant_id, workspace_id, knowledge_source_binding_ref)[:32]`.

### 8.5 Provider metadata resolution

Provider metadata (`provider_id`, `integration_kind`, `source_kind`, `connection_ref`, scope) is resolved **only** from the referenced tenant `KnowledgeSourceBinding` at create and sync time. LKW does not persist or reconstruct these fields.

**Vendor projection for sync** (ephemeral, not stored in LKW):

```python
def to_knowledge_source_ref(
    binding: WorkspaceIndexedSourceBinding,
    tenant_binding: KnowledgeSourceBinding,
) -> KnowledgeSourceRef:
    return to_source_ref(tenant_binding)  # intergrax/runtime/vendor_knowledge/bindings.py
```

### 8.6 Tenant binding lifecycle effects on workspace binding

| Tenant binding state | Workspace indexed binding | Future sync | Existing indexed data |
|---------------------|---------------------------|-------------|----------------------|
| `ACTIVE` | `ACTIVE` (if workspace authorized) | Allowed | Preserved |
| `DISABLED` | -> `UNAVAILABLE` | Blocked | Preserved |
| `REVOKED` | -> `UNAVAILABLE` | Blocked | Preserved |
| `EXPIRED` | -> `UNAVAILABLE` | Blocked | Preserved |
| Missing | -> `UNAVAILABLE` | Blocked | Preserved |
| Corrupt | -> `UNAVAILABLE` | Blocked | Preserved |

Do not silently reconstruct provider identity from stale LKW fields.

### 8.7 Detach semantics (non-destructive)

First-milestone removal means **logical detach**, not physical indexed-data deletion.

| Action | Effect |
|--------|--------|
| `PATCH .../indexed-sources/{id}` with `status=disabled` | New binding revision with `DISABLED`; Source record unchanged; future sync blocked |
| `DELETE .../indexed-sources/{id}` | **Logical detach only** - same as disable; does not delete Documents, Chunks or Vectors |

On detach:

- `WorkspaceIndexedSourceBinding` -> `DISABLED` (new revisioned version).
- `WorkspaceSource` record remains unchanged.
- `WorkspaceSource.status` remains its last indexing/processing status.
- Documents, Chunks and Vectors remain.
- Future synchronization is denied because binding is not ACTIVE.

Physical source-owned cleanup belongs to `LKW-KNOWLEDGE-LIFECYCLE-1` or a separately reviewed safe-removal operation. `ManagedWorkspaceRepository.delete_source()` removes only the Source metadata row - it does **not** delete Documents, Chunks or Vectors. Physical indexed-data deletion is outside `LKW-KNOWLEDGE-ACCESS-1`.

**Live permission:** indexed authorization does **not** create `WorkspaceLiveAccessBinding`.

---

## 9. Live Access Binding contract

### 9.1 Typed capability descriptor (authoritative read-only classification)

Read-only enforcement must **not** depend primarily on capability ID suffixes. Names such as `mail.send`, `jira.issue.transition`, `databricks.job.run`, `powerbi.refresh` may have side effects without ending in `.write`.

```python
class CapabilityEffectV1(StrEnum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


class LiveCapabilityDescriptorV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capability_id: str = Field(..., min_length=1, max_length=128)
    provider_id: str = Field(..., min_length=1, max_length=64)
    integration_kind: IntegrationCategory

    effect: CapabilityEffectV1
    read_only: bool

    resource_scope_required: bool
    supported_resource_types: tuple[str, ...] = ()

    request_schema_ref: str = Field(..., min_length=1, max_length=256)
    result_schema_ref: str = Field(..., min_length=1, max_length=256)

    max_result_items: int | None = Field(default=None, ge=1)
    max_result_bytes: int | None = Field(default=None, ge=1)
    available: bool = True
```

**First-milestone acceptance rule:** only capabilities where `effect == READ` **and** `read_only == True` **and** `available == True` may be bound.

**Suffix checking** (`.write`, `.create`, `.delete`, `.update`) may remain as defense-in-depth only. It must not be the authoritative read-only classification.

```python
class TenantLiveCapabilityCatalogPort(Protocol):
    def list_capabilities(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        remote_resource_id: str | None,
    ) -> tuple[LiveCapabilityDescriptorV1, ...]:
        ...
```

Task `1C-2` establishes capability discovery/catalog. Task `1D` may persist Live Access Bindings only against validated read-only descriptors. Arbitrary capability IDs from the frontend are rejected.

### 9.2 Model (to be implemented in 1B)

```python
class LiveAccessBindingStatusV1(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    REVOKED = "revoked"


class WorkspaceLiveAccessBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    live_access_binding_id: str = Field(..., min_length=1, max_length=128)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)

    connection_ref: str = Field(..., min_length=1, max_length=128)
    remote_resource_id: str | None = Field(default=None, max_length=256)
    allowed_capability_ids: tuple[str, ...] = Field(..., min_length=1)

    # Server-derived at create time (not accepted from frontend request):
    derived_provider_id: str = Field(..., min_length=1, max_length=64)
    derived_integration_kind: IntegrationCategory
    derived_resource_type: str | None = Field(default=None, max_length=64)
    derived_safe_display_label: str = Field(..., min_length=1, max_length=256)

    status: LiveAccessBindingStatusV1 = LiveAccessBindingStatusV1.ACTIVE

    mutation_id: str = Field(..., min_length=1, max_length=128)
    effective_revision: int = Field(..., ge=1)

    semantic_identity_hash: str = Field(..., min_length=64, max_length=64)

    created_at: datetime
    updated_at: datetime
```

**Validation rules:**

- Every `allowed_capability_id` must exist in `TenantLiveCapabilityCatalogPort.list_capabilities()` for the same `(tenant_id, connection_ref, remote_resource_id)`.
- Capabilities with `effect != READ` or `read_only != True` or `available != True` -> **rejected** (400 `capability_not_read_only`).
- `remote_resource_id` required when any selected descriptor has `resource_scope_required=True`.
- Unknown capability -> 400 `capability_not_found`.
- Unknown connection, unauthorized workspace, resource outside connection -> fail closed (404 workspace/connection, 400 validation).
- Duplicate binding same semantic identity -> return existing binding via `EXISTING_RESULT` (200).
- Does **not** imply durable ingestion rights.

**Semantic identity:**

```text
(tenant_id, workspace_id, connection_ref, normalized_remote_resource_id, normalized_capability_set)
```

Normalized capability set: trim IDs, sort, deduplicate.

---

## 10. Query Policy contract

### 10.1 First-milestone subset

| Mode | Supported in 1E |
|------|-----------------|
| `indexed_only` | **Yes** |
| `live_only` | **Yes** |
| `hybrid` | **No** - explicit 400 `query_policy_mode_unsupported` |
| `automatic` | **No** - explicit 400 `query_policy_mode_unsupported` |

### 10.2 Model (to be implemented in 1B)

```python
class QueryPolicyModeV1(StrEnum):
    INDEXED_ONLY = "indexed_only"
    LIVE_ONLY = "live_only"


class LiveResultRetentionV1(StrEnum):
    EPHEMERAL = "ephemeral"
    RECEIPT_ONLY = "receipt_only"


class WorkspaceQueryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)

    mode: QueryPolicyModeV1 = QueryPolicyModeV1.INDEXED_ONLY

    allowed_connection_refs: tuple[str, ...] = ()
    allowed_capability_ids: tuple[str, ...] = ()

    max_live_calls: int = Field(default=0, ge=0, le=50)
    max_total_duration_ms: int = Field(default=30_000, ge=1, le=300_000)
    max_result_items: int = Field(default=50, ge=1, le=500)
    max_result_bytes: int = Field(default=1_048_576, ge=1, le=16_777_216)

    live_result_retention: LiveResultRetentionV1 = LiveResultRetentionV1.EPHEMERAL

    mutation_id: str = Field(..., min_length=1, max_length=128)
    effective_revision: int = Field(..., ge=1)

    updated_at: datetime
```

**Removed from v1:** `prefer_indexed_evidence`, `allow_live_fallback` - belong to future `hybrid` / `automatic` modes.

### 10.3 Cross-field invariants (enforced at model validation)

**`indexed_only`:**

```text
allowed_connection_refs == ()
allowed_capability_ids == ()
max_live_calls == 0
live_result_retention == ephemeral
```

**`live_only`:**

```text
allowed_connection_refs not empty
allowed_capability_ids not empty
max_live_calls >= 1
all capabilities validated as read-only via TenantLiveCapabilityCatalogPort
```

**Unsupported modes:** `hybrid`, `automatic` -> explicit 400 `query_policy_mode_unsupported`. Do not accept-and-ignore future fields.

---

## 11. Workspace Knowledge Configuration aggregate

### 11.1 Representation

**Projection** assembled by `WorkspaceKnowledgeConfigurationService` from:

- `WorkspaceConnectionAttachment` (0..n)
- `WorkspaceIndexedSourceBinding` (0..n)
- `WorkspaceLiveAccessBinding` (0..n)
- `WorkspaceQueryPolicy` (0..1)
- existing `WorkspaceSource` / `Workspace` records (with visibility rules for connected Sources)

Not one mutable JSON blob.

### 11.2 Revision head (monotonic aggregate version)

One durable revision-head record - **not** derived from child versions. Child `effective_revision` values are not the aggregate concurrency token.

```python
class WorkspaceKnowledgeConfigurationHead(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)

    committed_revision: int = Field(default=0, ge=0)

    pending_revision: int | None = Field(default=None, ge=1)
    pending_mutation_id: str | None = Field(default=None, max_length=128)

    last_committed_mutation_id: str | None = Field(default=None, max_length=128)

    updated_at: datetime
```

**Persistence:**

```text
partition: lkw.managed_workspace:{tenant_id}:knowledge_configuration_head
row key: {workspace_id}
```

**Invariants:**

Idle head:

```text
pending_revision is None
pending_mutation_id is None
```

Pending head:

```text
pending_revision == committed_revision + 1
pending_mutation_id is non-empty
```

Committed revisions are strictly monotonic. A workspace may have only one pending configuration mutation.

**Empty workspace:** head missing -> logical revision `0`. First mutation requires `If-Match: WKC/0`.

**First-head creation:** the first workspace mutation must use `ConditionalDocumentStore.put_if_absent()` to create the idle head:

```text
committed_revision = 0
pending_revision = None
pending_mutation_id = None
```

Required first-mutation sequence:

```text
1. Read head.
2. When absent, put_if_absent(revision 0 idle head).
3. Read the winning head.
4. Continue through normal CAS acquisition.
```

Two concurrent first mutations must not both acquire revision `1`. A normal `DocumentStore.put()` is not sufficient.

When `ConditionalDocumentStore` is unavailable:

```text
configuration mutations are unavailable -> fail closed
```

Safe error: `configuration_conditional_store_required`. Recommended HTTP mapping for future routes: `503 Service Unavailable`.

**Publication point:** the single head CAS from pending to committed is the only publication point. Do not roll the committed revision backwards.

### 11.3 Projection model

```python
class WorkspaceKnowledgeConfigurationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    workspace_id: str
    configuration_revision: int  # from head.committed_revision only

    connection_attachments: tuple[WorkspaceConnectionAttachment, ...]
    indexed_sources: tuple[WorkspaceIndexedSourceBinding, ...]
    live_access_bindings: tuple[WorkspaceLiveAccessBinding, ...]
    query_policy: WorkspaceQueryPolicy | None

    updated_at: datetime
```

**Concurrency:** every Workspace Knowledge Configuration mutation requires:

- `If-Match: WKC/{committed_revision}`
- `Idempotency-Key: <opaque-value>` (HTTP header only; see Section 12.5)

| Condition | Behavior |
|-----------|----------|
| Missing `If-Match` | 428 `precondition_required`; stable error `knowledge_configuration_if_match_required` |
| Missing `Idempotency-Key` | 428 `precondition_required`; stable error `knowledge_configuration_idempotency_key_required` |
| Invalid `Idempotency-Key` | 400 `knowledge_configuration_idempotency_key_invalid` |
| Mismatch | 409 `configuration_revision_conflict` |
| Committed idempotency replay | May return stored result even when caller sends older `If-Match`, but only after idempotency key hash and normalized request hash are proven identical |

**Deterministic ordering:** sort attachments by `connection_ref`, indexed by `indexed_source_binding_id`, live by `live_access_binding_id`.
**Empty state:** all child collections empty, `query_policy=None`, `configuration_revision=0` (head missing).
**Integrity:** all child records must belong to same `tenant_id`/`workspace_id`; corrupt cross-workspace child records -> fail closed.

### 11.4 Workspace connection attachment (to be implemented in 1B)

```python
class WorkspaceConnectionAttachmentStatusV1(StrEnum):
    ATTACHED = "attached"
    UNAVAILABLE = "unavailable"
    DETACHED = "detached"


class WorkspaceConnectionAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attachment_id: str = Field(..., min_length=1, max_length=128)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    connection_ref: str = Field(..., min_length=1, max_length=128)
    safe_display_label: str = Field(..., min_length=1, max_length=256)
    status: WorkspaceConnectionAttachmentStatusV1

    mutation_id: str = Field(..., min_length=1, max_length=128)
    effective_revision: int = Field(..., ge=1)

    created_at: datetime
    updated_at: datetime
```

---

## 12. Durable mutation and idempotency record

### 12.1 Exact enums (to be implemented in 1B)

```python
class WorkspaceKnowledgeMutationOperationV1(StrEnum):
    ATTACH_CONNECTION = "attach_connection"
    DETACH_CONNECTION = "detach_connection"
    CREATE_INDEXED_SOURCE = "create_indexed_source"
    DISABLE_INDEXED_SOURCE = "disable_indexed_source"
    CREATE_LIVE_ACCESS_BINDING = "create_live_access_binding"
    DISABLE_LIVE_ACCESS_BINDING = "disable_live_access_binding"
    UPDATE_QUERY_POLICY = "update_query_policy"


class WorkspaceKnowledgeMutationStatusV1(StrEnum):
    RESERVED = "reserved"
    PREPARED = "prepared"
    COMMITTED = "committed"
    ABORTED = "aborted"
    RECOVERY_REQUIRED = "recovery_required"


class WorkspaceKnowledgeMutationOutcomeV1(StrEnum):
    APPLIED = "applied"
    EXISTING_RESULT = "existing_result"
```

### 12.2 Exact model (to be implemented in 1B)

```python
class WorkspaceKnowledgeMutationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mutation_id: str = Field(..., min_length=1, max_length=128)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)

    operation: WorkspaceKnowledgeMutationOperationV1

    idempotency_key_hash: str = Field(..., min_length=64, max_length=64)
    normalized_request_hash: str = Field(..., min_length=64, max_length=64)
    semantic_identity_hash: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )

    target_revision: int | None = Field(default=None, ge=1)
    committed_revision: int | None = Field(default=None, ge=0)

    status: WorkspaceKnowledgeMutationStatusV1
    outcome: WorkspaceKnowledgeMutationOutcomeV1 | None = None

    result_entity_type: str | None = Field(default=None, max_length=64)
    result_entity_id: str | None = Field(default=None, max_length=128)

    error_code: str | None = Field(default=None, max_length=128)

    created_at: datetime
    updated_at: datetime
    committed_at: datetime | None = None
```

### 12.2.1 Status invariants

**`RESERVED`:**

```text
target_revision is None
committed_revision is None
outcome is None
```

**`PREPARED`:**

```text
target_revision is not None
committed_revision is None
outcome is None
```

**`COMMITTED` with `APPLIED`:**

```text
target_revision is not None
committed_revision == target_revision
outcome == APPLIED
result reference is present when the operation creates or changes an entity
```

**`COMMITTED` with `EXISTING_RESULT`:**

```text
target_revision is None
committed_revision is the current already-committed workspace revision
outcome == EXISTING_RESULT
result reference points to the existing entity
```

**`ABORTED`:**

```text
committed_revision is None
outcome is None
```

**`RECOVERY_REQUIRED`:**

The record preserves all known revision and result information. Do not insert placeholder revisions. Do not assign `target_revision=1` before the writer slot is acquired.

### 12.3 Persistence

```text
partition: lkw.managed_workspace:{tenant_id}:knowledge_configuration_mutation
row key: {workspace_id}:{operation}:{sha256(normalized_idempotency_key)}
```

Do not put the raw idempotency key in the row key. Persist only `sha256(normalized Idempotency-Key)`. The raw key must not appear in row keys, models, logs, traces, error messages or API responses. The mutation row is the durable idempotency record. Do not introduce a second idempotency storage design.

### 12.4 Idempotency behavior

Normalize the request before hashing. Normalization includes:

```text
trim opaque references
canonical enum values
sort capability IDs
deduplicate capability IDs
canonical JSON keys
exclude timestamps
exclude server-derived display fields
exclude generated IDs
exclude Idempotency-Key header value
```

**No existing mutation record:** create `RESERVED` with `target_revision=None`, `committed_revision=None`, `outcome=None` using `put_if_absent`. If the conditional insert loses a race, reload the winning record.

**Existing record with different request hash:** return `409 configuration_idempotency_conflict`. This applies regardless of status and **before** evaluating `If-Match`.

**Existing COMMITTED record with the same request hash:** return the previously committed result. Do not increment revision again. Stale `If-Match` is permitted after exact request match.

**Existing RESERVED or PREPARED record with the same request hash:** run deterministic recovery - inspect the head; inspect staged records; complete publication when staged state is complete and valid; otherwise abort and clean the staged mutation. Do not create a second mutation for the same idempotency key while the previous one is pending.

**Existing ABORTED record with the same request hash:** a retry may replace the aborted record through conditional compare-and-swap with new `mutation_id`, `status = RESERVED`, `target_revision=None`. The same row key remains.

**Existing RECOVERY_REQUIRED record:** return `503 configuration_recovery_required` until recovery completes.

### 12.5 Canonical Idempotency-Key header (to be implemented in 1B)

Every Workspace Knowledge Configuration mutation requires:

```http
Idempotency-Key: <opaque-value>
```

The HTTP header is the **only** canonical idempotency input. Remove `idempotency_key` from mutation request bodies. Request bodies with `idempotency_key` are rejected by `extra="forbid"`.

**Header validation:**

```text
minimum length: 1
maximum length: 256
trim surrounding whitespace
empty after trim -> reject
control characters -> reject
raw value must not be logged or persisted
```

Recommended safe pattern: printable opaque string without control characters. Do not require UUID format.

| Condition | Behavior |
|-----------|----------|
| Missing header | 428 `knowledge_configuration_idempotency_key_required` |
| Invalid header | 400 `knowledge_configuration_idempotency_key_invalid` |

The domain service must not depend directly on HTTP headers. A future transport adapter may normalize and hash the header before invoking the service.

### 12.6 Semantic no-op flow (to be implemented in 1B)

After validation and idempotency reservation, but **before** acquiring the writer slot:

```text
1. Calculate semantic identity.
2. Read the current committed projection.
3. Find the existing logical entity.
4. Confirm that its committed state already satisfies the request.
5. Conditionally replace RESERVED mutation with COMMITTED.
6. Store outcome=EXISTING_RESULT.
7. Store committed_revision=current head.committed_revision.
8. Store result_entity_type and result_entity_id.
9. Keep target_revision=None.
10. Return the existing result.
```

No writer slot is acquired. No child row is written. No Source is created. No configuration revision is incremented.

**Race handling:** between semantic lookup and committing the no-op mutation record, the workspace head may change. Capture committed revision `N`; verify semantic state against projection `N`; before recording `EXISTING_RESULT`, re-read the head; if committed revision changed, retry semantic evaluation once; after repeated change, return `configuration_projection_unstable`. Do not persist `EXISTING_RESULT` against an unverified stale projection.

---

## 13. Revisioned child-record publication

All Workspace Knowledge Configuration child mutations must be revisioned. Affected record families:

```text
WorkspaceConnectionAttachment
WorkspaceIndexedSourceBinding
WorkspaceLiveAccessBinding
WorkspaceQueryPolicy
```

Do not overwrite the currently committed version before publishing the new revision.

### 13.1 Required row-key form

Use immutable revision rows:

```text
{workspace_id}:{entity_id}:rev:{revision_padded_to_20_digits}
```

Example:

```text
workspace-1:idx-123:rev:00000000000000000007
```

Query Policy uses:

```text
{workspace_id}:query-policy:rev:{revision_padded_to_20_digits}
```

Each versioned child record must include `mutation_id`, `effective_revision` and its domain fields.

### 13.2 Projection rule

To assemble revision `N`:

```text
1. Read head once.
2. Use head.committed_revision as N.
3. Ignore every child version with effective_revision > N.
4. For each logical entity, select the highest effective_revision <= N.
5. Apply logical status such as ACTIVE, DISABLED or DETACHED.
6. Sort deterministically.
```

A pending revision must never be visible through the configuration projection.

### 13.3 Logical disable as revisioned mutation

Indexed Source detach must create a new version of `WorkspaceIndexedSourceBinding` with `status = DISABLED`, `effective_revision = target revision`, `mutation_id = current mutation`. Do not overwrite or delete the prior committed version before publication.

Live Access Binding disable, Connection detach and Query Policy replacement follow the same revisioned pattern.

### 13.4 Mutation ownership and immutable row validation

Every revisioned child row must contain `mutation_id` and `effective_revision`. Recovery must never infer ownership solely from entity ID, row-key prefix or revision number.

Required exact ownership check:

```text
record.mutation_id == mutation.mutation_id
record.effective_revision == mutation.target_revision
record.tenant_id == mutation.tenant_id
record.workspace_id == mutation.workspace_id
```

A mismatch is corruption or ownership conflict. Do not clean it automatically. Return `configuration_recovery_required`.

---

## 14. Exact prepare and publish protocol

Freeze the following algorithm for every successful mutation that requires an actual state change. This is the only allowed transaction design for `1B`. Semantic no-op mutations (Section 12.6) exit before Step 5.

### Step 1 — validate required headers

Validate `If-Match` and `Idempotency-Key`. Do not mutate durable state yet.

### Step 2 — normalize request

Calculate `normalized_request_hash`, `semantic_identity_hash`, deterministic entity IDs.

### Step 3 — reserve mutation record

Create `RESERVED` with `target_revision=None`, `committed_revision=None`, `outcome=None` using `put_if_absent`. Resolve replay or conflict when the row already exists.

### Step 4 — semantic no-op detection

Check current committed projection. When state already satisfies the request: commit mutation as `EXISTING_RESULT`, return existing result. No writer slot.

### Step 5 — validate If-Match against current committed head

When mutation requires an actual state change: provided `If-Match` must equal current committed revision. Mismatch -> `409 configuration_revision_conflict`.

### Step 6 — acquire writer slot

CAS idle head:

```text
N / no pending
->
N / pending N+1 / mutation ID
```

Use `replace_if_match`. If CAS fails -> `409 configuration_revision_conflict` unless the competing pending mutation requires recovery, in which case -> `503 configuration_recovery_required`.

### Step 7 — assign target revision

Conditionally replace mutation: `RESERVED` -> `RESERVED` with `target_revision=N+1`. The assignment must be persisted **before** staged rows are written. Do not assign placeholder `target_revision=1` before writer-slot acquisition.

### Step 8 — write and validate staged records

Write all required immutable child versions with `effective_revision = N + 1`, `mutation_id = current mutation`. For Indexed Source creation, also create the pending connected Source according to Section 15 using `put_if_absent`. These records remain invisible because `effective_revision > head.committed_revision`.

Read back all staged records. Validate correct tenant, workspace, `mutation_id`, target revision, expected semantic identity, expected result ID, expected Source relationship, no forbidden provider/credential fields.

### Step 9 — mark PREPARED

Use `replace_if_match` to set mutation status `PREPARED`.

### Step 10 — publish head

CAS the head:

```text
before:
committed_revision = N
pending_revision = N + 1
pending_mutation_id = mutation_id

after:
committed_revision = N + 1
pending_revision = None
pending_mutation_id = None
last_committed_mutation_id = mutation_id
```

This single head replacement is the publication point. After this succeeds, the configuration mutation is committed. Do not roll the committed revision backwards.

`last_committed_mutation_id` may remain as diagnostic hint, fast-path optimization and recent mutation trace. It must **not** be the only proof that a mutation was published.

### Step 11 — finalize mutation

Replace the mutation record `PREPARED -> COMMITTED`, `outcome=APPLIED`, `committed_revision=R`. Store `result_entity_type`, `result_entity_id`, `committed_at`.

When mutation-record finalization fails after the head has committed:

```text
- do not roll back the head;
- repair the mutation record to COMMITTED using publication proof (Section 16.5);
- replay returns the committed result.
```

Do not depend only on `last_committed_mutation_id` for post-publication repair.

### Multi-record Source creation ordering

For Indexed Source creation use this exact staged set:

```text
1. WorkspaceIndexedSourceBinding revision row
2. WorkspaceSource row with visibility revision and creation_mutation_id
3. mutation record
4. revision head
```

The deterministic IDs must be calculated before writing. A retry must not create another Source.

---

## 15. Connected WorkspaceSource publication

### 15.1 Source status decision (frozen)

```text
WorkspaceSource.status is processing/indexing health.
WorkspaceIndexedSourceBinding.status is workspace authorization.
```

Logical detach must not misuse `WorkspaceSourceStatus.ERROR`. `1B` does not add `WorkspaceSourceStatus.DISABLED`.

### 15.2 Required WorkspaceSource fields (to be implemented in 1B)

```python
knowledge_configuration_creation_mutation_id: str | None = Field(
    default=None,
    min_length=1,
    max_length=128,
)

knowledge_configuration_visibility_revision: int | None = Field(
    default=None,
    ge=1,
)
```

**Field invariants:**

For legacy and non-connected Sources:

```text
creation_mutation_id is None
visibility_revision is None
```

For a connected Source created by Workspace Knowledge Configuration:

```text
creation_mutation_id is not None
visibility_revision is not None
source_type == CONNECTED_SOURCE
path == ""
recursive is False
```

Do not allow only one of the two knowledge-configuration fields to be present.

**Visibility meaning:**

```text
None -> legacy/non-connected Source, always visible under existing rules
integer N -> connected Source is product-visible only when
  WorkspaceKnowledgeConfigurationHead.committed_revision >= N
```

### 15.3 Source creation (to be implemented in 1B)

Create the deterministic connected Source with `put_if_absent`. Exact identity:

```text
source_id =
"src:connected:" +
sha256(tenant_id, workspace_id, knowledge_source_binding_ref)[:32]
```

When `put_if_absent` returns false:

1. Load the existing Source.
2. Validate tenant, workspace, source type and deterministic identity.
3. Determine whether it is:
   - the same staged Source owned by the current mutation;
   - an already committed Source from an earlier mutation;
   - a conflicting or corrupt Source.

**Same staged Source:** continue idempotent recovery.

**Already committed Source:** do not overwrite it. Do not change its creation mutation ID. Reuse it for reactivation or semantic replay when valid.

**Conflicting Source:** fail closed with `connected_source_identity_conflict`. Do not replace the record.

New connected Source creation:

```text
source_type = CONNECTED_SOURCE
path = ""
recursive = False
knowledge_configuration_creation_mutation_id = current mutation_id
knowledge_configuration_visibility_revision = target revision
```

### 15.4 Visibility rule

Product services that list or expose Sources must hide a connected Source when `visibility revision > committed configuration revision`. Internal repository and recovery code may access hidden staged Sources. This prevents the Source from becoming visible before the configuration head is committed.

### 15.5 Logical detach

On Indexed Source detach:

```text
WorkspaceIndexedSourceBinding -> DISABLED
WorkspaceSource record remains unchanged
WorkspaceSource.status remains its last indexing/processing status
Documents remain
Chunks remain
Vectors remain
future synchronization is denied because binding is not ACTIVE
```

The binding is the authorization gate. Do not set Source status to `ERROR`. Do not delete Source metadata. Do not delete indexed data. Physical cleanup remains in `LKW-KNOWLEDGE-LIFECYCLE-1`.

---

## 16. Failure and recovery protocol

### 16.1 Failure before writer-slot acquisition

Mutation may be marked `ABORTED`. No head cleanup is needed.

### 16.2 Failure after writer-slot acquisition but before staged writes

CAS the exact pending head back to idle. Mark mutation `ABORTED`.

### 16.3 Failure after staged writes

Delete exact owned staged records with `ConditionalDocumentStore.delete_if_match()` using the exact previously read staged record. Do **not** use unconditional `DocumentStore.delete()` for mutation rollback.

**Child cleanup condition:** delete a child row only when the current stored record still exactly matches expected row key, expected data, `mutation_id == current mutation`, `effective_revision == target revision`.

**Source cleanup condition:** delete a connected Source only when the stored record exactly matches the staged expected record and `knowledge_configuration_creation_mutation_id == current mutation_id` and `knowledge_configuration_visibility_revision == target_revision`. Never delete a Source merely because it has the deterministic `source_id`.

After successful conditional deletes: read each staged row again and confirm absence. Only then may recovery CAS head from pending to idle and mark mutation `ABORTED`.

Do not delete the mutation/idempotency record during normal cleanup. It must remain as `ABORTED` so future retries and request conflicts remain deterministic.

### 16.4 Cleanup compare failure

When `delete_if_match` returns false:

```text
- do not clear pending_revision;
- do not mark mutation ABORTED;
- mark RECOVERY_REQUIRED when possible;
- block subsequent workspace configuration mutations;
- return configuration_recovery_required.
```

This prevents an abandoned staged record from becoming visible under a future revision.

### 16.5 Failure after publication

Once Step 10 succeeds, the mutation is committed. Do not delete committed child rows. Do not remove Source. Do not decrement the revision. Repair only the mutation/idempotency record through publication proof.

A `PREPARED` mutation with target revision `R` is considered already published when all are true:

```text
head.committed_revision >= R
the expected immutable child rows exist
each expected row has:
  - mutation_id equal to the mutation record;
  - effective_revision equal to R;
  - correct tenant_id;
  - correct workspace_id;
  - correct deterministic entity identity;
the staged connected Source, when required, exists with:
  - creation_mutation_id equal to the mutation;
  - visibility_revision equal to R
```

When all conditions hold: `PREPARED -> COMMITTED`, `outcome = APPLIED`, `committed_revision = R`, `committed_at` set. This recovery remains valid even when later revisions have already been committed.

`last_committed_mutation_id` may remain as diagnostic hint, fast-path optimization and recent mutation trace. It must **not** be the only proof that a mutation was published.

### 16.6 Required recovery cases

**Head still pending for the same mutation** (`pending_mutation_id == mutation_id`, `pending_revision == target_revision`):

Recovery either completes publication when all staged records are valid and mutation is `PREPARED`, or cleans incomplete staged state and aborts.

**Head committed revision is at or above target revision:**

Validate immutable rows for that exact mutation and revision. When valid: repair mutation to `COMMITTED`. When missing or inconsistent: `RECOVERY_REQUIRED`. Do not delete rows from an already published revision merely because the mutation record is incomplete.

**Head below target revision and not pending for mutation:**

The mutation was not published. Clean only exact staged rows owned by that mutation using `delete_if_match`.

### 16.7 Required recovery operation (to be implemented in 1B)

Freeze a service-level recovery method:

```python
def recover_workspace_knowledge_mutation(
    *,
    tenant_id: str,
    workspace_id: str,
) -> RecoveryResult:
    ...
```

It must:

```text
inspect head pending mutation
load mutation record
load staged rows with exact mutation ownership validation
either complete valid PREPARED publication via immutable row proof
or clean incomplete staged state with delete_if_match
release the head only after successful cleanup verification
repair COMMITTED mutation record when head already committed it
```

Do not depend only on `last_committed_mutation_id` for publication proof.

This is an internal service operation in `1B`. No HTTP recovery endpoint is required in `1B`.

---

## 17. Reader consistency rules

The contract does not claim a single cross-partition snapshot when the DocumentStore does not provide one.

**Required read algorithm:**

```text
1. Read the head and capture committed_revision N.
2. Read child version rows.
3. Select only rows with effective_revision <= N.
4. Select latest applicable version per logical entity.
5. Validate all selected records belong to tenant/workspace.
6. Build deterministic projection.
7. Optionally re-read head.
8. When committed_revision changed during the read, retry once or return a
   safe retryable configuration_read_conflict.
```

Recommended behavior: retry once on head change. After repeated change: `503 configuration_projection_unstable`.

---

## 18. Persistence design

### 18.1 Records

| Record | Partition | Row key | Notes |
|--------|-----------|---------|-------|
| `WorkspaceKnowledgeConfigurationHead` | `lkw.managed_workspace:{tenant_id}:knowledge_configuration_head` | `{workspace_id}` | one per workspace |
| `WorkspaceKnowledgeMutationRecord` | `lkw.managed_workspace:{tenant_id}:knowledge_configuration_mutation` | `{workspace_id}:{operation}:{sha256(normalized_idempotency_key)}` | durable idempotency |
| `WorkspaceConnectionAttachment` | `lkw.managed_workspace:{tenant_id}:connection_attachment` | `{workspace_id}:{attachment_id}:rev:{revision_padded}` | immutable revisions |
| `WorkspaceIndexedSourceBinding` | `lkw.managed_workspace:{tenant_id}:indexed_source_binding` | `{workspace_id}:{indexed_source_binding_id}:rev:{revision_padded}` | immutable revisions |
| `WorkspaceLiveAccessBinding` | `lkw.managed_workspace:{tenant_id}:live_access_binding` | `{workspace_id}:{live_access_binding_id}:rev:{revision_padded}` | immutable revisions |
| `WorkspaceQueryPolicy` | `lkw.managed_workspace:{tenant_id}:query_policy` | `{workspace_id}:query-policy:rev:{revision_padded}` | immutable revisions |
| `WorkspaceSource` (existing) | `lkw.managed_workspace:{tenant_id}:source` | `{workspace_id}:{source_id}` | visibility revision for connected Sources |

### 18.2 Repository

Extend `ManagedWorkspaceRepository` with typed put/get/list methods mirroring existing Source/Operation patterns. Configuration mutations require `ConditionalDocumentStore`. No new DocumentStore implementation.

### 18.3 Migrations

Additive partitions only. Existing workspaces: empty knowledge configuration (head missing, revision 0). `CONNECTED_SOURCE` sources absent until explicit binding create.

### 18.4 Deletion semantics

| Action | Effect |
|--------|--------|
| Disable/detach indexed binding | Binding -> `DISABLED` (new revision); Source unchanged; future sync blocked; **Documents, Chunks, Vectors preserved** |
| Disable live binding | Binding -> `DISABLED` (new revision); no Document changes |
| Detach connection | `status=detached` (new revision); indexed/live bindings for that `connection_ref` -> `unavailable` |
| Delete workspace | Existing `delete_workspace()` extended to purge new partitions |

Physical source-owned cleanup (Documents + Chunks + Vectors) belongs to `LKW-KNOWLEDGE-LIFECYCLE-1`, not `LKW-KNOWLEDGE-ACCESS-1`.

---

## 19. Service and repository boundaries

```text
WorkspaceKnowledgeConfigurationService (LKW)
├── ManagedWorkspaceRepository (durable LKW records + revision head)
├── ManagedWorkspaceService (workspace existence authority)
├── TenantKnowledgeSourceBindingPort (read-only tenant binding lookup - 1B)
├── TenantConnectionPort (read-only; platform - 1C-1 catalog + 1C-2 discovery)
├── RemoteResourceDiscoveryPort (wraps VendorKnowledgeFacade inspect/list - 1C-2)
├── TenantLiveCapabilityCatalogPort (typed read-only capability catalog - 1C-2)
└── WorkspaceKnowledgeAuthorizationService (tenant + workspace + binding checks)

VendorKnowledgeFacadeService (Tier-1, unchanged)
├── ConnectionAwareVendorResolver
├── KnowledgeAdapterRegistry
└── existing integrations (no LKW import)
```

LKW services **must not** import provider packages (`jira`, `confluence`, `ms365_graph`).

### 19.1 Indexed binding create flow

```text
request
-> validate headers (If-Match, Idempotency-Key) (Step 1)
-> validation and normalization (Step 2)
-> reserve mutation RESERVED with target_revision=None (Step 3)
-> semantic no-op detection (Step 4)
-> validate If-Match (Step 5)
-> ensure head (put_if_absent when absent)
-> acquire writer slot (Step 6)
-> assign target_revision=N+1 (Step 7)
-> write staged binding + connected Source via put_if_absent (Step 8)
-> mark PREPARED (Step 9)
-> publish head (Step 10)
-> finalize mutation COMMITTED outcome=APPLIED (Step 11)
```

`DocumentStore` has no cross-record transaction. Atomicity is provided by the revision-head CAS publication point plus staged-record cleanup on failure. Do not claim multi-record mutation atomicity beyond the frozen protocol.

---

## 20. Public API proposal

Base prefix: `/v1/local_workspace`. All endpoints require resolved `tenant_id`. Workspace-scoped endpoints return **404** when workspace unknown (including cross-tenant).

Every mutating endpoint requires `If-Match: WKC/{committed_revision}` and `Idempotency-Key: <opaque-value>`. Initial empty configuration uses `If-Match: WKC/0`.

### 20.1 List safe Connections

| | |
|--|--|
| Method / path | `GET /connections` |
| Response | `SafeConnectionListResponseV1 { connections: list[SafeConnectionSummaryV1] }` |
| Auth | Tenant context |
| Pagination | `limit` 1-100, optional `page_token` |
| Secrets | Projection only - no `credential_ref` |
| Errors | 401 unauthenticated |

### 20.2 Inspect Connection

| | |
|--|--|
| Method / path | `GET /connections/{connection_ref}` |
| Response | `SafeConnectionSummaryV1` |
| Errors | 404 `connection_not_found` (including cross-tenant) |

### 20.3 Discover Remote Resources

| | |
|--|--|
| Method / path | `GET /connections/{connection_ref}/remote-resources` |
| Query | `source_kind`, `limit`, `page_token`, optional `filter` (max 128 chars) |
| Response | `RemoteResourceListResponseV1 { items, next_page_token }` |
| Auth | Tenant + connection ownership |
| Errors | 404 connection; 400 unsupported `source_kind`; 503 `connection_unavailable` |

### 20.4 Attach Connection to workspace

| | |
|--|--|
| Method / path | `PUT /workspaces/{workspace_id}/connections/{connection_ref}` |
| Request | `AttachConnectionRequestV1 { safe_display_label? }` |
| Headers | **Required** `If-Match: WKC/{committed_revision}`; **Required** `Idempotency-Key` |
| Response | `WorkspaceConnectionAttachment` + `configuration_revision` |
| If-Match | Missing -> 428 `knowledge_configuration_if_match_required`; mismatch -> 409 |
| Idempotency-Key | Missing -> 428 `knowledge_configuration_idempotency_key_required`; invalid -> 400 |
| Idempotency replay | Same key + same request -> 200 existing result; no new revision |
| Idempotency conflict | Same key + different request -> 409 `configuration_idempotency_conflict` (before If-Match) |
| Semantic duplicate | Same `(tenant_id, workspace_id, connection_ref)` ACTIVE -> 200 existing via `EXISTING_RESULT`; no new revision |
| Resulting revision | `committed_revision + 1` on success |
| Errors | 404 workspace/connection; 409 conflict; 428 missing headers |

### 20.5 Detach Connection from workspace

| | |
|--|--|
| Method / path | `PATCH /workspaces/{workspace_id}/connections/{connection_ref}` with `{ "status": "detached" }` or `DELETE .../connections/{connection_ref}` |
| Response | 204 (logical detach) |
| Headers | **Required** `If-Match`; **Required** `Idempotency-Key` |
| If-Match | Missing -> 428; mismatch -> 409 |
| Idempotency-Key | Missing -> 428; invalid -> 400 |
| Idempotency replay | Same key + same request -> 200 existing result; no new revision |
| Semantic duplicate | N/A for detach |
| Resulting revision | `committed_revision + 1` on success |
| Effect | Attachment -> `DETACHED` (new revision); indexed/live bindings for connection -> `unavailable` |
| Errors | 404; 409; 428 |

### 20.6 Read Workspace Knowledge Configuration

| | |
|--|--|
| Method / path | `GET /workspaces/{workspace_id}/knowledge-configuration` |
| Response | `WorkspaceKnowledgeConfigurationV1` |
| Errors | 404 workspace; 503 `configuration_projection_unstable` on repeated head change |

### 20.7 Create Indexed Source binding

| | |
|--|--|
| Method / path | `POST /workspaces/{workspace_id}/indexed-sources` |
| Request | `CreateWorkspaceIndexedSourceRequestV1 { knowledge_source_binding_ref, sync_mode }` |
| Headers | **Required** `If-Match`; **Required** `Idempotency-Key` |
| Server-derived | `provider_id`, `integration_kind`, `source_kind`, `connection_ref`, scope, `safe_display_label` - from tenant binding; **not in request schema** |
| Response | 201 `WorkspaceIndexedSourceBinding` |
| If-Match | **Required** `WKC/{committed_revision}`; first mutation `WKC/0`; missing -> 428; mismatch -> 409 |
| Idempotency-Key | Missing -> 428; invalid -> 400 |
| Idempotency replay | Same key + same request -> 200 stored result; no new revision |
| Idempotency conflict | Same key + different request -> 409 (before If-Match) |
| Semantic duplicate | Same `knowledge_source_binding_ref` ACTIVE -> 200 existing binding via `EXISTING_RESULT`; no new revision |
| Resulting revision | `committed_revision + 1` on success |
| Errors | 400 validation; 404 workspace/tenant-binding; 409 idempotency/revision conflict; 428 missing headers |

```python
class CreateWorkspaceIndexedSourceRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_source_binding_ref: str = Field(..., min_length=1, max_length=128)
    sync_mode: IndexedSourceSyncModeV1 = IndexedSourceSyncModeV1.INCREMENTAL
```

### 20.8 Disable / detach Indexed Source binding

| | |
|--|--|
| Method / path | `PATCH /workspaces/{workspace_id}/indexed-sources/{indexed_source_binding_id}` with `{ "status": "disabled" }` |
| Alternative | `DELETE /workspaces/{workspace_id}/indexed-sources/{indexed_source_binding_id}` - **logical detach only**, not physical indexed-data deletion |
| Response | 204 (logical detach) |
| Headers | **Required** `If-Match`; **Required** `Idempotency-Key` |
| If-Match | Missing -> 428; mismatch -> 409 |
| Idempotency-Key | Missing -> 428; invalid -> 400 |
| Idempotency replay | Same key + same request -> 200 existing result; no new revision |
| Resulting revision | `committed_revision + 1` on success |
| Effect | Binding -> `DISABLED` (new revision); Source unchanged; Documents/Chunks/Vectors **preserved** |
| Errors | 404; 409; 428 |

### 20.9 Create Live Access Binding

| | |
|--|--|
| Method / path | `POST /workspaces/{workspace_id}/live-access-bindings` |
| Request | `CreateWorkspaceLiveAccessBindingRequestV1 { connection_ref, remote_resource_id, allowed_capability_ids }` |
| Headers | **Required** `If-Match`; **Required** `Idempotency-Key` |
| Server-derived | `provider_id`, `integration_kind`, `resource_type`, `safe_display_label`, capability effect classification - validated via `TenantLiveCapabilityCatalogPort` |
| Response | 201 `WorkspaceLiveAccessBinding` |
| If-Match | **Required** `WKC/{committed_revision}`; missing -> 428; mismatch -> 409 |
| Idempotency-Key | Missing -> 428; invalid -> 400 |
| Idempotency replay | Same key + same request -> 200 existing result; no new revision |
| Semantic duplicate | Same semantic identity ACTIVE -> 200 existing via `EXISTING_RESULT`; no new revision |
| Resulting revision | `committed_revision + 1` on success |
| Errors | 400 `capability_not_read_only` / `capability_not_found`; 404 workspace/connection; 409 revision conflict; 428 |

```python
class CreateWorkspaceLiveAccessBindingRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_ref: str = Field(..., min_length=1, max_length=128)
    remote_resource_id: str | None = Field(default=None, max_length=256)
    allowed_capability_ids: tuple[str, ...] = Field(..., min_length=1)
```

### 20.10 Disable / detach Live Access Binding

| | |
|--|--|
| Method / path | `PATCH /workspaces/{workspace_id}/live-access-bindings/{live_access_binding_id}` with `{ "status": "disabled" }` or `DELETE .../live-access-bindings/{live_access_binding_id}` |
| Response | 204 (logical detach, non-destructive) |
| Headers | **Required** `If-Match`; **Required** `Idempotency-Key` |
| If-Match | Missing -> 428; mismatch -> 409 |
| Idempotency-Key | Missing -> 428; invalid -> 400 |
| Idempotency replay | Same key + same request -> 200 existing result; no new revision |
| Resulting revision | `committed_revision + 1` on success |
| Effect | Binding -> `DISABLED` (new revision); no Document changes |
| Errors | 404; 409; 428 |

### 20.11 Update Query Policy

| | |
|--|--|
| Method / path | `PUT /workspaces/{workspace_id}/query-policy` |
| Request | `UpdateQueryPolicyRequestV1` (policy fields only; no idempotency field) |
| Headers | **Required** `If-Match`; **Required** `Idempotency-Key` |
| If-Match | **Required** `WKC/{committed_revision}` (aggregate revision); missing -> 428; mismatch -> 409 |
| Idempotency-Key | Missing -> 428; invalid -> 400 |
| Idempotency replay | Same key + same request -> 200 existing result; no new revision |
| Resulting revision | `committed_revision + 1` on success |
| Response | `WorkspaceQueryPolicy` |
| Errors | 400 unsupported mode / invariant violation; 409 `configuration_revision_conflict`; 428 missing headers |

---

## 21. Authorization and safe error behavior

| Boundary | Behavior |
|----------|----------|
| Tenant | `resolve_tenant_id()`; data queries always include `tenant_id` partition |
| Workspace | `get_workspace()` None -> **404** |
| Principal | Request context principal (when present) must match tenant; future fine-grained workspace ACL hooks at service layer |
| Connection | `connection_ref` must resolve for same `tenant_id`; else 404 |
| Resource | `remote_resource_id` must be discovered under connection before binding |
| Capability | Must be in catalog with `effect=READ`, `read_only=True`, `available=True` |
| Fail closed | Unknown/stale/unauthorized -> 404 or 400; never silent downgrade |

Safe errors: stable snake_case `detail` string; no `connection_ref` in error messages (matches `test_connections.py` pattern).

---

## 22. Idempotency, semantic identity and configuration-revision semantics

### 22.1 Two separate identities

| Identity | Key components | Purpose |
|----------|---------------|---------|
| Request (idempotency) | `(tenant_id, workspace_id, operation, sha256(normalized Idempotency-Key))` via `WorkspaceKnowledgeMutationRecord` | Detect replay vs conflict |
| Semantic (logical resource) | Indexed: `(tenant_id, workspace_id, knowledge_source_binding_ref)`; Live: `(tenant_id, workspace_id, connection_ref, normalized_resource_scope, normalized_capability_set)`; Connection: `(tenant_id, workspace_id, connection_ref)` | Prevent duplicate logical bindings |

### 22.2 If-Match and idempotency evaluation order

Freeze exact evaluation order for all mutation API sections:

```text
1. Validate required headers (If-Match, Idempotency-Key)
2. Normalize request
3. Reserve mutation record (RESERVED, target_revision=None)
4. Semantic no-op detection
5. Validate If-Match against current committed head (when state change required)
6. Acquire writer slot
7. Assign target revision
8. Write staged records
9. Mark PREPARED
10. Publish head
11. Finalize mutation
```

For an existing mutation record:

| Scenario | Behavior |
|----------|----------|
| Same key + different request hash | Return idempotency conflict **before** evaluating `If-Match` |
| Same key + same request, COMMITTED | Return stored result even when `If-Match` is stale |
| Same key + same request, RESERVED/PREPARED | Run recovery; do not start a new mutation |
| No mutation record | Validate semantic state and current revision according to frozen processing order |

### 22.3 Idempotency behavior summary

| Scenario | Behavior |
|----------|----------|
| Same key + same request, COMMITTED | Return stored result; no new revision; stale If-Match permitted |
| Same key + different request | 409 `configuration_idempotency_conflict` (before If-Match) |
| Same key + same request, RESERVED/PREPARED | Deterministic recovery; no second mutation |
| Same key + same request, ABORTED | Retry may CAS-replace to RESERVED with new mutation_id |
| Same key, RECOVERY_REQUIRED | 503 `configuration_recovery_required` |
| Different key + same semantic identity, ACTIVE, state satisfies request | Durable `EXISTING_RESULT`; no writer slot; no new revision |
| DISABLED + reactivation request | New revision with ACTIVE; same logical entity ID |

### 22.4 Configuration revision

- Single monotonic `committed_revision` on `WorkspaceKnowledgeConfigurationHead`.
- Every successful state-changing mutation increments revision via head CAS publication (Step 10).
- Semantic no-op (`EXISTING_RESULT`) does **not** increment configuration revision.
- Aggregate projection reads revision from head record only - children do not define aggregate version.
- Child records store `effective_revision` and `mutation_id` on immutable revision rows.

---

## 23. No-duplication proof (future acceptance test - `1F`)

### 23.1 Scenario

```text
one durable TenantConnection (connection_ref = conn-proof-1) persisted in catalog
-> application restart with empty in-memory KnowledgeConnectionRegistry
-> restart rehydration reloads Connection, resolves credential_ref, registers one integration
-> one tenant KnowledgeSourceBinding (binding_ref = bind-proof-1) references conn-proof-1
-> attached to workspace W
-> WorkspaceIndexedSourceBinding I (references bind-proof-1)
-> WorkspaceLiveAccessBinding L (references conn-proof-1)
```

The proof must use a **durable Connection reconstructed after restart**, not only an object manually inserted into the registry.

### 23.2 Required invariants

| Invariant | Observable check |
|-----------|-------------------|
| Durable Connection survives restart | `TenantConnectionRepository` returns same `connection_ref` after process restart |
| Registry rehydrated | Empty registry at start; exactly one registration after bootstrap |
| One tenant binding | `TenantKnowledgeSourceBindingPort.get_binding` count == 1 |
| Indexed authorization references tenant binding | `I.knowledge_source_binding_ref == bind-proof-1`; no `provider_id`/`connection_ref` on I |
| Live authorization references same connection | `L.connection_ref == conn-proof-1` (derived from tenant binding at create) |
| No provider identity copied into indexed authorization | Serialized LKW indexed binding record lacks `provider_id`, `integration_kind`, `source_kind`, `credential_ref` |
| No credential duplication | `SecretsStore` lookup count == 1 per operation window; no `credential_ref` in LKW records |
| Same integration registration | `KnowledgeConnectionRegistry.resolve` returned object `id()` equal for indexed sync stub and live stub |
| No second vendor client | Integration constructor counter == 1 per `(tenant_id, connection_ref)` |
| Independent authorization | Disable I -> live still allowed; disable L -> indexed sync still allowed |
| Same workspace boundary | Cross-workspace binding attempt -> 404 |
| Missing secret does not delete Connection | Invalid `credential_ref` -> unavailable projection; Connection record remains |

### 23.3 Test harness sketch

Inject instrumented `TenantKnowledgeSourceBindingPort`, `KnowledgeConnectionRegistry`, `ConnectionAwareVendorResolver`, and `SecretsStore` fake into wiring used by configuration service and facade. Use existing vendor knowledge fakes from `tests/unit/runtime/vendor_knowledge/_fakes.py`. Assert absence of credential/provider-scope duplication in serialized LKW records.

---

## 24. Security threat review

| Threat | Boundary | Prevention | Failure | Test |
|--------|----------|------------|---------|------|
| Credential leakage | LKW persistence / API | Forbidden field scan on serialize; no secret keys in models | 500 corrupt record / reject write | Unit: binding store secret rejection pattern |
| Cross-tenant connection ref | Resolver + repository | Partition + tenant match on all reads | 404 | Integration: tenant A ref in tenant B workspace |
| Cross-workspace binding | Workspace service | `workspace_id` on all records | 404 | API test |
| Capability escalation | Live binding create | `TenantLiveCapabilityCatalogPort` rejects `effect != READ` | 400 `capability_not_read_only` | Unit validator |
| Write/execute/admin capability | Live binding create | Typed `CapabilityEffectV1` check (suffix check defense-in-depth only) | 400 | Contract test |
| Resource reference substitution | Binding create | Resource must be discovered under same `connection_ref` | 400 | Integration |
| Unsafe provider locator | Remote resource / evidence | `KnowledgeSourceScope` safe mapping rules | 400 validation | Reuse vendor_knowledge model tests |
| Provider identity spoofing | Indexed/live create | Request schema excludes `provider_id`, `integration_kind`, etc. | 422 validation | Schema scan |
| Stale tenant binding | Indexed sync | Re-resolve via `TenantKnowledgeSourceBindingPort`; unavailable -> block sync | `UNAVAILABLE` | Integration test |
| Provider permission loss | Discovery + execution | `availability` enum + binding `unavailable` | Fail closed | Simulated adapter denial |
| Live result persisted | Ask / executor | `ephemeral` retention default; no Document write path | Assert no `put_document_ref` | Integration |
| Duplicate vendor client | Connection registry | Single registration per ref | Constructor count | **1F proof** |
| MCP arbitrary exposure | LKW domain | MCP not in configuration models | N/A | Schema scan |
| Oversized provider result | Query policy | `max_result_bytes`, `max_result_items` | Truncate + receipt | Policy unit test |
| Abandoned staged record | Publication protocol | Cleanup `delete_if_match` failure -> RECOVERY_REQUIRED; head stays pending | 503 `configuration_recovery_required` | Failure-injection test |
| Premature Source visibility | Connected Source staging | `knowledge_configuration_visibility_revision` + `creation_mutation_id` gate | Source hidden until head commit | Failure-injection test |
| Raw idempotency key leakage | API / persistence / logs | Persist only `idempotency_key_hash`; header not in bodies | Raw key absent from records, logs, errors | Failure-injection test |
| Stale recovery via head pointer only | Post-publication repair | Publication proof from immutable rows + Source ownership; `last_committed_mutation_id` is hint only | `configuration_recovery_required` on inconsistent proof | Failure-injection test |
| Unconditional rollback delete | Mutation cleanup | All rollback deletions use `delete_if_match` with exact staged record | `RECOVERY_REQUIRED` on compare failure | Failure-injection test |

---

## 25. Migration and backward-compatibility impact

- **Additive only** - new DocumentStore partitions and routes.
- Existing workspaces without knowledge configuration: valid empty projection.
- `WorkspaceSourceType.CONNECTED_SOURCE` enum already exists; first binding implementation activates it.
- No change to existing intake kinds (`web_url`, `managed_file`, etc.).
- Ask remains indexed-only until Hybrid Ask (out of scope).
- Roadmap status `LKW-KNOWLEDGE-ACCESS-1 -> NEXT` unchanged.

---

## 26. Implementation decomposition

### 26.1 `LKW-KNOWLEDGE-ACCESS-1A` (this task)

**Outcome:** Freeze implementation contract and audit foundations.
**Dependencies:** Accepted architecture.
**Code areas:** docs only.
**Non-goals:** Production code.
**Tests:** `git diff --check`, encoding validation, manual link/symbol verification.
**Gate:** `READY_FOR_REVIEW` on this document.

### 26.2 `LKW-KNOWLEDGE-ACCESS-1B` - provider-neutral durable workspace authorization foundation

**Outcome:** Implement the durable provider-neutral workspace authorization foundation with tenant-binding references, immutable revisioned child records, one CAS-protected configuration head, one durable mutation/idempotency record, hidden staged connected Sources with exact mutation ownership, deterministic crash recovery, persistent semantic no-op outcomes and non-destructive detach.

**Dependencies:** 1A-C3.

**Exact scope:**

```text
WorkspaceKnowledgeConfigurationHead
WorkspaceKnowledgeMutationRecord
WorkspaceKnowledgeMutationOutcomeV1
nullable target_revision during RESERVED state
persistent EXISTING_RESULT no-op records
stable semantic no-op evaluation
revisioned child row contracts
TenantKnowledgeSourceBindingPort
WorkspaceIndexedSourceBinding
WorkspaceLiveAccessBinding storage shape
WorkspaceConnectionAttachment storage shape
WorkspaceQueryPolicy storage shape
WorkspaceSource creation_mutation_id
WorkspaceSource visibility_revision
put_if_absent Source creation
delete_if_match cleanup
canonical Idempotency-Key handling
raw idempotency-key confidentiality
ManagedWorkspaceRepository extensions
WorkspaceKnowledgeConfigurationService
CAS prepare/publish/recovery protocol
post-publication proof independent of head last-mutation pointer
idempotency replay/conflict
semantic duplicate detection
reader consistency
C3 failure-injection tests
```

Although HTTP routes remain outside `1B`, shared service request contracts may carry the already-normalized and hashed idempotency key from a future transport adapter. The domain service must not depend directly on HTTP headers.

**Non-goals:**

```text
HTTP route implementation
Connection catalog
Remote Resource discovery
capability catalog implementation
provider execution
Hybrid Ask
Slack
provider-specific imports
physical Source deletion
vector deletion
```

Do not move capability discovery from `1C-2` into `1B`.

**Required failure-injection tests:**

| Scenario | Expected outcome |
|----------|------------------|
| Two concurrent first mutations | One head created; one writer acquires revision 1; other receives revision conflict |
| CAS idle -> pending fails | No staged rows written |
| RESERVED mutation created | `target_revision` is None; no placeholder revision |
| Writer slot acquired at N+1 | Mutation `target_revision` conditionally becomes N+1; staged rows not written before assignment persists |
| Failure after binding version write | Binding staged; Source not staged -> `delete_if_match` cleanup binding; head returns idle; mutation ABORTED |
| Failure after Source write | Binding + Source staged; PREPARED not reached -> both removed via `delete_if_match`; head returns idle |
| Failure marking PREPARED | All staged written; PREPARED CAS fails -> `delete_if_match` cleanup; head returns idle only after successful cleanup verification |
| Cleanup compare failure | `delete_if_match` returns false -> head remains pending; mutation RECOVERY_REQUIRED; subsequent mutations fail closed |
| Failure publishing head | PREPARED records exist; head commit CAS fails -> `delete_if_match` cleanup staged records; release head |
| Failure finalizing mutation after head commit | Head committed; mutation still PREPARED -> recovery repairs to COMMITTED via immutable row proof; result replayed |
| Older PREPARED repair after later commits | M1 publishes revision 1; M1 finalization fails; M2 publishes revision 2; recovery of M1 validates M1 rows at revision 1; repairs M1 to COMMITTED; does not depend on `last_committed_mutation_id` |
| Semantic no-op result | Different idempotency key; same semantic identity; existing ACTIVE entity -> mutation COMMITTED; outcome EXISTING_RESULT; target_revision None; committed_revision N; no writer slot; no new child row; no new Source |
| No-op replay | Same key + same request -> same stored existing result; no semantic re-evaluation required |
| No-op conflict | Same key + different request -> 409 |
| No-op race | Head changes during semantic duplicate evaluation -> retry once; persist result only against stable committed projection |
| Idempotency replay | Same key + same request after commit -> stored result; no new revision |
| Idempotency conflict | Same key + different request -> 409; no new revision |
| Source ownership | Staged Source contains `creation_mutation_id=M`, `visibility_revision=R` |
| Committed Source reuse | `put_if_absent` loses because committed deterministic Source exists -> Source validated; not overwritten; not deleted |
| Source conflict | Deterministic Source ID exists with incompatible ownership -> fail closed; no overwrite |
| Child delete_if_match | `delete_if_match` succeeds only for exact staged record |
| Source delete_if_match | `delete_if_match` requires exact `creation_mutation_id` and `visibility_revision` |
| Source visibility | Source staged at N+1; head still N -> hidden; head committed N+1 -> visible |
| Non-destructive detach | Binding DISABLED; Source remains; document refs remain; Documents remain; vector deletion not called; future sync rejected |
| Reader retry | Head changes during projection -> retry once -> return stable revision or safe unstable error |
| Missing idempotency header | Missing `Idempotency-Key` -> 428 |
| Invalid idempotency header | Blank or control-bearing key -> 400 |
| Raw-key confidentiality | Raw `Idempotency-Key` absent from stored records, row keys, logs, errors, serialized responses |
| Body-key rejection | Request body containing `idempotency_key` -> rejected by `extra="forbid"` |

**Gate:** One tenant binding reference; no provider identity duplication; monotonic revision; zero provider-specific imports; publication protocol passes all failure-injection tests.

### 26.3 `LKW-KNOWLEDGE-ACCESS-1C-1` - DURABLE TENANT CONNECTION CATALOG AND RESTART REHYDRATION

**Outcome:** A tenant Connection is stored durably with safe configuration and an opaque credential reference, and the application can reconstruct its single runtime integration registration after restart without storing credentials in LKW.

**Dependencies:** 1B (for workspace binding resolution only — catalog itself is platform-owned).

**Exact scope:**

```text
TenantConnection model
TenantConnectionRepository
DocumentStore-backed repository
TenantConnectionService
administrative lifecycle (ACTIVE | DISABLED | REVOKED)
configuration-version concurrency
safe projection
SecretsStore credential_ref resolution
runtime integration factory boundary
KnowledgeConnectionRegistry rehydration
restart and unavailable-state tests
```

**Non-goals:**

```text
provider resource discovery
workspace attachment mutations
Indexed Source creation
Live Access Binding creation
Hybrid Ask
Slack commands
raw secret persistence
provider-specific business workflows
```

**Gate:** Create durable Connection; restart with empty in-memory registry; reload Connection; resolve secret by `credential_ref`; construct one integration instance; register one runtime connection; preserve the same `connection_ref`; expose safe status; store no raw secret.

### 26.4 `LKW-KNOWLEDGE-ACCESS-1C-2` - SAFE CONNECTION / REMOTE RESOURCE DISCOVERY AND TYPED CAPABILITY CATALOG

**Outcome:** LKW can list safe durable tenant Connections, discover ephemeral Remote Resources and expose only validated read-only capability descriptors.

**Dependencies:** 1B, 1C-1.

**Exact scope:**

```text
TenantConnectionPort
RemoteResourceDiscoveryPort
TenantLiveCapabilityCatalogPort
safe Connection reads
Remote Resource discovery
typed read-only capability descriptors
cross-tenant fail-closed behavior
```

**Non-goals:** Indexed/live binding mutations.

**Gate:** Discovery returns descriptors without secrets; only read-only capabilities listed.

### 26.5 `LKW-KNOWLEDGE-ACCESS-1D` - HTTP create/disable for bindings with server-derived metadata

**Outcome:** HTTP create/disable for connection attachment, Indexed Source authorization, Live Access Binding; server-derived metadata; no physical indexed-data deletion.
**Dependencies:** 1B, 1C-1, 1C-2.
**Code areas:** routes, schemas, `WorkspaceKnowledgeConfigurationService`.
**Non-goals:** Actual sync or live execution.
**Tests:** API acceptance, idempotency, semantic duplicate, independent binding authorization, non-destructive detach, If-Match on every mutation.
**Gate:** `CONNECTED_SOURCE` workspace Source created; indexed detach preserves Documents; no live binding auto-created.

### 26.6 `LKW-KNOWLEDGE-ACCESS-1E` - Query Policy and complete configuration projection

**Outcome:** Query policy CRUD + `GET knowledge-configuration` aggregate with revision head.
**Dependencies:** 1D.
**Non-goals:** Hybrid/automatic modes.
**Tests:** Unsupported mode rejection; cross-field invariant enforcement; deterministic ordering; revision concurrency; reader consistency retry.
**Gate:** Full projection matches stored records; revision from head only.

### 26.7 `LKW-KNOWLEDGE-ACCESS-1F` - one tenant binding / one connection indexed-live reuse proof

**Outcome:** Observable proof test - one durable tenant Connection reconstructed after restart, one tenant binding, one connection, one integration instance, no credential or provider-identity copy.
**Dependencies:** 1E + 1C-1 rehydration + minimal live executor stub OR facade-only proof with shared resolver instrumentation.
**Non-goals:** Production live queries.
**Tests:** Instrumented acceptance test per section 23.
**Gate:** All invariants green in CI proof module.

---

## 27. First implementation task

### Recommended: `LKW-KNOWLEDGE-ACCESS-1B` - PROVIDER-NEUTRAL DURABLE WORKSPACE AUTHORIZATION FOUNDATION

**One-sentence outcome:** Implement the durable provider-neutral workspace authorization foundation with tenant-binding references, immutable revisioned child records, one CAS-protected configuration head, one durable mutation/idempotency record, hidden staged connected Sources with exact mutation ownership, persistent semantic no-op outcomes, deterministic crash recovery and non-destructive detach.

**Expected scope:**

```text
WorkspaceKnowledgeConfigurationHead
WorkspaceKnowledgeMutationRecord
WorkspaceKnowledgeMutationOutcomeV1
nullable target_revision during RESERVED state
persistent EXISTING_RESULT no-op records
stable semantic no-op evaluation
revisioned child row contracts
TenantKnowledgeSourceBindingPort
WorkspaceIndexedSourceBinding
WorkspaceLiveAccessBinding storage shape
WorkspaceConnectionAttachment storage shape
WorkspaceQueryPolicy storage shape
WorkspaceSource creation_mutation_id
WorkspaceSource visibility_revision
put_if_absent Source creation
delete_if_match cleanup
canonical Idempotency-Key handling
raw idempotency-key confidentiality
ManagedWorkspaceRepository extensions
WorkspaceKnowledgeConfigurationService
CAS prepare/publish/recovery protocol
post-publication proof independent of head last-mutation pointer
idempotency replay/conflict
semantic duplicate detection
reader consistency
C3 failure-injection tests
```

**Explicit non-goals:**

```text
HTTP routes
Connection catalog implementation
Remote Resource discovery
live capability execution
Hybrid Ask
Slack
provider imports
physical Source data deletion
vector deletion
vendor client construction
```

**Required acceptance gate:**

```text
one tenant binding reference
no provider identity duplication
no credential duplication
monotonic workspace revision via head CAS
one pending writer per workspace
durable mutation/idempotency record
safe concurrent conflict
idempotency replay and conflict
semantic duplicate prevention
prepare/publish/recovery protocol
failure-injection tests pass
zero provider-specific imports
```

---

## 28. Explicit non-goals

Hybrid Ask, live Jira/Confluence/Graph queries, MCP execution, provider sync workers in LKW, write capabilities, second credential store, provider-specific LKW pipelines, automatic promotion of live evidence, generic provider execution endpoints.

---

## 29. Open blockers

| Blocker | Severity | Mitigation |
|---------|----------|------------|
| No durable tenant `TenantConnection` catalog | Medium | **`LKW-KNOWLEDGE-ACCESS-1C-1`** — model, repository, service, rehydration |
| No typed live capability catalog | Medium | `TenantLiveCapabilityCatalogPort` in **`1C-2`**; bindings-only in `1D` |
| No live capability executor | Medium | Executor in later platform task |
| `list_source_candidates` not implemented on facade | Low | Use `inspect_scope` + adapter list in **`1C-2`** |
| `CONNECTED_SOURCE` ingestion processor not wired | Medium | Separate intake task after configuration stable |

**Explicitly rejected (do not workaround):** storing raw tokens in LKW database; storing integration objects in DocumentStore; treating `KnowledgeConnectionRegistry` as durable; using `IntegrationProfile` as tenant catalog; copying Connection config into workspaces; copying credentials into attachments; auto-persisting discovered Remote Resources; manual Connection recreation after every restart; separate indexed/live clients; moving deployment topology into workspace state.

**Contract freeze status:** Not blocked - gaps are explicit and sequenced; C3 mutation semantics unchanged.

---

## 30. Final architecture verdict

The repository supports the intended design when LKW stores workspace authorization references to tenant `KnowledgeSourceBinding` records (not duplicated provider identity), maintains one monotonic `committed_revision` via CAS-protected head record with a single pending writer and publication point, uses immutable revisioned child records and a durable `WorkspaceKnowledgeMutationRecord` with `WorkspaceKnowledgeMutationOutcomeV1` for idempotency (including persistent `EXISTING_RESULT` semantic no-ops), assigns `target_revision` only after writer-slot acquisition, proves post-publication recovery from immutable revision rows and staged Source ownership (not solely `last_committed_mutation_id`), uses `delete_if_match` for all rollback deletions, requires canonical `Idempotency-Key` HTTP header for every mutation, reuses `to_source_ref(tenant_binding)` / `ConnectionAwareVendorResolver` / `VendorKnowledgeFacadeService` for indexed paths, validates live capabilities through typed `LiveCapabilityDescriptorV1`, and keeps live execution on a future shared executor. One `WorkspaceSource` continues to own all persisted Documents. Indexed detach is non-destructive via binding status. Connected Sources remain hidden until head publication and carry exact `creation_mutation_id` ownership. Every configuration mutation requires aggregate `If-Match`. Provider-specific LKW models and credential duplication are rejected. Durable tenant `TenantConnection` persistence and restart rehydration are owned by **`LKW-KNOWLEDGE-ACCESS-1C-1`**; safe discovery and capability catalog by **`1C-2`**. `KnowledgeConnectionRegistry` remains runtime-only. `IntegrationProfile` remains application composition. Raw secrets remain in `SecretsStore` only.

**STATUS: `READY_FOR_REVIEW`**
