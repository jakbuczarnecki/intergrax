# Workspace Knowledge Access — Implementation Contract

**Status:** `READY_FOR_REVIEW`  
**Task:** `LKW-KNOWLEDGE-ACCESS-1A — IMPLEMENTATION CONTRACT FREEZE AND EXISTING FOUNDATION AUDIT`  
**Classification:** docs-only architecture-to-implementation contract  
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
- Bounded decomposition of `LKW-KNOWLEDGE-ACCESS-1` into subtasks `1B`–`1F`.
- Security threat review and observable no-duplication proof design.

### 1.3 Out of scope

Hybrid Ask, Knowledge Query Orchestrator, live provider execution, Slack UI, new vendor adapters, MCP as domain model, provider-specific LKW tables, second credential stores, second connection registries, second Source systems, production code changes.

### 1.4 Repository note

`intergrax/applications/local_workspace/` does **not** exist. LKW product code lives under `applications/local_workspace_application/`. Tier-1 Vendor Knowledge lives under `intergrax/runtime/vendor_knowledge/`.

---

## 2. Inspected repository foundations

### 2.1 LKW workspace domain and persistence

| Concern | Verified owner |
|---------|----------------|
| Workspace identity | `Workspace.workspace_id` (`uuid.uuid4()` on create) — `applications/local_workspace_application/workspaces/models.py` |
| Tenant identity | `Workspace.tenant_id`; resolved via `resolve_tenant_id()` in `serving/workspace_routes.py` |
| Principal / authorization | Request context `get_request_context(request).tenant_id` preferred; workspace existence checked via `ManagedWorkspaceService.require_workspace()` → **404** when unknown or cross-tenant |
| Workspace repository | `ManagedWorkspaceRepository` — `workspaces/repository.py` |
| Workspace service | `ManagedWorkspaceService` — `workspaces/service.py` |
| FastAPI routes | `mount_managed_workspace_routes()` — `serving/workspace_routes.py`, prefix `/v1/local_workspace` |
| Public response conventions | `serving/workspace_schemas.py` (`*ResponseV1`, `extra="forbid"` on requests) |
| Error normalization | HTTP `detail` string codes (`not_found`, `workspace_not_found`, domain `error_code` on operations); Ask uses `WorkspaceAskLookupError` → 404 |
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
| `WorkspaceSource` | Durable per-workspace Source; types include `CONNECTED_SOURCE` enum value but **not implemented** |
| Source ID generation | `uuid.uuid4()` for local folder; deterministic `src:knowledge_input_source:{input_id}` for intake-derived sources |
| Tenant/workspace validation | `KnowledgeIntakeService.accept()` and resolvers check `repository.get_workspace()` |
| Source metadata storage | `ManagedWorkspaceRepository.put_source()` row key `{workspace_id}:{source_id}` |
| Provider correlation | **Not present** on `WorkspaceSource` today; `CONNECTED_SOURCE` path is deferred per `docs/plan/KNOWLEDGE_SOURCE_INTEGRATIONS.md` §6 |
| `KnowledgeInputKind` | `managed_file`, `uploaded_folder_snapshot`, `source_candidate`, `web_url` — **no new provider-specific kind required** for first connected-source milestone |
| Document ownership | `WorkspaceDocumentReference` — every indexed document references exactly one `source_id` |
| Operation state | `WorkspaceOperation` + `KnowledgeInput` linked by `operation_id` / `input_id` |

### 2.3 Vendor Knowledge foundation

| Symbol | Role | Reuse for LKW |
|--------|------|---------------|
| `KnowledgeSourceRef` | Tenant-scoped vendor-neutral source identity with optional `connection_ref` | **Direct** — build from LKW bindings for sync/live resolution |
| `KnowledgeSourceScope` | `remote_scope_id`, `remote_scope_type`, `safe_display_name`, safe `parameters` | **Direct** — maps to Remote Resource scope |
| `KnowledgeScopeInfo` | Output of `inspect_scope` | **Direct** — discovery / inspect projection |
| `KnowledgeSourceBinding` | Tenant-scoped durable sync binding (`connection_ref`, optional `credential_ref`) | **Reuse at tenant layer** — not workspace-scoped; LKW must not duplicate |
| `KnowledgeConnectionRegistry` | Instance-local `(tenant_id, connection_ref) → integration` | **Direct** — prevents second client when registered |
| `ConnectionAwareVendorResolver` | Registry-first resolver with profile fallback | **Direct** |
| `IntegrationProfileVendorResolver` | Profile-only; **rejects** `connection_ref` | Fallback path only |
| `VendorKnowledgeFacadeService` | Durable/indexed read path | **Direct** for indexed sync; not for live |
| `KnowledgeAdapterRegistry` | `(provider_id, integration_kind, source_kind) → adapter` | **Direct** |
| `DocumentStoreKnowledgeSourceBindingRepository` | Tenant binding persistence | **Do not duplicate** in LKW |

**Implemented adapters (verified):** Jira issues, Confluence pages, MS365 Graph drive/mail/teams_channel.

**Missing (gap):** durable tenant Connection catalog, `list_source_candidates`, live capability executor, provider-neutral live capability IDs.

### 2.4 Integration and registry foundation

| Path | Role |
|------|------|
| `intergrax/integrations/registry/profile.py` — `IntegrationProfile` | Application composition; `resolve(IntegrationCategory)` returns constructed integration |
| `intergrax/integrations/contracts/secrets_store.py` — `SecretsStore` | Credential storage; secrets never in LKW state |
| `intergrax/tools/registry/runtime.py` — `ToolRegistry` | Tool execution registry (future live path, not LKW domain model) |

**Reusable resolution path (frozen):**

```text
connection_ref + tenant_id + provider_id + integration_kind
→ KnowledgeConnectionRegistry.resolve()   # when registered
→ OR IntegrationProfileVendorResolver     # profile fallback; connection_ref must be None
→ existing integration instance (single)
→ VendorKnowledgeAdapter (indexed) OR LiveCapabilityAdapter (future, platform)
```

LKW configuration stores only `connection_ref`. It never constructs vendor clients.

### 2.5 Representative provider descriptors

| Provider | `provider_id` | `integration_kind` | `source_kind` (examples) | Scope type | Implemented read |
|----------|---------------|--------------------|---------------------------|------------|------------------|
| Jira | `jira` | `issue_tracker` | `jira.issues` | `jira_project` | inventory, content, reconciliation |
| Confluence | `confluence` | `wiki_knowledge` | `confluence.pages` | `confluence_space` | inventory, rich_text content |
| MS365 Graph | `ms365_graph` | `collaboration_suite` | `msgraph.drive`, `msgraph.mail`, `msgraph.teams_channel` | `msgraph_drive`, etc. | delta/incremental reads |

Capabilities are declared per adapter via `KnowledgeAdapterCapabilities`. Live capability IDs are **planned** in architecture docs, **not implemented** as a registry.

### 2.6 Public API and security conventions (existing LKW)

| Rule | Verified behavior |
|------|-------------------|
| Tenant resolution | Auth context > `X-Tenant-Id` > body > `"default"` |
| Cross-tenant | **404** `not_found`, never 403 for workspace/resource existence |
| Validation errors | HTTP 400/422 with stable string `detail` |
| Pagination | List endpoints use repository `limit` (typically 500–2000); no cursor pagination on workspace lists today |
| Opaque IDs | `workspace_id`, `source_id`, `operation_id`, `run_id` |
| Secret redaction | `SourceSummaryResponseV1` omits raw `path` for list; web URL locators keep private URL `repr=False` |
| Response size | Managed file upload max from settings; Ask `limit` capped at 100 |

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
| `KnowledgeSourceBinding` + `DocumentStoreKnowledgeSourceBindingRepository` | Tenant-level sync bindings (optional convergence, not LKW duplicate) |
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
| `KnowledgeSourceBinding` repository at LKW tier | Tenant sync binding ≠ workspace indexed/live authorization |
| `IntegrationProfile` per workspace | Application composition, not workspace product state |
| `SecretsStore` / credential blobs | Credentials stay in integration foundation |
| `VendorKnowledgeFacadeService` inside LKW routes | LKW calls ports; facade stays Tier-1 |
| Provider-specific LKW models (`JiraWorkspaceSource`, etc.) | Rejected by architecture |
| Generic `POST /execute-provider` | Unbounded provider access |

---

## 5. Ownership matrix

| Concern | Owner | Verified / decision |
|---------|-------|---------------------|
| Raw credentials and tokens | Integration / `SecretsStore` | Confirmed — not in LKW models |
| Global tenant Connection | Platform connection foundation | **Gap:** durable catalog not implemented; runtime `KnowledgeConnectionRegistry` + opaque `connection_ref` contract frozen |
| Vendor API client | Existing integration instance | Confirmed via `IntegrationProfile` / connection registry |
| Remote Resource discovery | Vendor Knowledge adapters + future list port | `inspect_scope` exists; list candidates planned |
| Workspace connection attachment | **LKW** | New `WorkspaceConnectionAttachment` record |
| Indexed Source authorization | **LKW** | New `WorkspaceIndexedSourceBinding` + `WorkspaceSource(CONNECTED_SOURCE)` |
| Live Access Binding | **LKW** | New `WorkspaceLiveAccessBinding` |
| Query Policy | **LKW** | New `WorkspaceQueryPolicy` |
| Durable Source | **LKW** | `WorkspaceSource` + `WorkspaceDocumentReference` |
| Documents, chunks, vectors | **LKW** | Existing indexing pipeline |
| Live capability implementation | Shared integration/tool foundation (future) | Not in LKW domain |
| Live execution authorization | LKW + platform policy | LKW binding allowlist + executor gate |
| Live evidence normalization | Provider-neutral boundary (future executor) | Ephemeral by default |
| Slack presentation | Slack frontend only | Confirmed thin-client architecture |

**Discrepancy:** Architecture docs describe a durable tenant Connection record (`connection_id`, `credential_ref`, `status`). Repository has only opaque `connection_ref` on bindings and an **instance-local** registry. **Smallest correction:** introduce a platform `TenantConnectionPort` (read-only for LKW) in task `1C`; LKW never persists Connection metadata beyond cached safe labels on attachments.

---

## 6. Connection implementation decision

### 6.1 What is a Connection in the current codebase?

A **Connection** is not a standalone durable LKW entity today. It is:

1. An opaque **`connection_ref`** string carried on `KnowledgeSourceRef` / `KnowledgeSourceBinding`.
2. A runtime registration in **`KnowledgeConnectionRegistry`** mapping `(tenant_id, connection_ref)` to an already-constructed integration instance with matching `provider_id` and `integration_kind`.
3. An architectural tenant-owned record (documented in `docs/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md` §7.1) **not yet persisted** in code.

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
| `tenant_id` | Must match resolver tenant; cross-tenant ref → fail closed |
| `provider_id` / `integration_kind` | From platform Connection metadata |
| Health / capability projection | Platform port + adapter registry |
| Delete semantics | Removing LKW attachment does not delete tenant Connection |
| Unavailable | `status=unavailable` → discovery and binding mutations rejected; existing bindings → `unavailable` state, no credential copy |

---

## 7. Remote Resource contract

### 7.1 Durability

| Contract | Durability |
|----------|------------|
| `RemoteResourceDescriptorV1` | **Ephemeral** discovery output |
| Optional future `RemoteResourceSnapshotV1` | **Not in 1B** — defer cached snapshots |

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
**Pagination:** cursor `next_page_token: str | None`, `limit` 1–100, opaque token max 4096 chars.  
**Permission loss:** `availability=permission_denied`; bindings transition to `unavailable`, execution fail closed.  
**Duplicate discovery:** dedupe by identity; deterministic sort by `(connection_ref, remote_resource_id, source_kind)`.  
**Unsafe metadata:** map to `KnowledgeSourceScope.parameters` rules — secret keys forbidden, URL credential embedding forbidden.

**Implementation mapping:** build from `KnowledgeScopeInfo` + adapter-specific list operations when added in `1C`.

---

## 8. Indexed Source contract

### 8.1 Decision

**Dedicated workspace binding** `WorkspaceIndexedSourceBinding` that **creates or references** one durable `WorkspaceSource` with `source_type=CONNECTED_SOURCE`. Not a direct unstructured extension of `WorkspaceSource` fields alone.

### 8.2 Model

```python
class IndexedSourceSyncModeV1(StrEnum):
    FULL = "full"
    INCREMENTAL = "incremental"


class IndexedSourceStatusV1(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    SYNCING = "syncing"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class WorkspaceIndexedSourceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indexed_source_binding_id: str = Field(..., min_length=1, max_length=128)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    source_id: str = Field(..., min_length=1, max_length=128)
    connection_ref: str = Field(..., min_length=1, max_length=128)
    remote_resource_id: str = Field(..., min_length=1, max_length=256)
    provider_id: str = Field(..., min_length=1, max_length=64)
    integration_kind: IntegrationCategory
    source_kind: str = Field(..., min_length=1, max_length=64)
    resource_type: str = Field(..., min_length=1, max_length=64)
    safe_display_label: str = Field(..., min_length=1, max_length=256)
    safe_description: str = Field(default="", max_length=1024)
    sync_mode: IndexedSourceSyncModeV1 = IndexedSourceSyncModeV1.INCREMENTAL
    status: IndexedSourceStatusV1 = IndexedSourceStatusV1.ACTIVE
    configuration_version: int = Field(..., ge=1)
    idempotency_key: str = Field(..., min_length=1, max_length=256)
    created_at: datetime
    updated_at: datetime
```

**Source relationship:** exactly one `WorkspaceSource` per binding; `WorkspaceSource.source_type = connected_source`, `path=""`, `recursive=false` (existing validator).  
**Idempotency:** `indexed_source_binding_id = sha256(tenant_id, workspace_id, idempotency_key)` prefixed `idx:`; same key + same payload → 200; conflicting payload → 409 `indexed_source_idempotency_conflict`.  
**Delete:** remove binding + LKW Source metadata; **does not** delete remote provider resource or tenant `KnowledgeSourceBinding`.  
**Live permission:** **not implied** — indexed binding does not create `WorkspaceLiveAccessBinding`.

**Vendor projection for sync:**

```python
def to_knowledge_source_ref(binding: WorkspaceIndexedSourceBinding) -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id=binding.tenant_id,
        provider_id=binding.provider_id,
        integration_kind=binding.integration_kind,
        source_kind=binding.source_kind,
        connection_ref=binding.connection_ref,
        scope=KnowledgeSourceScope(
            remote_scope_id=binding.remote_resource_id,
            remote_scope_type=binding.resource_type,
            safe_display_name=binding.safe_display_label,
            parameters={},
        ),
    )
```

---

## 9. Live Access Binding contract

### 9.1 Model

```python
class LiveAccessBindingStatusV1(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    REVOKED = "revoked"


class WorkspaceLiveAccessBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    live_access_binding_id: str = Field(..., min_length=1, max_length=128)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    connection_ref: str = Field(..., min_length=1, max_length=128)
    remote_resource_id: str | None = Field(default=None, max_length=256)
    resource_type: str | None = Field(default=None, max_length=64)
    provider_id: str = Field(..., min_length=1, max_length=64)
    integration_kind: IntegrationCategory
    allowed_capability_ids: tuple[str, ...] = Field(..., min_length=1)
    read_only_mode: Literal[True] = True
    status: LiveAccessBindingStatusV1 = LiveAccessBindingStatusV1.ACTIVE
    policy_reference: str | None = Field(default=None, max_length=128)
    configuration_version: int = Field(..., ge=1)
    idempotency_key: str = Field(..., min_length=1, max_length=256)
    created_at: datetime
    updated_at: datetime
```

**Validation rules:**

- Every `allowed_capability_id` must match `^[a-z][a-z0-9._-]{1,127}$`.
- Write-capable capability IDs (suffix `.write`, `.create`, `.delete`, `.update`) → **rejected** at create/update.
- `remote_resource_id` required when any capability is resource-scoped (adapter declares scope requirement).
- Unknown connection, unauthorized workspace, unknown capability, resource outside connection → **fail closed** (404 workspace/connection, 400 validation).
- Duplicate binding same `(workspace_id, connection_ref, remote_resource_id, capability set)` → 409 `live_access_binding_duplicate`.
- Does **not** imply durable ingestion rights.

---

## 10. Query Policy contract

### 10.1 First-milestone subset

| Mode | Supported in 1E |
|------|-----------------|
| `indexed_only` | **Yes** |
| `live_only` | **Yes** |
| `hybrid` | **No** — explicit 400 `query_policy_mode_unsupported` |
| `automatic` | **No** — explicit 400 `query_policy_mode_unsupported` |

### 10.2 Model

```python
class QueryPolicyModeV1(StrEnum):
    INDEXED_ONLY = "indexed_only"
    LIVE_ONLY = "live_only"


class LiveResultRetentionV1(StrEnum):
    EPHEMERAL = "ephemeral"
    RECEIPT_ONLY = "receipt_only"


class WorkspaceQueryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    mode: QueryPolicyModeV1 = QueryPolicyModeV1.INDEXED_ONLY
    prefer_indexed_evidence: bool = True
    allow_live_fallback: bool = False
    allowed_connection_refs: tuple[str, ...] = ()
    allowed_capability_ids: tuple[str, ...] = ()
    max_live_calls: int = Field(default=5, ge=0, le=50)
    max_total_duration_ms: int = Field(default=30_000, ge=1, le=300_000)
    max_result_items: int = Field(default=50, ge=1, le=500)
    max_result_bytes: int = Field(default=1_048_576, ge=1, le=16_777_216)
    live_result_retention: LiveResultRetentionV1 = LiveResultRetentionV1.EPHEMERAL
    configuration_version: int = Field(..., ge=1)
    updated_at: datetime
```

**Defaults:** `mode=indexed_only`, `live_result_retention=ephemeral`.  
**Unsupported mode behavior:** reject request; never accept-and-ignore.

---

## 11. Workspace Knowledge Configuration aggregate

### 11.1 Representation

**Projection** assembled by `WorkspaceKnowledgeConfigurationService` from:

- `WorkspaceConnectionAttachment` (0..n)
- `WorkspaceIndexedSourceBinding` (0..n)
- `WorkspaceLiveAccessBinding` (0..n)
- `WorkspaceQueryPolicy` (0..1)
- existing `WorkspaceSource` / `Workspace` records

Not one mutable JSON blob.

### 11.2 Projection model

```python
class WorkspaceKnowledgeConfigurationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    workspace_id: str
    configuration_version: int = Field(..., ge=1)
    connection_attachments: tuple[WorkspaceConnectionAttachment, ...]
    indexed_sources: tuple[WorkspaceIndexedSourceBinding, ...]
    live_access_bindings: tuple[WorkspaceLiveAccessBinding, ...]
    query_policy: WorkspaceQueryPolicy | None
    updated_at: datetime
```

**Versioning:** `configuration_version = max(child.configuration_version)`; bump on any child mutation.  
**Concurrency:** `If-Match: WKC/{configuration_version}` header on mutating endpoints; mismatch → 409 `configuration_version_conflict`.  
**Deterministic ordering:** sort attachments by `connection_ref`, indexed by `indexed_source_binding_id`, live by `live_access_binding_id`.  
**Empty state:** all child collections empty, `query_policy=None`, `configuration_version=1`.  
**Partial updates:** resource-scoped endpoints mutate one child; projection recomputed.

### 11.3 Workspace connection attachment

```python
class WorkspaceConnectionAttachmentStatusV1(StrEnum):
    ATTACHED = "attached"
    UNAVAILABLE = "unavailable"
    DETACHED = "detached"


class WorkspaceConnectionAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attachment_id: str = Field(..., min_length=1, max_length=128)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    connection_ref: str = Field(..., min_length=1, max_length=128)
    safe_display_label: str = Field(..., min_length=1, max_length=256)
    status: WorkspaceConnectionAttachmentStatusV1
    configuration_version: int = Field(..., ge=1)
    idempotency_key: str = Field(..., min_length=1, max_length=256)
    created_at: datetime
    updated_at: datetime
```

---

## 12. Persistence design

### 12.1 Records

| Record | Partition | Row key | Unique constraint |
|--------|-----------|---------|-------------------|
| `WorkspaceConnectionAttachment` | `lkw.managed_workspace:{tenant_id}:connection_attachment` | `{workspace_id}:{attachment_id}` | `(tenant_id, workspace_id, connection_ref)` |
| `WorkspaceIndexedSourceBinding` | `lkw.managed_workspace:{tenant_id}:indexed_source_binding` | `{workspace_id}:{indexed_source_binding_id}` | `(tenant_id, workspace_id, idempotency_key)` |
| `WorkspaceLiveAccessBinding` | `lkw.managed_workspace:{tenant_id}:live_access_binding` | `{workspace_id}:{live_access_binding_id}` | `(tenant_id, workspace_id, idempotency_key)` |
| `WorkspaceQueryPolicy` | `lkw.managed_workspace:{tenant_id}:query_policy` | `{workspace_id}` | one per workspace |
| `WorkspaceSource` (existing) | `lkw.managed_workspace:{tenant_id}:source` | `{workspace_id}:{source_id}` | — |

### 12.2 Repository

Extend `ManagedWorkspaceRepository` with typed put/get/list/delete methods mirroring existing Source/Operation patterns. No new DocumentStore implementation.

### 12.3 Migrations

Additive partitions only. Existing workspaces: empty knowledge configuration (backward compatible). `CONNECTED_SOURCE` sources absent until explicit binding create.

### 12.4 Deletion semantics

| Action | Effect |
|--------|--------|
| Delete indexed binding | Remove binding + workspace Source + document refs via existing workspace delete patterns for that source; no upstream delete |
| Delete live binding | Remove binding only; no Document changes |
| Detach connection | `status=detached`; indexed/live bindings for that `connection_ref` → `unavailable` |
| Delete workspace | Existing `delete_workspace()` extended to purge new partitions |

---

## 13. Service and repository boundaries

```text
WorkspaceKnowledgeConfigurationService (LKW)
├── ManagedWorkspaceRepository (durable LKW records)
├── ManagedWorkspaceService (workspace existence authority)
├── TenantConnectionPort (read-only; platform — new in 1C)
├── RemoteResourceDiscoveryPort (wraps VendorKnowledgeFacade inspect/list — 1C)
└── WorkspaceKnowledgeAuthorizationService (tenant + workspace + binding checks)

VendorKnowledgeFacadeService (Tier-1, unchanged)
├── ConnectionAwareVendorResolver
├── KnowledgeAdapterRegistry
└── existing integrations (no LKW import)
```

LKW services **must not** import provider packages (`jira`, `confluence`, `ms365_graph`).

---

## 14. Public API proposal

Base prefix: `/v1/local_workspace`. All endpoints require resolved `tenant_id`. Workspace-scoped endpoints return **404** when workspace unknown (including cross-tenant).

### 14.1 List safe Connections

| | |
|--|--|
| Method / path | `GET /connections` |
| Response | `SafeConnectionListResponseV1 { connections: list[SafeConnectionSummaryV1] }` |
| Auth | Tenant context |
| Pagination | `limit` 1–100, optional `page_token` |
| Secrets | Projection only — no `credential_ref` |
| Errors | 401 unauthenticated |

### 14.2 Inspect Connection

| | |
|--|--|
| Method / path | `GET /connections/{connection_ref}` |
| Response | `SafeConnectionSummaryV1` |
| Errors | 404 `connection_not_found` (including cross-tenant) |

### 14.3 Discover Remote Resources

| | |
|--|--|
| Method / path | `GET /connections/{connection_ref}/remote-resources` |
| Query | `source_kind`, `limit`, `page_token`, optional `filter` (max 128 chars) |
| Response | `RemoteResourceListResponseV1 { items, next_page_token }` |
| Auth | Tenant + connection ownership |
| Errors | 404 connection; 400 unsupported `source_kind`; 503 `connection_unavailable` |

### 14.4 Attach Connection to workspace

| | |
|--|--|
| Method / path | `PUT /workspaces/{workspace_id}/connections/{connection_ref}` |
| Request | `AttachConnectionRequestV1 { idempotency_key, safe_display_label? }` |
| Response | `WorkspaceConnectionAttachment` + `configuration_version` |
| Idempotency | Same `idempotency_key` → 200 replay |
| Errors | 404 workspace/connection; 409 conflict |

### 14.5 Read Workspace Knowledge Configuration

| | |
|--|--|
| Method / path | `GET /workspaces/{workspace_id}/knowledge-configuration` |
| Response | `WorkspaceKnowledgeConfigurationV1` |
| Errors | 404 workspace |

### 14.6 Create Indexed Source binding

| | |
|--|--|
| Method / path | `POST /workspaces/{workspace_id}/indexed-sources` |
| Request | `CreateIndexedSourceRequestV1` (connection_ref, remote_resource_id, source_kind, resource_type, safe_display_label, sync_mode, idempotency_key) |
| Response | 201 `WorkspaceIndexedSourceBinding` |
| Idempotency | `Idempotency-Key` header or body key |
| Errors | 400 validation; 404 workspace/connection/resource; 409 idempotency conflict |

### 14.7 Create Live Access Binding

| | |
|--|--|
| Method / path | `POST /workspaces/{workspace_id}/live-access-bindings` |
| Request | `CreateLiveAccessBindingRequestV1` |
| Response | 201 `WorkspaceLiveAccessBinding` |
| Errors | 400 write capability / unknown capability; 404 |

### 14.8 Update Query Policy

| | |
|--|--|
| Method / path | `PUT /workspaces/{workspace_id}/query-policy` |
| Request | `UpdateQueryPolicyRequestV1` + optional `If-Match` |
| Response | `WorkspaceQueryPolicy` |
| Errors | 400 unsupported mode; 409 version conflict |

### 14.9 Delete Indexed Source binding

| | |
|--|--|
| Method / path | `DELETE /workspaces/{workspace_id}/indexed-sources/{indexed_source_binding_id}` |
| Response | 204 |
| Errors | 404 |

### 14.10 Delete Live Access Binding

| | |
|--|--|
| Method / path | `DELETE /workspaces/{workspace_id}/live-access-bindings/{live_access_binding_id}` |
| Response | 204 |
| Errors | 404 |

---

## 15. Authorization and safe error behavior

| Boundary | Behavior |
|----------|----------|
| Tenant | `resolve_tenant_id()`; data queries always include `tenant_id` partition |
| Workspace | `get_workspace()` None → **404** |
| Principal | Request context principal (when present) must match tenant; future fine-grained workspace ACL hooks at service layer |
| Connection | `connection_ref` must resolve for same `tenant_id`; else 404 |
| Resource | `remote_resource_id` must be discovered under connection before binding |
| Capability | Must be in allowlist and declared read-only |
| Fail closed | Unknown/stale/unauthorized → 404 or 400; never silent downgrade |

Safe errors: stable snake_case `detail` string; no `connection_ref` in error messages (matches `test_connections.py` pattern).

---

## 16. Idempotency and configuration-version semantics

| Operation | Key | Conflict |
|-----------|-----|----------|
| Attach connection | `(tenant_id, workspace_id, idempotency_key)` | 409 |
| Create indexed source | `(tenant_id, workspace_id, idempotency_key)` | 409 if payload differs |
| Create live binding | same | 409 |
| Update query policy | `If-Match: WKC/{version}` | 409 `configuration_version_conflict` |

Child record `configuration_version` starts at 1; increments on update. Workspace aggregate version derived as max.

---

## 17. No-duplication proof (future acceptance test — `1F`)

### 17.1 Scenario

```text
one tenant Connection (connection_ref = conn-proof-1)
→ attached to workspace W
→ remote resource R selected
→ WorkspaceIndexedSourceBinding I
→ WorkspaceLiveAccessBinding L
```

### 17.2 Required invariants

| Invariant | Observable check |
|-----------|-------------------|
| Same `connection_ref` on I and L | Assert record fields |
| Same credential reference | Spy `SecretsStore` lookup count == 1 per operation window |
| Same integration registration | `KnowledgeConnectionRegistry.resolve` call count; returned object `id()` equal for indexed sync stub and live stub |
| No second vendor client | Patch integration constructor counter == 1 per `(tenant_id, connection_ref)` |
| Independent authorization | Disable L → indexed sync still allowed; disable I → live still allowed if policy permits |
| Same workspace boundary | Cross-workspace binding attempt → 404 |

### 17.3 Test harness sketch

Inject instrumented `KnowledgeConnectionRegistry`, `ConnectionAwareVendorResolver`, and `SecretsStore` fake into wiring used by configuration service and facade. Use existing vendor knowledge fakes from `tests/unit/runtime/vendor_knowledge/_fakes.py`.

---

## 18. Security threat review

| Threat | Boundary | Prevention | Failure | Test |
|--------|----------|------------|---------|------|
| Credential leakage | LKW persistence / API | Forbidden field scan on serialize; no secret keys in models | 500 corrupt record / reject write | Unit: binding store secret rejection pattern |
| Cross-tenant connection ref | Resolver + repository | Partition + tenant match on all reads | 404 | Integration: tenant A ref in tenant B workspace |
| Cross-workspace binding | Workspace service | `workspace_id` on all records | 404 | API test |
| Capability escalation | Live binding create | Deny write suffix capabilities | 400 | Unit validator |
| Write capability exposed | Live executor (future) | Read-only allowlist + executor gate | 403/400 | Contract test |
| Resource reference substitution | Binding create | Resource must be discovered under same `connection_ref` | 400 | Integration |
| Unsafe provider locator | Remote resource / evidence | `KnowledgeSourceScope` safe mapping rules | 400 validation | Reuse vendor_knowledge model tests |
| Stale capability descriptor | Live execution | Re-validate against adapter registry at execution | `unavailable` | Future executor test |
| Provider permission loss | Discovery + execution | `availability` enum + binding `unavailable` | Fail closed | Simulated adapter denial |
| Live result persisted | Ask / executor | `ephemeral` retention default; no Document write path | Assert no `put_document_ref` | Integration |
| Duplicate vendor client | Connection registry | Single registration per ref | Constructor count | **1F proof** |
| MCP arbitrary exposure | LKW domain | MCP not in configuration models | N/A | Schema scan |
| Oversized provider result | Query policy | `max_result_bytes`, `max_result_items` | Truncate + receipt | Policy unit test |

---

## 19. Migration and backward-compatibility impact

- **Additive only** — new DocumentStore partitions and routes.
- Existing workspaces without knowledge configuration: valid empty projection.
- `WorkspaceSourceType.CONNECTED_SOURCE` enum already exists; first binding implementation activates it.
- No change to existing intake kinds (`web_url`, `managed_file`, etc.).
- Ask remains indexed-only until Hybrid Ask (out of scope).
- Roadmap status `LKW-KNOWLEDGE-ACCESS-1 → NEXT` unchanged.

---

## 20. Implementation decomposition

### 20.1 `LKW-KNOWLEDGE-ACCESS-1A` (this task)

**Outcome:** Freeze implementation contract and audit foundations.  
**Dependencies:** Accepted architecture.  
**Code areas:** docs only.  
**Non-goals:** Production code.  
**Tests:** `git diff --check`, manual link/symbol verification.  
**Gate:** `READY_FOR_REVIEW` on this document.

### 20.2 `LKW-KNOWLEDGE-ACCESS-1B` — provider-neutral durable configuration foundation

**Outcome:** Typed LKW contracts, repository records, configuration service, idempotency and versioning unit tests.  
**Dependencies:** 1A.  
**Code areas:** `workspaces/models.py` (or `knowledge_access_models.py`), `workspaces/repository.py`, new `workspaces/knowledge_configuration_service.py`, tests under `tests/workspaces/`.  
**Non-goals:** Provider discovery, live calls, HTTP routes, Slack.  
**Tests:** Repository round-trip, idempotency conflict, version conflict, secret-field rejection.  
**Gate:** All 1B unit tests green; no provider imports in LKW.

### 20.3 `LKW-KNOWLEDGE-ACCESS-1C` — safe Connection listing and Remote Resource discovery

**Outcome:** `TenantConnectionPort`, safe connection list/inspect, remote resource discovery HTTP endpoints.  
**Dependencies:** 1B, `KnowledgeConnectionRegistry` wiring.  
**Code areas:** `serving/workspace_routes.py`, `serving/workspace_schemas.py`, host wiring, platform connection read port.  
**Non-goals:** Indexed/live binding mutations.  
**Tests:** API tests with fakes; cross-tenant 404; pagination.  
**Gate:** Discovery returns `RemoteResourceDescriptorV1` without secrets.

### 20.4 `LKW-KNOWLEDGE-ACCESS-1D` — Indexed Source and Live Access Binding HTTP configuration

**Outcome:** Create/delete indexed and live bindings via HTTP.  
**Dependencies:** 1B, 1C.  
**Code areas:** routes, schemas, `WorkspaceKnowledgeConfigurationService`.  
**Non-goals:** Actual sync or live execution.  
**Tests:** API acceptance, idempotency, independent binding authorization.  
**Gate:** `CONNECTED_SOURCE` workspace Source created; no live binding auto-created.

### 20.5 `LKW-KNOWLEDGE-ACCESS-1E` — Query Policy and complete configuration projection

**Outcome:** Query policy CRUD + `GET knowledge-configuration` aggregate.  
**Dependencies:** 1D.  
**Non-goals:** Hybrid/automatic modes.  
**Tests:** Unsupported mode rejection; deterministic ordering; version concurrency.  
**Gate:** Full projection matches stored records.

### 20.6 `LKW-KNOWLEDGE-ACCESS-1F` — one-Connection indexed/live reuse proof

**Outcome:** Observable proof test — one connection, one integration instance, no credential copy.  
**Dependencies:** 1E + minimal live executor stub OR facade-only proof with shared resolver instrumentation.  
**Non-goals:** Production live queries.  
**Tests:** Instrumented acceptance test per §17.  
**Gate:** All invariants green in CI proof module.

---

## 21. First implementation task

### Recommended: `LKW-KNOWLEDGE-ACCESS-1B`

**Ready-to-use task summary for next session:**

Implement provider-neutral durable Workspace Knowledge Configuration foundations in LKW: add `WorkspaceConnectionAttachment`, `WorkspaceIndexedSourceBinding`, `WorkspaceLiveAccessBinding`, and `WorkspaceQueryPolicy` Pydantic models with frozen validation rules; extend `ManagedWorkspaceRepository` with partitioned DocumentStore persistence and optimistic `configuration_version` handling; introduce `WorkspaceKnowledgeConfigurationService` with tenant/workspace fail-closed checks, deterministic idempotency IDs, and projection assembly without provider imports; cover repository round-trips, idempotency conflicts, version conflicts, and secret-field rejection in unit tests. Exclude HTTP routes, provider discovery, live execution, and Slack.

---

## 22. Explicit non-goals

Hybrid Ask, live Jira/Confluence/Graph queries, MCP execution, provider sync workers in LKW, write capabilities, second credential store, provider-specific LKW pipelines, automatic promotion of live evidence, generic provider execution endpoints.

---

## 23. Open blockers

| Blocker | Severity | Mitigation |
|---------|----------|------------|
| No durable tenant Connection catalog | Medium | `TenantConnectionPort` + runtime registry for proofs (`1C`) |
| No live capability executor / capability ID registry | Medium | Bindings-only in `1D`; executor in later platform task |
| `list_source_candidates` not implemented on facade | Low | Use `inspect_scope` + adapter list in `1C` |
| `CONNECTED_SOURCE` ingestion processor not wired | Medium | Separate intake task after configuration stable |

**Contract freeze status:** Not blocked — gaps are explicit and sequenced.

---

## 24. Final architecture verdict

The repository supports the intended design when LKW stores only opaque `connection_ref` values and workspace-scoped bindings, reuses `KnowledgeSourceRef` / `ConnectionAwareVendorResolver` / `VendorKnowledgeFacadeService` for indexed paths, and keeps live execution on a future shared executor. One `WorkspaceSource` continues to own all persisted Documents. Provider-specific LKW models and credential duplication are rejected.

**STATUS: `READY_FOR_REVIEW`**
