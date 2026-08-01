# Workspace Knowledge Access - Implementation Contract

**Status:** `READY_FOR_REVIEW`
**Task:** `LKW-KNOWLEDGE-ACCESS-1A-C2 - UTF-8 REPAIR, REVISIONED PUBLICATION PROTOCOL AND IDEMPOTENCY CONTRACT CLOSEOUT`
**Prior task:** `LKW-KNOWLEDGE-ACCESS-1A-C1` (commit `8d9c1be78af7ab62750d6d52df4329f9acf59986`)
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
| `WorkspaceSource` | Durable per-workspace Source; types include `CONNECTED_SOURCE` enum value but **not implemented** |
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
| `KnowledgeConnectionRegistry` | Instance-local `(tenant_id, connection_ref) -> integration` | **Direct** - prevents second client when registered |
| `ConnectionAwareVendorResolver` | Registry-first resolver with profile fallback | **Direct** |
| `IntegrationProfileVendorResolver` | Profile-only; **rejects** `connection_ref` | Fallback path only |
| `VendorKnowledgeFacadeService` | Durable/indexed read path | **Direct** for indexed sync; not for live |
| `KnowledgeAdapterRegistry` | `(provider_id, integration_kind, source_kind) -> adapter` | **Direct** |
| `DocumentStoreKnowledgeSourceBindingRepository` | Tenant binding persistence | **Do not duplicate** in LKW |

**Implemented adapters (verified):** Jira issues, Confluence pages, MS365 Graph drive/mail/teams_channel.

**Missing (gap):** durable tenant Connection catalog, `list_source_candidates`, live capability executor, provider-neutral live capability IDs.

### 2.4 Integration and registry foundation

| Path | Role |
|------|------|
| `intergrax/integrations/registry/profile.py` - `IntegrationProfile` | Application composition; `resolve(IntegrationCategory)` returns constructed integration |
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

| Concern | Owner | Verified / decision |
|---------|-------|---------------------|
| Raw credentials and tokens | Integration / `SecretsStore` | Confirmed - not in LKW models |
| Global tenant Connection | Platform connection foundation | **Gap:** durable catalog not implemented; runtime `KnowledgeConnectionRegistry` + opaque `connection_ref` contract frozen |
| Vendor API client | Existing integration instance | Confirmed via `IntegrationProfile` / connection registry |
| Remote Resource discovery | Vendor Knowledge adapters + future list port | `inspect_scope` exists; list candidates planned |
| Workspace connection attachment | **LKW** | New `WorkspaceConnectionAttachment` record |
| Tenant knowledge source binding | **Vendor Knowledge** (`KnowledgeSourceBinding`) | Authoritative provider/resource/scope/connection; LKW references via `knowledge_source_binding_ref` |
| Indexed Source authorization | **LKW** | `WorkspaceIndexedSourceBinding` (authorization reference) + `WorkspaceSource(CONNECTED_SOURCE)` |
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
3. An architectural tenant-owned record (documented in `docs/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md` Section 7.1) **not yet persisted** in code.

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
| Health / capability projection | Platform port + adapter registry |
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

**Implementation mapping:** build from `KnowledgeScopeInfo` + adapter-specific list operations when added in `1C`.

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
(tenant_id, workspace_id, operation, idempotency_key)
```

- Same key + same normalized request -> replay existing result (200).
- Same key + different normalized request -> 409 `configuration_idempotency_conflict`.

**Semantic identity** (logical resource):

```text
(tenant_id, workspace_id, knowledge_source_binding_ref)
```

First milestone: one Indexed Source authorization per tenant binding per workspace (`sync_mode` excluded from semantic identity).

**Duplicate behavior (frozen):** second request with a different `idempotency_key` but the same semantic identity and an ACTIVE committed record -> **return existing binding** (200) with stable `indexed_source_binding_id`. Do not create a revision.

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

Task `1C` (or a separate prerequisite) establishes capability discovery/catalog. Task `1D` may persist Live Access Bindings only against validated read-only descriptors. Arbitrary capability IDs from the frontend are rejected.

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
- Duplicate binding same semantic identity -> return existing binding (200).
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

**Concurrency:** every Workspace Knowledge Configuration mutation requires `If-Match: WKC/{committed_revision}`.

| Condition | Behavior |
|-----------|----------|
| Missing `If-Match` | 428 `precondition_required`; stable error `knowledge_configuration_if_match_required` |
| Mismatch | 409 `configuration_revision_conflict` |
| Committed idempotency replay | May return existing result even when caller sends older `If-Match`, but only after idempotency key and normalized request hash are proven identical |

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
    semantic_identity_hash: str | None = Field(default=None, min_length=64, max_length=64)

    target_revision: int = Field(..., ge=1)
    status: WorkspaceKnowledgeMutationStatusV1

    result_entity_type: str | None = Field(default=None, max_length=64)
    result_entity_id: str | None = Field(default=None, max_length=128)

    error_code: str | None = Field(default=None, max_length=128)

    created_at: datetime
    updated_at: datetime
    committed_at: datetime | None = None
```

### 12.3 Persistence

```text
partition: lkw.managed_workspace:{tenant_id}:knowledge_configuration_mutation
row key: {workspace_id}:{operation}:{sha256(idempotency_key)}
```

Do not put the raw idempotency key in the row key. The mutation row is the durable idempotency record. Do not introduce a second idempotency storage design.

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
```

**No existing mutation record:** create `RESERVED` using `put_if_absent`. If the conditional insert loses a race, reload the winning record.

**Existing record with different request hash:** return `409 configuration_idempotency_conflict`. This applies regardless of status.

**Existing COMMITTED record with the same request hash:** return the previously committed result. Do not increment revision again.

**Existing RESERVED or PREPARED record with the same request hash:** run deterministic recovery - inspect the head; inspect staged records; complete publication when staged state is complete and valid; otherwise abort and clean the staged mutation. Do not create a second mutation for the same idempotency key while the previous one is pending.

**Existing ABORTED record with the same request hash:** a retry may replace the aborted record through conditional compare-and-swap with new `mutation_id`, new target revision, `status = RESERVED`. The same row key remains.

**Existing RECOVERY_REQUIRED record:** return `503 configuration_recovery_required` until recovery completes.

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

---

## 14. Exact prepare and publish protocol

Freeze the following algorithm for every successful mutation. This is the only allowed transaction design for `1B`.

### Phase 1 - validation and normalization

Before any persistent mutation:

```text
require workspace
authorize tenant/workspace
validate referenced tenant binding / connection / capability
normalize request
calculate request hash
calculate semantic identity
resolve deterministic entity IDs
validate If-Match
```

No durable child record is changed in this phase.

### Phase 2 - reserve idempotency

Create or resolve `WorkspaceKnowledgeMutationRecord`. The target revision is not finalized until the writer slot is acquired.

### Phase 3 - ensure head

Ensure the idle revision-0 head exists using `put_if_absent`.

### Phase 4 - acquire the single writer slot

Given idle head with committed revision `N`, CAS:

```text
before:
committed_revision = N
pending_revision = None
pending_mutation_id = None

after:
committed_revision = N
pending_revision = N + 1
pending_mutation_id = mutation_id
```

Use `replace_if_match`. If CAS fails -> `409 configuration_revision_conflict` unless the competing pending mutation requires recovery, in which case -> `503 configuration_recovery_required`.

Set mutation `target_revision = N + 1`.

### Phase 5 - write hidden staged records

Write all required immutable child versions with `effective_revision = N + 1`, `mutation_id = current mutation`.

For Indexed Source creation, also create the pending connected Source according to Section 15. These records remain invisible because `effective_revision > head.committed_revision`.

### Phase 6 - validate staged state

Read back all staged records. Validate correct tenant, workspace, `mutation_id`, target revision, expected semantic identity, expected result ID, expected Source relationship, no forbidden provider/credential fields.

Only after successful validation set mutation status `PREPARED` through conditional replacement.

### Phase 7 - atomic publication point

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

### Phase 8 - finalize idempotency result

Replace the mutation record `PREPARED -> COMMITTED`. Store `result_entity_type`, `result_entity_id`, `committed_at`.

When mutation-record finalization fails after the head has committed:

```text
- do not roll back the head;
- retry/recovery observes last_committed_mutation_id;
- repair the mutation record to COMMITTED;
- replay returns the committed result.
```

### Multi-record Source creation ordering

For Indexed Source creation use this exact staged set:

```text
1. WorkspaceIndexedSourceBinding revision row
2. WorkspaceSource row with visibility revision
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

### 15.2 Required WorkspaceSource field (to be implemented in 1B)

```python
knowledge_configuration_visibility_revision: int | None = Field(default=None, ge=1)
```

Meaning:

```text
None -> legacy/non-connected Source, always visible under existing rules
integer N -> connected Source is product-visible only when
  WorkspaceKnowledgeConfigurationHead.committed_revision >= N
```

New connected Source creation:

```text
source_type = CONNECTED_SOURCE
path = ""
recursive = False
knowledge_configuration_visibility_revision = target revision
```

### 15.3 Visibility rule

Product services that list or expose Sources must hide a connected Source when `visibility revision > committed configuration revision`. Internal repository and recovery code may access hidden staged Sources. This prevents the Source from becoming visible before the configuration head is committed.

### 15.4 Logical detach

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

### 16.1 Failure before head publication

When any failure occurs before Phase 7:

```text
1. Delete every staged child revision belonging to mutation_id.
2. Delete any staged connected WorkspaceSource belonging to mutation_id.
3. Verify that staged records no longer exist.
4. CAS head from pending mutation back to idle revision N.
5. Mark mutation ABORTED.
```

The pending writer slot must not be released before cleanup succeeds.

### 16.2 Cleanup failure

When any staged record cannot be removed:

```text
- do not clear pending_revision;
- do not allow another mutation;
- mark mutation RECOVERY_REQUIRED when possible;
- return configuration_recovery_required.
```

This prevents an abandoned staged record from becoming visible under a future revision.

### 16.3 Failure after head publication

Once Phase 7 succeeds, the mutation is committed. Do not delete committed child rows. Do not decrement the revision. Repair only the mutation/idempotency record.

### 16.4 Required recovery operation (to be implemented in 1B)

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
load staged rows
either complete valid PREPARED publication
or clean incomplete staged state
release the head only after successful cleanup
repair COMMITTED mutation record when head already committed it
```

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
| `WorkspaceKnowledgeMutationRecord` | `lkw.managed_workspace:{tenant_id}:knowledge_configuration_mutation` | `{workspace_id}:{operation}:{sha256(idempotency_key)}` | durable idempotency |
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
├── TenantConnectionPort (read-only; platform - 1C)
├── RemoteResourceDiscoveryPort (wraps VendorKnowledgeFacade inspect/list - 1C)
├── TenantLiveCapabilityCatalogPort (typed read-only capability catalog - 1C)
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
-> validation and normalization (Phase 1)
-> reserve idempotency (Phase 2)
-> ensure head (Phase 3)
-> acquire writer slot (Phase 4)
-> write staged binding + connected Source (Phase 5)
-> validate staged state (Phase 6)
-> publish head (Phase 7)
-> finalize mutation record (Phase 8)
```

`DocumentStore` has no cross-record transaction. Atomicity is provided by the revision-head CAS publication point plus staged-record cleanup on failure. Do not claim multi-record mutation atomicity beyond the frozen protocol.

---

## 20. Public API proposal

Base prefix: `/v1/local_workspace`. All endpoints require resolved `tenant_id`. Workspace-scoped endpoints return **404** when workspace unknown (including cross-tenant).

Every mutating endpoint requires `If-Match: WKC/{committed_revision}`. Initial empty configuration uses `If-Match: WKC/0`.

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
| Request | `AttachConnectionRequestV1 { idempotency_key, safe_display_label? }` |
| Response | `WorkspaceConnectionAttachment` + `configuration_revision` |
| If-Match | **Required** `WKC/{committed_revision}`; missing -> 428; mismatch -> 409 |
| Idempotency replay | Same key + same request -> 200 existing result; no new revision |
| Idempotency conflict | Same key + different request -> 409 `configuration_idempotency_conflict` |
| Semantic duplicate | Same `(tenant_id, workspace_id, connection_ref)` ACTIVE -> 200 existing; no new revision |
| Resulting revision | `committed_revision + 1` on success |
| Errors | 404 workspace/connection; 409 conflict; 428 missing If-Match |

### 20.5 Detach Connection from workspace

| | |
|--|--|
| Method / path | `PATCH /workspaces/{workspace_id}/connections/{connection_ref}` with `{ "status": "detached" }` or `DELETE .../connections/{connection_ref}` |
| Response | 204 (logical detach) |
| If-Match | **Required** `WKC/{committed_revision}`; missing -> 428; mismatch -> 409 |
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
| Request | `CreateWorkspaceIndexedSourceRequestV1 { knowledge_source_binding_ref, sync_mode, idempotency_key }` |
| Server-derived | `provider_id`, `integration_kind`, `source_kind`, `connection_ref`, scope, `safe_display_label` - from tenant binding; **not in request schema** |
| Response | 201 `WorkspaceIndexedSourceBinding` |
| If-Match | **Required** `WKC/{committed_revision}`; first mutation `WKC/0`; missing -> 428; mismatch -> 409 |
| Idempotency replay | Same key + same request -> 200 existing result; no new revision |
| Idempotency conflict | Same key + different request -> 409 `configuration_idempotency_conflict` |
| Semantic duplicate | Same `knowledge_source_binding_ref` ACTIVE -> 200 existing binding; no new revision |
| Resulting revision | `committed_revision + 1` on success |
| Errors | 400 validation; 404 workspace/tenant-binding; 409 idempotency/revision conflict; 428 missing If-Match |

```python
class CreateWorkspaceIndexedSourceRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_source_binding_ref: str = Field(..., min_length=1, max_length=128)
    sync_mode: IndexedSourceSyncModeV1 = IndexedSourceSyncModeV1.INCREMENTAL
    idempotency_key: str = Field(..., min_length=1, max_length=256)
```

### 20.8 Disable / detach Indexed Source binding

| | |
|--|--|
| Method / path | `PATCH /workspaces/{workspace_id}/indexed-sources/{indexed_source_binding_id}` with `{ "status": "disabled" }` |
| Alternative | `DELETE /workspaces/{workspace_id}/indexed-sources/{indexed_source_binding_id}` - **logical detach only**, not physical indexed-data deletion |
| Response | 204 (logical detach) |
| If-Match | **Required** `WKC/{committed_revision}`; missing -> 428; mismatch -> 409 |
| Idempotency replay | Same key + same request -> 200 existing result; no new revision |
| Resulting revision | `committed_revision + 1` on success |
| Effect | Binding -> `DISABLED` (new revision); Source unchanged; Documents/Chunks/Vectors **preserved** |
| Errors | 404; 409; 428 |

### 20.9 Create Live Access Binding

| | |
|--|--|
| Method / path | `POST /workspaces/{workspace_id}/live-access-bindings` |
| Request | `CreateWorkspaceLiveAccessBindingRequestV1 { connection_ref, remote_resource_id, allowed_capability_ids, idempotency_key }` |
| Server-derived | `provider_id`, `integration_kind`, `resource_type`, `safe_display_label`, capability effect classification - validated via `TenantLiveCapabilityCatalogPort` |
| Response | 201 `WorkspaceLiveAccessBinding` |
| If-Match | **Required** `WKC/{committed_revision}`; missing -> 428; mismatch -> 409 |
| Idempotency replay | Same key + same request -> 200 existing result; no new revision |
| Semantic duplicate | Same semantic identity ACTIVE -> 200 existing; no new revision |
| Resulting revision | `committed_revision + 1` on success |
| Errors | 400 `capability_not_read_only` / `capability_not_found`; 404 workspace/connection; 409 revision conflict; 428 |

```python
class CreateWorkspaceLiveAccessBindingRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_ref: str = Field(..., min_length=1, max_length=128)
    remote_resource_id: str | None = Field(default=None, max_length=256)
    allowed_capability_ids: tuple[str, ...] = Field(..., min_length=1)
    idempotency_key: str = Field(..., min_length=1, max_length=256)
```

### 20.10 Disable / detach Live Access Binding

| | |
|--|--|
| Method / path | `PATCH /workspaces/{workspace_id}/live-access-bindings/{live_access_binding_id}` with `{ "status": "disabled" }` or `DELETE .../live-access-bindings/{live_access_binding_id}` |
| Response | 204 (logical detach, non-destructive) |
| If-Match | **Required** `WKC/{committed_revision}`; missing -> 428; mismatch -> 409 |
| Idempotency replay | Same key + same request -> 200 existing result; no new revision |
| Resulting revision | `committed_revision + 1` on success |
| Effect | Binding -> `DISABLED` (new revision); no Document changes |
| Errors | 404; 409; 428 |

### 20.11 Update Query Policy

| | |
|--|--|
| Method / path | `PUT /workspaces/{workspace_id}/query-policy` |
| Request | `UpdateQueryPolicyRequestV1` |
| If-Match | **Required** `WKC/{committed_revision}` (aggregate revision); missing -> 428; mismatch -> 409 |
| Idempotency replay | Same key + same request -> 200 existing result; no new revision |
| Resulting revision | `committed_revision + 1` on success |
| Response | `WorkspaceQueryPolicy` |
| Errors | 400 unsupported mode / invariant violation; 409 `configuration_revision_conflict`; 428 missing If-Match |

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
| Request (idempotency) | `(tenant_id, workspace_id, operation, idempotency_key)` via `WorkspaceKnowledgeMutationRecord` | Detect replay vs conflict |
| Semantic (logical resource) | Indexed: `(tenant_id, workspace_id, knowledge_source_binding_ref)`; Live: `(tenant_id, workspace_id, connection_ref, normalized_resource_scope, normalized_capability_set)`; Connection: `(tenant_id, workspace_id, connection_ref)` | Prevent duplicate logical bindings |

### 22.2 Idempotency behavior summary

| Scenario | Behavior |
|----------|----------|
| Same key + same request, COMMITTED | Return existing result; no new revision |
| Same key + different request | 409 `configuration_idempotency_conflict` |
| Same key + same request, RESERVED/PREPARED | Deterministic recovery; no second mutation |
| Same key + same request, ABORTED | Retry may CAS-replace to RESERVED with new mutation_id |
| Same key, RECOVERY_REQUIRED | 503 `configuration_recovery_required` |
| Different key + same semantic identity, ACTIVE | Return existing entity; no new revision |
| DISABLED + reactivation request | New revision with ACTIVE; same logical entity ID |

### 22.3 Configuration revision

- Single monotonic `committed_revision` on `WorkspaceKnowledgeConfigurationHead`.
- Every successful mutation increments revision via head CAS publication (Phase 7).
- Aggregate projection reads revision from head record only - children do not define aggregate version.
- Child records store `effective_revision` and `mutation_id` on immutable revision rows.

---

## 23. No-duplication proof (future acceptance test - `1F`)

### 23.1 Scenario

```text
one tenant KnowledgeSourceBinding (binding_ref = bind-proof-1)
-> references connection_ref = conn-proof-1
-> attached to workspace W
-> WorkspaceIndexedSourceBinding I (references bind-proof-1)
-> WorkspaceLiveAccessBinding L (references conn-proof-1)
```

### 23.2 Required invariants

| Invariant | Observable check |
|-----------|-------------------|
| One tenant binding | `TenantKnowledgeSourceBindingPort.get_binding` count == 1 |
| Indexed authorization references tenant binding | `I.knowledge_source_binding_ref == bind-proof-1`; no `provider_id`/`connection_ref` on I |
| Live authorization references same connection | `L.connection_ref == conn-proof-1` (derived from tenant binding at create) |
| No provider identity copied into indexed authorization | Serialized LKW indexed binding record lacks `provider_id`, `integration_kind`, `source_kind`, `credential_ref` |
| No credential duplication | `SecretsStore` lookup count == 1 per operation window; no `credential_ref` in LKW records |
| Same integration registration | `KnowledgeConnectionRegistry.resolve` returned object `id()` equal for indexed sync stub and live stub |
| No second vendor client | Integration constructor counter == 1 per `(tenant_id, connection_ref)` |
| Independent authorization | Disable I -> live still allowed; disable L -> indexed sync still allowed |
| Same workspace boundary | Cross-workspace binding attempt -> 404 |

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
| Abandoned staged record | Publication protocol | Cleanup failure -> RECOVERY_REQUIRED; head stays pending | 503 `configuration_recovery_required` | Failure-injection test |
| Premature Source visibility | Connected Source staging | `knowledge_configuration_visibility_revision` gate | Source hidden until head commit | Failure-injection test |

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

**Outcome:** Implement the durable provider-neutral workspace authorization foundation with tenant-binding references, immutable revisioned child records, one CAS-protected configuration head, one durable mutation/idempotency record, hidden staged connected Sources, deterministic crash recovery and non-destructive detach.

**Dependencies:** 1A-C2.

**Exact scope:**

```text
WorkspaceKnowledgeConfigurationHead
WorkspaceKnowledgeMutationRecord
revisioned child row contracts
TenantKnowledgeSourceBindingPort
WorkspaceIndexedSourceBinding
WorkspaceLiveAccessBinding storage shape
WorkspaceConnectionAttachment storage shape
WorkspaceQueryPolicy storage shape
WorkspaceSource visibility revision
ManagedWorkspaceRepository extensions
WorkspaceKnowledgeConfigurationService
CAS prepare/publish/recovery protocol
idempotency replay/conflict
semantic duplicate detection
reader consistency
failure-injection tests
```

**Non-goals:**

```text
HTTP routes
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

Do not move capability discovery from `1C` into `1B`.

**Required failure-injection tests:**

| Scenario | Expected outcome |
|----------|------------------|
| Two concurrent first mutations | One head created; one writer acquires revision 1; other receives revision conflict |
| CAS idle -> pending fails | No staged rows written |
| Failure after binding version write | Binding staged; Source not staged -> cleanup binding; head returns idle; mutation ABORTED |
| Failure after Source write | Binding + Source staged; PREPARED not reached -> both removed; head returns idle |
| Failure marking PREPARED | All staged written; PREPARED CAS fails -> cleanup; head returns idle only after successful cleanup |
| Cleanup failure | One staged delete fails -> head remains pending; mutation RECOVERY_REQUIRED; subsequent mutations fail closed |
| Failure publishing head | PREPARED records exist; head commit CAS fails -> cleanup staged records; release head |
| Failure finalizing mutation after head commit | Head committed; mutation still PREPARED -> recovery repairs to COMMITTED; result replayed |
| Idempotency replay | Same key + same request after commit -> existing result; no new revision |
| Idempotency conflict | Same key + different request -> 409; no new revision |
| Semantic duplicate | Different key + same semantic identity -> existing entity; no duplicate Source; no new revision when state matches |
| Source visibility | Source staged at N+1; head still N -> hidden; head committed N+1 -> visible |
| Non-destructive detach | Binding DISABLED; Source remains; document refs remain; Documents remain; vector deletion not called; future sync rejected |
| Reader retry | Head changes during projection -> retry once -> return stable revision or safe unstable error |

**Gate:** One tenant binding reference; no provider identity duplication; monotonic revision; zero provider-specific imports; publication protocol passes all failure-injection tests.

### 26.3 `LKW-KNOWLEDGE-ACCESS-1C` - tenant ports, connection discovery and typed capability catalog

**Outcome:** `TenantConnectionPort`, `TenantKnowledgeSourceBindingPort` adapter, `TenantLiveCapabilityCatalogPort`, Remote Resource discovery, safe connection/resource HTTP reads.
**Dependencies:** 1B, `KnowledgeConnectionRegistry` wiring.
**Code areas:** `serving/workspace_routes.py`, `serving/workspace_schemas.py`, host wiring, platform ports.
**Non-goals:** Indexed/live binding mutations.
**Tests:** API tests with fakes; cross-tenant 404; capability catalog read-only validation.
**Gate:** Discovery returns descriptors without secrets; only read-only capabilities listed.

### 26.4 `LKW-KNOWLEDGE-ACCESS-1D` - HTTP create/disable for bindings with server-derived metadata

**Outcome:** HTTP create/disable for connection attachment, Indexed Source authorization, Live Access Binding; server-derived metadata; no physical indexed-data deletion.
**Dependencies:** 1B, 1C.
**Code areas:** routes, schemas, `WorkspaceKnowledgeConfigurationService`.
**Non-goals:** Actual sync or live execution.
**Tests:** API acceptance, idempotency, semantic duplicate, independent binding authorization, non-destructive detach, If-Match on every mutation.
**Gate:** `CONNECTED_SOURCE` workspace Source created; indexed detach preserves Documents; no live binding auto-created.

### 26.5 `LKW-KNOWLEDGE-ACCESS-1E` - Query Policy and complete configuration projection

**Outcome:** Query policy CRUD + `GET knowledge-configuration` aggregate with revision head.
**Dependencies:** 1D.
**Non-goals:** Hybrid/automatic modes.
**Tests:** Unsupported mode rejection; cross-field invariant enforcement; deterministic ordering; revision concurrency; reader consistency retry.
**Gate:** Full projection matches stored records; revision from head only.

### 26.6 `LKW-KNOWLEDGE-ACCESS-1F` - one tenant binding / one connection indexed-live reuse proof

**Outcome:** Observable proof test - one tenant binding, one connection, one integration instance, no credential or provider-identity copy.
**Dependencies:** 1E + minimal live executor stub OR facade-only proof with shared resolver instrumentation.
**Non-goals:** Production live queries.
**Tests:** Instrumented acceptance test per section 23.
**Gate:** All invariants green in CI proof module.

---

## 27. First implementation task

### Recommended: `LKW-KNOWLEDGE-ACCESS-1B` - PROVIDER-NEUTRAL DURABLE WORKSPACE AUTHORIZATION FOUNDATION

**One-sentence outcome:** Implement the durable provider-neutral workspace authorization foundation with tenant-binding references, immutable revisioned child records, one CAS-protected configuration head, one durable mutation/idempotency record, hidden staged connected Sources, deterministic crash recovery and non-destructive detach.

**Expected scope:**

```text
WorkspaceKnowledgeConfigurationHead
WorkspaceKnowledgeMutationRecord
revisioned child row contracts
TenantKnowledgeSourceBindingPort
WorkspaceIndexedSourceBinding
WorkspaceLiveAccessBinding storage shape
WorkspaceConnectionAttachment storage shape
WorkspaceQueryPolicy storage shape
WorkspaceSource visibility revision
ManagedWorkspaceRepository extensions
WorkspaceKnowledgeConfigurationService
CAS prepare/publish/recovery protocol
idempotency replay/conflict
semantic duplicate detection
reader consistency
failure-injection tests
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
| No durable tenant Connection catalog | Medium | `TenantConnectionPort` + runtime registry for proofs (`1C`) |
| No typed live capability catalog | Medium | `TenantLiveCapabilityCatalogPort` in `1C`; bindings-only in `1D` |
| No live capability executor | Medium | Executor in later platform task |
| `list_source_candidates` not implemented on facade | Low | Use `inspect_scope` + adapter list in `1C` |
| `CONNECTED_SOURCE` ingestion processor not wired | Medium | Separate intake task after configuration stable |

**Contract freeze status:** Not blocked - gaps are explicit and sequenced.

---

## 30. Final architecture verdict

The repository supports the intended design when LKW stores workspace authorization references to tenant `KnowledgeSourceBinding` records (not duplicated provider identity), maintains one monotonic `committed_revision` via CAS-protected head record with a single pending writer and publication point, uses immutable revisioned child records and a durable `WorkspaceKnowledgeMutationRecord` for idempotency, reuses `to_source_ref(tenant_binding)` / `ConnectionAwareVendorResolver` / `VendorKnowledgeFacadeService` for indexed paths, validates live capabilities through typed `LiveCapabilityDescriptorV1`, and keeps live execution on a future shared executor. One `WorkspaceSource` continues to own all persisted Documents. Indexed detach is non-destructive via binding status. Connected Sources remain hidden until head publication. Every configuration mutation requires aggregate `If-Match`. Provider-specific LKW models and credential duplication are rejected.

**STATUS: `READY_FOR_REVIEW`**
