# Vendor Knowledge Plugin Author Guide

**Status:** canonical developer guide · **VK-EXT-4** · **PLATFORM-PLUGIN-DOCS-6**
**Architecture owner:** [`docs/project/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md)
**Integration canon:** [`docs/project/architecture/INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md)
**Maintainer plan:** [`docs/project/maintainers/plans/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../maintainers/plans/KNOWLEDGE_SOURCE_INTEGRATIONS.md)

This guide is the practical implementation and qualification manual for Vendor
Knowledge provider plugins. It describes the **accepted** extension ABI from
VK-EXT-1, VK-EXT-2 and VK-EXT-3. It does not redesign the architecture.

This is an authoring guide, not a provider qualification claim. Qualification
evidence remains in the reference tests cited in §21.

**Audience:** senior engineer, staff/principal engineer, external integration
author, maintainer reviewing a new provider. Assumes Python and integration
design; does not assume Intergrax internals.

---

## Developer journey (D1–D16)

| D | Topic | Status | Section |
|---|-------|--------|---------|
| D1 | Purpose | COMPLETE | §1 |
| D2 | Public contract | COMPLETE | §2 |
| D3 | Minimal implementation | COMPLETE | §20 · installable example §4.1 |
| D4 | External package | COMPLETE | §4.1 · §6 |
| D5 | Local / host path | COMPLETE | §1 · §4.2 |
| D6 | Configuration | COMPLETE | §12 `KnowledgeSourceBinding` |
| D7 | Secrets | COMPLETE | §10 |
| D8 | DI | COMPLETE | §2 contribution catalog model |
| D9 | Registration/discovery | COMPLETE | §6 — separate from Tier-0 catalog |
| D10 | Qualification | COMPLETE | §22 |
| D11 | Runtime use | COMPLETE | §15 |
| D12 | Lifecycle | COMPLETE | §17 restart/rehydration |
| D13 | Failure behavior | COMPLETE | §18 |
| D14 | Testing | COMPLETE | §21 |
| D15 | Production checklist | COMPLETE | §19 · §25 |
| D16 | Troubleshooting | COMPLETE | §26 |

**Overall:** **COMPLETE** for the external-EP and host-builder paths supported today. Vendor Knowledge is **not** the Tier-0 plugin catalog — see §1 and §4.2.

---

## Public vs internal labeling

Throughout this guide:

| Label | Meaning |
|---|---|
| **PUBLIC / AUTHOR-FACING** | Symbols and flows a third-party plugin may depend on today. Stability is best-effort; breaking changes require a new contribution contract version. |
| **INTERNAL / DO NOT DEPEND ON DIRECTLY** | Composition, registry builders, host wiring, and incidental implementation details. Use only when qualifying inside this repository. |

Do not couple to incidental Python modules for convenience. If a symbol is not
listed as public here, treat it as internal.

---

## 1. Purpose and extension model

Vendor Knowledge extensions add **external provider packages** without editing
generic Vendor Knowledge core or generic LKW composition.

```text
external provider package (installable distribution)
        |
        v
Python entry point  (group: intergrax.vendor_knowledge.providers)
        |
        v
VendorKnowledgeProviderContribution  (immutable contribution bundle)
        |
        v
VendorKnowledgeContributionCatalog  (instance-local publication snapshot)
        |
        v
generic runtime registries  (adapters, source plugins, factories, Live)
        +
generic application composition  (LKW discovery/materializer registries)
```

**Frozen rule:** adding a provider must **not** require provider-specific edits
to generic Vendor Knowledge core or generic LKW composition. Provider business
logic lives in the plugin package and its contribution factory.

### Built-in vs external providers

| Kind | Registration | Discovery |
|---|---|---|
| **Built-in** | Deterministic builders inside `intergrax.runtime.vendor_knowledge.*_contribution` | Always loaded by `build_default_vendor_knowledge_contribution_catalog()` |
| **External** | Same `VendorKnowledgeProviderContribution` ABI | Loaded only when `discover_entry_points=True` on catalog bootstrap |

Built-ins and externals feed the **same** catalog and registries. External
provider installation is **not** built-in provider registration. Do not edit the
built-in aggregator to register an external vendor.

The Acme reference plugin in
`examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin/`
is the **installable worked example**. Repository qualification tests install
the same package from that path. A real provider lives in its own
installable distribution with the same entry-point contract and qualification
expectations.

**Vendor Knowledge is not the Tier-0 plugin catalog.** There is no
`register_vendor_knowledge_plugin()` global catalog registration. Flow:

```text
Python package (pip install)
        → VendorKnowledgeProviderContribution (EP factory)
        → intergrax.vendor_knowledge.providers
        → discover_vendor_knowledge_contributions() when discover_entry_points=True
        → VendorKnowledgeContributionCatalog (instance-local snapshot)
        → generic VK registries (adapters, factories, materializers, Live)
        → host application composition (LKW discovery/materializer registries)
        → KnowledgeSourceBinding (tenant/workspace/connection scoped)
        → runtime knowledge source (Durable / INDEXED / optional LIVE)
```

Do not unify this into Tool/Skill catalog semantics or a universal Platform
Plugin runtime wrapper.

---

## 2. Public extension surfaces

### `VendorKnowledgeProviderContribution` — PUBLIC

**Owns:** one immutable bundle per `(provider_id, integration_category)`:
adapters, source plugins, connection factories, discovery contributions,
indexed materializers, optional Live bundles, `contract_version`.

**Must not own:** tenant connections, secrets, workspace configuration,
registry mutation, service locator behavior, or runtime singletons.

**Identity:** `provider_id` + `integration_category` on the contribution must
match every nested component.

**Lifecycle:** constructed once by the provider factory; validated in
`__post_init__`; registered into a catalog snapshot; consumed by generic
registry builders.

**Security:** declarative only; no credential fields.

**Module:** `intergrax.runtime.vendor_knowledge.contribution`

### `VendorKnowledgeSourceIdentity` — PUBLIC

**Owns:** stable tuple `(provider_id, integration_category, source_kind)`.

**Must not own:** tenant/workspace/connection scope; that belongs in
`KnowledgeSourceRef` / bindings.

**Identity requirements:** non-empty trimmed strings; `provider_id` ≤ 64;
`source_kind` ≤ 128; `integration_category` is `IntegrationCategory`.

**Lifecycle:** immutable; key is `(provider_id, integration_category, source_kind)`.

**Security:** identity only; no secrets.

**Module:** `intergrax.runtime.vendor_knowledge.plugin`

### `VendorKnowledgeSourcePlugin` — PUBLIC

**Owns:** declarative mode capabilities (`DURABLE`, `INDEXED`, `LIVE`) per
source kind: `contract_version`, `operations`, `runtime_ref`, optional
`capability_refs`, secret-free `metadata`.

**Must not own:** sync execution, indexing, Search, Ask, credential loading.

**Identity:** `identity` must align with contribution `provider_id` /
`integration_category`.

**Lifecycle:** registered from contribution into
`VendorKnowledgeSourcePluginRegistry` at catalog build.

**Security:** metadata forbids credential/tenant keys (see `_FORBIDDEN_METADATA_KEYS`
in `plugin.py`).

### `VendorKnowledgeAdapter` — PUBLIC (protocol)

**Owns:** mapping vendor integration instances into canonical
`KnowledgeScopeInfo`, `KnowledgePage`, `KnowledgeContent`, `KnowledgePermissions`.

**Must not own:** credential loading, Search, Ask, index ownership, application
routing, global persistence, LKW imports.

**Identity:** `provider_id`, `integration_kind`, `source_kind` properties must
match a registered source plugin.

**Lifecycle:** receives resolved `integration` from `KnowledgeConnectionRegistry`;
invoked by `VendorKnowledgeFacadeService`.

**Security:** bounded reads; safe error messages; no secret leakage in metadata.

**Module:** `intergrax.runtime.vendor_knowledge.contracts`

### `VendorKnowledgeConnectionFactoryContribution` — PUBLIC

**Owns:** `(provider_id, integration_category)` factory hook.

**Must not own:** tenant state, raw credentials, or connection persistence.

**Identity:** must match parent contribution `provider_key`.

**Lifecycle:** factory registry rebuilt from catalog; used on connection create
and restart rehydration.

**Security:** receives `credential` only at factory invocation from
`SecretsStore`; must not persist it.

### `VendorKnowledgeDiscoveryContribution` — PUBLIC

**Owns:** provider-owned discovery factory for one `VendorKnowledgeSourceIdentity`.

**Must not own:** generic LKW routing, opaque-ref signing keys, or binding
persistence. Factory receives **application-owned** host context
(`APPLICATION_OWNED_EXTENSION_SURFACE`).

**Identity:** `identity` must match a registered source plugin on the same
contribution.

**Lifecycle:** catalog augmentation wires factory into LKW discovery registry
when application composition runs.

**Security:** candidates must use signed opaque refs; safe display labels only.

### `VendorKnowledgeIndexedMaterializerContribution` — PUBLIC

**Owns:** `identity`, `runtime_ref`, zero-argument materializer factory for one
INDEXED source.

**Must not own:** vector indexing pipeline, Search, Ask.

**Identity:** `runtime_ref` must equal the source plugin INDEXED capability
`runtime_ref`; unique per catalog.

**Lifecycle:** materializer registry resolves by `runtime_ref` at sync time.

**Security:** provenance on output documents; no secrets in markdown/metadata.

### `LiveRegistrationBundleV1` — PUBLIC (optional)

**Owns:** `descriptor`, `handler`, `request_schema`, `result_schema` for one
Live capability.

**Must not own:** durable sync, indexing, Search, Ask.

**Identity:** descriptor and handler must align with LIVE capability refs on
the source plugin.

**Lifecycle:** validated at contribution construction;
`VendorKnowledgeLiveRegistrationRegistry` built from catalog.

**Security:** read-only semantics; capability identity collision fails closed.

**Module:** `intergrax.runtime.vendor_knowledge.live.registration`

### INTERNAL composition (do not depend on)

| Symbol | Role |
|---|---|
| `VendorKnowledgeContributionCatalog` | Instance-local snapshot; conflict detection across contributions |
| `build_default_vendor_knowledge_contribution_catalog` | Built-in + optional EP discovery |
| `build_vendor_knowledge_*_registry` | Generic registry builders from catalog |
| `build_default_vendor_knowledge_application_contribution_catalog` | Application-owned discovery/materializer augmentation |
| `VendorKnowledgeApplicationExtensionContext` | Typed host resources for discovery factories |
| `TenantConnectionRehydrator` | Restart path from durable `TenantConnection` rows |
| `KnowledgeAdapterRegistry`, `ConnectedSourceContentMaterializerRegistry` | Instance-local registries |

---

## 3. Provider identity model

Stable identity tuple:

```text
provider_id          — identifies the vendor/provider (e.g. acme_reference, slack)
IntegrationCategory  — generic integration class (e.g. wiki_knowledge, conversation_channel)
source_kind          — provider-owned source surface within the category (e.g. acme_documents)
```

The same triple must align across:

```text
VendorKnowledgeSourceIdentity
VendorKnowledgeAdapter  (provider_id, integration_kind, source_kind)
VendorKnowledgeSourcePlugin
VendorKnowledgeConnectionFactoryContribution  (provider_id, integration_category) — per provider/category, not per source
VendorKnowledgeDiscoveryContribution
VendorKnowledgeIndexedMaterializerContribution
LiveRegistrationBundleV1 descriptor  (when LIVE is claimed)
```

**Fail-closed on mismatch:** contribution validation raises
`VendorKnowledgeContributionError` with codes such as
`source_plugin_identity_mismatch`, `adapter_identity_mismatch`,
`discovery_identity_mismatch`, `materializer_runtime_ref_mismatch`,
`live_capability_registration_missing`.

Duplicate `(provider_id, integration_category, source_kind)` within one
contribution or across catalog registrations raises conflict errors.

---

## 4. Choosing integration category

**Reuse an existing `IntegrationCategory`.** Do not create a new category
merely because a new vendor exists.

| Concept | Identifies |
|---|---|
| `provider_id` | The vendor/provider implementation |
| `IntegrationCategory` | The generic integration class from platform canon |

Canonical categories include `wiki_knowledge`, `issue_tracker`,
`conversation_channel`, `collaboration_suite`, `relational_store`, etc. See
`IntegrationCategory` in `intergrax.integrations.contracts.base` and
[`INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md).

Examples from accepted providers:

| Provider | `provider_id` | Category |
|---|---|---|
| Acme reference | `acme_reference` | `wiki_knowledge` |
| Slack | `slack` | `conversation_channel` |
| MS365 Graph | `ms365_graph` | `collaboration_suite` |
| Jira | `jira` | `issue_tracker` |
| Databricks | `databricks` | `relational_store` (connection-only) |

---

## 5. Package layout

Recommended layout (Acme reference as worked example):

```text
my_vendor_plugin/
  pyproject.toml
  src/my_vendor_plugin/
    __init__.py
    constants.py          # provider_id, source_kind, runtime_refs, scope types
    integration.py        # vendor integration class
    backend.py            # optional in-memory / HTTP client
    adapter.py            # VendorKnowledgeAdapter implementation
    factory.py            # TenantConnectionIntegrationFactory
    discovery.py          # optional discovery strategy + factory
    materializer.py       # optional INDEXED materializer
    contribution.py       # build_*_contribution()
```

Acme reference paths:

```text
examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin/
```

Relocated to:

```text
examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin/
  pyproject.toml
  src/acme_reference_vk_plugin/
    constants.py
    integration.py
    backend.py
    adapter.py
    factory.py
    discovery.py
    materializer.py
    contribution.py
```

**Optional files:** `discovery.py`, `materializer.py`, `backend.py`, Live
handlers — include only when the provider genuinely implements those surfaces.
`Databricks` contributes only `connection_factories` (see §7).

This file split is **recommended**, not mandatory. One module may hold multiple
surfaces if identity and boundaries remain clear.

---

## 6. Python entry point

**Group (exact):**

```text
intergrax.vendor_knowledge.providers
```

**Complete `pyproject.toml` example** (from Acme reference):

```toml
[project]
name = "acme-reference-vk-plugin"
version = "0.0.1"
requires-python = ">=3.12"
dependencies = []

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[project.entry-points."intergrax.vendor_knowledge.providers"]
acme_reference = "acme_reference_vk_plugin.contribution:build_acme_reference_contribution"
```

**Accepted target forms:**

1. `VendorKnowledgeProviderContribution` instance (returned directly from `load()`)
2. Zero-argument callable returning `VendorKnowledgeProviderContribution`
   (if `load()` returns a factory, catalog discovery calls it once)

**Discovery timing:**

| `discover_entry_points` | Behavior |
|---|---|
| `False` (default) | Built-ins only; external EPs not loaded |
| `True` | `discover_vendor_knowledge_contributions()` loads EP group |

Application catalog:
`build_default_vendor_knowledge_application_contribution_catalog(discover_entry_points=...)`
mirrors the flag.

**Explicit prohibitions:**

```text
no filesystem scanning for plugins
no arbitrary environment import strings
no import-time global registration into runtime registries
```

Entry-point discovery uses `importlib.metadata.entry_points` only when enabled.

---

## 4.1 External package quickstart

Install the reference package from this repository:

```bash
uv pip install ./examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin
```

Build a wheel:

```bash
uv build --wheel --project examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin
```

Enable discovery when building the contribution catalog:

```python
from intergrax.runtime.vendor_knowledge.contribution_catalog import (
    build_default_vendor_knowledge_contribution_catalog,
)

catalog = build_default_vendor_knowledge_contribution_catalog(
    discover_entry_points=True,
)
```

`installed` ≠ `discovered` ≠ `enabled` ≠ `production-qualified`. Package
installation alone does not bind tenant connections or qualify the provider for
production.

---

## 4.2 Host builder path (not Tier-0 catalog)

Hosts compose Vendor Knowledge through **domain-owned builders**, not
`bootstrap_catalogs()`:

| Step | API | Notes |
|------|-----|-------|
| Contribution catalog | `build_default_vendor_knowledge_contribution_catalog(discover_entry_points=…)` | Built-ins + optional EP |
| Adapter registry | `build_vendor_knowledge_adapter_registry(catalog)` | From catalog snapshot |
| Factory registry | `build_tenant_connection_integration_factory_registry(catalog)` | Credential resolution at factory invoke |
| Application augmentation | `build_default_vendor_knowledge_application_contribution_catalog(…)` | LKW discovery/materializer wiring |
| Tenant binding | `KnowledgeSourceBinding` + `credential_ref` on `TenantConnection` | Host owns secrets |

External providers must not call global `register_*` catalog functions. They
expose a contribution factory via EP only.

---

## 7. Contribution factory

Minimal pattern (syntactically aligned with current contracts):

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeConnectionFactoryContribution,
    VendorKnowledgeDiscoveryContribution,
    VendorKnowledgeIndexedMaterializerContribution,
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.contribution_builder import build_durable_source_plugin
from intergrax.runtime.vendor_knowledge.plugin import VendorKnowledgeSourceIdentity


def build_my_vendor_contribution() -> VendorKnowledgeProviderContribution:
    category = IntegrationCategory.WIKI_KNOWLEDGE
    provider_id = "my_vendor"
    source_kind = "my_documents"
    identity = VendorKnowledgeSourceIdentity(
        provider_id=provider_id,
        integration_category=category,
        source_kind=source_kind,
    )
    return VendorKnowledgeProviderContribution(
        provider_id=provider_id,
        integration_category=category,
        adapters=(MyVendorAdapter(),),
        source_plugins=(
            build_durable_source_plugin(
                provider_id=provider_id,
                integration_category=category,
                source_kind=source_kind,
                runtime_ref="knowledge-adapter:my_vendor:wiki_knowledge:my_documents",
                indexed_runtime_ref="indexed-source:my_vendor:my_documents",
            ),
        ),
        connection_factories=(
            VendorKnowledgeConnectionFactoryContribution(
                provider_id=provider_id,
                integration_category=category,
                factory=MyVendorConnectionFactory(),
            ),
        ),
        discovery_contributions=(
            VendorKnowledgeDiscoveryContribution(
                identity=identity,
                factory=build_my_vendor_discovery_strategy,
            ),
        ),
        indexed_materializers=(
            VendorKnowledgeIndexedMaterializerContribution(
                identity=identity,
                runtime_ref="indexed-source:my_vendor:my_documents",
                factory=MyVendorMaterializer,
            ),
        ),
        # live_contributions=()  — omit unless LIVE is genuinely implemented
    )
```

**Partial contributions are valid.** Unsupported surfaces should be **absent**,
not faked.

Evidence: Databricks connection-only contribution
(`intergrax/runtime/vendor_knowledge/databricks_contribution.py`) registers only
`connection_factories` — no adapters, source plugins, discovery, or materializers.

Acme reference full contribution:
`examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin/src/acme_reference_vk_plugin/contribution.py`

**Contract version:** default `vendor-knowledge.provider-contribution.v1`
(`VENDOR_KNOWLEDGE_PROVIDER_CONTRIBUTION_CONTRACT_VERSION`). Target a supported
version; do not assume forward compatibility beyond the published contract id.

---

## 8. Source plugin modes

Modes (`VendorKnowledgeMode`):

| Mode | Purpose |
|---|---|
| `DURABLE` | Bounded inventory/reconciliation into generic durable coordinator |
| `INDEXED` | Eligible for LKW indexed sync and materialization |
| `LIVE` | Optional read-only live capabilities |

Declare modes **only when genuinely implemented**.

| Pattern | Example |
|---|---|
| DURABLE only | Hypothetical archive-only provider |
| DURABLE + INDEXED | Acme reference, Slack, Graph mail, Jira issues |
| DURABLE + INDEXED + LIVE | Slack (built-in Live bundles) |

Not every provider must support all three.

### Capability `runtime_ref` and collisions

Each declared mode carries a unique `runtime_ref` string (≤ 256 chars). INDEXED
`runtime_ref` on the source plugin must equal the materializer contribution
`runtime_ref`. Duplicate `runtime_ref` across catalog materializers fails closed
(`duplicate_materializer_runtime_ref`).

LIVE mode requires non-empty `capability_refs`; each ref must resolve to a
`LiveRegistrationBundleV1` on the same contribution.

Helper: `build_durable_source_plugin(...)` in
`intergrax.runtime.vendor_knowledge.contribution_builder` — optional
`indexed_runtime_ref` adds INDEXED capability alongside DURABLE.

---

## 9. Adapter implementation

Implement `VendorKnowledgeAdapter` (protocol in `contracts.py`).

**Responsibilities:**

```text
translate vendor primitives into canonical Vendor Knowledge records
bounded reads (respect limit, cursors)
stable remote IDs and revisions
permissions/provenance where supported
safe_display_name / scope validation
```

**Explicitly forbidden:**

```text
credential loading
Search implementation
Ask implementation
index ownership
application routing
global persistence
```

Acme reference adapter:
`examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin/src/acme_reference_vk_plugin/adapter.py`

Key patterns:

- `inspect_scope`, `read_page`, `fetch_content`, `fetch_permissions`
- `KnowledgeAdapterCapabilities` declares what the adapter actually supports
- `VendorKnowledgeError` with `VendorKnowledgeErrorCode` for fail-closed scope/remote errors
- Integration type check in `_require_integration` — adapter receives resolved integration, not raw credentials

---

## 10. Connection factory / credentials

Lifecycle:

```text
TenantConnection (durable row)
        |
        v
credential_ref  (stored on connection; not the secret itself)
        |
        v
SecretsStore.get_secret(credential_ref)
        |
        v
TenantConnectionIntegrationFactory.create_integration(...)
        |
        v
provider integration instance
        |
        v
KnowledgeConnectionRegistry.register / resolve
```

**Rules:**

```text
raw credentials must not be persisted in TenantConnection
plugin metadata must remain secret-free
secret_free_config holds only non-sensitive connection parameters
```

`TenantConnectionIntegrationFactory` protocol (`tenant_connection_rehydration.py`):

```python
def create_integration(
    self,
    *,
    tenant_id: str,
    connection_ref: str,
    provider_id: str,
    integration_kind: IntegrationCategory,
    credential_ref: str,
    credential: str,
    secret_free_config: Mapping[str, JsonValue],
) -> object:
    ...
```

Acme factory example:
`examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin/src/acme_reference_vk_plugin/factory.py`

**Restart/rehydration:** `TenantConnectionRehydrator` lists durable connections,
loads secrets via `SecretsStore`, invokes the **contribution-discovered** factory
registry, and repopulates `KnowledgeConnectionRegistry`. Manual runtime injection
after restart does **not** qualify.

---

## 11. Discovery

Provider-owned discovery via `VendorKnowledgeDiscoveryContribution`. Factory
signature: callable accepting `VendorKnowledgeApplicationExtensionContext`,
returning a strategy implementing `list_remote_resources` and
`revalidate_candidate_label`.

For providers compatible with generic scoped sources, use:

```text
RemoteResourceTypeV1.VENDOR_KNOWLEDGE_SCOPED_SOURCE
```

Encode candidates with `RemoteResourceOpaqueRefCodec.encode_vendor_knowledge_scoped_source_candidate`.

**Generic candidate fields** (opaque payload `VendorKnowledgeScopedSourceCandidatePayload`):

```text
provider_id
integration_kind   (IntegrationCategory value string)
source_kind
scope_type         (provider-owned semantic type)
scope_id           (provider-owned semantic id)
tenant_id
workspace_id
connection_ref
safe_display_label
```

**Generic LKW interprets:** tenant/workspace/connection ownership fences;
`provider_id`, `integration_kind`, `source_kind` for binding; opaque signature
and schema version; `safe_display_label` for UI.

**Generic LKW deliberately does not interpret:** `scope_type` and `scope_id`
semantics beyond storing them on `KnowledgeSourceScope` — provider adapter
validates scope on read.

Acme discovery:
`examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin/src/acme_reference_vk_plugin/discovery.py`

Do not add provider-specific LKW routes. Use the generic scoped-source seam.

---

## 12. Opaque ref and binding

Flow:

```text
provider discovery
        → signed opaque candidate (HMAC codec)
        → generic candidate validation
        → KnowledgeSourceBinding
```

**Ownership fences** (fail-closed):

```text
tenant_id      payload must match request tenant
workspace_id   payload must match request workspace
connection_ref payload must match attached connection
provider_id    validated against contribution catalog on persist
integration_kind must be valid IntegrationCategory
source_kind    must be registered for provider/category
```

`scope_type` and `scope_id` are provider-owned semantic identity values.
Generic binding (`_scoped_vendor_binding` in `connected_source_tenant_binding.py`)
stores them opaquely on `KnowledgeSourceScope`.

Wrong tenant/workspace/connection → `ConnectedSourceBindingError`
(`workspace_not_found`, `connection_not_attached`).

Unknown provider/source on persist → fail-closed (see scoped-source qualification
tests).

---

## 13. Durable

Provider data enters the generic Durable path through:

```text
VendorKnowledgeAdapter.read_page / fetch_content
        → generic durable coordinator
        → checkpoint / continuation via KnowledgeCursor
        → stable remote IDs and revision semantics
        → replay / idempotency at item level
        → source isolation by (provider_id, source_kind, scope)
```

**Do not create a provider-specific Durable repository** unless architecture
explicitly requires one in the future. Use the generic coordinator and canonical
`KnowledgeChange` / `KnowledgeItemDescriptor` models.

Acme proof: `read_page` returns bounded changes with `proposed_checkpoint`;
adapter maps documents to `KnowledgeChangeKind.UPSERT` with provenance.

---

## 14. INDEXED / materializer

`VendorKnowledgeIndexedMaterializerContribution` wires provider transformation
into indexed sync.

**Runtime-ref alignment (required):**

```text
source plugin INDEXED capability runtime_ref
==
materializer contribution runtime_ref
==
materializer class runtime_ref attribute
```

Materializer `materialize(...)` returns `MaterializedConnectedSourceDocument`
containing `KnowledgeDocument` (via `build_materialized_connected_source_document`).

**Provenance requirements:** identity, binding, workspace, revision, permissions
when available — generic indexing and Search evidence depend on canonical
document provenance.

No provider-specific indexing pipeline. LKW generic sync invokes materializer
by `runtime_ref`; vector publication is generic.

Acme materializer:
`examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin/src/acme_reference_vk_plugin/materializer.py`

---

## 15. Search / Ask

Providers do **not** implement Search or Ask.

Canonical flow:

```text
provider adapter
        → Vendor Knowledge durable fetch
        → INDEXED materialization
        → KnowledgeDocument in workspace index
        → generic Search
        → generic Ask (with evidence / provenance gates)
```

Search and Ask evidence requirements are fail-closed in generic RAG/LKW paths.
Acme E2E proof uses `ACME_REFERENCE_MARKER` in indexed content and verifies
Search hit and Ask response (`test_acme_reference_external_provider_proof.py`).

---

## 16. LIVE — optional

Live is **optional**. Implement only when the provider claims `LIVE` on the
source plugin.

If implemented, contribute `LiveRegistrationBundleV1` entries in
`live_contributions`:

```text
descriptor   — LiveCapabilityDescriptorV1
handler        — read-only LiveCapabilityHandlerV1
request_schema — SchemaRegistrationV1
result_schema  — SchemaRegistrationV1
```

Validation at contribution construction:

```text
read-only semantics (CapabilityEffectV1)
capability identity alignment with plugin capability_refs
schema identity registration
collision handling — duplicate capability keys fail closed
```

Built-in Slack Live bundles are reference implementations inside
`intergrax.runtime.vendor_knowledge.live.*` — external authors follow the same
bundle contract.

Do not require Live for provider qualification unless the provider claims it.

---

## 17. Restart / rehydration

Restart proof is **mandatory** for external provider qualification.

Required lifecycle after process restart:

```text
1. plugin rediscovery          (discover_entry_points=True when qualifying externals)
2. catalog rebuild             VendorKnowledgeContributionCatalog
3. factory registry rebuild    TenantConnectionIntegrationFactoryRegistry from catalog
4. TenantConnection rehydration TenantConnectionRehydrator + SecretsStore
5. KnowledgeConnectionRegistry repopulated from factories
6. existing KnowledgeSourceBinding remains usable without manual injection
```

**Manual runtime injection after restart does not qualify.**

Canonical test: `test_acme_reference_restart_rehydration_search_ask` in
`tests/integration/vendor_knowledge/test_acme_reference_external_provider_proof.py`

Steps exercised:

- Simulated host restart (new catalog/registry instances)
- Rehydration from durable tenant connection rows
- Discovery, binding, sync, Search, Ask without re-attaching connection manually

---

## 18. Error / conflict behavior

Fail-closed cases:

| Area | Error / code |
|---|---|
| Malformed entry point | `VendorKnowledgePluginLoadError` (`entry_point_name_invalid`, `external_contribution_load_failed`, `external_contribution_invalid`) |
| Wrong return type | `external_contribution_invalid` |
| Duplicate EP name | `VendorKnowledgePluginConflict` (`duplicate_entry_point_name`) |
| Duplicate provider/category | `conflicting_provider_contribution` |
| Cross-contribution duplicate source | `duplicate_source_plugin`, `duplicate_adapter`, `duplicate_materializer_runtime_ref`, `duplicate_live_capability` |
| Source mismatch in contribution | `source_plugin_identity_mismatch`, `adapter_source_plugin_missing`, `discovery_source_plugin_missing` |
| Adapter identity mismatch | `adapter_identity_mismatch` |
| Materializer runtime_ref conflict | `materializer_runtime_ref_mismatch`, `materializer_mode_not_declared` |
| Live capability conflict | `live_capability_registration_missing`, `duplicate_live_capability_identity` |
| Unknown provider/source at runtime | `VendorKnowledgeErrorCode.ADAPTER_NOT_FOUND`; binding persist failures in scoped-source tests |
| Opaque ref tampering | `ConnectedSourceDiscoveryError`, `RemoteResourceOpaqueRefCodecError` |

**Diagnostics:** use safe, redacted messages. Never include credentials, tokens,
or raw secret values in errors or logs surfaced to operators.

---

## 19. Security checklist

```text
☐ credential_ref only on TenantConnection — never raw credential persistence
☐ SecretsStore is the only credential resolution path
☐ plugin metadata and secret_free_config are secret-free
☐ tenant isolation on discovery candidates and bindings
☐ workspace isolation on discovery candidates and bindings
☐ connection ownership — connection_ref must match attached connection
☐ signed opaque refs (HMAC codec; wrong key / tamper rejected)
☐ bounded reads in adapter (limit, cursor completion)
☐ safe metadata on candidates (safe_display_label, no PII leakage in errors)
☐ safe diagnostics — VendorKnowledgeError.safe_message pattern
☐ Search evidence fail-closed in generic path
☐ Ask evidence fail-closed in generic path
☐ no import-time global registry side effects
```

---

## 20. Minimal provider example

Compact illustration (not a second qualification artifact):

```python
"""Minimal single-source DURABLE-only external provider sketch."""

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeConnectionFactoryContribution,
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.contribution_builder import build_durable_source_plugin
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeAdapterCapabilities,
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgePage,
    KnowledgePermissions,
    KnowledgeScopeInfo,
    KnowledgeVisibility,
)


PROVIDER_ID = "minimal_vendor"
SOURCE_KIND = "minimal_docs"
RUNTIME_REF = f"knowledge-adapter:{PROVIDER_ID}:wiki_knowledge:{SOURCE_KIND}"


class MinimalAdapter:
    provider_id = PROVIDER_ID
    integration_kind = IntegrationCategory.WIKI_KNOWLEDGE
    source_kind = SOURCE_KIND
    capabilities = KnowledgeAdapterCapabilities(
        full_inventory=True,
        content_fetch=True,
        structured_content=True,
    )

    async def inspect_scope(self, *, integration, source) -> KnowledgeScopeInfo:
        return KnowledgeScopeInfo(source=source, capabilities=self.capabilities)

    async def read_page(self, *, integration, source, cursor, limit) -> KnowledgePage:
        return KnowledgePage(changes=(), next_cursor=None, has_more=False)

    async def fetch_content(self, *, integration, source, item) -> KnowledgeContent:
        return KnowledgeContent(mode=KnowledgeContentMode.STRUCTURED_RECORD, structured_record={})

    async def fetch_permissions(self, *, integration, source, item) -> KnowledgePermissions:
        return KnowledgePermissions(visibility=KnowledgeVisibility.TENANT)


class MinimalFactory:
    def create_integration(self, *, tenant_id, connection_ref, provider_id,
                           integration_kind, credential_ref, credential,
                           secret_free_config):
        if provider_id != PROVIDER_ID:
            raise ValueError("provider_id mismatch")
        return object()  # replace with real integration


def build_minimal_vendor_contribution() -> VendorKnowledgeProviderContribution:
    category = IntegrationCategory.WIKI_KNOWLEDGE
    return VendorKnowledgeProviderContribution(
        provider_id=PROVIDER_ID,
        integration_category=category,
        adapters=(MinimalAdapter(),),
        source_plugins=(
            build_durable_source_plugin(
                provider_id=PROVIDER_ID,
                integration_category=category,
                source_kind=SOURCE_KIND,
                runtime_ref=RUNTIME_REF,
            ),
        ),
        connection_factories=(
            VendorKnowledgeConnectionFactoryContribution(
                provider_id=PROVIDER_ID,
                integration_category=category,
                factory=MinimalFactory(),
            ),
        ),
    )
```

Entry point:

```toml
[project.entry-points."intergrax.vendor_knowledge.providers"]
minimal_vendor = "minimal_vendor_plugin.contribution:build_minimal_vendor_contribution"
```

For INDEXED, discovery, and materializers, extend the Acme reference — not
this sketch.

---

## 21. Full qualification example

Point readers to:

| Artifact | Path |
|---|---|
| Reference package | `examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin/` |
| Unit qualification | `tests/unit/runtime/vendor_knowledge/test_acme_reference_plugin.py` |
| E2E external proof | `tests/integration/vendor_knowledge/test_acme_reference_external_provider_proof.py` |
| Scoped-source seam | `applications/local_workspace_application/tests/workspaces/test_vendor_knowledge_scoped_source_seam_qualification.py` |

**What the reference proof demonstrates:**

```text
entry-point discovery with discover_entry_points=True
contribution validation and identity alignment
connection create via credential_ref + factory
VENDOR_KNOWLEDGE_SCOPED_SOURCE discovery and opaque ref roundtrip
generic binding without provider branches in VK core
Durable sync through generic coordinator
INDEXED materialization to KnowledgeDocument
generic Search and Ask with provenance marker (ACME_REFERENCE_MARKER)
restart rehydration — catalog, factory registry, connection registry, bindings
built-in parity when external discovery disabled
no acme_reference identifiers in generic Vendor Knowledge core
```

---

## 22. Qualification matrix

| Row | Required? | What to prove | Canonical reference test |
|---|---|---|---|
| Contribution validation | Yes | Identity alignment, mode/materializer/Live consistency | `test_reference_contribution_identity_and_modes` |
| Entry-point enabled | Yes (external) | EP loads contribution; catalog counts increment | `test_enabled_catalog_increments_builtin_counts`, `test_entry_point_discovery_loads_reference_contribution` |
| Entry-point disabled | Yes | External provider unavailable; built-ins unchanged | `test_external_discovery_disabled_keeps_reference_unavailable`, `test_builtin_parity_with_external_discovery_disabled` |
| Malformed plugin | Yes | Fail-closed load errors | `test_malformed_external_plugin_fails_closed` |
| Conflict behavior | Yes | Duplicate provider/source/runtime_ref conflicts | `test_duplicate_external_provider_key_conflicts`, `test_materializer_registry_duplicate_runtime_ref_conflicts_fail_closed` |
| Connection creation | Yes | Factory from credential_ref | `test_reference_factory_creates_integration_from_credential_ref` |
| Credential isolation | Yes | No raw secret on TenantConnection | Acme E2E + factory tests |
| Restart rehydration | Yes | Full lifecycle without manual injection | `test_acme_reference_restart_rehydration_search_ask` |
| Discovery | Yes | Scoped-source candidates | `test_acme_reference_external_provider_full_proof` |
| Binding | Yes | Opaque ref → KnowledgeSourceBinding | `test_scoped_source_opaque_ref_roundtrip`, `test_second_synthetic_identity_encode_decode_and_binding` |
| Wrong tenant | Yes | Fence on candidate scope | `test_scoped_source_binding_ownership_fence` |
| Wrong workspace | Yes | Fence on candidate scope | `test_scoped_source_binding_ownership_fence` |
| Wrong connection | Yes | Fence on candidate scope | `test_scoped_source_binding_ownership_fence` |
| Durable initial sync | Yes | Items reach durable path | `test_acme_reference_external_provider_full_proof` |
| Durable replay | Yes | Idempotent reconciliation semantics | Acme adapter checkpoint / E2E sync |
| Indexed materialization | Yes | KnowledgeDocument with provenance | E2E proof + `test_materializer_registry_runtime_ref_resolution_enforced` |
| Search | Yes | Generic search hits indexed content | `test_acme_reference_external_provider_full_proof` |
| Ask | Yes | Generic ask with evidence | `test_acme_reference_restart_rehydration_search_ask` |
| Live if claimed | If claimed | Live bundle registration | Built-in Slack + `test_builtin_parity_and_live_unchanged_with_external_plugin` |
| Built-in parity | Yes | Built-ins unchanged when EP off | `test_builtin_parity_and_live_unchanged_with_external_plugin` |
| No generic provider branch | Yes | No provider name in VK core | Scoped-source + E2E grep contract |
| Opaque ref security | Yes | Tamper / wrong key rejected | `test_scoped_source_opaque_ref_tampering_rejected`, `test_scoped_source_opaque_ref_wrong_signing_key_rejected` |
| Unknown provider/source persist | Yes | Fail-closed binding persist | `test_scoped_source_provider_and_category_mismatch_fail_on_persist`, `test_scoped_source_unknown_source_kind_fails_on_persist` |

---

## 23. No-go patterns

Explicitly forbidden:

```text
editing generic composition to add provider name
provider-specific if/switch in LKW for your vendor id
manual global registry registration at import time
filesystem scanning for plugins
raw secrets in plugin config / metadata / TenantConnection
provider-specific Search implementation
provider-specific Ask implementation
provider-specific indexing pipeline
fake capability declaration (INDEXED/LIVE without implementation)
new IntegrationCategory just for vendor identity
test module registered as runtime plugin
manual connection injection after restart to pass qualification
import-time side effects in entry-point target modules
coupling discovery to undocumented internal registry APIs
```

---

## 24. Author workflow

Operational recipe (ordered):

```text
 1. identity / category     — provider_id, IntegrationCategory, source_kind constants
 2. integration / backend   — vendor client; category integration class
 3. adapter                 — VendorKnowledgeAdapter; bounded reads; stable IDs
 4. source plugin modes     — declare only implemented modes; runtime_refs
 5. connection factory      — TenantConnectionIntegrationFactory; secret-free config
 6. discovery               — optional; VENDOR_KNOWLEDGE_SCOPED_SOURCE if generic seam fits
 7. materializer            — optional; INDEXED only; runtime_ref alignment
 8. contribution            — VendorKnowledgeProviderContribution assembly
 9. entry point             — pyproject.toml group intergrax.vendor_knowledge.providers
10. unit qualification      — contribution validation, EP load, conflict cases
11. application E2E         — connection, discovery, bind, sync (Acme E2E pattern)
12. restart                 — rehydration proof without manual injection
13. security / conflict     — opaque ref, ownership fences, unknown provider/source
14. built-in parity         — discover_entry_points=False unchanged
15. maintainer review       — reviewer checklist §25
```

---

## 25. Reviewer checklist

Short maintainer review list:

```text
☐ provider-neutral VK core unchanged — no new provider branches
☐ identity alignment across adapter, plugin, factory, discovery, materializer, Live
☐ secret handling — credential_ref only; factory-only credential access
☐ restart — catalog + factory registry + connection registry + bindings without manual injection
☐ failure modes — malformed EP, conflicts, unknown provider/source fail closed
☐ capability truthfulness — modes match real implementation
☐ provenance — durable changes and KnowledgeDocument provenance present
☐ tests — unit, E2E, scoped-source seam as appropriate to claimed surfaces
☐ entry point — correct group; no filesystem scanning; discovery opt-in documented
☐ partial contributions — absent surfaces not faked
```

---

## 26. Troubleshooting

| Symptom | Likely cause | What to check |
|---------|--------------|---------------|
| Provider missing after `pip install` | EP discovery disabled | `discover_entry_points=True` on catalog bootstrap |
| `duplicate_entry_point_name` / `conflicting_provider_contribution` | Duplicate EP or provider key | Unique EP names; one `(provider_id, integration_category)` per contribution |
| Connection create fails | Missing durable `TenantConnection` | `credential_ref` on connection; `SecretsStore` resolves secret at factory invoke |
| Binding persist fails (ownership fence) | Tenant/workspace/connection mismatch | Scoped candidate `tenant_id`, `workspace_id`, `connection_ref` |
| Materializer not invoked | `runtime_ref` mismatch | Source plugin INDEXED `runtime_ref` == materializer contribution |
| Provider in catalog but no indexed content | Source not bound | `KnowledgeSourceBinding` + workspace sync |
| Qualification withheld | Host semantic admission not granted | `installed` ≠ `production-qualified` |
| Built-ins affected by external test | Incorrect conflict expectation | External discovery **adds** contributions when keys differ |

**Qualification layers:** (1) package/EP load, (2) provider/domain contract, (3) live connection/source binding where applicable.

---

## Versioning

Current contribution contract:

```text
vendor-knowledge.provider-contribution.v1
```

Source plugin default:

```text
vendor-knowledge.source-plugin.v1
```

Plugin packages should target a **supported** contribution contract version.
Intergrax does not promise undocumented forward compatibility. A future contract
version would require explicit platform release notes and migration guidance.

---

## Document map

| Document | Role |
|---|---|
| This guide | Practical authoring and qualification |
| [`KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md) | Architecture authority; §19 contribution composition |
| [`INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md) | Integration category canon |
| [`RAG_EXTENSION_GUIDE.md`](RAG_EXTENSION_GUIDE.md) | RAG extension (Search/Ask/indexing boundary) |
| Acme reference package | [`examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin/`](../../../../examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin/) |
