# Memory Store Plugin Author Guide

**Status:** canonical developer guide · **PLATFORM-PLUGIN-DOCS-4**
**Architecture owner:** [`docs/project/architecture/MEMORY.md`](../../architecture/MEMORY.md)
**Platform catalog:** [`EXTENSION_AUTHOR_GUIDE.md`](EXTENSION_AUTHOR_GUIDE.md) · [`PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md)

This guide documents **three separate public memory plugin surfaces**. There is no single `MemoryPlugin` protocol — do not collapse factory contracts.

---

## Developer journey (D1–D16)

| D | Topic | Status | Section |
|---|-------|--------|---------|
| D1 | Purpose | COMPLETE | §1 |
| D2 | Public contract | COMPLETE | §2 |
| D3 | Minimal implementation | COMPLETE | §3 |
| D4 | External package | COMPLETE | §4 |
| D5 | Local / host path | PARTIAL | §5 |
| D6 | Configuration | COMPLETE | §6 |
| D7 | Secrets / credentials | COMPLETE | §7 |
| D8 | DI / composition | COMPLETE | §8 |
| D9 | Registration / discovery | COMPLETE | §9 |
| D10 | Qualification | COMPLETE | §10 |
| D11 | Runtime use | PARTIAL | §11 |
| D12 | Lifecycle / cleanup | COMPLETE | §12 |
| D13 | Failure behavior | COMPLETE | §13 |
| D14 | Testing | COMPLETE | §14 |
| D15 | Production checklist | COMPLETE | §15 |
| D16 | Troubleshooting | COMPLETE | §16 |

**Overall:** **PARTIAL** — `SessionTurnIndexStorePlugin` has a wired host path; `UserProfileStorePlugin` and `SessionStoragePlugin` have public contracts and EP discovery counting but **no shipped Tier-3 factory resolution** (see `RUNTIME_CAPABILITY_GAP` in §11).

---

## 1. Purpose — what is pluggable

Memory plugins swap **store backends** behind Tier-1 facades (`SessionManager`, `UserProfileManager`). They are not Context plugins, RAG components, or Integration manifests.

| Surface | Protocol | Factory method | Replaces |
|---------|----------|----------------|----------|
| User profile store | `UserProfileStorePlugin` | `create_user_profile_store(**kwargs)` | `InMemoryUserProfileStore`, SQLite/Mongo bundles |
| Session storage | `SessionStoragePlugin` | `create_session_storage(**kwargs)` | `InMemorySessionStorage`, SQLite bundle |
| Session turn vector index | `SessionTurnIndexStorePlugin` | `create_session_turn_index(**kwargs)` | `VectorSessionTurnIndexStore` over host vector stack |

Entry point group (all three share one group; dispatch is by factory method shape):

```text
intergrax.memory_stores
```

**Vector memory:** LTM and episodic indexes reuse the host **integration vector store** (`EmbeddingManager`, `VectorstoreManager`). Plugins supply index adapters — not vendor SDK clients owned by the plugin package.

**Shared truths:**

- `installed` ≠ `discovered` ≠ `enabled` ≠ `production-qualified`
- Trusted in-process Python
- Qualification is host-owned semantic approval
- Secrets stay in host configuration, not EP metadata
- No universal Platform Plugin lifecycle/unload manager

---

## 2. Public contracts

Import from `intergrax.memory.contracts`:

### UserProfileStorePlugin

```python
# intergrax/memory/contracts/memory_store_plugin.py
@runtime_checkable
class UserProfileStorePlugin(Protocol):
    @classmethod
    def plugin_id(cls) -> str: ...

    @classmethod
    def create_user_profile_store(cls, **kwargs: Any) -> UserProfileStore: ...
```

### SessionStoragePlugin

```python
@runtime_checkable
class SessionStoragePlugin(Protocol):
    @classmethod
    def plugin_id(cls) -> str: ...

    @classmethod
    def create_session_storage(cls, **kwargs: Any) -> SessionStorage: ...
```

### SessionTurnIndexStorePlugin

```python
# intergrax/memory/contracts/session_turn_index.py
@runtime_checkable
class SessionTurnIndexStore(Protocol):
    async def upsert_turn(...) -> None: ...
    async def tombstone_turn(self, entry_id: str) -> None: ...
    async def search_turns(...) -> list[dict[str, Any]]: ...

@runtime_checkable
class SessionTurnIndexStorePlugin(Protocol):
    @classmethod
    def plugin_id(cls) -> str: ...

    @classmethod
    def create_session_turn_index(cls, **kwargs: Any) -> SessionTurnIndexStore: ...
```

`SessionTurnIndexStore` is an index **over** `SessionStorage`, not a replacement for session persistence.

There is **no** `register_memory_store_plugin()` helper.

---

## 3. Minimal implementation

### User profile store (test fixture pattern)

Reference: `tests/fixtures/plugin_packages/memory_store_plugin/memory_store_plugin/plugin.py` (**test fixture — packaging reference, not production sample**)

```python
from typing import Any

from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_store import UserProfileStore


class ExternalInMemoryUserProfileStorePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "external.in_memory_user_profile"

    @classmethod
    def create_user_profile_store(cls, **kwargs: Any) -> UserProfileStore:
        _ = kwargs
        return InMemoryUserProfileStore()
```

### Session turn index (wired surface)

Reference: `tests/fixtures/plugin_packages/session_turn_index_plugin/session_turn_index_plugin/plugin.py` (**test fixture**)

```python
from typing import Any

from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore


class ExternalSessionTurnIndexStorePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "external.session_turn_index"

    @classmethod
    def create_session_turn_index(cls, **kwargs: Any) -> VectorSessionTurnIndexStore:
        return VectorSessionTurnIndexStore(
            embedding_manager=kwargs["embedding_manager"],
            vectorstore_manager=kwargs["vectorstore_manager"],
            index_roles=kwargs.get("index_roles", ("user", "assistant")),
            tenant_id=str(kwargs.get("tenant_id") or "default"),
            vector_index_namespace=kwargs.get("vector_index_namespace"),
        )
```

Host passes `embedding_manager`, `vectorstore_manager`, `tenant_id`, and optional `vector_index_namespace` / `index_roles` via `**kwargs`.

---

## 4. External package

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-intergrax-memory-stores"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["Intergrax-ai==0.1.0"]

[tool.hatch.build.targets.wheel]
packages = ["src/my_intergrax_memory_stores"]

[project.entry-points."intergrax.memory_stores"]
acme_user_profile = "my_intergrax_memory_stores.user_profile:ExternalInMemoryUserProfileStorePlugin"
acme_session_turn = "my_intergrax_memory_stores.session_turn:ExternalSessionTurnIndexStorePlugin"
```

One package may expose multiple EP targets. Dispatch at bootstrap counts plugins by which factory method exists (`create_user_profile_store`, `create_session_storage`, or `create_session_turn_index`).

Discovery is **opt-in** (`discover_entry_points=True` or `INTERGRAX_DISCOVER_PLUGINS=true`).

---

## 5. Local / host path

| Surface | Local path | Status |
|---------|------------|--------|
| Session turn index | Pass `session_turn_index_plugins=[MyPlugin]` to `build_session_turn_index_store(...)` | **Supported** |
| User profile store | Compose `MemoryPlatformWiring(user_profile_store=...)` and pass to `build_session_manager_from_environment(memory_wiring=...)` | **Host composition** — no EP resolver |
| Session storage | Same `MemoryPlatformWiring(session_storage=...)` pattern | **Host composition** — no EP resolver |

There is **no** `register_memory_store_plugin()` and **no** scaffold CLI.

Default Tier-3 path (`resolve_memory_platform_wiring`) selects SQLite, MongoDB, or in-memory backends from `IntegrationProfile` — **not** from `intergrax.memory_stores` EP discovery.

---

## 6. Configuration — MemoryProfile

`MemoryProfile` on `ApplicationEnvironmentProfile` toggles memory features:

| Field | Role |
|-------|------|
| `enable_user_memory` / `enable_long_term_memory` | User profile manager + optional LTM vector index |
| `enable_org_memory` | Organization profile store |
| `enable_task_memory` | Task KV (not a memory-store plugin surface) |
| `enable_session_vector_index` | Episodic turn index (`SessionTurnIndexStore`) |
| `session_index_top_k`, `session_index_score_threshold` | Episodic search defaults |
| `vector_index_namespace` | Collection namespace for vector indexes |
| `session_index_roles` | Roles indexed for episodic search |
| `include_cross_session_episodic` | Cross-session episodic recall |
| `retention_days`, `scope_boundary`, `consolidation_mode` | Policy fields |

Vector memory flags require a resolvable RAG stack with vector backends (`assert_memory_vector_backend_available`).

---

## 7. Secrets and credentials

Memory plugins should consume **host-resolved dependencies** passed as factory `**kwargs` or constructor arguments. Do not encourage arbitrary `os.environ` lookup in plugin factories unless your domain contract explicitly documents that pattern.

Integration credentials belong in `IntegrationProfile` and provider bundles — the memory plugin receives constructed clients/managers from the host.

---

## 8. DI and composition

| Dependency | Who provides |
|------------|--------------|
| `UserProfileStore` / `SessionStorage` instances | Host wiring or custom `MemoryPlatformWiring` |
| `embedding_manager`, `vectorstore_manager` | Host RAG stack (`RagStack`) for vector indexes |
| `tenant_id` | Runtime tenant scope from host |
| `rag_stack` | `resolve_rag_stack_for_memory_wiring` / `create_default_rag_stack` |

Plugins return store instances; managers (`UserProfileManager`, `SessionManager`) remain Tier-1.

---

## 9. Registration and discovery — bootstrap semantics (critical)

```python
from intergrax.core.memory_bootstrap import bootstrap_memory_stores

result = bootstrap_memory_stores(
    discover_entry_points=True,
    user_profile_plugins=(),
    session_storage_plugins=(),
    session_turn_index_plugins=(),
)
# result.user_profile_plugins — int count only
# result.session_storage_plugins — int count only
# result.session_turn_index_plugins — int count only
```

### What `bootstrap_memory_stores` does

- Scans `intergrax.memory_stores` entry points when `discover_entry_points=True`
- Classifies each loaded class by factory method presence
- Returns **counts** (discovered + explicit sequences)
- Exposes `discover_session_turn_index_plugin_types()` for turn-index classes

### What it does NOT do

- Does **not** create a universal registered catalog of active stores
- Does **not** select which plugin backs a running host
- Does **not** materialize `UserProfileStore` or `SessionStorage` for Tier-3 wiring today

**Calling bootstrap alone does not activate a memory provider.**

---

## 10. Qualification

Platform qualification primitives exist (`check_platform_compatibility`, production gates). Memory-specific production qualification is host-owned. Compatible EP metadata ≠ production-qualified.

External memory plugins do not receive automatic live-backend qualification for vector indexes — that evidence is separate (RAG/integration qualification).

---

## 11. Runtime use

### Session turn index (wired path)

```text
MemoryProfile.enable_session_vector_index=True
  → resolve_rag_stack_for_memory_wiring(env, tenant_id=…)
  → build_session_turn_index_store(env, tenant_id=…, rag_stack=…)
      → discover_session_turn_index_plugin_types()  # EP scan
      → plugin.create_session_turn_index(**kwargs)  # first match
      → else VectorSessionTurnIndexStore (default)
  → SessionManager(session_turn_index_store=…)
```

Explicit local override:

```python
from intergrax.applications._shared.memory_vector_wiring import build_session_turn_index_store

index = build_session_turn_index_store(
    env,
    tenant_id="tenant-a",
    rag_stack=rag_stack,
    session_turn_index_plugins=[ExternalSessionTurnIndexStorePlugin],
)
```

### User profile + session storage (default host path)

```text
IntegrationProfile (sqlite | mongodb | none)
  → resolve_memory_platform_wiring(env)
  → MemoryPlatformWiring(session_storage, user_profile_store, …)
  → build_session_manager_from_environment(env, memory_wiring=wiring, rag_stack=…)
```

### RUNTIME_CAPABILITY_GAP

| Surface | Gap |
|---------|-----|
| `UserProfileStorePlugin` | Public contract + EP counting exist; **no shipped resolver** invokes `create_user_profile_store` from discovered EPs in `memory_wiring.py` |
| `SessionStoragePlugin` | Same — **no shipped resolver** invokes `create_session_storage` from EPs |
| `SessionTurnIndexStorePlugin` | **Wired** via `build_session_turn_index_store` |

Until a host resolver ships, user profile and session storage plugins require **explicit host composition** (`MemoryPlatformWiring`) or custom Tier-3 wiring.

---

## 12. Lifecycle and cleanup

- **Store lifetime:** factories return instances owned by the host wiring (`SessionManager`, `UserProfileManager`)
- **Vector clients:** when using `VectorSessionTurnIndexStore`, embedding/vector managers follow RAG stack lifecycle
- **No universal unload:** process-global EP import side effects only; no Platform Plugin shutdown API

If a factory returns a persistent client pool, the host owns closing it on application shutdown.

---

## 13. Failure behavior

| Failure | Result |
|---------|--------|
| EP not discovered | Count stays 0; default wiring used |
| Misunderstanding bootstrap counts | Stores not active — bootstrap is diagnostic/counting only |
| `enable_session_vector_index` without tenant | `MemoryVectorBackendUnavailableError(reason="tenant_required")` |
| Vector flags without RAG stack | `MemoryVectorBackendUnavailableError(reason="vector_backend_unavailable")` |
| Wrong factory return type | `TypeError` / attribute errors at wiring time |
| Multiple turn-index EPs | First discovered plugin wins in `build_session_turn_index_store` |

---

## 14. Testing

| Test | Path |
|------|------|
| Bootstrap counting | `tests/unit/core/plugins/test_memory_store_bootstrap.py` |
| Vector episodic wiring | `tests/integration/applications/test_memory_vector_ltm_wiring.py` |
| Memory platform wiring | `tests/unit/applications/test_memory_wiring.py` |

Author pattern for bootstrap counts:

```python
result = bootstrap_memory_stores(
    discover_entry_points=False,
    user_profile_plugins=(MyUserProfilePlugin,),
)
assert result.user_profile_plugins == 1
```

---

## 15. Production checklist

- [ ] Correct protocol for your surface (do not mix factory method names)
- [ ] Stable `plugin_id` per class
- [ ] For episodic index: handle `embedding_manager`, `vectorstore_manager`, `tenant_id` kwargs
- [ ] Do not assume `bootstrap_memory_stores()` activates your store
- [ ] For user/session stores today: plan explicit `MemoryPlatformWiring` until resolver ships
- [ ] Tombstone semantics for vector indexes when primary store deletes entries
- [ ] Tenant isolation enforced in store implementation
- [ ] No secrets in plugin metadata

---

## 16. Troubleshooting

| Symptom | Check |
|---------|-------|
| EP installed but store unchanged | Expected for user/session plugins — no resolver yet; use `MemoryPlatformWiring` |
| Bootstrap count > 0 but no effect | Bootstrap only counts; see §9 |
| Episodic index always default | No turn-index EP discovered; pass `session_turn_index_plugins=` explicitly |
| `tenant_required` | Pass non-empty `tenant_id` to session manager wiring |
| `vector_backend_unavailable` | Enable integration vector store; resolve `rag_stack` before memory wiring |
| Wrong store type at runtime | Inspect `resolve_memory_platform_wiring` path (SQLite/Mongo/in-memory) |

---

**Next:** [`EXTENSION_AUTHOR_GUIDE.md`](EXTENSION_AUTHOR_GUIDE.md) · [`MEMORY.md`](../../architecture/MEMORY.md) §5.3
