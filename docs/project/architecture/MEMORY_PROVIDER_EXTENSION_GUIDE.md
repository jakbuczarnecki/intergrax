# Memory provider extension guide

**Canonical architecture:** [`MEMORY_ARCHITECTURE.md`](MEMORY_ARCHITECTURE.md)  
**Detailed plugin surfaces:** [`MEMORY_STORE_PLUGIN_AUTHOR_GUIDE.md`](../technical/guides/MEMORY_STORE_PLUGIN_AUTHOR_GUIDE.md)

This guide covers **adding or replacing a UserProfile (or other) memory store provider** on the MEM-ENT composition path.

## Principles

- Platform code depends on **contracts**, not vendor SDKs.
- **Persistence** is reached through providers/materialization — not `if mongodb` in domain services.
- `installed` ≠ `discovered` ≠ `qualified` ≠ `production-wired`.

## Step-by-step — new UserProfile provider

### 1. Implement contract

Implement `UserProfileStore` and/or `UserProfileStorePlugin` (`intergrax/memory/contracts/memory_store_plugin.py`, `user_profile_store.py`).

### 2. Register / discover plugin

- Entry point group: `intergrax.memory_stores`
- Ensure `discover_classified_memory_store_plugins` can classify your factory method shape.

### 3. Materialization

Host passes `MemoryStoreMaterializationContext` (`resolver/materialization.py`) with `env`, `tenant_id`, `integration_profile`, optional `rag_stack`.

Resolution: `materialize_user_profile_store` / `resolve_memory_platform_wiring`.

### 4. Qualification

Run host qualification suite (`provider_qualification/runner.py`):

- **Contract** checks — required methods and types
- **Behavioral** checks — semantic parity with reference
- **Durable** checks — use SQLite harness pattern for real reopen; in-memory-only recreation is **not** durable proof

External Mongo/document integrations may exist for wiring experiments; **MEM-ENT-15 did not certify full external vendor durable E2E**.

### 5. Lifecycle expectations

- Canonical mutations go through `UserProfileManager`, not direct store access from Tier-3 features.
- After primary write, **projections** run via `UserProfileMemoryLifecycleCoordinator`.
- Projection failure leaves canonical committed — reconcile later.

### 6. Durability evidence

- **SQLite:** reference durable qualification path (MEM-ENT-13C).
- Document what your provider proves (reopen, delete, tenant isolation) in tests.

### 7. Tests

- Unit tests against store contract
- Qualification check module or harness registration
- Integration: wiring via `MemoryProfile.user_profile_store_plugin_id`

## Configuration

Set on `MemoryProfile`:

- `user_profile_store_plugin_id`
- `session_storage_plugin_id`
- `entity_temporal_memory_store_plugin_id`
- `procedural_memory_store_plugin_id`
- `long_horizon_memory_store_plugin_id`

Only flags that exist in `sub_profiles.MemoryProfile` are valid — do not invent config keys.

## Composition owner

`intergrax/applications/_shared/memory_wiring.py` — `resolve_memory_platform_wiring`, `MemoryPlatformWiring`.

Strict mode: `assert_strict_memory_bootstrap_acceptable` fails closed on plugin admission errors.

## New specialized memory store (checklist)

When adding a new domain store (not only UserProfile):

1. Define **contract** under `intergrax/memory/contracts/`
2. Define **canonical vs derived** authority (see architecture doc)
3. Define **lifecycle owner** (manager or service)
4. Define **projection** role if synced from UserProfile
5. Wire **governance** (`memory_specialized_*` where applicable)
6. Add **qualification** checks
7. Add **observability** terminals
8. Define **recovery** / reconcile if derived

## DO NOT

- Branch Memory services on vendor type
- Skip qualification and claim production-ready
- Expose store mutation bypassing control plane / manager
