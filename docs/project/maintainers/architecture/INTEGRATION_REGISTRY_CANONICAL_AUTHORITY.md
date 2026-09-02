# Integration Registry Canonical Authority (INTEGRATIONS-REGISTRY-CANONICALIZATION-1)

**Status:** Accepted  
**Date:** 2026-09-02

## Problem

`intergrax/integrations/registry/catalog.py` was already the runtime authority for provider resolution, but `intergrax/runtime/integrations/registry_v2.py` independently discovered providers via `SLUG_CATEGORY`, `provider_import_path`, module import, and reflection. That created dual-registration / metadata drift risk.

## Decision

**The open catalog is the single authoritative provider registration lifecycle.**

`registry_v2` is a **derived contract projection / read model** built from canonical catalog entries. It must not register providers that are absent from the catalog, and it must not maintain an independent discovery universe.

## Canonical authority

| Concern | Authority |
|--------|-----------|
| Register / unregister provider | `register_integration()` · `register_from_manifest()` · `register_integration_plugin()` |
| Catalog storage | `intergrax.integrations.registry.catalog` |
| Runtime resolution | `IntegrationProfile` → `resolve_from_profile()` → `get_entry()` → `entry.factory()` |
| Typed contract metadata at registration | `IntegrationContractSpec` on `IntegrationEntry.contract_specs` |
| Built-in contract capture (transitional) | `intergrax.integrations.registry.contract_capture` at registration time only |
| Contract projection / inspection | `build_contract_registry_snapshot()` in `registry_v2.py` |

## Runtime resolution flow

```text
Host IntegrationProfile
  → resolve_from_profile(category)
  → catalog.get_entry(slug)
  → IntegrationEntry.factory(...)
  → provider integration instance
```

Qualification and Collaborative Work continue to use this path. Qualification core does **not** import `registry_v2` for execution dispatch.

## Contract projection flow

```text
register_from_manifest / register_integration_plugin (once)
  → catalog IntegrationEntry (+ contract_specs)
  → build_contract_registry_snapshot()
  → immutable IntegrationRegistry projection rows
```

Projection snapshots may become stale after catalog mutation; rebuild the snapshot to refresh.

## External plugin flow

```text
IntegrationPlugin + IntegrationManifest
  → register_integration_plugin(cls, contract_specs=...)  # explicit typed specs when needed
  → catalog row
  → runtime resolve + optional contract projection
```

No `SLUG_CATEGORY` edit, no `registry_v2.register()` call, and no central vendor enum are required.

## Typed-wiring rules

**Authoritative projection path (`registry_v2.py`):** no `vars()`, `getattr`, `hasattr`, `__dict__`, class-name scanning, factory-name string dispatch, or `TypeError` probing.

**P2 transitional debt (registration-time only):** `contract_capture.py` still uses built-in package-layout reflection once during `register_from_manifest` for shipped providers. External plugins must supply explicit `IntegrationContractSpec` rows.

## Identity validation

Projection fails with `IntegrationContractProjectionError` when canonical `provider_id` metadata does not match catalog slug (unless `allow_provider_slug_alias` metadata is explicitly set).

## Qualification relationship

Provider Qualification execution remains vendor-neutral and catalog-backed. Contract projection may support inspection (“what capabilities does this provider expose?”) but is not an execution dispatch registry.

## Compatibility

- `build_contract_registry()` remains as an alias of `build_contract_registry_snapshot()`.
- `IntegrationRegistry` class name retained; instances are immutable/local projections, not provider authority.
- INTEGRATIONS-3B (registry-backed runtime binding) remains **planned** and **out of scope** for this change.

## Future removal

Phase 2 may eliminate built-in `contract_capture` reflection once all providers publish explicit registration metadata on catalog entries.
