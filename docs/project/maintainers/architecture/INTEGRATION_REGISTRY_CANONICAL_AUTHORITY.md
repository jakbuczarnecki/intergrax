# Integration Registry Canonical Authority (INTEGRATIONS-REGISTRY-CANONICALIZATION-1)

**Status:** CLOSED (architecture accepted · environment regression closed R1)  
**Date:** 2026-09-02  
**Architecture ancestor:** `e6ec82d13993dd8abb01dc33f32dbaf940364513`  
**Environment regression HEAD:** `b575cb6fdaa55231ca59df60350f4cc5696a4051`

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

**P2 transitional debt (registration-time only):** `contract_capture.py` remains a migration-only fallback for built-ins not yet publishing explicit `IntegrationContractSpec` rows. New providers and migrated built-ins must use provider-owned explicit declarations via `register_from_manifest(..., contract_specs=...)`. Runtime reflection discovery is prohibited as canonical authority.

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

## Environment regression closure (R1)

**Task:** `INTEGRATIONS-REGISTRY-CANONICALIZATION-1-R1`  
**Scope:** environment restoration + mandatory PostgreSQL/Mongo regression only (no architecture changes).

### Canonical dependency provisioning

```bash
uv sync --extra integrations-postgresql --extra integrations-mongodb
uv run python -c "import psycopg; print(psycopg.__version__)"
```

- **CLIENT_DRIVER_VERSION (psycopg):** `3.3.4` (driver availability only; not backend provenance)
- **PostgreSQL service:** `intergrax-postgresql` · image `postgres:16.6` · port `5434` · healthy
- **MongoDB service (optional qual evidence):** `intergrax-mongodb` · image `mongo:7.0` · port `27017` · healthy
- **Sanitized DSN env:** `INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN=postgresql://***@localhost:5434/intergrax`

### Regression commands and results (HEAD `b575cb6f`)

| Suite | Command | Result |
|-------|---------|--------|
| Canonical registry + external plugin + slack boundaries | `uv run pytest tests/unit/integrations/test_registry.py tests/unit/runtime/integrations/test_contract_registry_v2.py tests/unit/runtime/integrations/test_canonical_registry_projection.py tests/unit/integrations/test_external_plugin.py tests/unit/integrations/providers/conversation_channel/slack/test_registry_boundaries.py -q` | **48 passed**, 0 skipped |
| CW unit | `uv run pytest tests/unit/collaborative_work/ -q` | **218 passed**, 0 skipped |
| Real PostgreSQL CW repository | `uv run pytest tests/integration/collaborative_work/test_postgresql_repository.py -m "integration and network" -q` | **15 passed**, 0 skipped |
| CW full E2E | `uv run pytest tests/e2e/collaborative_work/ -m e2e -q` | **35 passed**, 0 skipped |
| CW PostgreSQL E2E | `uv run pytest tests/e2e/collaborative_work/ -m "e2e and integration and network" -q` | **11 passed**, 0 skipped |
| Qualification units | `uv run pytest tests/unit/core/qualification/ -q` | **224 passed**, 0 skipped |
| Real PostgreSQL qualification | `uv run pytest tests/integration/core/qualification/test_provider_qualification_execution_postgresql.py -m "integration and network" -q` | **1 passed**, 0 skipped |
| Multi-provider qualification | `uv run pytest tests/integration/core/qualification/test_provider_qualification_multi_provider_proof.py -q` | **7 passed**, 0 skipped |
| Vendor abstraction gates | `uv run pytest tests/unit/core/qualification/test_provider_qualification_vendor_abstraction_gate.py tests/unit/collaborative_work/test_vendor_neutrality.py tests/e2e/collaborative_work/test_architecture_gates.py tests/unit/runtime/integrations/test_canonical_registry_projection.py -q` | **16 passed**, 0 skipped |
| Mongo qualification persistence (PROVIDER-QUAL-9 evidence) | `uv run pytest tests/integration/core/qualification/test_provider_qualification_discovery_mongo.py tests/integration/core/qualification/test_provider_qualification_validity_mongo.py tests/integration/core/qualification/test_provider_qualification_persistence_durable_reopen.py -q` | **6 passed**, 0 skipped |

**Packaging acceptance:** `uv sync --extra integrations-postgresql` supplies `psycopg[binary]` sufficient for all mandatory PostgreSQL tests; no manual driver install required.

**Session logs:** `.tmp/session/INTEGRATIONS-REGISTRY-CANONICALIZATION-1-R1/`

**Verdict:** `INTEGRATIONS-REGISTRY-CANONICALIZATION-1` — **READY_TO_CLOSE** (pending independent R1 evidence/SHA audit).
