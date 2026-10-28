# EBH-6 — Enterprise Surface Inventory

| Field | Value |
| --- | --- |
| **Stage** | EBH-6 — Final Architecture Recertification |
| **IMPLEMENTATION_SHA** | `badb7a086d4d4d4b374535b8698efb06eb8fa6db` |
| **Discovery SSOT** | [`tests/qualification/ebh_6/_closed_world_discovery.py`](../../../tests/qualification/ebh_6/_closed_world_discovery.py) |

Closed-world inventory of composition roots, sanctioned production sites, application wiring, and integration bundles for L3 reachability (roadmap §3.0.8).

## Composition root hints (Tier-0..3)

Mechanical discovery: `discover_composition_root_paths()` — extends EBH-5 hints including:

- `intergrax/integrations/registry/factory.py`
- `intergrax/integrations/execution_bound_integration_resolution.py`
- `intergrax/llm_adapters/registry/profile.py`
- `intergrax/applications/_shared/memory_wiring.py`
- `intergrax/applications/_shared/runtime_event_delivery_wiring.py`
- `intergrax/tools/registry/wiring.py`
- `intergrax/rag/embedding/registry/profile.py`
- `intergrax/integrations/existing_capability_configuration_facade.py`

## Sanctioned production composition roots

Source: TRACE-X `PRODUCTION_COMPOSITION_SITE_REGISTRY` — `discover_sanctioned_production_composition_roots()`.

## Application wiring entrypoints

Pattern: `applications/**/*_wiring.py` — `discover_application_wiring_paths()`.

## Integration provider bundles

Pattern: `intergrax/integrations/providers/**/bundle.py` — `discover_integration_bundle_paths()`.

## SQLite production touchpoints

Replay: EBH-5 annex + HEAD diff — `assert_ebh6_sqlite_annex_accountability()` (38 registered paths @ qualification run).

## Universal audit layers (§3.0.8)

Sixteen closed-world layer ids in `EBH6_AUDIT_LAYERS` — each mapped to `EBH6-LAYER-*` findings in [`EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md`](EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md).

## Explicit non-claims

- Not a duplicate of `VENDOR_BACKEND_DEPENDENCY_INVENTORY` (EBH-5 SSOT).
- Does not assert production qualification or live vendor connectivity.
