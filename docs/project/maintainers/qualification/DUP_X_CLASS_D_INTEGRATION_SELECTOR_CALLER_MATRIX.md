# DUP-X — Class D integration selector caller matrix (DUP-10)

| Field | Value |
| --- | --- |
| **Gap** | DUPX-GAP-CLASS-D |
| **Candidate** | DUPX-CAND-03 |
| **Wave** | DUP-X-03 Wave 1 |
| **Canonical owner** | `intergrax/integrations/_shared/config.py` (env key) · `intergrax/integrations/registry/factory.py` (effective slug/profile resolution) |

## Disposition summary

| Symbol / module | Role | Disposition | Evidence |
| --- | --- | --- | --- |
| `read_integration_slug_from_env` | `INTERGRAX_INTEGRATION_<CATEGORY>` parse | **CANONICAL** | Sole env slug reader; re-exported from `_shared` for tests only |
| `resolve_slug` / `build_profile_from_env` / `resolve` / `resolve_from_profile` | Category slug → catalog materialization | **CANONICAL** | SSOT in `factory.py`; mechanical gate `test_dup_x_class_d_no_shadow_category_slug_reads_in_intergrax` |
| `execution_bound_integration_resolution` | Execution-bound adoption pin + provenance | **THIN** | Delegates `resolve` / `resolve_from_profile` via `ExecutionBoundIntegrationMaterializationPort`; no parallel env reads |
| `existing_capability_configuration_facade` | Governed INT-CONFIG-REAL-X public entry | **DISTINCT** | Authorization + realization service; does not select integration slugs |
| `integration_vectorstore.bootstrap` | RAG vectorstore bootstrap | **CALLER** | Imports `build_profile_from_env` / `resolve` from factory — no shadow selector |

## Caller graph (production-effective)

```text
Tier-3 composition roots / agents
  → resolve_from_profile(profile, category)  [factory.py]
  → resolve(category, profile=…, config=…)   [factory.py]
       → resolve_slug → profile slug | read_integration_slug_from_env | error

Execution-bound adoption (TRACE-X-P5-R2-P3)
  → execution_bound_integration_resolution
       → materialization_port.resolve_from_profile / resolve_catalog
            → factory.resolve / factory.resolve_from_profile

Existing-capability configuration (governance)
  → existing_capability_configuration_facade.realize
       → authorization_port + realization_service (no slug selection)

Env overlay
  → build_profile_from_env(defaults=…)
       → read_integration_slug_from_env per IntegrationCategory
```

## Negative bypass proof

- Wave B gate: no `read_integration_slug_from_env` call sites under `intergrax/` outside `factory.py` and `_shared/config.py` (`discover_integration_env_slug_reads_outside_factory()`).
- No BLOCKER: production remediation (OD-DUPX-01) not required for qualification closure — audit records bounded ownership only.

## Operator notes

- **OD-DUPX-01** remains **OPEN** — no production code delta in Wave 1.
- Future convergence (if any) must go through factory + CONFIG-X contracts, not parallel registries.
