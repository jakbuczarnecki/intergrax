# DUP-X — Closed-World Inventory

| Field | Value |
| --- | --- |
| **AUDITED_HEAD** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` |
| **Mechanical SSOT** | `tests/qualification/dup_x/_closed_world_inventory.py` |
| **Families** | **15** |
| **Status** | **INVENTORY FROZEN @ IMPLEMENTATION WORKTREE** — independent audit pending |

## Discovery spine

```text
ebh_5/COMPOSITION_ROOT_HINTS → ebh_6/closed_world → dup_x/inventory
TRACE-X PRODUCTION_COMPOSITION_SITE_REGISTRY → composition union
CONFIG-X heuristic ∪ hints (DUP-14 qual SSOT)
```

## Responsibility families (summary)

| family_id | semantic_responsibility | implementation_owner_path | reachability |
| --- | --- | --- | --- |
| RF-EXE-01 | Execution ingress / dispatch | `intergrax/runtime/execution/` | production-effective |
| RF-BGW-01 | Background workers | `agents/` | production-effective |
| RF-GOV-01 | Governance admission | `intergrax/runtime/governance/` | production-effective |
| RF-CAP-01 | Capability selection | `intergrax/integrations/registry/factory.py` | production-effective |
| RF-PRV-01 | Provider bundles | `intergrax/integrations/providers/` | production-effective |
| RF-REG-01 | Registries | `intergrax/tools/registry/` | production-effective |
| RF-RCV-01 | Retry / recovery | `intergrax/runtime/enterprise_reliability/` | production-effective |
| RF-STA-01 | State / persistence | `intergrax/runtime/state/` | production-effective |
| RF-PRV-02 | Provenance / evidence | `intergrax/runtime/events/` | production-effective |
| RF-OBS-01 | Observability | `intergrax/runtime/observability/` | production-effective |
| RF-PLG-01 | Plugin lifecycle | `intergrax/integrations/registry/plugin_register.py` | production-effective |
| RF-TEN-01 | Tenant / context | `intergrax/runtime/tenant/` | production-effective |
| RF-LEG-01 | Legacy compat | `intergrax/tools/tools_base.py` | legacy-reachable |
| RF-CMP-01 | Composition roots | hints + TRACE-X + wiring | production-effective |
| RF-T3-01 | Tier-3 copies | `applications/**` | unknown — L3 per app |

## Mechanical census (@ HEAD)

| Enumerator | Count (indicative) | Notes |
| --- | ---: | --- |
| `*Registry` classes | mechanical | `discover_registry_class_census()` |
| `*Resolver` classes | mechanical | `discover_resolver_class_census()` |
| Dispatch-bearing modules | mechanical | `discover_dispatcher_symbol_census()` |
| Composition root union | mechanical | `discover_normative_composition_root_union()` |
| Production `tools_base` importers | **0** | AST gate |

## Candidate cross-links

See [`DUP_X_FINDINGS_REGISTER.md`](DUP_X_FINDINGS_REGISTER.md) · [`DUP_X_CLASS_COVERAGE_MATRIX.md`](DUP_X_CLASS_COVERAGE_MATRIX.md).

**Gap IDs:** `DUPX-GAP-CLASS-D` · `DUPX-GAP-L3-TIER3-DEDUPE`
