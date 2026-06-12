---
id: IJ-2026-06-11-047
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-OPS-4
status: completed
commit: pending
adr: none — ops inventory over existing manifests; no Nexus registry changes
---

# Sprint 22 — ApplicationRegistry + EnvironmentRegistry + CLI (APP-OPS-4)

## Operator request

Continue Tier-3 application architecture sprint queue: APP-OPS-4 — typed application/environment registries with CLI and CI sync gate; closes APP-OPS register.

## Summary

- `application_registry.py` — `ApplicationRegistry`, `EnvironmentRegistry`, deployment and entry contracts.
- `registry_ops_wiring.py` — build/sync/load, `list_applications`, `get_application`, `register_application`, `check_platform_registries`.
- `intergrax/cli/apps.py` — `apps list|show|sync`.
- `intergrax/cli/envs.py` — `envs list|show`.
- Artifacts: `build/application_registry.json`, `build/environment_registry.json`.
- `scripts/check_application_registry.py` wired into production gates.

## Project impact

Platform ops now has a canonical inventory of product applications and STRICT environments with package refs, ownership, health scores, and deployment metadata. APP-OPS-1..4 complete — Tier-3 reference platform canon is feature-complete per §50.5 freeze boundary.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §50.4 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` APP-OPS-4 · §6.2y step 14 |

## Changed artifacts

- `intergrax/applications/contracts/application_registry.py`
- `intergrax/applications/_shared/registry_ops_wiring.py`
- `intergrax/cli/apps.py`
- `intergrax/cli/envs.py`
- `intergrax/cli/main.py`
- `scripts/check_application_registry.py`
- `scripts/check_application_production_gates.py`
- `tests/unit/applications/test_registry_ops_wiring.py`
- `tests/unit/scripts/test_check_application_registry.py`

## Verification

```bash
uv run pytest tests/unit/applications/test_registry_ops_wiring.py \
  tests/unit/scripts/test_check_application_registry.py \
  tests/unit/scripts/test_check_application_production_gates.py -q
uv run python scripts/check_application_registry.py
python scripts/check_implementation_journal.py
```

Result: pass.

## Risks and follow-ups

- Registry JSON embeds full package/manifest snapshots — multi-tenant pluggable store deferred.
- Next §6.2y queue: APP-CON-DX.* author guide + audit prompt.
