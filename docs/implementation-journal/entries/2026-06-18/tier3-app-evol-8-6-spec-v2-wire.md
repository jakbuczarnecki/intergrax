---
id: IJ-2026-06-18-001
date: 2026-06-18
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-EVOL-8.6
status: completed
commit: pending
adr: ADR-APP-003 — M3 implementation; no new ADR
---

# Tier-3 — APP-EVOL-8.6 spec_version 2.0 nested canonical wire

## Operator request

Implement open P1 `APP-EVOL-8.6` for `TIER3_APPLICATION_ENVIRONMENT` after Mode A audit identified drift on M3 profile wire.

## Summary

Shipped `spec_version` `2.0.0` nested canonical JSON wire: `uses_nested_profile_wire()`, `migrate_profile_dict_to_spec_v2()`, `ApplicationEnvironmentProfile.with_spec_v2_wire()`, extended `ProfileMigration` validation for 1.x→2.x, `standard_profile_spec_v2_migration()` + `apply_profile_migration()`, schema gate coverage, and plan migration guide.

## Project impact

Tier-3 authors can opt into nested canonical profile JSON without Nexus changes; 1.x flat wire remains default for reference hosts until product cutover.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §22.6.4 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` — APP-EVOL-8.6 |
| ADR | `docs/adr/entries/2026-06-17/ADR-APP-003.md` |

## Changed artifacts

- `intergrax/applications/contracts/environment_profile/normalization.py`
- `intergrax/applications/contracts/environment_profile/root.py`
- `intergrax/applications/contracts/environment_profile/__init__.py`
- `intergrax/applications/_shared/migration_wiring.py`
- `tests/unit/applications/test_environment_profile_bundles.py`
- `tests/unit/applications/test_migration_wiring.py`
- `scripts/check_environment_profile_bundle_schema.py`

## Verification

```bash
uv run pytest tests/unit/applications/test_environment_profile_bundles.py tests/unit/applications/test_migration_wiring.py -q
uv run python scripts/check_environment_profile_bundle_schema.py
```

## Remaining

- Reference hosts still default to `spec_version` 1.x flat wire until explicit product migration.
- P3 `test_research_application_exposes_mcp_mount` and stale tier3 audit prompt (separate backlog).

## Risks and follow-ups

- STRICT hosts adopting 2.0 require golden replay per plan migration guide §44.
- External digest consumers must treat `spec_version` as wire metadata, not semantic drift.
