---
id: IJ-2026-06-11-040
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-EVOL-2
  - APP-EVOL-2b
status: completed
commit: pending
adr: none — formalizes §49.2 contracts; no Nexus/runtime behavior change
---

# Sprint 15 — ApplicationMigration schema and typed sub-migration validators

## Operator request

Continue Tier-3 application architecture sprint queue: APP-EVOL-2 and APP-EVOL-2b — migration contracts and CI validation for breaking environment bumps.

## Summary

- `application_migration.py` — `ApplicationMigration`, `MigrationStep`, `ProfileMigration`, `GraphSpecMigration`, `OrgEnvelopeMigration`.
- `migration_wiring.py` — load JSON migrations from `applications/*/migrations/`, typed validators, semver range checks, manifest coverage.
- `scripts/check_application_migrations.py` wired into `check_application_production_gates.py`.
- Unit tests for contracts, wiring, and CI script.

## Project impact

Tier-3 hosts can declare versioned environment migrations with typed primitive sub-schemas; CI enforces document shape, step order, breaking script refs, and manifest version coverage when migration files are present.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §49.2 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` APP-EVOL-2 · APP-EVOL-2b · §6.2y step 12 |

## Changed artifacts

- `intergrax/applications/contracts/application_migration.py`
- `intergrax/applications/_shared/migration_wiring.py`
- `scripts/check_application_migrations.py`
- `scripts/check_application_production_gates.py`
- `tests/unit/applications/test_application_migration_contracts.py`
- `tests/unit/applications/test_migration_wiring.py`
- `tests/unit/scripts/test_check_application_migrations.py`

## Verification

```bash
uv run pytest tests/unit/applications/test_application_migration_contracts.py \
  tests/unit/applications/test_migration_wiring.py \
  tests/unit/scripts/test_check_application_migrations.py \
  tests/unit/scripts/test_check_application_production_gates.py -q
uv run python scripts/check_application_migrations.py
python scripts/check_implementation_journal.py
```

Result: pass.

## Risks and follow-ups

- Reference hosts ship without `migrations/` yet — gate is no-op until authors add JSON migrations on version bumps.
- Next queue item: APP-EVOL-3 capability alias routing.
