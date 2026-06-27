---
id: IJ-2026-06-11-046
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-OPS-3
status: completed
commit: pending
adr: none — ops score rollup over existing gates; no new runtime primitive
---

# Sprint 21 — EnvironmentHealthScore + doctor health-app (APP-OPS-3)

## Operator request

Continue Tier-3 application architecture sprint queue: APP-OPS-3 — continuous environment health score with CLI and release CI gate.

## Summary

- `environment_health_score.py` — `EnvironmentHealthScore`, `HealthDimensionScore`, `ApplicationHealthScore`, `HealthDimension`.
- `health_score_wiring.py` — rollup from APP-PROD/EVOL/OPS gates across nine dimensions; `check_strict_product_health_scores`.
- `doctor_health_app.py` — `intergrax doctor health-app` with `--json`, `--write`, `--fail-below`.
- `scripts/maintenance/check_application_health_score.py` wired into production gates.

## Project impact

STRICT product hosts now expose a 0–1 ops health score per environment with dimension evidence and production-ready rollup. Operators can publish score artifacts on release tags via `doctor health-app --write`.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §50.3 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` APP-OPS-3 · §6.2y step 14 |

## Changed artifacts

- `intergrax/applications/contracts/environment_health_score.py`
- `intergrax/applications/_shared/health_score_wiring.py`
- `intergrax/cli/doctor_health_app.py`
- `intergrax/cli/doctor.py`
- `scripts/maintenance/check_application_health_score.py`
- `scripts/gates/check_application_production_gates.py`
- `tests/unit/applications/test_health_score_wiring.py`
- `tests/unit/scripts/test_check_application_health_score.py`

## Verification

```bash
uv run pytest tests/unit/applications/test_health_score_wiring.py \
  tests/unit/scripts/test_check_application_health_score.py \
  tests/unit/scripts/test_check_application_production_gates.py -q
uv run python scripts/maintenance/check_application_health_score.py
python scripts/maintenance/check_implementation_journal.py
```

Result: pass.

## Risks and follow-ups

- `policy_coverage` uses structural proxies (policy rules / org envelope / security profile) — UC-A7 golden replay remains future hardening.
- Next queue item: APP-OPS-4 ApplicationRegistry + EnvironmentRegistry.
