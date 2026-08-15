---
id: IJ-2026-06-11-044
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-EVOL-6
status: completed
commit: pending
adr: none — diff contract + CLI; no Nexus or profile primitive changes
---

# Sprint 19 — ApplicationEnvironmentDiff + doctor diff-app (APP-EVOL-6)

## Operator request

Continue Tier-3 application architecture sprint queue: APP-EVOL-6 — typed environment diff for pre-deploy review with CLI and CI gate.

## Summary

- `application_environment_diff.py` — `ApplicationEnvironmentDiff`, `StructuredDiff`, `RosterEntryChange`, `DiffRiskLevel`.
- `environment_diff_wiring.py` — `diff_profile`, `diff_graph`, `diff_envelope`, `diff_roster`, `build_application_environment_diff`, `assess_diff_risk`, `format_application_environment_diff`.
- `doctor_diff_app.py` — `intergrax doctor diff-app` with `--app`, `--left`, `--right`, `--json`, `--fail-on-high`.
- `doctor.py` — subcommand routing for `diff-app` alongside default harness checks.
- `scripts/maintenance/check_application_environment_diff.py` wired into production gates.

## Project impact

STRICT product hosts can be diffed for deploy review: profile/graph/envelope/roster deltas, aggregate risk classification, and CI smoke across product manifests. Foundation for incident comparison using `profile_snapshot_id` from APP-EVOL-1.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §49.6 |
| Plan | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` APP-EVOL-6 · §6.2y step 13 |

## Changed artifacts

- `intergrax/applications/contracts/application_environment_diff.py`
- `intergrax/applications/_shared/environment_diff_wiring.py`
- `intergrax/cli/doctor_diff_app.py`
- `intergrax/cli/doctor.py`
- `scripts/maintenance/check_application_environment_diff.py`
- `scripts/gates/check_application_production_gates.py`
- `tests/unit/applications/test_environment_diff_wiring.py`
- `tests/unit/scripts/test_check_application_environment_diff.py`

## Verification

```bash
uv run pytest tests/unit/applications/test_environment_diff_wiring.py \
  tests/unit/scripts/test_check_application_environment_diff.py \
  tests/unit/scripts/test_check_application_production_gates.py -q
uv run python scripts/maintenance/check_application_environment_diff.py
python scripts/maintenance/check_implementation_journal.py
```

Result: pass.

## Risks and follow-ups

- `doctor diff-app` currently varies manifest `version` labels against the same resolved environment — full two-deploy diff needs paired snapshot artifacts (deferred).
- Next queue item: APP-EVOL-7 ApplicationPackage.
