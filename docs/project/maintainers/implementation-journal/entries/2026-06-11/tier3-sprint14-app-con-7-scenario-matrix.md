---
id: IJ-2026-06-11-039
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-CON-7
status: completed
commit: pending
adr: none — registry and CI gate over existing tests; no new runtime primitive
---

# Sprint 14 — Tier-3 scenario matrix gate (APP-CON-7)

## Operator request

Continue Tier-3 application architecture sprint queue: APP-CON-7 — scenario matrix gate with UC-A* minimum per reference host posture.

## Summary

- `tier3_scenario_matrix_wiring.py` — §44 scenario catalog, §35 UC-A* mapping, reference host posture matrix (7 hosts).
- `scripts/maintenance/check_tier3_scenario_matrix.py` — CI gate validating evidence test paths and UC-A coverage.
- `test_tier3_scenario_matrix.py` — pytest `-m tier3_scenario` + gate marker.
- Wired into `check_application_production_gates.py`; registered `tier3_scenario` pytest marker.

## Project impact

Reference Tier-3 hosts now have a declarative minimum scenario matrix tied to existing unit-test evidence — closing §35/§44 planned gap and enabling posture-class completeness checks in CI.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §35 · §44 |
| Plan | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` APP-CON-7 · §6.2y step 11 |

## Changed artifacts

- `intergrax/applications/_shared/tier3_scenario_matrix_wiring.py`
- `scripts/maintenance/check_tier3_scenario_matrix.py`
- `scripts/gates/check_application_production_gates.py`
- `tests/unit/applications/test_tier3_scenario_matrix.py`
- `tests/unit/scripts/test_check_tier3_scenario_matrix.py`
- `pyproject.toml` (marker)
- `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md`
- `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md`

## Verification

```bash
uv run pytest tests/unit/applications/test_tier3_scenario_matrix.py \
  tests/unit/scripts/test_check_tier3_scenario_matrix.py \
  tests/unit/scripts/test_check_application_production_gates.py -q
python scripts/maintenance/check_tier3_scenario_matrix.py
python scripts/maintenance/check_implementation_journal.py
```

Result: pass.

## Risks and follow-ups

- Matrix is evidence-path based; integration smoke per host (§44 daemon row) remains separate from this gate.
- Next queue item: APP-EVOL-2/2b migrations.
