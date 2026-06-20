---
id: IJ-2026-06-11-026
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-PROD-9
status: completed
commit: pending
adr: none — CI wiring only
---

# Sprint 1 — APP-PROD-9 wire application production gates to gate CI

## Operator request

Execute Tier-3 implementation Sprint 1: wire `check_application_production_gates.py` into `pytest -m gate` and CI.

## Summary

Added `tests/unit/scripts/test_check_application_production_gates.py` with `pytest.mark.gate`. Added script to GitHub Actions `gate-governance-tier` job. Marked APP-PROD-9 Done in plan.

## Project impact

Tier-3 host factory/manifest/wiring rules enforced on every gate run and PR CI tier-audit job — closes APP-PROD-9 fidelity gap.

## Traceability

| Link | Target |
|------|--------|
| Architecture | TIER3 §40.2 APP-PROD register |
| Plan | Sprint 1 · APP-PROD-9 |

## Changed artifacts

- `tests/unit/scripts/test_check_application_production_gates.py`
- `.github/workflows/unit-tests.yml`
- `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md`

## Verification

```bash
uv run pytest tests/unit/scripts/test_check_application_production_gates.py -q
python scripts/check_application_production_gates.py
uv run pytest tests/unit/scripts/test_check_application_production_gates.py -m gate -q
```

Result: pass.

## Risks and follow-ups

- Sprint 2: APP-CON-3 env state lifecycle sync on hooks.
