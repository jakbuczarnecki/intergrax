---
id: IJ-2026-06-11-045
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-EVOL-7
status: completed
commit: pending
adr: none — packaging contract + resolver; no marketplace or Nexus changes
---

# Sprint 20 — ApplicationPackage + dependency resolver (APP-EVOL-7)

## Operator request

Continue Tier-3 application architecture sprint queue: APP-EVOL-7 — formal application package model, dependency closure resolver, scaffold emission, and CI gate.

## Summary

- `application_package.py` — `ApplicationPackage`, `ApplicationDependency`, `ApplicationDistribution`.
- `package_wiring.py` — collect/build/validate package closure; `package_gate_environment` for CI wiring without vendor drivers.
- `package_emit.py` — scaffold `package.json` emission from `new-application` / `new-stack`.
- `wire_application_environment` validates package closure when `conformance_check=True`.
- `scripts/maintenance/check_application_package.py` wired into production gates.

## Project impact

Tier-3 hosts now declare immutable package artifacts with direct agent/skill/tool/integration/profile dependencies. Scaffold emits `package.json`; STRICT product hosts pass package closure smoke in CI. Closes APP-EVOL evolution register (§49.8).

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §49.7 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` APP-EVOL-7 · §6.2y step 13 |

## Changed artifacts

- `intergrax/applications/contracts/application_package.py`
- `intergrax/applications/_shared/package_wiring.py`
- `intergrax/applications/_shared/environment_wiring.py`
- `intergrax/scaffold/package_emit.py`
- `intergrax/scaffold/new_application.py`
- `intergrax/scaffold/new_stack.py`
- `scripts/maintenance/check_application_package.py`
- `scripts/gates/check_application_production_gates.py`
- `tests/unit/applications/test_package_wiring.py`
- `tests/unit/scripts/test_check_application_package.py`
- `tests/unit/scaffold/test_minimal_stack_scaffold.py`

## Verification

```bash
uv run pytest tests/unit/applications/test_package_wiring.py \
  tests/unit/scripts/test_check_application_package.py \
  tests/unit/scaffold/test_minimal_stack_scaffold.py \
  tests/unit/scripts/test_check_application_production_gates.py -q
uv run python scripts/maintenance/check_application_package.py
python scripts/maintenance/check_implementation_journal.py
```

Result: pass.

## Risks and follow-ups

- CI package closure uses lab integration swap (`package_gate_environment`) — runtime deploy still uses product integration profiles.
- Marketplace/registry distribution channels are schema-only; APP-OPS-3/4 next in §6.2y queue.
