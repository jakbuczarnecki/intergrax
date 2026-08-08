---
id: IJ-2026-06-11-043
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-EVOL-5
status: completed
commit: pending
adr: none — profile contract + documentation; no Nexus recovery fork
---

# Sprint 18 — ApplicationRecoveryContract on ReliabilityProfile (APP-EVOL-5)

## Operator request

Continue Tier-3 application architecture sprint queue: APP-EVOL-5 — typed recovery contract on product `ReliabilityProfile` with architecture documentation and CI gate.

## Summary

- `application_recovery_contract.py` — `ApplicationRecoveryContract` and recovery action enums.
- `recovery_contract` field on `ReliabilityProfile`; default on `product_defaults`.
- `recovery_contract_wiring.py` — validation, ARCHITECTURE.md markers, task metadata attach.
- `apply_reliability_task_defaults` wires `application_recovery_contract.v1` on tasks.
- Product ARCHITECTURE.md recovery sections (legal, research, dispute_sim, local_workspace).
- `scripts/maintenance/check_application_recovery_contract.py` wired into production gates.

## Project impact

STRICT product hosts now declare explicit recovery actions per failure scenario; CI validates contract/reliability consistency and product architecture recovery documentation.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §49.5 |
| Plan | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` APP-EVOL-5 · §6.2y step 13 |

## Changed artifacts

- `intergrax/applications/contracts/application_recovery_contract.py`
- `intergrax/applications/contracts/environment_profile.py`
- `intergrax/applications/_shared/recovery_contract_wiring.py`
- `intergrax/applications/_shared/reliability_wiring.py`
- `applications/*/ARCHITECTURE.md` (×4 product hosts)
- `scripts/maintenance/check_application_recovery_contract.py`
- `scripts/gates/check_application_production_gates.py`

## Verification

```bash
uv run pytest tests/unit/applications/test_recovery_contract_wiring.py \
  tests/unit/scripts/test_check_application_recovery_contract.py \
  tests/unit/scripts/test_check_application_production_gates.py -q
uv run python scripts/maintenance/check_application_recovery_contract.py
python scripts/maintenance/check_implementation_journal.py
```

Result: pass.

## Risks and follow-ups

- Contract is declarative — runtime enforcement of each action remains in existing checkpoint/HITL paths.
- Next queue item: APP-EVOL-6 ApplicationEnvironmentDiff.
