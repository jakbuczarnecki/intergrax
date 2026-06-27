---
id: IJ-2026-06-11-042
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-EVOL-4
status: completed
commit: pending
adr: none — formalizes §49.4 contracts; complements APP-OPS-1 lifecycle blast-radius gate
---

# Sprint 17 — AgentCertification STRICT roster gate (APP-EVOL-4)

## Operator request

Continue Tier-3 application architecture sprint queue: APP-EVOL-4 — agent certification records and STRICT product roster gate blocking non-approved lifecycle states.

## Summary

- `agent_governance.py` — `AgentApprovalPolicy`, `AgentCertificationRecord`, `AgentGovernanceProfile`.
- `agent_governance_profile` on `ApplicationEnvironmentProfile`.
- `agent_certification_wiring.py` — roster certification materialization, STRICT validation, product gate helper.
- Reference STRICT product manifests declare roster certifications via `apply_roster_agent_governance`.
- `scripts/maintenance/check_agent_certification_roster.py` wired into `check_application_production_gates.py`.

## Project impact

STRICT product hosts now require explicit certification records for STAGING/PRODUCTION roster agents and reject experimental/deprecated lifecycle states — closing §49.4 enforcement beyond capability-graph blast-radius checks.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §49.4 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` APP-EVOL-4 · §6.2y step 13 |

## Changed artifacts

- `intergrax/applications/contracts/agent_governance.py`
- `intergrax/applications/contracts/environment_profile.py`
- `intergrax/applications/_shared/agent_certification_wiring.py`
- `applications/*/manifest.py` (×4 product hosts)
- `scripts/maintenance/check_agent_certification_roster.py`
- `scripts/gates/check_application_production_gates.py`
- `tests/unit/applications/test_agent_certification_gate.py`
- `tests/unit/scripts/test_check_agent_certification_roster.py`

## Verification

```bash
uv run pytest tests/unit/applications/test_agent_certification_gate.py \
  tests/unit/scripts/test_check_agent_certification_roster.py \
  tests/unit/scripts/test_check_application_production_gates.py -q
uv run python scripts/maintenance/check_agent_certification_roster.py
python scripts/maintenance/check_implementation_journal.py
```

Result: pass.

## Risks and follow-ups

- STAGING remains allowed on STRICT reference hosts with reference-host certification placeholders — promotion to PRODUCTION requires updated records and ACP-PROD evidence.
- Next queue item: APP-EVOL-5 ApplicationRecoveryContract.
