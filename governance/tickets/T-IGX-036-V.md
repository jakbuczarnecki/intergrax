# T-IGX-036-V — PDC-X P6 batch B-P6-G: ENT-AUDIT impact-map coverage

## Scope

Declarative product-domain rules in the CHANGED-SINCE impact map so that the PDC-touched product modules select domain checks (XD finding 8 from ticket T-IGX-036-S, split out of T-IGX-036-T). Product code: yes (`intergrax/platform/enterprise_audit/`). Plan: `PLAN_T-IGX-036-V.md`.

## Plan digest

Plan digest: 93923ce2c4c985dbd70d25619ce3772c6565c9c2a7af9133794333913c198203

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-V.md`](plans/PLAN_T-IGX-036-V.md).

## Approved

Approved: operator-delegate @ 2026-11-13T16:18:33+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P6-IMPACT.md@67a4dd1374b6362fb3defac799b2b30f52d1c99e

## Test contract

```text
uv run pytest tests/qualification/pdc_x/p6_impact tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py tests/qualification/ent_audit_x -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- The 85-check manifest rule and the manifest JSON, new checks, IIAC header and activation.
- FRZ changes, OD signatures and certification.

## Lifecycle state

plan_approved
