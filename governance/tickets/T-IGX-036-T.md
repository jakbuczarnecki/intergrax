# T-IGX-036-T — PDC-X P6 batch B-P6-R: integration repairs

## Scope

Small integration repairs that the P5 lanes could not fix inside their write sets: Windows `tzdata`, D3 capability flags through the evidence adapter, D2 freshness through multi-pass retrieval, the D4 launcher path, the observability circular import, D8 trace alias callers and feature-doc registration; D8 composition wiring and the XD and ECAP findings that close inside existing authorities (Amendment 1). Product code: yes. Plan: `PLAN_T-IGX-036-T.md`.

## Plan digest

Plan digest: 8962feba0fe726191b1026a5a3b78cb511a95341ab3810c0ba03667e1bb9edb8

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-T.md`](plans/PLAN_T-IGX-036-T.md).

## Approved

Approved: operator-delegate @ 2026-11-13T16:18:33+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P6-REPAIR.md@67a4dd1374b6362fb3defac799b2b30f52d1c99e

## Test contract

```text
uv run pytest tests/qualification/pdc_x/p6_repair -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- IIAC header and activation, compat_x baseline counters, the unused ollama manifest, queue provider identity mapping.
- FRZ changes, OD signatures and certification.

## Lifecycle state

plan_approved
