# T-IGX-036-T — PDC-X P6 batch B-P6-R: integration repairs

## Scope

Small integration repairs that the P5 lanes could not fix inside their write sets: Windows `tzdata`, D3 capability flags through the evidence adapter, D2 freshness through multi-pass retrieval, the D4 launcher path, the observability circular import, D8 trace alias callers and feature-doc registration; D8 composition wiring and the XD and ECAP findings that close inside existing authorities (Amendment 1). Product code: yes. Plan: `PLAN_T-IGX-036-T.md`.

## Plan digest

Plan digest: 5cdfbe6ecb8a40549275f275b877cb9d94afc822979621e4decce370c3b1afbc

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-T.md`](plans/PLAN_T-IGX-036-T.md).

## Approved

_Pending: delegate re-signing per Amendment 2 of [`DELEGATION-PDC-P6-REPAIR.md`](../../docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P6-REPAIR.md). The implementer never writes this line._

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

plan_pending
