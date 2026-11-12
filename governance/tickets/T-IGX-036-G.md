# T-IGX-036-G — PDC-X P4: implementation design (WBS, DAG, cross-domain test matrix; no product code)

## Scope

PDC-P4 implementation design: `scripts/qualification/pdc_x_wbs.py` with a batch table that renders the implementation WBS, dependency DAG and cross-domain test matrix XD-01..12; audit checkpoints; PDC-P5-REG placement; ECAP D8 note. Plan: `PLAN_T-IGX-036-G.md`. Roadmap stage `PDC-X` (§3.2).

## Plan digest

Plan digest: 05ee3467d1db3f4fa990d6ff4e35c18ea44a086c661fb10f84cedf2ea363d0fa

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-G.md`](plans/PLAN_T-IGX-036-G.md).

## Approved

_Pending: delegate session per [`DELEGATION-PDC-WAVE-A.md`](../../docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-WAVE-A.md). The implementer never writes this line._

## Test contract

```text
uv run python scripts/qualification/pdc_x_wbs.py --check
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run pytest tests/unit/governance/test_pdc_x_wbs.py tests/unit/governance/test_pdc_x_requirements.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Product code; P5 implementation; P5 ticket files; FRZ status changes; IIAC activation; OD signatures; merges.

## Lifecycle state

plan_pending
