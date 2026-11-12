# T-IGX-036-G — PDC-X P4: implementation design (WBS, DAG, cross-domain test matrix; no product code)

## Scope

PDC-P4 implementation design: `scripts/qualification/pdc_x_wbs.py` with a batch table that renders the implementation WBS, dependency DAG and cross-domain test matrix XD-01..12; audit checkpoints; PDC-P5-REG placement; ECAP D8 note. Plan: `PLAN_T-IGX-036-G.md`. Roadmap stage `PDC-X` (§3.2).

## Plan digest

Plan digest: 05ee3467d1db3f4fa990d6ff4e35c18ea44a086c661fb10f84cedf2ea363d0fa

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-G.md`](plans/PLAN_T-IGX-036-G.md).

## Approved

Approved: operator-delegate @ 2026-11-12T07:30:40+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-WAVE-A.md@b8cacd0d1ad3ef91d534e6468ec6797a0ceec9fb

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

quality_gate

## Checkpoint

| Field | Value |
| --- | --- |
| Branch | `pdc-x-p4` |
| `pdc_x_wbs.py --check` | `[OK] 107 rows assigned, 14 batches, no write conflicts` |
| Unit tests | `test_pdc_x_wbs.py`, `test_pdc_x_requirements.py`, `test_pdc_x_inventory.py`: 25 passed |
| Delegate notes addressed | Integration owner for shared hotspots stated and checked (one hotspot writer per wave). ECAP note placed in the ECAP baseline notes block. P5 ticket letters reserved in the WBS (I, J..S), as the plan specifies |
| Red-gate diagnosis | `PDC_REG_DIAGNOSIS.md`: 12 stale gates (B-REG-T, tests only), 1 product regression (B-REG-P, gated) |
