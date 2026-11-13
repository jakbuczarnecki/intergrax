# T-IGX-036-S — PDC-X P6 batch B-P6: cross-domain integration

## Scope

Cross-domain integration scenarios XD-01..XD-12, DUP-X and EBH-4 replay and the full ECAP change-delta on the integrated baseline. Requirements: PDC-D2-R018, PDC-D6-R019, PDC-D7-R014, PDC-D8-R019 and the scenario-level cross-domain cases XD-01..XD-12. Tests only; starts after all P5 batches are merged, which waits for MCP-SERVER-X P8 (OD-PDC-SEQ-01). Plan: `PLAN_T-IGX-036-S.md`.

## Plan digest

Plan digest: c7b967b58769de0c956962d8e7d9b76709d5d51a8b77231774118cf423a1e4a6

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-S.md`](plans/PLAN_T-IGX-036-S.md).

## Approved

_Pending: delegate session per [`DELEGATION-PDC-STANDING.md`](../../docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-STANDING.md), authorization 4. The implementer never writes this line._

## Test contract

```text
uv run pytest tests/qualification/pdc_x -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Product-code repairs of findings (separate integration-repair ticket).
- FRZ changes, OD signatures, IIAC activation and certification.

## Lifecycle state

plan_pending
