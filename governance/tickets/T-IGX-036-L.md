# T-IGX-036-L — PDC-X P5 batch B-D7: D7 Background & Scheduling

## Scope

Recurrence on SchedulerLedger, dead-letter state, identity on enqueue, two-process fencing. Requirements: PDC-D7-R006, R007, R008, R009, R012, R013, R016. Decision: ADR-PDC-07. Product code: yes; implemented in an isolated worktree, no merge before MCP-SERVER-X P8 (OD-PDC-SEQ-01). Plan: `PLAN_T-IGX-036-L.md`.

## Plan digest

Plan digest: 65d5e68109bc10808bda49ea712398412d429e5bd4a8ffef64677c78ada97660

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-L.md`](plans/PLAN_T-IGX-036-L.md).

## Approved

Approved: operator-delegate @ 2026-11-12T11:25:00+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-WAVE-1.md@2a4056374beefe91f567c07e201245f3baaec4e5

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d7 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Other batches and the `ollama` regression (B-REG-P).
- FRZ changes, OD signatures, merges and IIAC activation.

## Lifecycle state

plan_approved
