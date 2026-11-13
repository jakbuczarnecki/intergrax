# T-IGX-036-Q — PDC-X P5 batch B-D8: D8 Artifact Lifecycle

## Scope

Artifact taxonomy, mandatory digest, governed read and download port, durable store, retention, spill, dedupe. Requirements: PDC-D6-R022, PDC-D8-R001, R002, R003, R004, R005, R006, R007, R008, R009, R010, R012, R013, R014, R016, R017, R018. Decision: ADR-PDC-08. Product code: yes; implemented in an isolated worktree, no merge before MCP-SERVER-X P8 (OD-PDC-SEQ-01). Plan: `PLAN_T-IGX-036-Q.md`.

## Plan digest

Plan digest: f98b68477734ebe0aa89646231a1d95e0fcfe2fb743aacb6daab15bd9886f254

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-Q.md`](plans/PLAN_T-IGX-036-Q.md).

## Approved

Approved: operator-delegate @ 2026-11-13T06:26:23+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-WAVE-3B.md@55d4bf8fe369ad6f5ac25821e854be01c8fe331d

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d8 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Other batches (B-D4 follows) and the `ollama` regression (B-REG-P).
- FRZ changes, OD signatures, merges and IIAC activation.

## Lifecycle state

plan_approved
