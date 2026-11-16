# T-IGX-036-R — PDC-X P5 batch B-D4: D4 Host Runtime

## Scope

Generic lifecycle suite with crash and restart proof, lifecycle metrics, digest-preserving restart, stable public API, TL-FIX-C. Requirements: PDC-D4-R003, R004, R005, R006, R007, R009, R010, R014, R015. Decision: ADR-PDC-04. Product code: yes; implemented in an isolated worktree, no merge before MCP-SERVER-X P8 (OD-PDC-SEQ-01). Plan: `PLAN_T-IGX-036-R.md`.

## Plan digest

Plan digest: 0ebec21021956c44da73267bc2fdbc969df393334f6b8debdc4e1fcc1c9f5581

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-R.md`](plans/PLAN_T-IGX-036-R.md).

## Approved

Approved: operator-delegate @ 2026-11-13T06:42:05+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-WAVE-3C.md@8044e236dffedf305b832f309e3d5e5da26f3aeb

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d4 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Other batches and the `ollama` regression (B-REG-P).
- FRZ changes, OD signatures, merges and IIAC activation.

## Lifecycle state

plan_approved
