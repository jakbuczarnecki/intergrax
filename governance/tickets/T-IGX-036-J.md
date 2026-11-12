# T-IGX-036-J — PDC-X P5 batch B-D1: D1 Agent Lifecycle

## Scope

Agent upgrade/retire, skill lifecycle, discovery owner, adversarial lifecycle. Requirements: PDC-D1-R003, R004, R007, R011, R012, R013, R014, R017, R018, R020. Decision: ADR-PDC-01. Product code: yes; implemented in an isolated worktree, no merge before MCP-SERVER-X P8 (OD-PDC-SEQ-01). Plan: `PLAN_T-IGX-036-J.md`.

## Plan digest

Plan digest: e7029fb035c409d1363d4f827c5f4c96b3d2c1b1663836bc2511d734886dd4ad

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-J.md`](plans/PLAN_T-IGX-036-J.md).

## Approved

_Pending: delegate session per [`DELEGATION-PDC-WAVE-1.md`](../../docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-WAVE-1.md). The implementer never writes this line._

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d1 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Other batches and the `ollama` regression (B-REG-P).
- FRZ changes, OD signatures, merges and IIAC activation.

## Lifecycle state

plan_pending
