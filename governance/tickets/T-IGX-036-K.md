# T-IGX-036-K — PDC-X P5 batch B-D3: D3 Model Runtime

## Scope

Declared streaming/tool-streaming/cancel capabilities, conformance gate, catalog-owned context windows. Requirements: PDC-D3-R004, R006, R009, R011, R015, R022, R023. Decision: ADR-PDC-03. Product code: yes; implemented in an isolated worktree, no merge before MCP-SERVER-X P8 (OD-PDC-SEQ-01). Plan: `PLAN_T-IGX-036-K.md`.

## Plan digest

Plan digest: d9a3ffeda069190e1910ad5f8504eab760d2bf54b094854dec85bd3647713e90

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-K.md`](plans/PLAN_T-IGX-036-K.md).

## Approved

Approved: operator-delegate @ 2026-11-12T11:25:00+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-WAVE-1.md@2a4056374beefe91f567c07e201245f3baaec4e5

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d3 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Other batches and the `ollama` regression (B-REG-P).
- FRZ changes, OD signatures, merges and IIAC activation.

## Lifecycle state

plan_approved
