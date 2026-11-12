# T-IGX-036-P — PDC-X P5 batch B-D6: D6 Runtime Security

## Scope

Fail-closed signing and poisoning defaults, single tool-input defense path, durable audit trail, secret gate, plugin provenance. Requirements: PDC-D2-R008, PDC-D6-R004, R006, R007, R008, R009, R013, R014, R015, R016, R020, R021, R023, R024, R027. Decision: ADR-PDC-06. Product code: yes; implemented in an isolated worktree, no merge before MCP-SERVER-X P8 (OD-PDC-SEQ-01). Plan: `PLAN_T-IGX-036-P.md`.

## Plan digest

Plan digest: 68f1eb209906a3e5b988f962617b36826b325339fed606792ad33069cdac5bbb

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-P.md`](plans/PLAN_T-IGX-036-P.md).

## Approved

Approved: operator-delegate @ 2026-11-12T17:33:46+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-WAVE-3A.md@e7e123b43772a8b43bb3d6c86e009a366a9621c1

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d6 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Other batches (B-D8 and B-D4 follow) and the `ollama` regression (B-REG-P).
- FRZ changes, OD signatures, merges and IIAC activation.

## Lifecycle state

plan_approved
