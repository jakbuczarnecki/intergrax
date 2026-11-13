# T-IGX-036-M — PDC-X P5 batch B-REG-P: duplicate `ollama` catalog slug

## Scope

Fix the duplicate `ollama` catalog slug that breaks the full integration bootstrap (regression from `7aa9cfa18`). Requirements: PDC-D1-R022. Product code: yes; implemented in an isolated worktree, no merge before MCP-SERVER-X P8 (OD-PDC-SEQ-01). Plan: `PLAN_T-IGX-036-M.md`.

## Plan digest

Plan digest: d0ca223371410d5727a6e9bb6751e568f794f1c1979ba362ee85d19526c56bb2

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-M.md`](plans/PLAN_T-IGX-036-M.md).

## Approved

Approved: operator-delegate @ 2026-11-13T08:01:32+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-STANDING.md@33cd27da81a787548d77e3e5fc719b49e733e824

## Test contract

```text
uv run pytest tests/qualification/pdc_x/reg_p tests/unit/rag/test_rag_plugin_discovery.py tests/qualification/plug_03 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Other batches and the circular import in `runtime/integrations/observability.py`.
- FRZ changes, OD signatures, merges and IIAC activation.

## Lifecycle state

plan_approved
