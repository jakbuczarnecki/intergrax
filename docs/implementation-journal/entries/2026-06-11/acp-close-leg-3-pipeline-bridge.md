---
id: IJ-2026-06-11-007
date: 2026-06-11
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-CLOSE-LEG-3
status: completed
commit: pending
adr: none — module relocation; RuntimeEngine remains internal to framework bridge
---

# ACP-CLOSE LEG-3 — retire public uaep_pipeline module

## Operator request

Execute next ACP-CLOSE sprint: retire `intergrax/agents/uaep_pipeline.py` and keep RuntimeEngine on an internal-only bridge.

## Summary

Moved `pipeline_agent_steps`, `run_pipeline_step`, and `pipeline_step_complete` to `intergrax/agents/authoring/uaep_pipeline_bridge.py` (INTERNAL ONLY). Deleted public `uaep_pipeline.py`. Updated all Tier-2 `steps/pipeline.py` imports, lab mock agents, and scaffold templates. Extended `check_agent_fleet_migration.py` to reject restored public module and `RuntimeEngine` in `agents/`.

## Project impact

GAP-ACP-03 closed. Tier-2 `agents/` contains no `RuntimeEngine` references; pipeline-backed UAEP agents call the internal bridge only.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §13.5 · GAP-ACP-03 |
| Plan | `ACP-CLOSE-LEG-3` |
| ADR | none |

## Changed artifacts

- `intergrax/agents/authoring/uaep_pipeline_bridge.py` — internal bridge (new)
- `intergrax/agents/uaep_pipeline.py` — deleted
- `agents/*/steps/pipeline.py` — import path
- `agents/lab/mock_agents.py`, `intergrax/scaffold/new_agent.py`
- `scripts/maintenance/check_agent_fleet_migration.py` — LEG-3 regression guard
- `tests/unit/agents/authoring/test_uaep_pipeline_bridge.py` — unit test (new)

## Verification

```bash
uv run pytest tests/unit/agents/authoring/test_uaep_pipeline_bridge.py -q
python scripts/maintenance/check_agent_fleet_migration.py
```

## Risks and follow-ups

- Pipeline-backed agents still depend on UAEP + internal RuntimeEngine bridge; full migration to typed ACP `on_next_step` is a separate product decision.
- ACP-CLOSE-CI-1 may extend grep gates in gate workflow.
