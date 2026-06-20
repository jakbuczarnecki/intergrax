---
id: IJ-2026-06-11-015
date: 2026-06-11
tiers:
  - tier-0
  - tier-1
  - tier-2
scope: AGENT_CONTRACTS, TOOLS
plan_ref:
  - ACP-CLOSE-PAT-1
  - TOOL-ENG-6
status: completed
commit: pending
adr: ADR-TOOL-002
---

# ACP-CLOSE PAT-1 + TOOL-ENG-6 — unified tool loop and ReAct budget

## Operator request

Continue ACP-CLOSE sprint: deliver PAT-1 with cross-domain TOOL-ENG-6 — ReAct ↔ unified tool loop + shared `AcpBudgetState` keys.

## Summary

Added `run_bounded_tool_loop` (`tool_loop_step.py`) with `RuntimeConfig.max_tool_iterations` (default 1). `ToolsStep` delegates to the loop; multi-iteration paths use `ToolPlanningService.plan_native_round` and append `role=tool` messages. Extended `AcpBudgetState` with react iteration fields; `ReActAgent` mirrors counters via `react_budget.py`. Closed DEBT-ACP-18 and GAP-ACP-04. ADR-TOOL-002 accepted.

## Project impact

Pipeline and ReAct patterns share one bounded tool-loop implementation and budget vocabulary in `acp.state.v1.budget`. Hosts opt into multi-iteration tool chains via `max_tool_iterations > 1`.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `AGENT_CONTRACTS_AND_ASSEMBLY` §25.2 · §26.3; `TOOLS` multi-tool semantics |
| Plan | `ACP-CLOSE-PAT-1`, `TOOL-ENG-6` |
| ADR | `docs/adr/entries/2026-06-11/ADR-TOOL-002.md` |

## Changed artifacts

- `intergrax/runtime/nexus/runtime_steps/tool_loop_step.py` (new)
- `intergrax/runtime/nexus/runtime_steps/tools_step.py` — loop delegation
- `intergrax/runtime/nexus/tools/tool_planning_service.py` — `plan_native_round`
- `intergrax/runtime/nexus/config.py` — `max_tool_iterations`
- `intergrax/contracts/acp_state.py` — budget react fields
- `intergrax/agents/authoring/patterns/react.py`, `react_budget.py` (new)
- `tests/integration/runtime/test_tool_loop_integration.py` (new)
- `tests/unit/agents/authoring/patterns/test_react_budget.py` (new)

## Verification

```bash
uv run pytest tests/integration/runtime/test_tool_loop_integration.py tests/unit/agents/authoring/patterns/test_react_budget.py tests/unit/runtime/nexus/tools/test_tool_planning_constraints.py -m gate -q
```

## Risks and follow-ups

- JSON fallback planner remains single-iteration.
- ACP-CLOSE-PAT-2 (CVL reflection hooks) and CI-2 remain open.
