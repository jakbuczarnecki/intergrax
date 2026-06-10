---
id: IJ-2026-06-10-005
date: 2026-06-10
tier: tier-1
scope: TOOLS
plan_ref: TOOL-ENG-4, TOOL-ENG-11
status: completed
commit: 79fd5e81
adr: none — enforcement of existing step plan tool_ids contract
---

# Constrain tool planner by plan tool_ids and tools_context_scope

## Operator request

Ensure the Nexus tool planner cannot invoke tools outside the execution plan: honour `tool_ids` on the step plan and implement `tools_context_scope` so context and catalog visibility align with planner input.

## Summary

Extended `tool_planning_service`, `tool_planner_input`, and step handlers to constrain selectable tools to plan `tool_ids`. Implemented `tools_context_scope` propagation through `tools_step` and runtime state. Added `test_tool_planning_constraints.py` gate suite.

## Project impact

Execution graphs and planners can declare an explicit tool surface — reduces unbounded tool exposure, supports security/policy reviews, and aligns with LEG closeout (`tool_ids` canonical path).

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TOOLS.md` |
| Plan | `docs/plan/TOOLS.md` TOOL-ENG-4, TOOL-ENG-11 |
| Audit map | Layer 11 — Tool Library |

## Changed artifacts

- `intergrax/runtime/nexus/tools/tool_planning_service.py`
- `intergrax/runtime/nexus/tools/tool_planner_input.py`
- `tests/unit/runtime/nexus/tools/test_tool_planning_constraints.py`

## Verification

```bash
uv run pytest tests/unit/runtime/nexus/tools/test_tool_planning_constraints.py -q
```

Result: pass.

## Risks and follow-ups

- Tier-3 hosts must pass consistent `tool_ids` in graph specs; document in agent creation guide if gaps appear in AA audit.
