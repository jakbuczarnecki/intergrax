---
id: IJ-2026-06-10-003
date: 2026-06-10
tiers:
  - tier-1
scope: TOOLS
plan_ref:
  - TOOL-ENG-0
  - TOOL-ENG-3
status: completed
commit: 87fb3841
adr: none — wiring existing catalog planner into RuntimeContext.build
---

# Wire catalog tool planner and scope policy in RuntimeContext.build

## Operator request

Establish the foundation for catalog-native tool planning: Nexus must build tool planner inputs from runtime context and enforce scope policy before TOOL-ENG dispatch and constraint work lands.

## Summary

Connected catalog tool planner to `RuntimeContext.build`, introduced scope policy hooks, and aligned `tools_step` with the catalog planning path. Prepared runtime types and config sections for downstream TOOL-ENG-1/2/4 deliverables.

## Project impact

Tool execution moves from ad-hoc boolean flags toward plan-driven `tool_ids` and typed planner inputs — prerequisite for constrained planning (TOOL-ENG-4) and full-gateway routing (TOOL-ENG-1/2).

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TOOLS.md` |
| Plan | `docs/plan/TOOLS.md` TOOL-ENG phase rows |

## Changed artifacts

- `intergrax/runtime/nexus/tools/catalog_tool_planner.py`
- `intergrax/runtime/nexus/runtime_steps/tools_step.py`
- `intergrax/runtime/nexus/config.py`, `config_sections.py`

## Verification

```bash
uv run pytest tests/unit/runtime/nexus/tools/ -q
```

Result: pass (existing tools unit suite).

## Risks and follow-ups

- TOOL-ENG-1/2 (dispatch + routing) and TOOL-ENG-4 (plan constraints) required for full closeout.
