---
id: IJ-2026-06-11-017
date: 2026-06-11
tiers:
  - tier-0
  - tier-1
scope: AGENT_CONTRACTS_AND_ASSEMBLY
plan_ref:
  - ACP-CLOSE-CI-2
status: completed
commit: pending
adr: none — static enforcement of existing ACP-AP-02 boundary post TOOL-ENG-6
---

# ACP-CLOSE CI-2 — ACP-AP-02 tool loop boundary gate

## Operator request

Continue ACP-CLOSE sprint: close CI-2 — ensure Nexus graph orchestration does not schedule tool iterations (anti-pattern ACP-AP-02).

## Summary

Added `check_agent_acp_ap02_tool_loop_boundary.py` — static gate forbidding `run_bounded_tool_loop`, `tool_loop_step`, `plan_native_round`, and `ToolPlanningService` references in `intergrax/runtime/nexus/execution/` and `orchestration/`, and in Tier-2 `agents/`. Only `tools_step.py` may import `run_bounded_tool_loop` outside `tool_loop_step.py`. Wired into `check_agent_acp_close_ci.py` and CI matrix row CI-17. ACP-CLOSE wave marked complete.

## Project impact

Tool/ReAct loops stay in Plane 3 (`ToolsStep` / `run_bounded_tool_loop`) or Plane 2 (`ReActAgent`); `GraphExecutor` cannot regress into micromanaging tool iterations.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `AGENT_CONTRACTS_AND_ASSEMBLY` §28.4 ACP-AP-02 · §40.10 CI-17 |
| Plan | `ACP-CLOSE-CI-2` |
| ADR | none — enforces ADR-TOOL-002 placement |

## Changed artifacts

- `scripts/maintenance/check_agent_acp_ap02_tool_loop_boundary.py` (new)
- `scripts/gates/check_agent_acp_close_ci.py`
- `scripts/gates/check_acp_ci_conformance_matrix.py`
- `tests/unit/scripts/test_check_agent_acp_ap02_tool_loop_boundary.py` (new)
- `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`
- `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`

## Verification

```bash
uv run pytest tests/unit/scripts/test_check_agent_acp_ap02_tool_loop_boundary.py tests/unit/scripts/test_check_agent_acp_close_ci.py -m gate -q
uv run python scripts/maintenance/check_agent_acp_ap02_tool_loop_boundary.py
uv run python scripts/gates/check_acp_ci_conformance_matrix.py --scripts-only
python scripts/maintenance/check_implementation_journal.py
```

## Risks and follow-ups

- Gate is static (source scan); behavioral coverage remains in `test_tool_loop_integration.py`.
- ACP-CLOSE wave complete — next work defaults to gate maintenance in `PLATFORM_FOUNDATION`.
