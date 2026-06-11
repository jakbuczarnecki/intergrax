---
id: IJ-2026-06-11-025
date: 2026-06-11
tiers:
  - tier-0
  - tier-1
scope: AGENT_CONTRACTS_AND_ASSEMBLY
plan_ref:
  - ACP-CON-4
status: completed
commit: pending
adr: none — contract default aligns with scaffold/authoring baseline; test-only UAEP stub module
---

# ACP gate CI — budget default, fleet inventory fixture, UAEP test stubs

## Operator request

Restore failing regression-gate CI after ACP assembly validation and legacy pipeline removal: plugin catalog smoke, readiness scoreboard, graph/orchestration integration tests, and legal host factory gate.

## Summary

Set `AgentContract.max_steps` default to `10` so assembly validation matches scaffold and authoring baselines. Added session autouse pytest fixture to generate `build/agent_fleet_inventory.json` when gate tests run without governance scripts. Introduced `testing_support/uaep_gate_stubs.UaepPipelineStubAgent` and migrated graph, orchestration, debug intake, and critic gate tests off direct `RuntimeEngine` pipeline stubs (ACP-CLOSE-LEG-1). Fixed legal factory gate test to set `INTERGRAX_HARNESS_API_KEY`.

## Project impact

Gate CI and local `-m gate` runs no longer fail on missing budget bounds, missing fleet inventory, or non-UAEP test agents. Plugin catalog registration smoke and production readiness scoreboard tests pass in the gate-tests job without depending on parallel governance job artifacts.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §12 |
| Plan | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` ACP-CON-4 |
| ADR | none |

## Changed artifacts

- `intergrax/contracts/agent_contract_meta.py` — default `max_steps=10`
- `tests/conftest.py` — fleet inventory session fixture
- `testing_support/uaep_gate_stubs.py` — shared UAEP gate stub agent
- `tests/integration/runtime/test_*` — UAEP stub migration (graph, orchestration, planner)
- `tests/acceptance/agent_os/test_agent_os_scenarios.py` — graph acceptance stubs
- `tests/unit/debug/test_h3_*.py`, `test_h5_*.py` — intake execute paths
- `tests/unit/runtime/critic/test_critic_*_graph.py` — UAEP wiring
- `tests/unit/runtime/hooks/test_tool_and_selection_hooks.py` — harness API key env

## Verification

- `uv run python scripts/check_plugin_catalog.py` — OK
- `uv run python scripts/check_agent_acp_close_ci.py` — OK
- `uv run pytest … -m gate -q` (full gate paths, no xdist) — 1753 passed

## Risks and follow-ups

- `UaepPipelineStubAgent.run_count` / `run_log` are class-level; parallel xdist workers may need per-class isolation if flakes reappear.
- Consider generating fleet inventory in the `gate-tests` CI step explicitly (not only via pytest fixture) for clearer job ordering.
