---
id: IJ-2026-06-10-011
date: 2026-06-10
tiers:
  - tier-0
  - tier-3
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-DX-2
  - ACP-DX-3
  - ACP-DX-4
  - ACP-DX-5
  - ACP-CFG
status: completed
commit: pending
adr: none — implements §29–§30 merge order per ADR-AGENT-002
---

# ACP Wave 2 — run facade, environment merge, Nexus bridge

## Operator request

Deliver Wave 2: `merge_environment`, typed `IntergraxAgent.run(AgentRunRequest)`, Nexus `RuntimeRequest` bridge, `AgentBinding` profile slices, and reference harness profile injection.

## Summary

Implemented `EffectiveAgentRunEnvironment` and `merge_environment` with memory scope §30.9 validation, extended `AgentBinding` and `AgentContract` binding fields, `run_acp_session` loop wiring Wave 1 kernel, `IntergraxAgent.run` overload for `AgentRunRequest`, optional Nexus bridge via `acp.session.v1` metadata, and `build_lab_agent_runtime_config_from_merged` for ACP-CFG.

## Project impact

Authors can call `await agent.run(AgentRunRequest(...))` directly with merged Tier-3 slices. Nexus can opt into the same typed loop without forking UAEP default path. Lab/reference agents can build `RuntimeConfig` from merged profile flags.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §29–§30 |
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` Wave 2 |
| ADR | ADR-AGENT-002 |

## Changed artifacts

- `intergrax/agents/run_environment.py` — merge + namespace templates
- `intergrax/agents/authoring/acp_run.py` — session loop
- `intergrax/agents/authoring/base.py` — `run()` + hooks
- `intergrax/agents/runtime_request_bridge.py` — Nexus bridge
- `intergrax/agents/agent_engine.py` — ACP session routing
- `intergrax/applications/contracts/manifest.py` — binding slices
- `intergrax/contracts/memory_scope.py` — scope enum
- `intergrax/agents/reference_harness.py` — merged config builders
- `tests/unit/agents/test_run_environment.py` and related Wave 2 tests

## Verification

```bash
uv run pytest tests/unit/agents/ tests/unit/runtime/registry/ -m gate -q
```

52 gate tests green (agents + registry).

## Risks and follow-ups

- Nexus ACP bridge is opt-in (`acp.session.v1`) — default path remains UAEP until fleet migration Wave 8.
- `ACPSessionHostContext.binding` excluded from JSON serialization — hosts pass via in-process metadata.
- Next: Wave 3 observability (`ACP-OBS-1/2`), LLM routing, shared context.
