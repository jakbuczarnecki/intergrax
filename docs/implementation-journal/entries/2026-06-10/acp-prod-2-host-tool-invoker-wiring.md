---
id: IJ-2026-06-10-031
date: 2026-06-10
tiers:
  - tier-0
  - tier-1
  - tier-3
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-PROD-2
  - TOOL-ENG-6
status: completed
commit: pending
adr: none — host wiring for existing DeclarativeToolInvoker protocol
---

# ACP declarative catalog tool invoker host wiring (C4)

## Operator request

Continue the ACP production sprint sequence by wiring `declarative_tool_invoker` into Tier-3 hosts and `acp_run`, so declarative `requested_actions` execute real catalog tools instead of `skipped_no_invoker`.

## Summary

Added `CatalogDeclarativeToolInvoker` routing through `invoke_catalog_tool_request`, metadata helpers in `tool_invoker_wiring.py`, and Tier-3 `declarative_tool_wiring.py` to build invokers from `ApplicationToolWiring`. `run_acp_session` resolves and binds the invoker on `StepKernelContext`. `GraphExecutor` / `NexusLoop` / `build_harness_host_runtime` inject the host invoker into ACP session metadata alongside checkpoint wiring.

## Project impact

Lab and harness hosts with enabled tool catalogs now supply declarative tool I/O for ACP sessions routed through Nexus or direct `Agent.run(AgentRunRequest)` with metadata.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §32.8, §40.2 |
| Plan | Wave 7.2 ACP-PROD-2 depth · `plan/TOOLS.md` TOOL-ENG-6 |
| ADR | none |

## Changed artifacts

- `intergrax/agents/persistence/catalog_declarative_invoker.py`
- `intergrax/agents/persistence/tool_invoker_wiring.py`
- `intergrax/applications/_shared/declarative_tool_wiring.py`
- `intergrax/agents/authoring/acp_run.py`, `acp_session_host.py`
- `intergrax/runtime/nexus/execution/graph_executor.py`, `nexus_loop.py`
- `intergrax/applications/_shared/nexus_factory.py`, `harness_host_runtime.py`
- `intergrax/contracts/acp_metadata_keys.py`
- Unit tests under `tests/unit/agents/persistence/` and `tests/unit/applications/shared/`

## Verification

```bash
uv run pytest tests/unit/agents/persistence/test_catalog_declarative_invoker.py tests/unit/agents/persistence/test_tool_invoker_wiring.py tests/unit/applications/shared/test_declarative_tool_wiring.py -q
```

## Risks and follow-ups

- Minimal `RuntimeState` shim uses `MagicMock` for non-tool subsystems; full Nexus `RuntimeContext` reuse is a future optimization.
- Product hosts without `build_harness_host_runtime` must attach invoker manually via metadata or `ACPSessionHostContext`.
