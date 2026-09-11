# Platform Execution Unification — U3 Agent & Plugin Execution Closure

## Before

| Item | Verdict |
|------|---------|
| EP-13 | CANONICAL WITH GAP — `RuntimeToolInvoker.agent_runtime_governance` optional; production UAEP path skipped governance when unset |
| Plugin execution | No supported production plugin-local tool runtime (integration tests only) |

## After

| Item | Verdict |
|------|---------|
| EP-13 | **CANONICAL** — production composition requires `RuntimeConfig.agent_runtime_governance`; invoker fail-closed when `production_mode` and governance absent |
| Plugin execution | **QUALIFIED / NO CHANGE** — plugins supply contracts; side effects remain on canonical `RuntimeToolInvoker` path |

## EP-13 proof

```text
UAEP.run → RuntimeExecutionContext + BoundToolGateway
→ RuntimeToolGateway.for_state → invoke_catalog_tool_request
→ config.tool_invoker (RuntimeToolInvoker wired in RuntimeContext.build)
→ _require_agent_runtime_governance (mandatory in production_mode)
→ authority / scope / declarative policy / executor
```

## Reused owners

- `RuntimeExecutionContext.invoke_tool`
- `BoundToolGateway` / `RuntimeToolGateway`
- `RuntimeContext.build` + `RuntimeConfig.agent_runtime_governance`
- `AgentRuntimeGovernanceBoundary` / `AgentRuntimeGovernancePipeline`
- `intergrax/runtime/wiring/agent_runtime_governance_factory.py`
- `intergrax/applications/_shared/agent_runtime_governance_wiring.py` (roster grants)

## Tests

- `tests/unit/runtime/architecture/test_platform_execution_unification_u3_agent_plugin_execution_closure.py`
- `tests/unit/runtime/governance/test_runtime_context_agent_runtime_governance.py`
- Existing NPSC-4 agent governance + UAEP/tool gateway tests (regression)

## Remaining

- **BY-01**: UNCHANGED / DEFERRED TO U4
- **EP-17**: UNCHANGED / AMBIGUOUS — REQUIRES OWNER DECISION
