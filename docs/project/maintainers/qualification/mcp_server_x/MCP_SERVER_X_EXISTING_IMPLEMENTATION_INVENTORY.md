# MCP-SERVER-X — existing implementation inventory

**Closed-world pin:** `c228799785db8f35826f47ba0790d6a21d97fbe3`  
**Semantic owner today:** application hosting composition in `intergrax/applications/_shared/` (Tier-0 application shared layer). Execution semantics: canonical host execution and catalog child invocations — not a second ToolRuntime.

## Core modules

| Module | Responsibility | Contract / API | Input | Boundary |
| --- | --- | --- | --- | --- |
| [`intergrax/applications/_shared/fastapi_mcp.py`](../../../../intergrax/applications/_shared/fastapi_mcp.py) | Mount FastMCP `http_app` on FastAPI wrapper; lifespan; auth middleware; task-control and transport-detachment coupling | `couple_fastapi_with_mcp`, `couple_harness_host_fastapi_with_mcp` | ASGI `/mcp` (default), streamable HTTP | Transport only; runtime semantics outside module |
| [`intergrax/applications/_shared/mcp_nexus_server.py`](../../../../intergrax/applications/_shared/mcp_nexus_server.py) | `FastMCP(name)` + `list_agents`, `run_agent`; optional catalog mount | `build_nexus_mcp_server` | Host execution, registry, default tenant/user | `run_agent` → `resolve_mcp_run_agent_identity`, `execute_mcp_agent_task` |
| [`intergrax/applications/_shared/mcp_catalog_tools.py`](../../../../intergrax/applications/_shared/mcp_catalog_tools.py) | `list_catalog_tools`, `describe_catalog_tool`, `invoke_catalog_tool` | `mount_catalog_tools_on_mcp` | `ToolRegistry` + optional `McpCatalogInvocationComposition` | No selective exposure policy: enabled registry ⇒ full catalog on server |
| [`intergrax/applications/_shared/mcp_catalog_invocation_bridge.py`](../../../../intergrax/applications/_shared/mcp_catalog_invocation_bridge.py) | Catalog invocation bridge: task/run/step binding, tenant, child execution | `execute_mcp_catalog_tool_invocation` | Active execution | Not a second ToolRuntime |
| [`intergrax/applications/_shared/mcp_catalog_side_effect_enforcement.py`](../../../../intergrax/applications/_shared/mcp_catalog_side_effect_enforcement.py) | Side-effect authorization request | Governance request builder | Governance port | Policy decision outside MCP |
| [`intergrax/applications/_shared/mcp_harness_principal.py`](../../../../intergrax/applications/_shared/mcp_harness_principal.py) | Harness principal from FastMCP request; client tenant/user assertions rejected on conflict | `resolve_mcp_run_agent_identity` | HTTP context | Required auth without principal → error; local bypass uses arg/default tenant |
| [`intergrax/applications/_shared/mcp_host_execution_transport_cancel.py`](../../../../intergrax/applications/_shared/mcp_host_execution_transport_cancel.py) | Root cancel on transport drop | Admission + launch | Supervision port | Reason `mcp_transport_cancel`; no second child root |
| [`intergrax/applications/_shared/mcp_task_control_coupling.py`](../../../../intergrax/applications/_shared/mcp_task_control_coupling.py) | Task-control mutation boundary at mount | Resolver | Runtime/env | Governance control plane |
| [`intergrax/applications/_shared/mcp_transport_detachment_supervision_coupling.py`](../../../../intergrax/applications/_shared/mcp_transport_detachment_supervision_coupling.py) | Lifespan shutdown supervision coupling | Resolver + lifespan | Runtime | Execution supervision |
| [`intergrax/tools/mcp_result_bounds.py`](../../../../intergrax/tools/mcp_result_bounds.py) | Result byte limits | `bound_mcp_structured_result`, `emit_bounded_mcp_list` | Payload | Fail on limit exceeded |
| [`intergrax/applications/_shared/mcp_import_guard.py`](../../../../intergrax/applications/_shared/mcp_import_guard.py) | Lazy FastMCP import | `ensure_mcp_dependencies` | Import | No silent fallback |
| [`intergrax/tools/exporters/mcp.py`](../../../../intergrax/tools/exporters/mcp.py) | Export `ToolContract` → MCP JSON Schema | `to_mcp_tools` | Registry/contract | Schema export only — not server publication |

## Applications (Tier-3 thin servers)

| Application | Path | Notes |
| --- | --- | --- |
| dispute_sim, governed_contractor, intergrax_assistant, lab, legal, local_workspace, poc_template | `applications/*/mcp/server.py` | Thin `build_nexus_mcp_server` wrappers (name, default capability, default tenant/user) |
| research_application | `applications/research_application/mcp/server.py` | Adds `run_research_pipeline` via `execute_mcp_agent_task` |

## Dependency and transport

- **SDK:** `fastmcp>=3.3.1,<4` in [`pyproject.toml`](../../../../pyproject.toml).
- **Host transport:** streamable HTTP via `mcp.http_app` only; no `@mcp.resource` / `@mcp.prompt` under `intergrax/`.
- **Configuration:** `MCP_MOUNT_PATH` in [`intergrax/applications/_shared/settings_loader.py`](../../../../intergrax/applications/_shared/settings_loader.py).

## Tests (historical — reuse, not MCP-SERVER-X closure)

| Test area | Paths |
| --- | --- |
| FastAPI MCP host | `tests/unit/applications/test_fastapi_mcp.py` |
| Reference apps | `tests/unit/applications/test_reference_apps_mcp.py` |
| Surface opt-in / execution / principal / cancel | `test_mcp_surface_opt_in.py`, `test_mcp_canonical_execution.py`, `test_mcp_trust_a_principal_binding.py`, `test_mcp_cancel_a_transport_governed_cancel.py` |
| Catalog L3 | `test_mcp_catalog_f3_streamable_http_physical_l3.py`, cross-loop cancel |
| Fixture | `tests/fixtures/tool_life_x_l3_r1_mcp_qualification_server.py` |

## Observability

Correlation via `RequestIdentity` and existing execution trace. **No** dedicated MCP audit log (by design).
