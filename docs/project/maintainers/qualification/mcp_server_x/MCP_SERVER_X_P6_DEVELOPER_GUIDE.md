# MCP-SERVER-X P6 — developer guide

**Public API module:** `intergrax.applications._shared.mcp_developer_api`  
**Definition contract:** `intergrax.contracts.mcp_server_definition`  
**Reference qualification server:** `tests/qualification/mcp_server_x/reference_app/build.py`

## Quickstart

```python
from intergrax.applications._shared.mcp_developer_api import (
    compose_application_mcp_server,
    couple_mcp_fastmcp_on_harness_host,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from tests.fixtures.harness_host_task_execution import build_harness_host_task_execution

registry = AgentRegistry()
# ... nexus + host_execution ...
mcp = compose_application_mcp_server(
    display_name="My App MCP",
    host_execution=host_execution,
    registry=registry,
    default_capability="my.capability",
    tool_registry=tool_registry,  # optional
)
api = couple_mcp_fastmcp_on_harness_host(fastapi_app, mcp, mount_path="/mcp", mcp_enabled=True)
```

## Supported imports (Tier-3)

| Concern | Module |
| --- | --- |
| Compose application server | `mcp_developer_api.compose_application_mcp_server` |
| Custom host tools (e.g. research pipeline) | `execute_mcp_agent_task`, `research_pipeline_binding` |
| Typed definition / validation | `build_mcp_server_definition`, `McpServerDefinition`, validators |
| Host mount (streamable HTTP) | `couple_mcp_fastmcp_on_harness_host`, mount path helpers |
| Advanced composition | `compose_mcp_server_from_definition`, `McpServerCompositionDeps` |

Do **not** import `mcp_catalog_invocation_bridge`, `mcp_server_engine` internals, or catalog enforcement modules from Tier-3.

## Reference server walkthrough

The P6 reference server is test-only (`build_mcp_server_x_reference_qualification_server`). It exercises:

- Tools discovery (`list_tools` via streamable HTTP)
- Custom host binding (`qual_echo`)
- Host builtins (`list_agents`, `run_agent`)
- Harness mount at `/mcp`

L3 proof: `tests/qualification/mcp_server_x/test_mcp_server_x_p6_reference_l3_qualification.py`

## Transport and hosting

- **Supported:** FastMCP 3.x streamable HTTP via `couple_mcp_fastmcp_on_harness_host` / CONFIG mount pattern (P3).
- **Unsupported:** stdio, SSE (OD-MCP-03).

## Security and tenant integration

- Principal propagation and tenant isolation are qualified in P5 — use harness auth; do not bypass `MeaningfulSideEffectAuthorizationPort` for catalog tools.
- Client-supplied `tenant_id` on tools is an **assertion** reconciled with harness principal (P5).

## Versioning and compatibility

- Server definition contract version: `mcp-server-definition.v1` (`MCP_SERVER_DEFINITION_CONTRACT_VERSION`).
- Exposure changes: recompose definition; fingerprint via `compute_mcp_exposure_schema_fingerprint`.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Mount 404 | `mcp_enabled`, `normalize_mcp_mount_path` |
| Tool missing from discovery | `McpExposurePolicy.allowed_binding_ids`, retired bindings |
| `MCP_BINDING_UNKNOWN_TOOL` | Catalog tool not in registry at compose time |
| Research strict factory failure | `LIM-RESEARCH-STRICT-FACTORY-MEMORY` — not an MCP engine defect |

## MCP v1 capability matrix (announced subset)

| Capability | v1 status |
| --- | --- |
| Tools | **Supported** |
| Resources | **Unsupported by design** |
| Prompts | **Unsupported by design** |
| streamable HTTP | **Supported** |
| stdio / SSE | **Unsupported by design** |

Full matrix: [`MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md`](./MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md)
