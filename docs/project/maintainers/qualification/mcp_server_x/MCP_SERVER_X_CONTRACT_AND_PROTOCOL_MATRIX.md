# MCP-SERVER-X — contract and protocol matrix

**Status:** P3 transport pin (streamable HTTP only @ implementation SHA); L3 evidence @ MCP-P3-02 / P7  
**Supersedes scaffold:** [`MCP_SERVER_X_CONTRACT_MATRIX.md`](./MCP_SERVER_X_CONTRACT_MATRIX.md) (pointer only)

## Public contracts

| Surface | Version | Owner module (planned) | Fingerprint | COMPAT inventory |
| --- | --- | --- | --- | --- |
| `mcp-server-definition.v1` | v1 | `intergrax/contracts/mcp_server_definition.py` | `compute_mcp_exposure_schema_fingerprint` | COMPAT mechanical inventory |
| MCP tool schema export | existing | `intergrax/tools/exporters/mcp.py` | from `ToolContract` | historical |
| Streamable HTTP host adapter | FastMCP 3.x (`fastmcp>=3.3.1,<4`) | `fastapi_mcp.py` + `mcp_host_mount.py` | lockfile @ SHA | L2 `test_fastapi_mcp`; L3 streamable HTTP qual |

## Protocol / transport (v1 design)

| REQ-ID | Declared support | Runtime owner | Evidence task |
| --- | --- | --- | --- |
| REQ-MCP-P-01 | **Supported:** streamable HTTP via `FastMCP.http_app` only; revision = installed `fastmcp` @ qual SHA | `mcp_host_mount` + `fastapi_mcp` | MCP-P3-02 L3 (`test_mcp_catalog_f3_streamable_http_physical_l3`) |
| REQ-MCP-P-02 | Negotiation via SDK | FastMCP | DEFERRED |
| REQ-MCP-P-03 | stdio/SSE | — | **UNSUPPORTED_BY_DESIGN** (OD-MCP-03) |
| REQ-MCP-P-04 | Cancel on disconnect | Execution + transport cancel | MCP-P5-06 |
| REQ-MCP-P-05 | Contract error catalog | Server definition | MCP-P2-05 |

## MCP methods vs stage claims (observed @ P7, see MCP_SERVER_X_P7_QUALIFICATION_PACKAGE.md)

| Method / feature | v1 claim | Owner |
| --- | --- | --- |
| `tools/list`, `tools/call` | **Supported** (tools only) | Engine + ToolContract |
| Resources | **Advertised by FastMCP 3.3.1 but always empty** (`UNSUPPORTED_BY_DESIGN`: `resources/list` returns `[]`, reading an unknown resource fails) | OD-MCP-02 |
| Prompts | **Advertised by FastMCP 3.3.1 but always empty** (`UNSUPPORTED_BY_DESIGN`: `prompts/list` returns `[]`, getting an unknown prompt fails) | OD-MCP-02 |
| Cancel via transport | **Supported** (existing cancel path; evidenced by the P5/P7 tests) | Execution |
| Progress notifications | **Not claimed; not evidenced at P7** (no engine code and no P7 test mentions progress) | — |
| `logging` (`logging/setLevel`) | **Advertised by FastMCP, not claimed by v1** | FastMCP |
| UI extension `io.modelcontextprotocol/ui` | **Advertised by FastMCP, not claimed by v1** | FastMCP |
| `tasks`, `completions` | **Not advertised** (method not found, -32601) | — |

## Error codes (observed @ P7; 7 members of `McpServerDefinitionErrorCode`)

| Code class | Meaning | When |
| --- | --- | --- |
| `MCP_DEF_INVALID` | Definition failed validation | Pre-mount; also at invoke for a blank binding id |
| `MCP_BINDING_UNKNOWN` | Binding id not exposed: not defined, or not in the policy allowlist | Pre-mount (allowlist naming an undefined binding); discovery/invoke |
| `MCP_BINDING_UNKNOWN_TOOL` | Catalog binding names a `tool_id` that is not in the known catalog | Pre-mount validation |
| `MCP_BINDING_RETIRED` | Retired capability invoked | Invoke |
| `MCP_PRINCIPAL_REQUIRED` | Auth profile requires harness principal | Pre-execution |
| `MCP_TENANT_ASSERTION_CONFLICT` | Client assertion vs principal | Pre-execution |
| `MCP_RESULT_BOUND_EXCEEDED` | Payload over byte limit | Post-tool |

Enum names are final and verified by the P7 L1 tests (`test_acc_02_*`). `MCP_PRINCIPAL_REQUIRED`, `MCP_TENANT_ASSERTION_CONFLICT` and `MCP_RESULT_BOUND_EXCEEDED` are declared but not raised by the engine at P7 (residual R-P7-05); the wire errors actually observed are listed in the section "Wire errors observed at P7" below.

## Binding types

| Binding kind | Source of schema | Execution path |
| --- | --- | --- |
| `catalog_tool` | `ToolRegistry` / `ToolContract` | `execute_mcp_catalog_tool_invocation` (TL11 child) |
| `host_list_agents` | built-in | host registry |
| `host_run_agent` | built-in | `execute_mcp_agent_task` |
| `host_custom` | e.g. research pipeline | `execute_mcp_agent_task` or typed binding |

## Cross-links

- Gaps: [`MCP_SERVER_X_GAP_REGISTER.md`](./MCP_SERVER_X_GAP_REGISTER.md)
- Qualification: [`MCP_SERVER_X_QUALIFICATION_MATRIX.md`](./MCP_SERVER_X_QUALIFICATION_MATRIX.md)

## Wire errors observed at P7

Strings as asserted by the P7 tests (`tests/qualification/mcp_server_x/test_mcp_server_x_p7_*`); the enum above is the definition-time contract, this table is the runtime vocabulary. Exact payload shape is per test file.

| Wire error | Situation | MCP-ACC |
| --- | --- | --- |
| HTTP 401 | Missing or invalid credentials at the HTTP ingress | 10 |
| `unauthorized` | Invalid api key, invalid bearer key, missing credentials, rejected identity-provider token (tool error) | 10 |
| `forbidden` | Tenant required but the principal carries no tenant | 12 |
| `identity_conflict:` (prefix) | Client-asserted tenant or user conflicts with the verified principal (rejected before execution) | 11, 12 |
| `missing_caller_identity` | Catalog invoke without a verified caller | 10 |
| `unauthorized_execution_caller` | Same tenant, different user: not the execution owner | 10 |
| `wrong_tenant` | Execution owned by another tenant | 11 |
| `stale_run_binding` | Stale run id is rejected instead of silently correlated | 17 |
| `missing_execution_binding` | Unknown task id on catalog invoke (fail closed, invoker not run) | 13 |
| `unauthorized_side_effect_tool` | Side-effect tool without an authorization port | 10 |
| `side_effect_authorization_denied` | Side-effect denied by the authorization port | 10 |
| `Unknown tool: '<name>'` (result with `isError`) | Unknown tool name; no redirect | P7-05 |
| `mcp_concurrent_call_limit_exceeded` | Second concurrent call over the configured limit (immediate rejection, slot released) | P7-05 |
| `mcp_payload_byte_limit_exceeded` | Bounded result over the ceiling (`list_agents`, `run_agent`); not applied to custom tools (R-P7-03) | P7-05 |
