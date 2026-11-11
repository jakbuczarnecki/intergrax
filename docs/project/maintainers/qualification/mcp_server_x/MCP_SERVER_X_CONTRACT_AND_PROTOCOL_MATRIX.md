# MCP-SERVER-X — contract and protocol matrix

**Status:** P1 populated (architecture); implementation evidence columns filled @ P2–P7  
**Supersedes scaffold:** [`MCP_SERVER_X_CONTRACT_MATRIX.md`](./MCP_SERVER_X_CONTRACT_MATRIX.md) (pointer only)

## Public contracts

| Surface | Version | Owner module (planned) | Fingerprint | COMPAT inventory |
| --- | --- | --- | --- | --- |
| `mcp-server-definition.v1` | v1 | `intergrax/contracts/mcp_server_definition.py` | _pending P4-03_ | _pending P4-05_ |
| MCP tool schema export | existing | `intergrax/tools/exporters/mcp.py` | from `ToolContract` | historical |
| Streamable HTTP host adapter | FastMCP pin | `intergrax/applications/_shared/fastapi_mcp.py` | lockfile @ SHA | host unit tests |

## Protocol / transport (v1 design)

| REQ-ID | Declared support | Runtime owner | Evidence task |
| --- | --- | --- | --- |
| REQ-MCP-P-01 | Revision = FastMCP 3.x @ qual SHA | Transport adapter | MCP-P3-02 L3 |
| REQ-MCP-P-02 | Negotiation via SDK | FastMCP | DEFERRED |
| REQ-MCP-P-03 | stdio/SSE | — | **UNSUPPORTED_BY_DESIGN** (OD-MCP-03) |
| REQ-MCP-P-04 | Cancel on disconnect | Execution + transport cancel | MCP-P5-06 |
| REQ-MCP-P-05 | Contract error catalog | Server definition | MCP-P2-05 |

## MCP methods vs stage claims (target @ P7)

| Method / feature | v1 claim | Owner |
| --- | --- | --- |
| `tools/list`, `tools/call` | **Supported** (tools only) | Engine + ToolContract |
| Resources | **Not advertised** | OD-MCP-02 |
| Prompts | **Not advertised** | OD-MCP-02 |
| Progress / cancel via transport | **Supported** (existing cancel path) | Execution |

## Error codes (planned @ P2-05)

| Code class | Meaning | When |
| --- | --- | --- |
| `MCP_DEF_INVALID` | Definition failed validation | Pre-mount |
| `MCP_BINDING_UNKNOWN` | Binding id not in policy allowlist | Discovery/invoke |
| `MCP_BINDING_RETIRED` | Retired capability invoked | Invoke |
| `MCP_PRINCIPAL_REQUIRED` | Auth profile requires harness principal | Pre-execution |
| `MCP_TENANT_ASSERTION_CONFLICT` | Client assertion vs principal | Pre-execution |
| `MCP_RESULT_BOUND_EXCEEDED` | Payload over byte limit | Post-tool |

Exact enum names finalized in implementation; matrix SSOT updated @ P2 merge.

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
