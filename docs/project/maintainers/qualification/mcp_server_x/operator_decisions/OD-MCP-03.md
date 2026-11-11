# OD-MCP-03 — Transport scope v1

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-03 |
| **Status** | **PENDING** |
| **ADR** | [`ADR_MCP_03_TRANSPORT_STREAMABLE_HTTP.md`](../../../architecture/ADR_MCP_03_TRANSPORT_STREAMABLE_HTTP.md) |

## Question

Which transports are in scope for qualification and production exposure?

## Recommended option (A)

- **In scope:** streamable HTTP via FastMCP `>=3.3.1,<4` and `fastapi_mcp.py`.
- **Out of scope v1:** stdio, SSE.

## Operator record

_Decision status: **PENDING**. Agent MUST NOT write `Approved`._
