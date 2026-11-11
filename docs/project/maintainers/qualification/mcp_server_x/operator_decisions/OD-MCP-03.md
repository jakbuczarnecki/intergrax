# OD-MCP-03 — Transport scope v1

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-03 |
| **Status** | **APPROVED — Option A** |
| **ADR** | [`ADR_MCP_03_TRANSPORT_STREAMABLE_HTTP.md`](../../../architecture/ADR_MCP_03_TRANSPORT_STREAMABLE_HTTP.md) |

## Question

Which transports are in scope for qualification and production exposure?

## Recommended option (A)

- **In scope:** streamable HTTP via FastMCP `>=3.3.1,<4` and `fastapi_mcp.py`.
- **Out of scope v1:** stdio, SSE.

## Operator record

Approved: operator @ 2026-11-11T11:24:00+01:00

**Selected variant:** Option A — streamable HTTP via FastMCP `>=3.3.1,<4` and `fastapi_mcp.py` only; stdio/SSE out of scope. ADR-MCP-03 remains **Proposed**.
