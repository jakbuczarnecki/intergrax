# ADR-MCP-03 — Transport: streamable HTTP via FastMCP 3.x

| Field | Value |
| --- | --- |
| **Document ID** | ADR-MCP-03 |
| **Status** | **Proposed** (pending OD-MCP-03) |
| **Related OD** | OD-MCP-03 |

## Context

Host code mounts FastMCP `http_app` only. `pyproject.toml` pins `fastmcp>=3.3.1,<4`. stdio and SSE are not wired to the harness host.

## Decision (proposed)

- **Supported transport:** streamable HTTP through existing `fastapi_mcp.py` adapter.
- **Unsupported v1:** stdio, SSE, custom sockets — **UNSUPPORTED_BY_DESIGN**.
- **Protocol revision:** whatever the installed FastMCP version announces at qualification SHA — recorded in contract/protocol matrix, not hard-coded as “full MCP”.

Negotiation behavior is delegated to the SDK; the platform does not ship a second negotiator.

## Alternatives

| Option | Outcome |
| --- | --- |
| A. Add stdio for local CLIs | **Deferred** — multiplies P3 matrix and CI |
| B. Single HTTP transport | **Recommended** |

## Consequences

- MCP-P3-02 L3 evidence binds to HTTP client on loopback.
- REQ-MCP-P-01 gap closes via matrix + test, not new transport code in P1.
