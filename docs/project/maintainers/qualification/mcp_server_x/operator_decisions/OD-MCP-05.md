# OD-MCP-05 — Contract version and legacy facades

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-05 |
| **Status** | **APPROVED — Option A** |
| **ADR** | [`ADR_MCP_05_COMPATIBILITY_FACADES_AND_V1.md`](../../../architecture/ADR_MCP_05_COMPATIBILITY_FACADES_AND_V1.md) |

## Question

How is the new public surface versioned and how are existing builders treated?

## Recommended option (A)

- Version `mcp-server-definition.v1` + schema fingerprint.
- Keep `build_nexus_mcp_server` and app `mcp/server.py` as compatibility facades until migration guide retires them.
- No silent removal of host/catalog tools.

## Operator record

Approved: operator @ 2026-11-11T11:24:00+01:00

**Selected variant:** Option A — `mcp-server-definition.v1` + compatibility facades (`build_nexus_mcp_server`, app `mcp/server.py`) over engine until migration guide retires them. ADR-MCP-05 remains **Proposed**.
