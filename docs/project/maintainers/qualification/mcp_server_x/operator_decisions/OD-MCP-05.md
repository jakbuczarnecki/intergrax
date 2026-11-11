# OD-MCP-05 — Contract version and legacy facades

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-05 |
| **Status** | **PENDING** |
| **ADR** | [`ADR_MCP_05_COMPATIBILITY_FACADES_AND_V1.md`](../../../architecture/ADR_MCP_05_COMPATIBILITY_FACADES_AND_V1.md) |

## Question

How is the new public surface versioned and how are existing builders treated?

## Recommended option (A)

- Version `mcp-server-definition.v1` + schema fingerprint.
- Keep `build_nexus_mcp_server` and app `mcp/server.py` as compatibility facades until migration guide retires them.
- No silent removal of host/catalog tools.

## Operator record

_Decision status: **PENDING**. Agent MUST NOT write `Approved`._
