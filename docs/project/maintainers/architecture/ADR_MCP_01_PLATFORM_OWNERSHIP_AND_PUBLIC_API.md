# ADR-MCP-01 — Platform ownership and public MCP Server API

| Field | Value |
| --- | --- |
| **Document ID** | ADR-MCP-01 |
| **Status** | **Proposed** (pending OD-MCP-01) |
| **Campaign** | MCP-SERVER-X P1 |
| **Related OD** | OD-MCP-01 |

## Context

Eight Tier-3 applications mount MCP via duplicated kwargs around `build_nexus_mcp_server`. [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md) §3.2 requires a **platform capability** for shared exposure — applications name servers; they do not own MCP infrastructure.

## Decision (proposed)

1. **Public contract** for server shape lives in `intergrax/contracts/` (`mcp_server_definition.py`, version `mcp-server-definition.v1`).
2. **Composition engine** lives in `intergrax/applications/_shared/` (proposed module `mcp_server_engine.py`).
3. Applications import the **contract** and thin builders only — not FastMCP — except at the host ASGI adapter boundary (`fastapi_mcp.py`).
4. Tier-0 `intergrax/` core MUST NOT gain a second execution runtime package (e.g. `intergrax/runtime/mcp` as execution owner).

## Server definition shape (P2 target — specification only @ P1)

- `server_id`, `display_name`, `contract_version`
- `bindings[]`: `tool_id` | built-in host tool (`list_agents`, `run_agent`, catalog triad)
- `exposure_policy`: enabled flag + allowlisted binding ids
- Optional: default tenant/user **defaults** (overridden by harness principal in auth profiles)

No FastMCP types in the contract module.

## Alternatives considered

| Option | Outcome |
| --- | --- |
| A. MCP engine in each `applications/*/mcp/` | **Rejected** — violates §3.2, duplicates transport/auth |
| B. New Tier-1 `intergrax/runtime/mcp` execution package | **Rejected** — implies second runtime authority |
| C. Keep status quo forever | **Rejected** — no exposure policy, version, or DX API |

## Consequences

- P2 implementer has a single contracts module and single composition owner for `_shared`.
- COMPAT-X inventories the new public surface after P4.
- EVOL change-impact required when signatures change.

## References

- [`MCP_SERVER_X_EXISTING_IMPLEMENTATION_INVENTORY.md`](../qualification/mcp_server_x/MCP_SERVER_X_EXISTING_IMPLEMENTATION_INVENTORY.md)
- [`ADR_TLX_TL11_MCP_CATALOG_CHILD_EXECUTION_OWNERSHIP.md`](ADR_TLX_TL11_MCP_CATALOG_CHILD_EXECUTION_OWNERSHIP.md)
