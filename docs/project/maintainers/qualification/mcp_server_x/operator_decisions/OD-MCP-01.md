# OD-MCP-01 — Public API placement (contracts + _shared engine)

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-01 |
| **Stage** | MCP-SERVER-X |
| **Status** | **PENDING** |
| **ADR** | [`ADR_MCP_01_PLATFORM_OWNERSHIP_AND_PUBLIC_API.md`](../../../architecture/ADR_MCP_01_PLATFORM_OWNERSHIP_AND_PUBLIC_API.md) |

## Question

Where does the platform MCP Server Engine public API live, and what may Tier-3 applications import?

## Recommended option (A)

- Contract types: `intergrax/contracts/mcp_server_definition.py` (no FastMCP).
- Composition: `intergrax/applications/_shared/mcp_server_engine.py` (name finalized in ADR).
- Applications: contract + facade builders only; FastMCP only at host adapter.

## Rejected options

- **B.** Per-application MCP engines in Tier-3 — violates architecture §3.2.
- **C.** `intergrax/runtime/mcp` as execution owner — second runtime authority.

## If approved — consequences

P2 may implement contract + engine modules. Imports from applications to FastMCP remain limited to host bootstrap paths.

## Operator record

_Decision status: **PENDING**. Agent MUST NOT write `Approved`._
