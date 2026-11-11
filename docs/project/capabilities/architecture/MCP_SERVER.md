# MCP Server — architecture (multi-layer feature)

**Status:** P0 stub — campaign not implemented  
**Pair:** [plan/MCP_SERVER.md](../plan/MCP_SERVER.md)  
**Stage:** `MCP-SERVER-X` — Platform MCP Server Exposure Engine Certification  
**Dossier SSOT:** [`docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_INDEX.md`](../../maintainers/qualification/mcp_server_x/MCP_SERVER_X_INDEX.md)

## Purpose

Single platform-owned MCP **exposure engine** over canonical Execution and Governance — typed server definition, exposure policy, streamable HTTP (FastMCP 3.x), harness identity, compatibility versioning. Applications keep thin facades; they do not own MCP infrastructure ([`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md) §3.2).

## Planned logical components

1. **Server Definition** — `intergrax/contracts/mcp_server_definition.py` (no FastMCP import)
2. **Capability Binding** — references existing `tool_id` or built-in host tools
3. **Exposure Policy** — allowlisted bindings; CONFIG-X-driven recomposition
4. **Composition** — `_shared` engine composing definitions per application
5. **Transport Adapter** — existing `fastapi_mcp.py` (streamable HTTP)

Detail: campaign master plan WBS P2–P4; inventory @ [`MCP_SERVER_X_EXISTING_IMPLEMENTATION_INVENTORY.md`](../../maintainers/qualification/mcp_server_x/MCP_SERVER_X_EXISTING_IMPLEMENTATION_INVENTORY.md).
