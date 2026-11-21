# MCP Server — architecture (multi-layer feature)

**Status:** see the roadmap row `MCP-SERVER-X` (stage table and ledger), which decides stage status. At baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` it carries the label `CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED` (SCOPED, P0–P8; roadmap wording, quoted and not re-asserted here; not ARCH-FREEZE; no FRZ promotion). Closure record: [final closure report](../../maintainers/qualification/mcp_server_x/MCP_SERVER_X_FINAL_CLOSURE_REPORT.md). The earlier text of this line (P6 in progress, stage not CLOSED) is superseded.
**Pair:** [plan/MCP_SERVER.md](../plan/MCP_SERVER.md)  
**Stage:** `MCP-SERVER-X` — Platform MCP Server Exposure Engine Certification  
**Dossier SSOT:** [`docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_INDEX.md`](../../maintainers/qualification/mcp_server_x/MCP_SERVER_X_INDEX.md)

## Purpose

Single platform-owned MCP **exposure engine** over canonical Execution and Governance — typed server definition, exposure policy, streamable HTTP (FastMCP 3.x), harness identity, compatibility versioning. Applications keep thin facades; they do not own MCP infrastructure ([`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md) §3.2).

## Logical components

1. **Server Definition** — `intergrax/contracts/mcp_server_definition.py` (no FastMCP import)
2. **Capability Binding** — references existing `tool_id` or built-in host tools
3. **Exposure Policy** — allowlisted bindings; CONFIG-X-driven recomposition
4. **Composition** — `_shared` engine composing definitions per application
5. **Transport Adapter** — existing `fastapi_mcp.py` (streamable HTTP)

Note (EQO-A remediation draft, as of 2026-11-16): these components are implemented. At baseline the definition contract is `intergrax/contracts/mcp_server_definition.py`, the composition engine is `intergrax/applications/_shared/mcp_server_engine.py` (one of 17 `mcp_*.py` modules in `intergrax/applications/_shared/`), and the transport adapter is `intergrax/applications/_shared/fastapi_mcp.py`. This note does not map each component to one module. The ADR-MCP-01..07 files still carry `Proposed` in their own status field at baseline although OD-MCP-01..07 are recorded as APPROVED (Option A) in the Operator gate section below; the ADR status fields are outside this document.

**P1 SSOT:** [`MCP_SERVER_X_ARCHITECTURE_AND_OWNERSHIP.md`](../../maintainers/qualification/mcp_server_x/MCP_SERVER_X_ARCHITECTURE_AND_OWNERSHIP.md) · ADR-MCP-01..07 (**Proposed**) · [`MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md`](../../maintainers/qualification/mcp_server_x/MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md).

Detail: WBS P2–P4 in dossier; inventory @ [`MCP_SERVER_X_EXISTING_IMPLEMENTATION_INVENTORY.md`](../../maintainers/qualification/mcp_server_x/MCP_SERVER_X_EXISTING_IMPLEMENTATION_INVENTORY.md).

## Operator gate

OD-MCP-01..07 **APPROVED — Option A** @ 2026-11-11 — records under `operator_decisions/`. P3+ continues on published engine; see [`MCP_SERVER_X_WBS.md`](../../maintainers/qualification/mcp_server_x/MCP_SERVER_X_WBS.md).
