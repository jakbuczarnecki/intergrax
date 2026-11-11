# MCP-SERVER-X P6 — application migration matrix

**Ticket:** `T-IGX-035-G` · **Baseline:** `e48106740a4a243204cadccb6a85db9a8710d6af`

Format: `APPLICATION → LEGACY API → CANONICAL API → REQUIRED CHANGE → TEST → RESULT`

| Application | Legacy API | Canonical API | Required change @ P6 | Test | Result |
| --- | --- | --- | --- | --- | --- |
| lab_application | Ad-hoc FastMCP decorators / direct nexus helpers | `compose_application_mcp_server` | None — P4 facade retained | `test_lab_application_exposes_mcp_mount` | **PASS** — no code change |
| poc_template_application | Same | `compose_application_mcp_server` | None | `test_reference_apps_mcp` (subset) | **PASS** |
| legal_application | Same | `compose_application_mcp_server` | None | import policy + mount tests | **PASS** |
| dispute_sim_application | Same | `compose_application_mcp_server` | None | import policy | **PASS** |
| governed_contractor_application | Same | `compose_application_mcp_server` | None | import policy | **PASS** |
| local_workspace_application | Same | `compose_application_mcp_server` | None | import policy | **PASS** |
| intergrax_assistant_application | Same | `compose_application_mcp_server` | None | import policy | **PASS** |
| research_application | `execute_mcp_agent_task` via `mcp_nexus_server` | `mcp_application_server` / `mcp_developer_api` | Import consolidation only | `test_research_application_exposes_mcp_mount` | **PASS** — behavior unchanged |

## Unsupported legacy behavior (explicit)

| Behavior | Disposition |
| --- | --- |
| Per-app FastMCP tool decorators bypassing definition contract | **Unsupported** — use bindings + `compose_application_mcp_server` |
| Direct `mcp_catalog_invocation_bridge` from Tier-3 | **Forbidden** — engine-internal |
| Resources / Prompts MCP surfaces | **UNSUPPORTED_BY_DESIGN** v1 (OD-MCP-02) |
| stdio / SSE transport | **UNSUPPORTED_BY_DESIGN** v1 (OD-MCP-03) |
| Strict PRODUCT research factory + vendor MemoryProvider without PROD-Q | **Blocked** — `LIM-RESEARCH-STRICT-FACTORY-MEMORY` (cross-stage) |
