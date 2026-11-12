# PLAN T-IGX-038-A — MRC-1 identity and tenant admission (fail-closed MCP identity)

Parent: MCP-RC master plan (`agent-env/plans/pasted-content-id-703a-integrax-prancy-fiddle.md`), decisions `OD-MCP-RC-01`. Findings: R-P8-01 (G-ID-01), R-P7-06 (G-ID-06), G-ID-02, G-ID-03, G-ID-04, G-ID-07, G-ID-09; error codes MCP_PRINCIPAL_REQUIRED and MCP_TENANT_ASSERTION_CONFLICT (R-P7-05, part).

This batch changes product code under `intergrax/` and `applications/`. It is implemented on branch `mcp-rc` (worktree) and merged to `development` only together with the PDC-P7-00 ACT-N11 rebind (OD-MCP-P8-RESIDUAL-01, OD-MCP-RC-01 D-1/D-1b). The shared hotspot `intergrax/applications/_shared/` has a single writer: this campaign, serial order MRC-1, MRC-2, MRC-3, MRC-4.

## Deliverables

1. New typed `McpIdentityMode` (`HARNESS_BOUND` default, `LOCAL_DEV_EXPLICIT`) chosen **only** by an explicit composition parameter (code), never by an environment variable or configuration default; carried on `McpServerCompositionDeps` and by every facade builder as an additive keyword (default `HARNESS_BOUND`).
2. `resolve_mcp_run_agent_identity` fails closed in `HARNESS_BOUND`: no verified principal raises `MCP_PRINCIPAL_REQUIRED`; an asserted tenant or user that differs from the principal raises `MCP_TENANT_ASSERTION_CONFLICT`. Asserted values never create authority. `LOCAL_DEV_EXPLICIT` keeps the old behaviour only on that explicit opt-in, never for catalog invoke, and is guarded: it is admitted only for in-process calls (no HTTP request) or loopback clients (`127.0.0.1`, `::1`, `localhost`, test client); a non-loopback HTTP client is refused with `MCP_PRINCIPAL_REQUIRED`; under strict-production serving the existing strict caller-principal binding still raises (no principal is fabricated); a startup log warning is emitted when the mode is composed.
3. Principal gate for `list_agents`, `list_catalog_tools`, `describe_catalog_tool`, `invoke_catalog_tool` and the research pipeline tool in `HARNESS_BOUND`.
4. Coupling resolves harness auth at request time from the wrapper app, so factories that wire identity after coupling (lab, legal, research) are covered (no one-shot copy).
5. `run_research_pipeline` routes through the same identity helper and execution path as `run_agent`.
6. `invoke_catalog_tool` composes with the allowlist-filtered registry view.

## Write set

`intergrax/applications/_shared/mcp_identity_mode.py` (new), `mcp_harness_principal.py`, `mcp_server_engine.py`, `mcp_nexus_server.py`, `mcp_application_server.py`, `mcp_catalog_tools.py`, `mcp_catalog_invocation_bridge.py`, `fastapi_mcp.py`, `intergrax/contracts/mcp_server_definition.py` (error code wiring only), the eight `applications/*/mcp/server.py` facades (additive keyword), and tests under `tests/unit/applications/` and `tests/qualification/mcp_server_x/`.

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator.

## Protocol

1. Tests first (red), then the fix; reuse the canonical mechanisms named in the master plan, section D. No second identity, execution, deadline, trace or registry authority.
2. One pytest process per verification step, several paths in one command (`intergrax-test-process-budget`).
3. Commit with `Ref: governance/tickets/T-IGX-038-A.md`, branch `mcp-rc`; merge to `development` only together with the PDC-P7-00 rebind (OD-MCP-RC-01 D-1/D-1b), after independent audit, certification and five green CI contexts on the pinned head.
4. The implementer never writes `Approved:`, never audits, certifies or signs; P8 records stay untouched.
