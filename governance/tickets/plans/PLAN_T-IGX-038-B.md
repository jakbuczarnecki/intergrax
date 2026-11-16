# PLAN T-IGX-038-B — MRC-2 result bounds and canonical MCP error contract

Parent: MCP-RC master plan (`agent-env/plans/pasted-content-id-703a-integrax-prancy-fiddle.md`), decisions `OD-MCP-RC-01`. Findings: R-P7-03 (G-RB-01..04), G-ERR-03, R-P7-05 (MCP_RESULT_BOUND_EXCEEDED, G-ERR-01, G-ERR-02).

This batch changes product code under `intergrax/` and `applications/`. It is implemented on branch `mcp-rc` (worktree) and merged to `development` only together with the PDC-P7-00 ACT-N11 rebind (OD-MCP-P8-RESIDUAL-01, OD-MCP-RC-01 D-1/D-1b). The shared hotspot `intergrax/applications/_shared/` has a single writer: this campaign, serial order MRC-1, MRC-2, MRC-3, MRC-4.

## Deliverables

1. One bounded-result choke point: incremental bounded encoding in `intergrax/tools/mcp_result_bounds.py` (no full serialization before the check, UTF-8 byte accounting), applied to built-in tools, custom host tools, catalog tools and the cross-loop ingress serializer.
2. Overflow always surfaces as `ToolError("MCP_RESULT_BOUND_EXCEEDED: ...")` (`isError=true`) on every path; no `isError=false` overflow dicts.
3. A canonical wire-error mapping in `intergrax/contracts/mcp_server_definition.py`: every legacy lowercase code maps to a declared `MCP_*` code, the legacy string stays in the detail.
4. No `str(exc)` on the wire for internal errors; unknown-tool errors do not echo unbounded caller input.

## Write set

`intergrax/tools/mcp_result_bounds.py`, `intergrax/contracts/mcp_server_definition.py`, `mcp_server_engine.py`, `mcp_nexus_server.py`, `mcp_catalog_tools.py`, `mcp_catalog_invocation_bridge.py`, tests.

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator.

## Protocol

1. Tests first (red), then the fix; reuse the canonical mechanisms named in the master plan, section D. No second identity, execution, deadline, trace or registry authority.
2. One pytest process per verification step, several paths in one command (`intergrax-test-process-budget`).
3. Commit with `Ref: governance/tickets/T-IGX-038-B.md`, branch `mcp-rc`; merge to `development` only together with the PDC-P7-00 rebind (OD-MCP-RC-01 D-1/D-1b), after independent audit, certification and five green CI contexts on the pinned head.
4. The implementer never writes `Approved:`, never audits, certifies or signs; P8 records stay untouched.
