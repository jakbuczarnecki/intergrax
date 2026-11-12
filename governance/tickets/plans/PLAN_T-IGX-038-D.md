# PLAN T-IGX-038-D — MRC-4 MCP invocation correlation id

Parent: MCP-RC master plan (`agent-env/plans/pasted-content-id-703a-integrax-prancy-fiddle.md`), decisions `OD-MCP-RC-01`. Findings: R-P7-01 / REQ-MCP-OBS-01 (G-TR-01).

This batch changes product code under `intergrax/` and `applications/`. It is implemented on branch `mcp-rc` (worktree) and merged to `development` only together with the PDC-P7-00 ACT-N11 rebind (OD-MCP-P8-RESIDUAL-01, OD-MCP-RC-01 D-1/D-1b). The shared hotspot `intergrax/applications/_shared/` has a single writer: this campaign, serial order MRC-1, MRC-2, MRC-3, MRC-4.

## Deliverables

1. `mcp_invocation_id`: opaque per-call correlation key (uuid4 hex, prefixed `mcpi-`), generated once per tool call at the engine choke point, not derived from tenant or user, never an execution authority.
2. Returned additively in the `run_agent` payload and carried in task metadata so that it appears next to `task_id`, `run_id` and `execution_id` (ACC-17).

## Write set

`mcp_server_engine.py`, `mcp_nexus_server.py`, tests.

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator.

## Protocol

1. Tests first (red), then the fix; reuse the canonical mechanisms named in the master plan, section D. No second identity, execution, deadline, trace or registry authority.
2. One pytest process per verification step, several paths in one command (`intergrax-test-process-budget`).
3. Commit with `Ref: governance/tickets/T-IGX-038-D.md`, branch `mcp-rc`; merge to `development` only together with the PDC-P7-00 rebind (OD-MCP-RC-01 D-1/D-1b), after independent audit, certification and five green CI contexts on the pinned head.
4. The implementer never writes `Approved:`, never audits, certifies or signs; P8 records stay untouched.
