# PLAN T-IGX-038-C — MRC-3 server-side deadline, cancellation, admission and resilience

Parent: MCP-RC master plan (`agent-env/plans/pasted-content-id-703a-integrax-prancy-fiddle.md`), decisions `OD-MCP-RC-01`. Findings: R-P7-04 (G-DL-01, G-DL-02, G-DL-04, G-DL-06), G-DL-03, G-DL-05, F-P8-07.

This batch changes product code under `intergrax/` and `applications/`. It is implemented on branch `mcp-rc` (worktree) and merged to `development` only together with the PDC-P7-00 ACT-N11 rebind (OD-MCP-P8-RESIDUAL-01, OD-MCP-RC-01 D-1/D-1b). The shared hotspot `intergrax/applications/_shared/` has a single writer: this campaign, serial order MRC-1, MRC-2, MRC-3, MRC-4.

## Deliverables

1. Governing decision: `OD-MCP-RC-02` (operator chat 2026-11-13) records that the MCP call bound is a transport observer trigger, not a deadline authority. Canonical deadline authority (W1 finding, named): the root Execution deadline is owned by `ExecutionDeadlineAuthorityResolver` (`intergrax/runtime/execution/deadline_authority/resolver.py`), composed by the host through the run budget (`max_wall_time_seconds`) and read in `intergrax/runtime/execution/runtime.py`. MCP does not create, configure or override it, and no MCP-local deadline lifecycle is added. The MCP transport adds only an observer trigger, `max_call_seconds` (optional, default none), exactly like client disconnect: on expiry it calls the existing `governed_cancel_active_task` path of `run_host_execution_with_mcp_transport_cancel` (cooperative cancel owned by `CancellationCoordinator`), applies the bounded drain and the existing supervision handoff, then returns `MCP_DEADLINE_EXCEEDED` with an explicit outcome-unknown detail; never reported as success. A host that wants a hard run wall time sets the canonical run budget.
2. Catalog invoke gets the same bound and a configurable cross-loop wait; a timeout returns `MCP_DEADLINE_EXCEEDED` (outcome unknown), documented limit: a non-cooperative synchronous thread cannot be terminated in-process.
3. Admission: cancellation-safe slot accounting (no awaited lock in release), catalog invoke passes through admission, over-limit raises the declared `MCP_CONCURRENT_CALL_LIMIT_EXCEEDED`.
4. The disconnect probe no longer treats a probe failure as connected forever; dead `deps` assignments are wired (supervision) or documented as no-ops.

## Write set

`mcp_host_execution_transport_cancel.py`, `mcp_engine_concurrency.py`, `mcp_server_engine.py`, `mcp_nexus_server.py`, `mcp_catalog_invocation_bridge.py`, `intergrax/contracts/mcp_server_definition.py` (new codes), tests.

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator.

## Protocol

1. Tests first (red), then the fix; reuse the canonical mechanisms named in the master plan, section D. No second identity, execution, deadline, trace or registry authority.
2. One pytest process per verification step, several paths in one command (`intergrax-test-process-budget`).
3. Commit with `Ref: governance/tickets/T-IGX-038-C.md`, branch `mcp-rc`; merge to `development` only together with the PDC-P7-00 rebind (OD-MCP-RC-01 D-1/D-1b), after independent audit, certification and five green CI contexts on the pinned head.
4. The implementer never writes `Approved:`, never audits, certifies or signs; P8 records stay untouched.
