# PLAN T-IGX-038-E — MRC-5 R-P7-02 test repair, behavioural tests and mutation qualification

Parent: MCP-RC master plan (`agent-env/plans/pasted-content-id-703a-integrax-prancy-fiddle.md`), decisions `OD-MCP-RC-01`. Findings: R-P7-02 (G-TQ-01), F-P8-08 (G-TQ-02), G-TQ-03, G-TQ-04, G-TQ-05.

This batch changes product code under `intergrax/` and `applications/`. It is implemented on branch `mcp-rc` (worktree) and merged to `development` only together with the PDC-P7-00 ACT-N11 rebind (OD-MCP-P8-RESIDUAL-01, OD-MCP-RC-01 D-1/D-1b). The shared hotspot `intergrax/applications/_shared/` has a single writer: this campaign, serial order MRC-1, MRC-2, MRC-3, MRC-4.

## Deliverables

1. Per-test root cause for the failing tests in the three R-P7-02 files, fixed in fixtures or product, never by timeout inflation, xfail or exclusion without recorded evidence.
2. Self-referential and source-string assertions on critical invariants replaced by behavioural tests; fail-open tests rewritten for the new contract.
3. `scripts/qualification/mcp_rc_mutation.py`: eight mutants applied to a throw-away copy, killed or survivors dispositioned.

## Write set

`tests/unit/applications/`, `tests/qualification/mcp_server_x/`, `tests/fixtures/tool_life_x_tl11_*`, `scripts/qualification/mcp_rc_mutation.py`; product only if a test exposes a defect (recorded).

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator.

## Protocol

1. Tests first (red), then the fix; reuse the canonical mechanisms named in the master plan, section D. No second identity, execution, deadline, trace or registry authority.
2. One pytest process per verification step, several paths in one command (`intergrax-test-process-budget`).
3. Commit with `Ref: governance/tickets/T-IGX-038-E.md`, branch `mcp-rc`; merge to `development` only together with the PDC-P7-00 rebind (OD-MCP-RC-01 D-1/D-1b), after independent audit, certification and five green CI contexts on the pinned head.
4. The implementer never writes `Approved:`, never audits, certifies or signs; P8 records stay untouched.
