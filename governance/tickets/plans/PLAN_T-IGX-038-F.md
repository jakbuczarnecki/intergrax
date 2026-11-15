# PLAN T-IGX-038-F — MRC-6 protocol classification, compatibility record and corrective documentation

Parent: MCP-RC master plan (`agent-env/plans/pasted-content-id-703a-integrax-prancy-fiddle.md`), decisions `OD-MCP-RC-01`. Findings: G-PR-01..04, G-DOC-01..12, v1.1 compatibility and migration note.

This batch changes product code under `intergrax/` and `applications/`. It is implemented on branch `mcp-rc` (worktree) and merged to `development` only together with the PDC-P7-00 ACT-N11 rebind (OD-MCP-P8-RESIDUAL-01, OD-MCP-RC-01 D-1/D-1b). The shared hotspot `intergrax/applications/_shared/` has a single writer: this campaign, serial order MRC-1, MRC-2, MRC-3, MRC-4.

## Deliverables

1. Capability snapshot test and classification (misleading advertised capability, documentation-only per D-6) and stdio/SSE negative test.
2. `MCP_SERVER_X_RC_*` records (corrections record, gap register supplement, contract matrix v1.1, migration note, ADR-MCP-RC-01); P8 records are not edited.

## Write set

`docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_RC_*.md`, `docs/project/maintainers/architecture/ADR_MCP_RC_01*.md`, tests.

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator.

## Protocol

1. Tests first (red), then the fix; reuse the canonical mechanisms named in the master plan, section D. No second identity, execution, deadline, trace or registry authority.
2. One pytest process per verification step, several paths in one command (`intergrax-test-process-budget`).
3. Commit with `Ref: governance/tickets/T-IGX-038-F.md`, branch `mcp-rc`; merge to `development` only together with the PDC-P7-00 rebind (OD-MCP-RC-01 D-1/D-1b), after independent audit, certification and five green CI contexts on the pinned head.
4. The implementer never writes `Approved:`, never audits, certifies or signs; P8 records stay untouched.
