# T-IGX-038-D — MCP-RC MRC-4 MCP invocation correlation id

## Scope

MRC-4 MCP invocation correlation id. Findings: R-P7-01 / REQ-MCP-OBS-01 (G-TR-01). Decisions: `OD-MCP-RC-01` (operator chat, recorded). Product code: yes; branch `mcp-rc`; no merge to `development` before the PDC-P7-00 rebind. Plan: `PLAN_T-IGX-038-D.md`.

## Plan digest

Plan digest: c09933e0caa44f58a4030138026b2ddc69f2417df83d04ab5b1f87bb02e80dd0

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-038-D.md`](plans/PLAN_T-IGX-038-D.md).

## Approved

Approved: operator-delegate @ 2026-11-12T18:21:30+01:00
Delegation: docs/project/maintainers/qualification/mcp_server_x/operator_decisions/DELEGATION-MCP-RC-1.md@4ef1e09350c06018ec6beeb46eac645d49b1dc6f

## Test contract

```text
uv run pytest tests/qualification/mcp_server_x/test_mcp_server_x_p7_l2_security_trace.py tests/unit/applications/test_mcp_invocation_id.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- P8 record edits, GitHub rulesets, FRZ changes, ACT-N11/IIAC logic, new MCP protocol capabilities.
- Independent audit, certification, signatures and merges (separate sessions).

## Lifecycle state

plan_approved
