# T-IGX-038-A — MCP-RC MRC-1 identity and tenant admission (fail-closed MCP identity)

## Scope

MRC-1 identity and tenant admission (fail-closed MCP identity). Findings: R-P8-01 (G-ID-01), R-P7-06 (G-ID-06), G-ID-02, G-ID-03, G-ID-04, G-ID-07, G-ID-09; error codes MCP_PRINCIPAL_REQUIRED and MCP_TENANT_ASSERTION_CONFLICT (R-P7-05, part). Decisions: `OD-MCP-RC-01` (operator chat, recorded). Product code: yes; branch `mcp-rc`; no merge to `development` before the PDC-P7-00 rebind. Plan: `PLAN_T-IGX-038-A.md`.

## Plan digest

Plan digest: 94b40ae23a5be9f8e9e496dcc4cb5242e65cd3d063da5e53625c1881af60f843

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-038-A.md`](plans/PLAN_T-IGX-038-A.md).

## Approved

Approved: operator-delegate @ 2026-11-12T18:24:13+01:00
Delegation: docs/project/maintainers/qualification/mcp_server_x/operator_decisions/DELEGATION-MCP-RC-1.md@4ef1e09350c06018ec6beeb46eac645d49b1dc6f

## Test contract

```text
uv run pytest tests/unit/applications/test_mcp_trust_a_principal_binding.py tests/unit/applications/test_mcp_identity_mode_fail_closed.py tests/unit/applications/test_mcp_server_engine.py tests/unit/applications/test_mcp_canonical_execution.py tests/unit/applications/test_fastapi_mcp.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- P8 record edits, GitHub rulesets, FRZ changes, ACT-N11/IIAC logic, new MCP protocol capabilities.
- Independent audit, certification, signatures and merges (separate sessions).

## Lifecycle state

plan_approved
