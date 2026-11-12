# T-IGX-038-A — MCP-RC MRC-1 identity and tenant admission (fail-closed MCP identity)

## Scope

MRC-1 identity and tenant admission (fail-closed MCP identity). Findings: R-P8-01 (G-ID-01), R-P7-06 (G-ID-06), G-ID-02, G-ID-03, G-ID-04, G-ID-07, G-ID-09; error codes MCP_PRINCIPAL_REQUIRED and MCP_TENANT_ASSERTION_CONFLICT (R-P7-05, part). Decisions: `OD-MCP-RC-01` (operator chat, recorded). Product code: yes; branch `mcp-rc`; no merge to `development` before the PDC-P7-00 rebind. Plan: `PLAN_T-IGX-038-A.md`.

## Plan digest

Plan digest: 2acc9340f0529dd94eac42862e611f78aa0945c6a82c7bf5fe7f66e4e0937042

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-038-A.md`](plans/PLAN_T-IGX-038-A.md).

## Approved

(pending: written only by the operator or by a commissioned delegate session; not by the implementer)

## Test contract

```text
uv run pytest tests/unit/applications/test_mcp_trust_a_principal_binding.py tests/unit/applications/test_mcp_identity_mode_fail_closed.py tests/unit/applications/test_mcp_server_engine.py tests/unit/applications/test_mcp_canonical_execution.py tests/unit/applications/test_fastapi_mcp.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- P8 record edits, GitHub rulesets, FRZ changes, ACT-N11/IIAC logic, new MCP protocol capabilities.
- Independent audit, certification, signatures and merges (separate sessions).

## Lifecycle state

plan_pending
