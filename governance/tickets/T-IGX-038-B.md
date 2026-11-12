# T-IGX-038-B — MCP-RC MRC-2 result bounds and canonical MCP error contract

## Scope

MRC-2 result bounds and canonical MCP error contract. Findings: R-P7-03 (G-RB-01..04), G-ERR-03, R-P7-05 (MCP_RESULT_BOUND_EXCEEDED, G-ERR-01, G-ERR-02). Decisions: `OD-MCP-RC-01` (operator chat, recorded). Product code: yes; branch `mcp-rc`; no merge to `development` before the PDC-P7-00 rebind. Plan: `PLAN_T-IGX-038-B.md`.

## Plan digest

Plan digest: a68e5c97aab7641e4ee0bbcb66a607908b808bf87327af917882b0e58a20303a

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-038-B.md`](plans/PLAN_T-IGX-038-B.md).

## Approved

(pending: written only by the operator or by a commissioned delegate session; not by the implementer)

## Test contract

```text
uv run pytest tests/unit/applications/test_mcp_result_bounds_and_error_contract.py tests/qualification/mcp_server_x/test_mcp_server_x_p7_l1_contract.py tests/qualification/mcp_server_x/test_mcp_server_x_p7_failure_injection_l2.py tests/unit/applications/test_mcp_catalog_discovery_qualification.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- P8 record edits, GitHub rulesets, FRZ changes, ACT-N11/IIAC logic, new MCP protocol capabilities.
- Independent audit, certification, signatures and merges (separate sessions).

## Lifecycle state

plan_pending
