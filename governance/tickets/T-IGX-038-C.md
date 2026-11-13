# T-IGX-038-C — MCP-RC MRC-3 server-side deadline, cancellation, admission and resilience

## Scope

MRC-3 server-side deadline, cancellation, admission and resilience. Findings: R-P7-04 (G-DL-01, G-DL-02, G-DL-04, G-DL-06), G-DL-03, G-DL-05, F-P8-07. Decisions: `OD-MCP-RC-01` (operator chat, recorded). Product code: yes; branch `mcp-rc`; no merge to `development` before the PDC-P7-00 rebind. Plan: `PLAN_T-IGX-038-C.md`.

## Plan digest

Plan digest: fa0f444041e0e40a08d7a8ccbd77bc046dbc021b8d8fac399c818e0382f442c9

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-038-C.md`](plans/PLAN_T-IGX-038-C.md).

## Approved

Approved: operator-delegate @ 2026-11-13T06:31:46+01:00
Delegation: docs/project/maintainers/qualification/mcp_server_x/operator_decisions/DELEGATION-MCP-RC-1.md@4ef1e09350c06018ec6beeb46eac645d49b1dc6f

## Test contract

```text
uv run pytest tests/unit/applications/test_mcp_deadline_and_resilience.py tests/unit/applications/test_mcp_engine_concurrency.py tests/unit/applications/test_mcp_cancel_a_transport_governed_cancel.py tests/unit/applications/test_mcp_catalog_gate_b_cancel_c_qualification.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- P8 record edits, GitHub rulesets, FRZ changes, ACT-N11/IIAC logic, new MCP protocol capabilities.
- Independent audit, certification, signatures and merges (separate sessions).

## Lifecycle state

plan_approved
