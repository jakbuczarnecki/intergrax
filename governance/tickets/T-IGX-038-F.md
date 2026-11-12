# T-IGX-038-F — MCP-RC MRC-6 protocol classification, compatibility record and corrective documentation

## Scope

MRC-6 protocol classification, compatibility record and corrective documentation. Findings: G-PR-01..04, G-DOC-01..12, v1.1 compatibility and migration note. Decisions: `OD-MCP-RC-01` (operator chat, recorded). Product code: yes; branch `mcp-rc`; no merge to `development` before the PDC-P7-00 rebind. Plan: `PLAN_T-IGX-038-F.md`.

## Plan digest

Plan digest: 4dcc7adc54b5e8a53de0d81ccbfa3c59d86dead13861894953a09d27535c8d4f

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-038-F.md`](plans/PLAN_T-IGX-038-F.md).

## Approved

(pending: written only by the operator or by a commissioned delegate session; not by the implementer)

## Test contract

```text
uv run pytest tests/qualification/mcp_server_x/test_mcp_server_x_p7_matrix_consistency.py tests/qualification/mcp_server_x/test_mcp_server_x_p7_l3_protocol_subset.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- P8 record edits, GitHub rulesets, FRZ changes, ACT-N11/IIAC logic, new MCP protocol capabilities.
- Independent audit, certification, signatures and merges (separate sessions).

## Lifecycle state

plan_pending
