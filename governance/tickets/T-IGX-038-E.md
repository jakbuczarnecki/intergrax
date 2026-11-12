# T-IGX-038-E — MCP-RC MRC-5 R-P7-02 test repair, behavioural tests and mutation qualification

## Scope

MRC-5 R-P7-02 test repair, behavioural tests and mutation qualification. Findings: R-P7-02 (G-TQ-01), F-P8-08 (G-TQ-02), G-TQ-03, G-TQ-04, G-TQ-05. Decisions: `OD-MCP-RC-01` (operator chat, recorded). Product code: yes; branch `mcp-rc`; no merge to `development` before the PDC-P7-00 rebind. Plan: `PLAN_T-IGX-038-E.md`.

## Plan digest

Plan digest: fab8bc465dc37a5dfa7d50d97e99c617f20161311fcbbde9211bd7514a05ca07

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-038-E.md`](plans/PLAN_T-IGX-038-E.md).

## Approved

Approved: operator-delegate @ 2026-11-12T18:21:30+01:00
Delegation: docs/project/maintainers/qualification/mcp_server_x/operator_decisions/DELEGATION-MCP-RC-1.md@4ef1e09350c06018ec6beeb46eac645d49b1dc6f

## Test contract

```text
uv run pytest tests/unit/applications/test_mcp_surface_opt_in.py tests/unit/applications/test_mcp_surface_opt_in_runtime.py tests/unit/applications/test_mcp_catalog_e2e_hung_server_qualification.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- P8 record edits, GitHub rulesets, FRZ changes, ACT-N11/IIAC logic, new MCP protocol capabilities.
- Independent audit, certification, signatures and merges (separate sessions).

## Lifecycle state

plan_approved
