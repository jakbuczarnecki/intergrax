# MCP-SERVER-X P6 — ACT-N11 product Δ trigger

**Ticket:** `T-IGX-035-G`  
**PR:** #459  
**Candidate SHA:** `5ff8606b089828b334ebac133518ba47c9f6f7ce`  
**Prior IIAC `audit_execution_sha`:** `8fff85e3bfc5190929ff464b00fbe759dadcb5f8` (MCP ACT-N11 P4 rebind)

## Trigger

`verify_iiac_policy_activation.py --require-in-force` reports **ACT-N11** on PR #459:

| Path | Nature |
| --- | --- |
| `intergrax/applications/_shared/mcp_developer_api.py` | **New** product path |
| `intergrax/applications/_shared/mcp_application_server.py` | Modified (`execute_mcp_agent_task` export) |
| `applications/research_application/mcp/server.py` | Modified (import consolidation) |

CI job **`governance-fast`** fails on `tests/unit/governance/test_iiac_activation_gate.py` and related EVOL-X fresh-clone gates.

## Required governance (no bypass)

1. Independent **MCP ACT-N11 P6** product re-audit @ candidate SHA (separate session; not implementer).
2. Operator **IIIC/OD** re-bind of `audit_execution_sha` / evidence publication in `IIAC_POLICY_ACTIVATION_RECORD.json` (certifier session — implementer MUST NOT edit activation record).
3. Re-run mandatory CI on accepted candidate **before merge**.

## Out of scope

- Does not authorize MCP-SERVER-X stage CLOSED, FRZ promotion, or P7/P8 certification claims.
