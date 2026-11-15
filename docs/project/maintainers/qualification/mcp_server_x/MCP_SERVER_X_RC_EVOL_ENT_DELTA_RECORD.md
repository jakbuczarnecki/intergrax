# MCP-SERVER-X RC — EVOL change-impact and ENT delta record (MRC-7.1, implementer's input to the audit)

| Field | Value |
| --- | --- |
| **Tickets** | `T-IGX-038-A..F` |
| **Base** | `origin/development` `6a6279a49a9aa5a33936a1c6a577c7194b4bcebb` (product delta to the IIAC pin `2d3eceb6c`: empty before this campaign) |
| **Status** | Implementer's declaration. EVOL/ENT results are produced by the CI contexts on the **pinned PR head** and by the independent auditor; nothing here is a verdict |

## Change class

**Product change under `intergrax/` and `applications/`** (23 files). This is a deliberate ACT-N11 trigger: the active IIAC activation (`audit_execution_sha` `2d3eceb6c`) is invalid for this change until an independent delta audit and a two-phase rebind (`OD-MCP-P8-RESIDUAL-01`, `OD-MCP-RC-01` D-1: carried by PDC-P7-00). Merge to `development` before that rebind turns `governance-fast` red; the plan forbids it.

| Area | Files |
| --- | --- |
| New modules | `intergrax/applications/_shared/mcp_identity_mode.py`, `mcp_wire_errors.py`, `mcp_invocation_id.py` |
| Changed engine and bridges | `mcp_server_engine.py`, `mcp_nexus_server.py`, `mcp_application_server.py`, `mcp_harness_principal.py`, `mcp_catalog_tools.py`, `mcp_catalog_invocation_bridge.py`, `mcp_host_execution_transport_cancel.py`, `mcp_engine_concurrency.py`, `fastapi_mcp.py`, `mcp_developer_api.py` |
| Contract and tools | `intergrax/contracts/mcp_server_definition.py` (8 added error codes), `intergrax/tools/mcp_result_bounds.py` (incremental size check) |
| Facades (additive keyword) | the eight `applications/*/mcp/server.py` |
| Tests, fixtures, scripts | MRC-1..5 suites, `rc_selection.py`, `scripts/qualification/mcp_rc_mutation.py` (not product paths) |
| Records | OD-MCP-RC-01/02, DELEGATION-MCP-RC-1, ADR-MCP-RC-01, `MCP_SERVER_X_RC_*` |

## EVOL change-impact

| Question | Answer |
| --- | --- |
| New or changed public contract or API | Yes: `McpIdentityMode`, the `identity_mode` and `max_call_seconds` keywords (additive), `execute_mcp_agent_task_with_resolved_identity`, 8 error codes; **behaviour change** of the default (fail-closed identity, overflow `isError=true`, declared wire codes). Versioned as wire contract v1.1 (`OD-MCP-RC-01` D-4). `mcp-server-definition.v1` and its fingerprint are unchanged (pinned by the P7 L1 tests, which pass). |
| New or changed execution/governance seam | No new seam. Identity uses the existing Harness Auth principal resolution; cancel uses the existing `governed_cancel_active_task`; the canonical deadline stays with the host (`OD-MCP-RC-02`). The architecture gate test forbids authenticator or deadline-authority definitions in the engine. |
| Layer boundaries | No new upward import (`check_no_upward_application_imports.py` lists only pre-existing runtime violations, none in MCP files). |
| COMPAT inventory | Public MCP API surface test (`test_mcp_public_api_surface.py`) passes; `McpIdentityMode` and the helper are exported through `mcp_developer_api`. |
| FRZ criteria | No status change, no new family, counter unchanged at 225; no FRZ document is edited. Classification of the findings is the operator-approved D-7 (plan section C). |
| Other domains touched | None. `intergrax/runtime/`, `intergrax/runtime/governance/` and harness auth are read, not edited; the cross-loop ingress is bounded at the MCP boundary without a runtime edit. |

## Regression scope

`tests/qualification/mcp_server_x/rc_selection.py` (`MCP_SERVER_X_RC_SELECTION`): the P7 qualification and regression selections, the three R-P7-02 files and the four MRC suites. Run with one pytest process and `PYTHONPATH` set to the checkout under test.

Implementer's run on the implementation head: see `MCP_SERVER_X_RC_QUALIFICATION_PACKAGE.md` (exact SHA, counts, logs). The independent auditor re-runs it.

## ENT delta

- ENT-AUDIT-X FAST contexts (`ent-audit-fast`, `ent-audit-pr-gate`) and `evol-x-b4-preflight` run in CI on the pinned PR head; their results are taken from CI, not asserted here.
- The CLOSED stages (ENT-AUDIT-X, EVOL-X and earlier) are not invalidated by this change: no baseline, validator or freeze document is edited.
- IIAC: `verify_iiac_policy_activation --require-in-force` is expected to **fail** on any branch that contains the product change until the rebind; this is the intended ACT-N11 behaviour, not a defect.

## Non-claims

No unscoped PASS, no FRZ promotion, no production-readiness claim, no ARCH-FREEZE claim, no claim that the historical P8 `PASS_SCOPED` covers these changes. Independent audit, certification and terminal verification are separate sessions.
