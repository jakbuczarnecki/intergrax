# MCP-SERVER-X P7 — EVOL change-impact and ENT delta record (MCP-P7-07)

**Ticket:** `governance/tickets/T-IGX-035-H.md` · **TEST_EXECUTION_SHA:** `abbe1525450da77da6801b080619a600c20cd497` · **Base:** `origin/development` `d5c1383fc15c005daafd131eaf03b3c49393b792`

## Change class

Tests and documentation only. The P7 change set adds MCP qualification tests and records and corrects two documentation findings. It changes **no** path under `intergrax/`, `agents/` or `applications/` (checked with `git diff --name-only origin/development HEAD`), no validator, no activation record and no auditor artifact.

| Area | Files |
| --- | --- |
| New qualification tests | `tests/qualification/mcp_server_x/test_mcp_server_x_p7_*.py` (9 files), `p7_selection.py` |
| Test fixtures (additive) | `tests/qualification/mcp_server_x/reference_app/build.py` (optional kwargs, defaults unchanged), `http_serve.py`, `variants.py` |
| Records | `MCP_SERVER_X_P7_QUALIFICATION_PACKAGE.md`, this record, `iiac_evidence/MCP-SERVER-X/p7_qualification/`, matrix/ladder/gap/WBS/index/handoff updates, ticket `T-IGX-035-H` |
| Documentation corrected | `MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md` (D-P7-01, D-P7-02) |

## EVOL change-impact

| Question | Answer |
| --- | --- |
| New or changed public contract or API | None. The public surface (`McpServerDefinition`, `mcp_developer_api`) is exactly as qualified in P6; P7 only exercises it. |
| New or changed execution/governance seam | None. The tests drive the existing seams (principal binding, governed cancel, side-effect port, admission). |
| COMPAT inventory / schema fingerprint | Unchanged; the fingerprint and version constant are pinned by the P7 L1 tests (golden value for this tree). |
| FRZ criteria | No status change, no new family, counter unchanged. |
| TOOL-LIFE tests touched by the diff | None (no product or execution path in the diff). The MCP cancel, catalog and F3 regression files run as part of the regression selection. |

## Regression scope (selected by this record)

`MCP_SERVER_X_P7_REGRESSION_SELECTION` in `tests/qualification/mcp_server_x/p7_selection.py`: the 12 files of the P5 selection plus `test_mcp_public_api_surface.py`, `test_fastapi_mcp.py`, `test_mcp_server_engine_architecture_gates.py`, `test_mcp_catalog_cross_loop_ingress_qualification.py`, `test_mcp_catalog_f3_streamable_http_physical_l3.py`, `test_mcp_catalog_f3_cross_loop_cancel_c_physical_l3.py` and `test_mcp_server_x_p6_reference_l3_qualification.py`. Result: 141 passed at the `TEST_EXECUTION_SHA` and in the clean replay.

Not in scope and recorded as pre-existing failures (R-P7-02, same set at `c59892829`, `d5c1383fc` and the P7 head): `test_mcp_surface_opt_in.py`, `test_mcp_surface_opt_in_runtime.py`, `test_mcp_catalog_e2e_hung_server_qualification.py`. Evidence: `iiac_evidence/MCP-SERVER-X/p7_qualification/baseline_p7-base-*.log`.

## ENT delta (Mode A at this change)

- The change set contains no product path, so no ENT invariant input changes. The ENT-AUDIT-X FAST non-certifying audit runs in CI on the P7 publication head (`ent-audit-fast`, `ent-audit-pr-gate`); its result is taken from the check runs of that pull request and is **not** claimed here before it exists.
- The CLOSED stages (`ENT-AUDIT-X`, `EVOL-X` and earlier) are not invalidated: no baseline, validator or freeze document is changed by P7.
- IIAC activation: `verify_iiac_policy_activation --require-in-force` exits 0 on the P7 head; no ACT-N11 product path changed after the audit pin `2d3eceb6cf7e93dccd4d96919319a6c969b4444b`.

## Non-claims

No MCP-SERVER-X stage closure, no FRZ promotion, no P8 verdict. P8 audit and certification are separate sessions (ticket `T-IGX-035-I`).
