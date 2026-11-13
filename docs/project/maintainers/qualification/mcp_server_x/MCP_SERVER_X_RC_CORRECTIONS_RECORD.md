# MCP-SERVER-X RC — corrections record (supplement)

| Field | Value |
| --- | --- |
| **Stage** | MCP-SERVER-X corrective campaign (MCP-RC); additive to the closed P7/P8 records, which are unchanged |
| **Status** | **IMPLEMENTED, NOT AUDITED, NOT CERTIFIED.** Every disposition below is the implementer's claim with evidence; none is an independent finding. `CLOSED_IN_CODE` means "fixed and covered by tests on the implementation SHA", not "closed" in the P8 sense |
| **Decisions** | `OD-MCP-RC-01` (D-1..D-8), `OD-MCP-RC-02`; delegation `DELEGATION-MCP-RC-1` (tickets `T-IGX-038-A..F`) |
| **Plan** | `agent-env/plans/pasted-content-id-703a-integrax-prancy-fiddle.md` (sections B, C, E, I) |
| **Branch** | `mcp-rc` (not merged; merge to `development` only together with the PDC-P7-00 rebind) |
| **Contract** | `MCP_SERVER_X_RC_CONTRACT_MATRIX.md`, `ADR_MCP_RC_01_FAIL_CLOSED_IDENTITY_AND_WIRE_V1_1.md` |
| **Test contract** | `tests/qualification/mcp_server_x/rc_selection.py` |

## Finding dispositions (implementer claims)

Disposition vocabulary: `CLOSED_IN_CODE` (fixed + tests), `ACCEPTED_SCOPED` (documented limit, owner named), `TRANSFERRED` (outside MCP, owner named). The independent audit decides.

| ID | Source | Disposition | Mechanism and evidence |
| --- | --- | --- | --- |
| G-ID-01 / R-P8-01 | F-P8-02 | CLOSED_IN_CODE | `McpIdentityMode.HARNESS_BOUND` default; `MCP_PRINCIPAL_REQUIRED`; tests `test_mcp_identity_mode_fail_closed.py`, `test_acc_12_default_mode_refuses_*` |
| G-ID-02, G-ID-09 | code review | CLOSED_IN_CODE | request-time resolver in `couple_fastapi_with_mcp`; `test_identity_wired_after_coupling_still_binds_and_conflicts` (real streamable HTTP) |
| G-ID-03 | code review | CLOSED_IN_CODE (behavioural) | five facades without harness auth now reject by default; every facade takes `identity_mode`; `test_every_facade_defaults_to_fail_closed_*` (8 facades). Wiring auth into those hosts remains a host decision (migration note) |
| G-ID-04 | code review | CLOSED_IN_CODE | `run_research_pipeline` uses `execute_mcp_agent_task_with_resolved_identity`; two research tests |
| G-ID-05 | code review | ACCEPTED_SCOPED / TRANSFERRED | `apply_harness_auth_middleware` silent no-op is a harness-host defect (shared security module, PDC-X B-D6 neighbourhood); MCP per-call gate fails closed regardless |
| G-ID-06 / R-P7-06 | R-P7-06 | CLOSED_IN_CODE (after audit AF-03) | meaning of `harness_auth=None` = request-time resolution; every tool incl. custom host tools sits behind the principal gate by default, with an explicit `public_custom_tool_names` opt-out; `test_mcp_rc_audit_corrections.py` |
| G-ID-07 | code review | CLOSED_IN_CODE | invoke composition uses the allowlist-filtered registry; `test_invoke_catalog_composition_receives_the_allowlist_filtered_registry` |
| G-ERR-01 / R-P7-05 | R-P7-05 | CLOSED_IN_CODE | the three declared codes are now raised by runtime paths (identity tests, bound tests); 8 codes added in v1.1 |
| G-ERR-02 | code review | CLOSED_IN_CODE | `LEGACY_WIRE_CODE_MAP` + `mcp_code_for_legacy`; bridge payloads carry `mcp_code` |
| G-ERR-03 | code review | CLOSED_IN_CODE | overflow always `isError=true`; `test_run_agent_answer_over_the_bound_is_is_error_true_*` |
| G-RB-01 / R-P7-03 | R-P7-03 | CLOSED_IN_CODE | custom tools bounded; `test_custom_host_tool_*`, `test_p7_05_*_applied_to_the_custom_tool_qual_echo` (the former "recorded gap" test is inverted) |
| G-RB-02 | code review | CLOSED_IN_CODE | cross-loop payload bounded at the MCP boundary (`bound_cross_loop_catalog_payload`); the runtime ingress is unchanged (host-owned) |
| G-RB-03 | code review | CLOSED_IN_CODE | incremental `mcp_serialized_size_exceeds`; `test_size_check_aborts_without_materializing_the_whole_value` |
| G-RB-04 | code review | CLOSED_IN_CODE (after audit AF-01, AF-10) | every tool body runs under `sanitized_tool_errors`: raw exception text never reaches the wire on `run_agent`, discovery, invoke and custom tools; unknown-tool echo capped at 128 chars; unserializable (circular/deep) results are typed |
| G-DL-01 / R-P7-04 | R-P7-04 | CLOSED_IN_CODE (after audit AF-02) | `max_call_seconds` governed-cancel trigger, `MCP_DEADLINE_EXCEEDED` with explicit outcome (`cancelled`, `failed`, `outcome_unknown`), never success; `OD-MCP-RC-02`; `test_mcp_deadline_and_resilience.py`, `test_mcp_rc_audit_corrections.py`. Effective wall time is `max_call_seconds` plus the bounded drain (30 s default), see "Audit follow-up" |
| G-DL-02 | code review | ACCEPTED_SCOPED | catalog invoke bounded (`invoke_timeout_s`, outcome unknown); a non-cooperative synchronous thread cannot be terminated in-process |
| G-DL-03 | code review | ACCEPTED_SCOPED | the cap bounds MCP handler occupancy; supervised detached executions are bounded by the transport-detachment supervisor; catalog invoke now passes admission |
| G-DL-04 | code review | ACCEPTED_SCOPED | disconnect probe failure is indistinguishable from "connected"; the `CancelledError` path still cancels; physical disconnect proof stays in the F3 L3 files |
| G-DL-05 | code review | CLOSED_IN_CODE | non-awaiting slot accounting; `test_admission_release_is_cancellation_safe`, burst test |
| G-DL-06 | code review | CLOSED_IN_CODE | `deps.transport_detachment_supervision` is wired into `run_agent`; `auto_wire_catalog_transport_supervision` documented as a reserved no-op |
| G-TR-01 / R-P7-01 | R-P7-01 | CLOSED_IN_CODE | `mcp_invocation_id`; `test_mcp_invocation_id.py`, ACC-17 |
| G-TQ-01 / R-P7-02 | R-P7-02 | see "R-P7-02 root causes" | |
| G-TQ-02 / F-P8-08 | F-P8-08 | CLOSED_IN_CODE | self-referential assertions rewritten (ACC-14, architecture gate, accept-either, broad `pytest.raises`), see below |
| G-TQ-03 | code review | CLOSED_IN_CODE | fail-open tests rewritten for the new contract (ACC-12 split into explicit local-dev and default-refuses) |
| G-TQ-04 | code review | CLOSED_IN_CODE | facade-level identity tests for all 8 builders and real-HTTP composition-order tests |
| G-TQ-05 | code review | see "Mutation qualification" | |
| G-PR-01 | code review | ACCEPTED_SCOPED (D-6) | misleading advertised-empty capabilities stay as FastMCP advertises them; pinned by `test_acc_18_announced_capabilities_snapshot` |
| G-PR-02 | code review | ACCEPTED_SCOPED | stdio/SSE absence is by missing entry point and probed over HTTP (`test_acc_18_only_streamable_http_transport_is_reachable`) |
| G-PR-03 | code review | ACCEPTED_SCOPED | the `exec()` wrapper keeps literal defaults only; unchanged, now always used so bounding applies uniformly |
| G-PR-04 | code review | ACCEPTED_SCOPED | duplicate coupling/composition entry points unchanged (consolidation would be an API change); all entry points share the one identity helper |
| G-DOC-01..12 | P8 records | see "Documentation defects" | |

## R-P7-02 root causes (reproduced, then repaired; no timeout inflation, no exclusion)

Baseline at `6a6279a49`: 11 failed, 18 passed (`.tmp/session/mcp-rc/baseline_r_p7_02.log`, matches the recorded P7 baseline).

| Test(s) | Root cause | Class | Repair |
| --- | --- | --- | --- |
| 7 `test_tl11_e2e_*` (hung/crash/reconnect/cleanup) | Fault injection armed at `ToolRegistry` reads fired during server **composition** (the engine enumerates the registry), so the subprocess hung or exited before printing its port | invalid fixture assumption | faults armed after composition |
| same 7 (second cause) | After arming correctly, faults still never fired: the engine serves the catalog from an allowlist-filtered **plain `ToolRegistry` copy**, so a registry subclass is bypassed | invalid fixture assumption (engine behaviour since P3) | faults installed on the `ToolRegistry` class methods in the subprocess |
| `test_mcp_surface_opt_in*` (4) | The LKW product vehicle needs an explicit LLM provider, the optional `ollama` SDK, a host tenant, strict memory admission evidence and strict agent-contract authority; none relates to MCP opt-in | environment-dependent + invalid fixture assumption | vehicle changed to the lab application with an explicit provider and a stubbed optional SDK; behaviour checks run in a clean interpreter |
| `test_scaffold_defaults_mcp_to_opt_in[product]` | Product scaffold emission fails STRICT agent-contract authority resolution in `package_wiring` (not MCP) | product defect outside MCP | **TRANSFERRED**: `xfail(strict=True)` with the reason; the `lab` profile case passes. Owner: scaffold/package-wiring (PDC-X/PROD-Q). Needs independent acceptance |
| also found | subprocess fixtures import `intergrax` from `PYTHONPATH`, not from the checkout that launched pytest, when another checkout is installed editable | environment hazard | the RC test contract requires `PYTHONPATH` = the checkout under test |

## Self-referential assertions rewritten (F-P8-08)

ACC-14 (no `inspect.getsource`; asserts the host deadline authority is untouched by composing with a call bound), architecture gate (AST call check instead of a tautology; plus a no-second-authority gate), in-memory deadline test (`outcomes in (["completed"], ["cancelled"])`), TL11 auth negatives (HTTP 401 instead of a broad `pytest.raises(Exception)`), source-text scans in the opt-in tests (AST and a clean-interpreter behaviour check).

## Mutation qualification

`scripts/qualification/mcp_rc_mutation.py` defines eight mutants (tenant admission bypass, principal mismatch ignored, bound missing on custom tools, deadline propagation dropped, deadline not requesting the governed cancel, slot release lost, error mapping swapped, forced local-dev path). Implementer run at `e75748d34`: **8 of 8 mutants killed** (`mutation_report.json` under `iiac_evidence/MCP-SERVER-X/rc_qualification/`). A ninth defect, found by the TL11 hung-server tests, was a real regression introduced while adding the identity gate: catalog discovery became `async` and a slow host registry then blocked the server event loop; it is fixed (reads run in a worker thread) and pinned by `test_slow_catalog_registry_read_does_not_block_other_calls`, which fails when the read is moved back onto the loop (manually verified). The auditor re-runs the harness; survivors are findings, not passes.

## Documentation defects (G-DOC-01..12) — corrected additively here, P8 files untouched

1. "same set" / "11" wording of R-P7-02 (`MCP_SERVER_X_P7_EVOL_ENT_DELTA_RECORD.md:30`, `p7_selection.py`, `MCP_SERVER_X_QUALIFICATION_MATRIX.md` row 19, `T-IGX-035-H`): correct statement is "core 11, observed 11-13; root causes above".
2. Consistency-file count: 13 in the P7 package (ACC-20 test deselected before the package was committed) versus 14 at audit and terminal verification.
3. `MCP_SERVER_X_SHA_LADDER.md` `FINAL_REMOTE_HEAD` row is malformed (extra column with P6 text); `FORMAL_CLOSURE_SHA` `8864cee84` is the PR #480 head while the merge is `c11184164`; the certified manifest still carries 40 zeros with `PENDING_PUBLICATION_MERGE`.
4. Governance lists in the closure report and ARCH-FREEZE handoff omit P8-PROC-03, P7-PROC-01/02, F-MCP-P5-GOV-002 and F-MCP-P5-MEM-001.
5. `OD-MCP-P8-RESIDUAL-01` records "Yes, both" and then a deferral of OPEN-BP-01; it is unsigned. The deferral text governs.
6. No operator ruling is recorded for closure conditions C5 and C8.
7. The gap register is mostly a P1 snapshot; `MCP_SERVER_X_INDEX.md` still says OD-MCP-01..07 PENDING; the auditor register still reads `RECORDED_PENDING_OPERATOR_DISPOSITION`.
These need an operator or roadmap-sync owner to correct the P8 originals; this record does not edit them.

## Audit follow-up (independent audit G6, verdict PARTIAL at `93a089d3c`; implementer's response)

The audit dossier is `MCP_SERVER_X_RC_INDEPENDENT_AUDIT_DOSSIER.md` (branch `mcp-rc-audit`). Product or test changes after the audit void it for the changed paths; a delta re-audit is required before certification.

| Finding | Response |
| --- | --- |
| AF-01 raw host exception text from `run_agent` | **Fixed in code**: `sanitized_tool_errors` around every tool body; tests for `run_agent`, `list_agents`, catalog list/describe |
| AF-02 no `MCP_DEADLINE_EXCEEDED` when the host ends cancelled or failed after the bound | **Fixed in code**: outcomes `cancelled` / `failed`; two parametrized tests |
| AF-03 custom host tools outside the principal gate | **Fixed in code** (product option chosen): default gate, explicit `public_custom_tool_names` opt-out; HTTP tests (refused without principal and handler never runs; allowed with a key and with the opt-out) |
| AF-04 five surviving auditor mutants (A3, A3b, A11, A12, A13) | **Tests added** (`test_mcp_rc_audit_corrections.py`); eight matching mutants M9-M16 added to the harness |
| AF-05 no real factory composition test; vehicle replaced | **ACCEPTED_SCOPED**: real product-factory composition (strict local-workspace app) stays unverified for MCP opt-in; restoration is owned by the strict-product test-support/scaffold owner together with the transferred `[product]` xfail. Not closed by this campaign |
| AF-06 wall time is `max_call_seconds` + drain (31.2 s for 1.0 s at the default 30 s drain) | **Documented**: the effective bound is `max_call_seconds` plus the host drain; the deadline tests shorten the drain to 0.25 s. Not changed (the drain is the existing transport constant) |
| AF-07 "only a governed-cancel trigger" is not literally true for LOCAL_DEV `run_agent` and catalog invoke | **Wording amended**: with a protected cancel context the bound requests the governed cancel; without one (LOCAL_DEV) and for catalog invoke it cancels the awaited call (`asyncio.wait_for`). It never creates a deadline authority, never reports success, and never changes the canonical run budget |
| AF-08 loopback guard trusts `request.client.host` | **Documented**: do not use `LOCAL_DEV_EXPLICIT` behind a same-host proxy; the guard is a development convenience, not an access control |
| AF-09 `execute_mcp_agent_task` is public with caller-asserted identity; the `unverified` label is not enforced anywhere | **Documented**: the label is evidence for audit and governance, not a gate; `execute_mcp_agent_task` is a low-level entry for hosts that already resolved identity; prefer `execute_mcp_agent_task_with_resolved_identity` |
| AF-10 deep/circular results give uncoded errors | **Fixed in code** with AF-01 |
| AF-11..AF-14 | **Accepted as declared** (reference server defaults to LOCAL_DEV in fixtures; factories expose no `identity_mode` by design; `max_call_seconds` is a composition parameter not yet threaded through facades; `initialize`/`tools/list` unauthenticated on API-key hosts is pre-existing protocol behaviour) |
| AF-15 / AF-16 | INFO, accepted; the `[product]` xfail owner/ticket must be named by the operator before certification |

## Delta re-audit follow-up (independent delta re-audit G6, verdict PASS_SCOPED at `b0e19bf61`; implementer's response)

The re-audit dossier is `MCP_SERVER_X_RC_INDEPENDENT_REAUDIT_DOSSIER.md` (branch `mcp-rc-reaudit`, commit `efa74c545`). It required no product change.

| Finding | Response |
| --- | --- |
| RA-01 no test pins the `sanitized_tool_errors` wrapper of `invoke_catalog_tool` (auditor mutant R8 survived) | **Closed by a test** (`test_invoke_catalog_tool_unexpected_exception_text_is_not_echoed`, commit `b98dc7719`, test and mutation harness only). Red on the mutant: the harness mutant `M18` is killed (`iiac_evidence/MCP-SERVER-X/rc_qualification/mutation_report_M18_at_b98dc7719.*`); green on the product code. Exact-SHA contract at `b98dc7719`: 481 passed, 1 xfailed, exit 0 (`rc_contract_at_b98dc7719.log`) |
| RA-02 a `ToolError` authored by host code keeps its host text | **Open, accepted as declared** (a host-authored `ToolError` is the host's own wire contract; no product change made) |
| RA-03 the Tier-3 facade `compose_application_mcp_server` has no `public_custom_tool_names` parameter | **Open, accepted as declared** (the opt-out is a composition parameter of the engine; no in-repo host needs it) |
| RA-04 pre-existing `TimeoutError` mapping in the `LOCAL_DEV` path gives the label `outcome=cancelled` | **Open, accepted as declared** (wrong label only; no success is ever claimed) |
| RA-05..RA-07 (INFO) | Accepted as declared |

## Deviations from the approved plan (implementer-disclosed)

- `LOCAL_DEV_EXPLICIT` guards: in-process or loopback only, no fabricated principal under strict-production serving (existing binding still raises), task label `unverified`, startup warning; no environment variable (delegate review of plan A).
- The MCP call bound is a transport trigger, not a deadline authority (`OD-MCP-RC-02`; delegate review of plan C).
- G-ID-05 is not changed in the shared harness module (ownership outside MCP).
