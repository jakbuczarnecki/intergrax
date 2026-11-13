# MCP-SERVER-X RC — implementer's qualification package (exact SHA)

| Field | Value |
| --- | --- |
| **Status** | **IMPLEMENTER EVIDENCE ONLY. Not an audit, not a certification, not a verdict.** The independent audit (G6), certification and terminal verification are separate sessions and have not happened when this file is written |
| **BASE** | `6a6279a49a9aa5a33936a1c6a577c7194b4bcebb` (`origin/development` at campaign start; product delta to the IIAC pin `2d3eceb6c` was empty) |
| **Product SHA** | `b0e19bf61b346e6941b2e190b703c6922e708b34` (delta re-audit verdict PASS_SCOPED; `b98dc7719` changes tests and the mutation harness only) |
| **IMPLEMENTATION_SHA = TEST_EXECUTION_SHA (previous)** | `b0e19bf61b346e6941b2e190b703c6922e708b34` on branch `mcp-rc` (clean tree, 0 changes): the head after the corrective loop that answers the independent audit |
| **Audited SHA (superseded)** | `93a089d3cca941592f1ad4f65435333c86da82e5`: the independent audit (verdict PARTIAL, branch `mcp-rc-audit`, commit `4a459ed0b`) covers this SHA only. Product and test files changed after it, so the audit is **void for the changed paths** until a delta re-audit |
| **Later commits** | documentation and evidence only (this file, `iiac_evidence/MCP-SERVER-X/rc_qualification/`); any product or test change after an accepted audit voids it and needs requalification |
| **Delta since the audited SHA** | `git diff --stat 93a089d3c b0e19bf61 -- intergrax applications` (product): `mcp_wire_errors.py`, `mcp_server_engine.py`, `mcp_catalog_tools.py`, `mcp_host_execution_transport_cancel.py`; tests/fixtures/scripts: new `test_mcp_rc_audit_corrections.py`, fixture wait budgets, mutation harness (17 mutants), selection |
| **Tickets** | `T-IGX-038-A..F` (`quality_gate`; plans approved by a delegate session under `DELEGATION-MCP-RC-1`) |
| **Decisions** | `OD-MCP-RC-01`, `OD-MCP-RC-02` |
| **Product change** | 23 files under `intergrax/` and `applications/` (+718 / −170); total campaign diff 67 files |
| **ACT-N11** | triggered on purpose; the IIAC activation is invalid for this branch until the independent delta audit and the PDC-P7-00 two-phase rebind. Nothing was merged or pushed to `development` |

## Mandatory test contract and result

Contract: `MCP_SERVER_X_RC_SELECTION` in `tests/qualification/mcp_server_x/rc_selection.py` (38 files: the P7 qualification and regression selections, the three R-P7-02 files, the four MRC suites and three impact-sweep suites). One pytest process; `PYTHONPATH` = the checkout under test (subprocess fixtures import the project from `PYTHONPATH`).

| Run | Result | Log |
| --- | --- | --- |
| Exact SHA `b98dc7719` (current; adds the RA-01 test and mutant M18, no product change after `b0e19bf61`) | **481 passed, 1 xfailed, exit 0**, 691.7 s, 39 files, one pytest process | `iiac_evidence/MCP-SERVER-X/rc_qualification/rc_contract_at_b98dc7719.log` |
| Mutant M18 at `b98dc7719` | killed (the 17 earlier mutants were last run at `b0e19bf61`; the product files are unchanged since) | `mutation_report_M18_at_b98dc7719.json` / `.md` |
| Exact SHA `b0e19bf61` (previous) | **480 passed, 1 xfailed, exit 0**, 487.3 s, 39 files, one pytest process | `iiac_evidence/MCP-SERVER-X/rc_qualification/rc_contract_at_b0e19bf61.log` |
| Mutation harness at `b0e19bf61` | **17 of 17 mutants killed** (8 original + 8 from the audit findings + 1 found while fixing the deadline race) | `mutation_report_17_at_b0e19bf61.json` / `.md` |
| Previously audited SHA `93a089d3c` | 464 passed, 1 xfailed, exit 0 (the implementer's run; the auditor reproduced 464 passed / 1 xfailed) | `rc_contract_at_93a089d3c.log`; the auditor's log is on branch `mcp-rc-audit` |
| R-P7-02 baseline `6a6279a49` (3 files) | 11 failed, 18 passed, exit 1 (matches the recorded P7 baseline) | `baseline_r_p7_02_at_6a6279a49.log` |

Machine note: the workstation is shared with other sessions (20 to 60 python processes during these runs). Two intermediate runs at `0a6f07f8d` and `5d625bebc` failed under that load (12 TL11 tests by a 15 s subprocess start budget; one deadline race and one L3-R2 start budget); both were root-caused and fixed (see the corrections record) and are not hidden: the exact-SHA run above is the clean one.

The single `xfailed` is `test_scaffold_defaults_mcp_to_opt_in[product]`: `strict=True`, **TRANSFERRED** to the scaffold/package-wiring owner (STRICT agent-contract authority is not resolved in product scaffold emission). It is an explicit, classified exclusion that needs the auditor's acceptance; it is not counted as a pass.

## What the contract covers (level)

| Level | Evidence |
| --- | --- |
| L1 contract | P7 L1 (enum of 15 codes, definition fingerprint unchanged), MRC unit suites |
| L2 composition | real facade builders (8), request-time auth resolution, identity modes, bounds, deadline and admission, invocation id |
| L3 physical | streamable HTTP on loopback for identity, composition order, cancel; subprocess TL11 hung/crash/slow-server tests (now armed correctly) |
| Mutation | 17 invariants broken on a throw-away worktree, all detected |

## Known limits the auditor should weigh (see the corrections record)

- Findings marked `ACCEPTED_SCOPED` in `MCP_SERVER_X_RC_CORRECTIONS_RECORD.md`: non-cooperative synchronous catalog tool, detached-execution accounting, disconnect probe, advertised-empty capabilities, stdio/SSE absence, duplicate coupling entry points, G-ID-05 (shared harness module).
- `LOCAL_DEV_EXPLICIT` tasks carry `mcp_identity_assertion=unverified`; the admission guard is in-process or loopback only.
- Tests and fixtures that build in-process MCP servers without Harness Auth now opt in explicitly; the auditor should confirm that none of them hides a default-mode behaviour that should have been tested closed.
- The campaign was implemented by one session; the first independent audit (PARTIAL) led to a corrective loop whose changes have **not** been audited: AF-01/AF-10 (`sanitized_tool_errors`), AF-02 (deadline outcomes cancelled/failed), AF-03 (custom tools behind the principal gate, `public_custom_tool_names` opt-out), the `_bounded_observe_host_task` race fix, and the new tests.
- Open and accepted-scoped items after the audit are listed in the corrections record section "Audit follow-up" (AF-05 real-factory composition unverified, AF-06 wall time = bound + drain, AF-07 wording, AF-08 loopback guard behind proxies, AF-09 public low-level entry, AF-11..AF-14).

## Non-claims

No independent audit, no certification, no FRZ promotion (counter stays 225), no production readiness, no ARCH-FREEZE closure, no statement that the historical P8 `PASS_SCOPED` covers these changes.
