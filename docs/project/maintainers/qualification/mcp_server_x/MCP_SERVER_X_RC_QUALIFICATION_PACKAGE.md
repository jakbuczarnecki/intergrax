# MCP-SERVER-X RC — implementer's qualification package (exact SHA)

| Field | Value |
| --- | --- |
| **Status** | **IMPLEMENTER EVIDENCE ONLY. Not an audit, not a certification, not a verdict.** The independent audit (G6), certification and terminal verification are separate sessions and have not happened when this file is written |
| **BASE** | `6a6279a49a9aa5a33936a1c6a577c7194b4bcebb` (`origin/development` at campaign start; product delta to the IIAC pin `2d3eceb6c` was empty) |
| **IMPLEMENTATION_SHA = TEST_EXECUTION_SHA** | `93a089d3cca941592f1ad4f65435333c86da82e5` on branch `mcp-rc` (clean tree, 0 changes) |
| **Later commits** | documentation and evidence only (this file, `iiac_evidence/MCP-SERVER-X/rc_qualification/`); any product or test change after an accepted audit voids it and needs requalification |
| **Tickets** | `T-IGX-038-A..F` (`quality_gate`; plans approved by a delegate session under `DELEGATION-MCP-RC-1`) |
| **Decisions** | `OD-MCP-RC-01`, `OD-MCP-RC-02` |
| **Product change** | 23 files under `intergrax/` and `applications/` (+718 / −170); total campaign diff 67 files |
| **ACT-N11** | triggered on purpose; the IIAC activation is invalid for this branch until the independent delta audit and the PDC-P7-00 two-phase rebind. Nothing was merged or pushed to `development` |

## Mandatory test contract and result

Contract: `MCP_SERVER_X_RC_SELECTION` in `tests/qualification/mcp_server_x/rc_selection.py` (38 files: the P7 qualification and regression selections, the three R-P7-02 files, the four MRC suites and three impact-sweep suites). One pytest process; `PYTHONPATH` = the checkout under test (subprocess fixtures import the project from `PYTHONPATH`).

| Run | Result | Log |
| --- | --- | --- |
| Exact SHA `93a089d3c` | **464 passed, 1 xfailed, exit 0**, 386.6 s | `iiac_evidence/MCP-SERVER-X/rc_qualification/rc_contract_at_93a089d3c.log` |
| R-P7-02 baseline `6a6279a49` (3 files) | 11 failed, 18 passed, exit 1 (matches the recorded P7 baseline) | `baseline_r_p7_02_at_6a6279a49.log` |
| Mutation harness at `e75748d34` | 8 of 8 mutants killed | `mutation_report_at_e75748d34.json` / `.md` |

The single `xfailed` is `test_scaffold_defaults_mcp_to_opt_in[product]`: `strict=True`, **TRANSFERRED** to the scaffold/package-wiring owner (STRICT agent-contract authority is not resolved in product scaffold emission). It is an explicit, classified exclusion that needs the auditor's acceptance; it is not counted as a pass.

## What the contract covers (level)

| Level | Evidence |
| --- | --- |
| L1 contract | P7 L1 (enum of 15 codes, definition fingerprint unchanged), MRC unit suites |
| L2 composition | real facade builders (8), request-time auth resolution, identity modes, bounds, deadline and admission, invocation id |
| L3 physical | streamable HTTP on loopback for identity, composition order, cancel; subprocess TL11 hung/crash/slow-server tests (now armed correctly) |
| Mutation | eight invariants broken on a throw-away worktree, all detected |

## Known limits the auditor should weigh (see the corrections record)

- Findings marked `ACCEPTED_SCOPED` in `MCP_SERVER_X_RC_CORRECTIONS_RECORD.md`: non-cooperative synchronous catalog tool, detached-execution accounting, disconnect probe, advertised-empty capabilities, stdio/SSE absence, duplicate coupling entry points, G-ID-05 (shared harness module).
- `LOCAL_DEV_EXPLICIT` tasks carry `mcp_identity_assertion=unverified`; the admission guard is in-process or loopback only.
- Tests and fixtures that build in-process MCP servers without Harness Auth now opt in explicitly; the auditor should confirm that none of them hides a default-mode behaviour that should have been tested closed.
- The campaign was implemented by one session; no second implementer reviewed the code before this package.

## Non-claims

No independent audit, no certification, no FRZ promotion (counter stays 225), no production readiness, no ARCH-FREEZE closure, no statement that the historical P8 `PASS_SCOPED` covers these changes.
