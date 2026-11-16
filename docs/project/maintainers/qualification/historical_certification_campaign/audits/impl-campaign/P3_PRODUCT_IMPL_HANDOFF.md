# P3 product batch — implementation handoff

| Field | Value |
|---|---|
| Status | **P3 complete** (resume slice 2026-11-09) |
| START_HEAD | `fe09375177bd0fa0d9ebb661b7f9bf64e2f1ed25` |
| INTEGRATION_SHA | `3a6ed857329177cc8e3b7669dbda53be1800c3df` |
| Branch | `hist-cert/product-batch` |
| Worktree | `D:/Projekty/_wt/intergrax/hist-int` |
| Plan approvals | `hist-cert/plan-approvals` @ `c8169ac1a` (`hist-sign`) |

## Lane SHAs (branch tips)

| Lane | Branch | SHA |
|---|---|---|
| C1 T-IGX-030-B | `hist-cert/c1-030b` | `088c9c5db` (test); product `31a6c3fe1` |
| C2 T-IGX-030-C | `hist-cert/c2-030c` | `30f67c94a` (product); test `a5ff540d8` |
| C3 T-IGX-032-A | `hist-cert/c3-032a` | `b6a056e7d` (product); test `ac8f96662` |
| C4 T-IGX-033-A | `hist-cert/c4-033a` | `8bfee643f` |

## Local verification (heavy lock)

Re-run on `INTEGRATION_SHA` (resume slice 2026-11-09):

| Lane | Command (focused) | Result | Exit |
|---|---|---|---|
| C1 | `test_runtime_error_code_contract_home.py` + `test_ebh_4_r1_…zero` | 5 passed | 0 |
| C2 | `test_tool_life_x_c0_no_boundary_submit_failure_terminal.py` | 6 passed | 0 |
| C3 | `test_skill_execution_pinning_wiring_store_precedence.py` | 9 passed | 0 |
| C4 gates | `test_bnd03_tier0_upward_edges_gate.py` + `test_ebh_5_rpl_g_regression_gates.py` | 18 passed | 0 |
| C4 CLI host | `test_cli_verification_hardening.py` | 5 passed, 3 failed, 2 errors | 1 |

**C4 CLI note (not a lane blocker):** `create_failed:canonical_governance_identity_required` in `offline_demo.py` reproduces on **clean `fe0937517`** at `hist-base` (same failure mix: 3 failed, 4 passed, 2 errors). Treat as pre-existing / out of 033-A defect scope; auditor must record PASS_SCOPED or environment caveat per ticket plan.

## P2 / development

Do **not** merge `hist-cert/product-batch` to `development` until IIAC consent (`PROPOSED_NO_CONSENT`).
