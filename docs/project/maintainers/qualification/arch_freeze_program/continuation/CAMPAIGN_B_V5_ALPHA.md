# Campaign B V5 alpha — `HISTORICAL_ALPHA` @ baseline B

**Execution SHA (alpha context):** `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (`AUDIT_BASELINE_2`)  
**Baseline worktree:** `D:/Projekty/_wt/intergrax/af-v5-p2-b`  
**Runner worktree (manifest):** `D:/Projekty/_wt/intergrax/af-v5-p2-cb` @ branch `af-v5/p2-campaign-b`  
**Instance:** `af-v5-p2-cb`  
**Context:** `HISTORICAL_ALPHA` (diagnostic; does not close alpha, SCOPE, G2, or FCS)

Session copies: `.tmp/session/arch-freeze/campaign-b-v5/` (runner tees); canonical JSON under `.tmp/session/w2-recovery-03/campaign-b/`.

## PT-B01 (phase `b1`)

| Metric | Value |
| --- | ---: |
| Manifest rows executed | 19 |
| PASS | 8 |
| FAIL | 8 |
| `NOT_EXECUTABLE_AT_BASELINE` | 3 |
| Incomplete (FAIL + NEAB) | **11 / 19** |
| Runner exit code | **1** |
| `coverage_classification` | **PARTIAL_DIAGNOSTIC** |

Evidence: `.tmp/session/w2-recovery-03/campaign-b/b1/HISTORICAL_ALPHA/7960ae7f6c82/af-v5-p2-cb/RESULTS_B1.json`

### Disposition by case

| case_id | result | Cause class |
| --- | --- | --- |
| R00 | PASS | — |
| R01a | NOT_EXECUTABLE_AT_BASELINE | `scripts/governance/verify_rw_plan_phase_check3.py` absent @ B |
| R01b | FAIL | **assertion exit 1** — `check_agent_registry_bypass.py` reports Tier-2 agents importing `intergrax.integrations.tools` |
| R02 | NOT_EXECUTABLE_AT_BASELINE | `tests/unit/governance/test_arch_freeze_plan_traceability.py` absent @ B |
| R03 | PASS | — |
| R04 | FAIL | **assertion exit 1** — recovered `pp_ten.py` overlay; 7 probe failures (defects confirmed), 1 skip (`langchain_core` missing in cb venv) |
| R05 | NOT_EXECUTABLE_AT_BASELINE | `scripts/governance/lint_iiac_closure_regex.py` absent @ B |
| R06 | FAIL | **collection exit 2** — `ModuleNotFoundError: fastmcp` on four MCP test modules |
| R07 | FAIL | **assertion exit 1** — `test_txp6_q02_restart_resume_closed_world_parity` G-class `impact_map.py` |
| R08 | PASS | — |
| R09 | PASS | — |
| R10 | FAIL | **collection exit 2** — `ModuleNotFoundError: fastmcp` (`test_mcp_catalog_f3_streamable_http_physical_l3.py`) |
| R11 | PASS | — |
| R12 | PASS | — |
| R13 | PASS | — |
| R14 | PASS | — |
| R15 | FAIL | **assertion exit 1** — `test_replay_x_closure_counters_computed` counter mismatch |
| R16 | FAIL | **assertion exit 1** — `trace_x` suite: 8 failed nodes (closed-world / model-call surface gates) |
| R17 | FAIL | **assertion exit 1** — `render_ebh7_f10_normative_matrix.py --check` returned `STALE` |

**R04 note:** FAIL is a real alpha measurement of the vendored probe (not an undefined slot). Overlay path: `.tmp/campaign-b-overlay/probes/pp_ten.py` (gitignored under B).

## PT-B02 (phase `b2`)

| Metric | Value |
| --- | ---: |
| Manifest rows | 9 |
| PASS (runner) | 5 (`R21`, `R22`, `R23`, `R24`, `S02`) |
| FAIL (runner) | 3 (`R25`, `S01`, `S03` process exit) |
| `NOT_EXECUTABLE_AT_BASELINE` | 1 (`R20`) |
| Incomplete (FAIL + NEAB) | **4 / 9** |
| Runner exit code | **1** |
| `coverage_classification` | **PARTIAL_DIAGNOSTIC** |

Evidence: `.tmp/session/w2-recovery-03/campaign-b/b2/HISTORICAL_ALPHA/7960ae7f6c82/af-v5-p2-cb/RESULTS_B2.json`

| case_id | result | Notes |
| --- | --- | --- |
| R20 | NOT_EXECUTABLE_AT_BASELINE | `verify_rw_plan_phase_check3.py` absent @ B |
| R21 | PASS | traceability verifier |
| R22 | PASS | scoped path net-zero |
| R23 | PASS | domain pairs |
| R24 | PASS | Recovered **NT-19** probe (`pp_nt19.py` overlay @ B); 6 passed, 4 skipped (optional deps) |
| R25 | FAIL | **assertion exit 1** — `r25_impact12.py`: **9/12** lanes exit 1 (`CHANGED-SINCE fail-closed: unmapped evol_x paths`); **3/12** exit 0: **EVOL-X**, **ENT-AUDIT-X**, **MCP-SERVER-X** (historical X-A6-003; no impact-map edit) |
| S01 | FAIL | **assertion exit 1** — `test_whole_surface_forbids_contract_to_implementation_edges`: `intergrax.applications._shared.strict_host_dependency_concurrency_admission` imported from contract surface (`environment_profile/root.py`); historical diagnostic |
| S02 | PASS | W5 typed observability gate |
| S03 | **node: PASS** | **Node result (stdout):** `127 passed` in `S03/stdout.txt`. Runner exit **4294967295** = **process termination**, not a failing assertion. Does **not** authorize WF E2–E4 product edits. No S03 rerun in P2. |

### S03 disposition

Alpha cites **pytest stdout** (`127 passed`, `tests/qualification/harness_01`) as the node measurement. Runner JSON retains non-zero exit from external process termination only; that kill is **not** qualification evidence and **not** a substitute for a clean campaign runner exit.