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
| PASS | 5 |
| FAIL | 3 |
| `NOT_EXECUTABLE_AT_BASELINE` | 1 (`R20`) |
| Runner exit code | **1** |
| `coverage_classification` | **PARTIAL_DIAGNOSTIC** |

Evidence: `.tmp/session/w2-recovery-03/campaign-b/b2/HISTORICAL_ALPHA/7960ae7f6c82/af-v5-p2-cb/RESULTS_B2.json`

| case_id | result | Notes |
| --- | --- | --- |
| R20 | NOT_EXECUTABLE_AT_BASELINE | `verify_rw_plan_phase_check3.py` absent @ B |
| R21 | PASS | traceability verifier |
| R22 | PASS | scoped path net-zero |
| R23 | PASS | domain pairs |
| R24 | PASS | `pp_nt19.py` overlay (6 passed, 4 skipped deps) |
| R25 | FAIL | **assertion exit 1** — R25c `exit0=3/12` (9× CHANGED-SINCE unmapped `evol_x`; historical X-A6-003 pattern) |
| S01 | FAIL | **assertion exit 1** — contract→implementation edge on `environment_profile/root.py` |
| S02 | PASS | W5 typed observability gate |
| S03 | FAIL | **external abort exit 4294967295** (session kill after long stall); stdout captured **127 passed** — **not** accepted as alpha pass without exit 0 |

### S03 disposition

**`REMEASURE_REQUIRED`:** Runner row is **FAIL** (non-zero exit from external terminate). Stdout shows full suite green at capture time; that is **diagnostic signal only**, not qualification PASS. Does **not** authorize WF E2–E4. Class prior pip/setup **ERROR** pattern as `missing_extra` when remeasured with proper extras.