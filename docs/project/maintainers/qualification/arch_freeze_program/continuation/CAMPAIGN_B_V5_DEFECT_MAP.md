# Campaign B V5 — alpha defect map (historical @ baseline B)

**Alpha classification:** **`PARTIAL_DIAGNOSTIC`** — do **not** claim a green historical baseline.  
**Execution / baseline B:** `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (`AUDIT_BASELINE_2`)  
**Companion execution summary:** [`CAMPAIGN_B_V5_ALPHA.md`](CAMPAIGN_B_V5_ALPHA.md)

**Map purpose:** Classify manifest outcomes onto existing RW / qualification tracks for planning. **No product edit** is authorized in the commit that adds this map.

## NOT_EXECUTABLE_AT_BASELINE (`scripts absent @ B`)

Historical alpha must not back-fill later scripts into baseline **B**. These nodes are **`NOT_EXECUTABLE_AT_BASELINE`** because the target path does not exist on the B tree:

| case_id | Missing path @ B | Notes |
| --- | --- | --- |
| R01a | `scripts/governance/verify_rw_plan_phase_check3.py` | Do not copy from a later commit into B for replay |
| R02 | `tests/unit/governance/test_arch_freeze_plan_traceability.py` | Same rule |
| R05 | `scripts/governance/lint_iiac_closure_regex.py` | Same rule |
| R20 | Phase-check script (`verify_rw_plan_phase_check3.py`, b2 slot) | Same rule |

## Assertion and collection defects (map to RW tracks)

Real historical failures and diagnostics — routing only; remediation stays on existing RW / enterprise tracks:

| case_id | Outcome @ B | Defect summary | Suggested RW / track mapping |
| --- | --- | --- | --- |
| R01b | FAIL | Tier-2 agents import `intergrax.integrations.tools` (`check_agent_registry_bypass.py`) | Registry / seam discipline (PT-G01 family; Campaign B evidence for bypass) |
| R04 | FAIL | Recovered `pp_ten.py` overlay: **7** probe failures + **1** skip (`langchain_core` missing in cb venv) | TEN / optional-dep qualification |
| R07 | FAIL | `test_txp6_q02_restart_resume_closed_world_parity` — G-class `impact_map.py` closed-world | TXP / impact-map RW lane |
| R15 | FAIL | `test_replay_x_closure_counters_computed` — counter mismatch | Replay-X closure counters |
| R16 | FAIL | `trace_x` suite — **8** failed nodes (closed-world / model-call surface gates) | TRACE-X / closed-world |
| R17 | FAIL | `render_ebh7_f10_normative_matrix.py --check` returned **`STALE`** | EBH-7 matrix freshness |
| R25 | FAIL | `r25_impact12.py`: **9/12** lanes exit 1 (`CHANGED-SINCE fail-closed: unmapped evol_x paths`); **3/12** exit 0: **EVOL-X**, **ENT-AUDIT-X**, **MCP-SERVER-X** | EVOL-X impact-map (historical X-A6-003; no impact-map edit from this map) |
| S01 | FAIL | `test_whole_surface_forbids_contract_to_implementation_edges`: contract surface imports `intergrax.applications._shared.strict_host_dependency_concurrency_admission` (`environment_profile/root.py`) | Contract-surface / tier boundary |

## Environment class — `missing_extra` (eligible parent rerun later)

| case_id | Symptom @ B | Classification | This implementer action |
| --- | --- | --- | --- |
| R06 | Collection **exit 2** — `ModuleNotFoundError: fastmcp` (four MCP test modules) | **`missing_extra`** in alpha venv | **No rerun** |
| R10 | Collection **exit 2** — `ModuleNotFoundError: fastmcp` (`test_mcp_catalog_f3_streamable_http_physical_l3.py`) | **`missing_extra`** in alpha venv | **No rerun** |

Parent verification @ `7960ae7f6c82da60ca9acdc0ceebf63810002a42`: **`fastmcp`** is declared in `pyproject.toml` and `uv.lock` (extras **`dev-unit-cert`** and **`mcp`**). Alpha failure is venv/install surface, not absence from the lockfile. **One later parent rerun** of **only R06 and R10** may be authorized after installing the correct extras; this document does not authorize that rerun.

## S03 — node PASS vs runner kill

| case_id | Node measurement | Runner JSON |
| --- | --- | --- |
| S03 | **PASS** from stdout: **`127 passed`** (`tests/qualification/harness_01`) | Exit **`4294967295`** = external **process termination** (kill), **not** a failing assertion |

Do **not** treat the runner exit as qualification failure. Do **not** authorize an S03 rerun from this map.

## PT-B01 / PT-B02 / SCOPE

This defect map is **planning input only**. It does **not** close **PT-B01**, **PT-B02**, or **SCOPE**. Alpha remains **`PARTIAL_DIAGNOSTIC`** @ `7960ae7f6c82da60ca9acdc0ceebf63810002a42`.

## Explicit non-effects

- No product path changes from this document.
- Does **not** set **G2**, **FCS**, **M3**, **ENTRY**, or **ARCH-FREEZE CLOSED**.

**Recorded:** 2026-11-26 · Campaign B V5 defect map (`af-v5/residual-docs`).