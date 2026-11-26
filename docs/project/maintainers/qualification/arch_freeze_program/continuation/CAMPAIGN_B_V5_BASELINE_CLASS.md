# Campaign B V5 — baseline class (historical @ B)

**Classification:** **`PARTIAL_DIAGNOSTIC`** — do not claim a green historical baseline.  
**Execution SHA (baseline B):** `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (`AUDIT_BASELINE_2`)  
**Companion summaries:** [`CAMPAIGN_B_V5_ALPHA.md`](CAMPAIGN_B_V5_ALPHA.md) · [`CAMPAIGN_B_V5_DEFECT_MAP.md`](CAMPAIGN_B_V5_DEFECT_MAP.md)

**Purpose:** Lock baseline **class** outcomes for Campaign B manifest nodes @ B. **No product edit** and **no script back-fill** onto B in the commit that adds this file.

## NOT_EXECUTABLE_AT_BASELINE (scripts absent @ B)

Historical alpha must **not** copy later train-era scripts onto baseline **B**. These nodes remain **`NOT_EXECUTABLE_AT_BASELINE`** because the target path does not exist on the B tree:

| case_id | Missing path @ B |
| --- | --- |
| R01a | `scripts/governance/verify_rw_plan_phase_check3.py` |
| R02 | `tests/unit/governance/test_arch_freeze_plan_traceability.py` |
| R05 | `scripts/governance/lint_iiac_closure_regex.py` |
| R20 | `scripts/governance/verify_rw_plan_phase_check3.py` (PT-B02 phase-check slot) |

Do **not** copy governance scripts from a later commit into **B** for replay.

## `missing_extra` — R06, R10

| case_id | Symptom @ B | Classification |
| --- | --- | --- |
| R06 | Collection **exit 2** — `ModuleNotFoundError: fastmcp` (four MCP test modules) | **`missing_extra`** |
| R10 | Collection **exit 2** — `ModuleNotFoundError: fastmcp` (`test_mcp_catalog_f3_streamable_http_physical_l3.py`) | **`missing_extra`** |

Parent tree @ `7960ae7f6c82da60ca9acdc0ceebf63810002a42`: **`fastmcp`** is declared in **`pyproject.toml`** and **`uv.lock`** under extras **`dev-unit-cert`** and **`mcp`**. The alpha Campaign B venv did **not** install the extra; the failure is install surface, not lockfile absence.

A **later single rerun** of **only R06 and R10** (with correct extras installed) remains **outstanding** and is **not** performed in this commit.

## S03 — node PASS vs runner kill

| case_id | Node measurement | Runner exit |
| --- | --- | --- |
| S03 | **PASS** from stdout: **127 passed** (`tests/qualification/harness_01`) | **`4294967295`** — external **process kill**, not a failing assertion |

Do **not** treat the runner exit as qualification failure. This baseline-class file does **not** authorize an S03 rerun.

## PT-B01 / PT-B02 / SCOPE

This document is **planning and classification input only**. It does **not** close **PT-B01**, **PT-B02**, or **SCOPE**. Alpha remains **`PARTIAL_DIAGNOSTIC`** @ `7960ae7f6c82da60ca9acdc0ceebf63810002a42`.

## Explicit non-effects

- No product path changes from this document.
- Does **not** set **SCOPE**, **G2**, **FCS**, or **M3**.
- Does **not** back-fill missing B scripts from later commits.

**Recorded:** 2026-11-26 · ARCH-FREEZE V5 evidence-integrity pass (`af-v5/evidence-integrity`).