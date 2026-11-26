# Campaign B V5 — R10 remainder (historical baseline)

**Classification:** `PARTIAL_DIAGNOSTIC` (unchanged). Measurement on the pinned historical baseline is **environment recovery evidence** only — not a green historical baseline and not product remediation. Does not close alpha, SCOPE, G2, FCS, M3, PT-B01, or PT-B02.

## Execution context

| Field | Value |
| --- | --- |
| Historical baseline SHA | `7960ae7f6c82da60ca9acdc0ceebf63810002a42` |
| Baseline worktree (pytest cwd) | `D:/Projekty/_wt/intergrax/af-v5-p2-b` |
| Evidence worktree / branch | `D:/Projekty/_wt/intergrax/af-v5-r10` @ `af-v5/r10-rest` |
| Interpreter | `D:/Projekty/_wt/intergrax/_scratch/af-v5-b06/venv/Scripts/python.exe` |
| `UV_PROJECT_ENVIRONMENT` | `D:/Projekty/_wt/intergrax/_scratch/af-v5-b06/venv` |
| `UV_LINK_MODE` | `copy` |
| Baseline commits | **None** (read-only measurement) |
| `uv sync` | **Not run** (per operator lock) |

## R10 manifest nodes (AB-08)

Manifest source: `scripts/qualification/campaign_b_w2_manifest.v1.json` (`case_id` R10).

| # | Path | Measured in this commit |
| ---: | --- | --- |
| 1 | `tests/unit/runtime/tools/test_tools_side_effect_safety.py` | **Yes** |
| 2 | `tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py` | **No** — prior evidence only |
| 3 | `tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py` | **Yes** |

### Prior node — `test_mcp_catalog_f3_streamable_http_physical_l3.py`

Recorded in [`CAMPAIGN_B_V5_R06_R10.md`](CAMPAIGN_B_V5_R06_R10.md) (same baseline SHA, recovered venv):

| Metric | Value |
| --- | ---: |
| Pytest exit code | **0** |
| Passed | **5** |
| Failed | **0** |

**Rule:** This file is **not** re-run here. R10 counts that node from the R06/R10 evidence row above.

### Remainder pair — single combined pytest (this commit)

Both paths present at baseline `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (not copied from a later commit).

**Command:** `python -m pytest` (paths 1 and 3 above) `-q --tb=line`

| Metric | Value |
| --- | ---: |
| Pytest exit code | **0** |
| Passed | **43** |
| Failed | **0** |
| Collection | OK |
| Wall time | 2.42s |

Per-path exit codes are not separable in one invocation; **both paths collected and executed** under the single exit code **0** above.

## R10 disposition rule

| Condition | R10 case | Campaign alpha classification |
| --- | --- | --- |
| `test_mcp_catalog_f3_streamable_http_physical_l3.py` already **5 passed, exit 0** (R06/R10 doc) **and** remainder pair **exit 0** | **PASS** (all three manifest paths) | Stays **`PARTIAL_DIAGNOSTIC`** — does not close PT-B01, PT-B02, or SCOPE; does **not** call the historical baseline green |
| Either remainder path fails collection or assertion | R10 **partial** | **`PARTIAL_DIAGNOSTIC`** |

**This measurement:** remainder pair exit **0**; with prior MCP catalog node exit **0**, **whole R10 is PASS** under the manifest. Campaign alpha remains **`PARTIAL_DIAGNOSTIC`** per table above.