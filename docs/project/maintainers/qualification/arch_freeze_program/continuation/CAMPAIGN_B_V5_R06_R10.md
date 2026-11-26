# Campaign B V5 — R06 / R10 environment recovery rerun

**Classification:** `PARTIAL_DIAGNOSTIC` (unchanged). A PASS here is **environment recovery** only — not a green historical baseline and not product remediation. Does not close alpha, SCOPE, G2, FCS, M3, PT-B01, or PT-B02.

## Execution context

| Field | Value |
| --- | --- |
| Historical baseline SHA | `7960ae7f6c82da60ca9acdc0ceebf63810002a42` |
| Baseline worktree (pytest cwd) | `D:/Projekty/_wt/intergrax/af-v5-p2-b` |
| Evidence worktree / branch | `D:/Projekty/_wt/intergrax/af-v5-b06` @ `af-v5/b-r06` |
| Isolated venv | `D:/Projekty/_wt/intergrax/_scratch/af-v5-b06/venv` |
| `uv sync` | From baseline worktree; `UV_LINK_MODE=copy`; `--frozen` |
| Extra installed | **`mcp`** (smallest extra declaring `fastmcp` @ baseline lockfile) |
| Prior alpha defect class | `missing_extra` — `ModuleNotFoundError: fastmcp` |

## Nodes executed (authorized subset only)

### R06 (AB-08) — MCP identity + MCP-SERVER-X P7 L2/L3 quartet

Manifest source: `scripts/qualification/campaign_b_w2_manifest.v1.json` (`case_id` R06).

| # | Path |
| ---: | --- |
| 1 | `tests/unit/applications/test_mcp_identity_mode_fail_closed.py` |
| 2 | `tests/unit/applications/test_mcp_rc_audit_corrections.py` |
| 3 | `tests/qualification/mcp_server_x/test_mcp_server_x_p7_l2_security_trace.py` |
| 4 | `tests/qualification/mcp_server_x/test_mcp_server_x_p7_l3_invoke_security.py` |

**Command:** `python -m pytest` (above four paths) `-q --tb=line`

| Metric | Value |
| --- | ---: |
| Pytest exit code | **0** |
| Passed | **92** |
| Failed | **0** |
| Collection | OK |

### R10 (AB-08) — F3 streamable HTTP physical L3 node

| # | Path |
| ---: | --- |
| 1 | `tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py` |

**Command:** `python -m pytest` (above path) `-q --tb=line`

| Metric | Value |
| --- | ---: |
| Pytest exit code | **0** |
| Passed | **5** |
| Failed | **0** |
| Collection | OK |

## Disposition

| case_id | Alpha result | Rerun result | Notes |
| --- | --- | --- | --- |
| R06 | FAIL (collection exit 2, `fastmcp`) | **PASS** (assertion exit 0) | Venv surface recovered with `mcp` extra |
| R10 | FAIL (collection exit 2, `fastmcp`) | **PASS** (assertion exit 0) | Same |

Remaining Campaign B V5 alpha FAIL / NEAB rows are **not** re-run in this commit.