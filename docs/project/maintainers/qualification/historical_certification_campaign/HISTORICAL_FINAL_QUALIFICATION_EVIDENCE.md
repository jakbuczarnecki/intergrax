# HIST P6 Dimension A replay @ activated HEAD

| Field | Value |
|---|---|
| EXECUTION_SHA | `bf03550aa` (HEAD pending push; IIAC activated) |
| IIAC effective_from_sha | `8045915a353bb04ea5a290f9ef8b6def33dee96d` |
| Label | **POST_ACTIVATION** (P5-VERIFY green) |
| Command | `uv run pytest -p no:xdist` over 35 P5 Dimension A modules + `test_bnd03_tier0_upward_edges_gate.py` + `test_enterprise_scale_resilience_w2_c_retry_containment.py` |
| Result | **595 passed**, 0 failed |
| Exit code | **0** |
| Duration | 368.98s |
| Log | `HISTORICAL_P6_DIMENSION_A_REPLAY_2026-11-09.log` |
| Baseline reference | AUD-QUALX @ `d9ea0d6e8`: 630 passed (35+2 modules; manifest integrity nodes differ) |

Does not certify ENT Population E, ARCH-FREEZE, or campaign terminal closure.
