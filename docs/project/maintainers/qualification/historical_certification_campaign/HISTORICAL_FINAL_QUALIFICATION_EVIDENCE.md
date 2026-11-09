# HIST Phase 6 — QUAL-X Dimension A replay evidence

| Field | Value |
|---|---|
| Label | **`PRE_ACTIVATION_DIAGNOSTIC`** (P5-VERIFY blocked; no new IIAC activation on tree) |
| Replay SHA | `427646f41de0df992cd54c0e157de8f36b560f39` (`origin/hist-cert/product-batch` tip) |
| Worktree | `D:/Projekty/_wt/intergrax/hist-qual-a` (detached) |
| Session | `hist-p6-qual-a` |
| Record date | 2026-11-09 |

## Selection authority

- Dimension A modules: `qual_x_p5_dimension_a_pytest_module_targets()` → **35 modules**
- Additional modules per master plan: **BND-03** gate (`test_bnd03_tier0_upward_edges_gate.py`), **W2-C** (`test_enterprise_scale_resilience_w2_c_retry_containment.py`)
- Baseline comparison: 630 passed @ `d9ea0d6e87b939e4cefb9971c6bbd4fbcaecc0ce` (historical); this run collected **595** tests in **37** module files (post–C1–C4 product delta)

## Command

```text
uv run --extra dev-ci pytest -p no:xdist <35 Dimension A module paths> \
  tests/unit/architecture/test_bnd03_tier0_upward_edges_gate.py \
  tests/unit/llm_adapters/test_enterprise_scale_resilience_w2_c_retry_containment.py
```

## Outcome

| Metric | Value |
|---|---|
| Passed | 595 |
| Failed | 0 |
| Skipped | 0 (unexplained) |
| Exit code | **0** |
| Duration | 304.13s |
| Process exit | Confirmed (`EXIT_CODE=0` appended to log) |
| Heavy lock | `D:/Projekty/intergrax/.tmp/session/hist-cert-campaign/locks/heavy.lock` |
| Log path | `D:/Projekty/_wt/intergrax/hist-qual-a/.tmp/session/hist-p6/p6_dimension_a.log` |
| Log SHA-256 | `d42cd8b9810f2141dd0e4ff5741ad5ef8dfd21cec7d798bef8fa8f7ee2c73a5f` |

## Limits

- Does **not** certify ENT Population E rollup.
- Does **not** satisfy P6 promotion prerequisite until P5-VERIFY is green on activated `development`.
- Does **not** close FRZ rows; feeds P7 evidence pointers only.
