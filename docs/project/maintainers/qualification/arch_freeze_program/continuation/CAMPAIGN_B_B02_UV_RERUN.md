# Campaign B b2 — `uv` on PATH scaffold rerun (`af-fc/s03-env`)

**Base commit:** `66d0aef63af462535bac75ef377497b436844b8f`  
**Worktree:** `D:/Projekty/_wt/intergrax/af-fc-s03env` (branch `af-fc/s03-env`)  
**Product files changed:** none (PATH-only remediation; no tracked source edits)

## PATH fact

Prepended `%USERPROFILE%\.local\bin` to `PATH` so `uv` resolves for the test subprocess.

```text
uv 0.13.4 (7456b60bc 2026-10-20 x86_64-pc-windows-msvc)
```

## Scoped single-node rerun (prior S03 MISSING_EXECUTABLE)

**Command:** `uv run pytest tests/qualification/harness_01/test_harness_01_public_authoring_surfaces.py::test_harness_01_new_agent_scaffold_generates_nexus_free_authoring_package -v --tb=short`  
**Evidence log:** `D:/Projekty/_wt/intergrax/_scratch/af-fc-s03env/s03_scaffold_uv_on_path.txt`

```text
tests/qualification/harness_01/test_harness_01_public_authoring_surfaces.py::test_harness_01_new_agent_scaffold_generates_nexus_free_authoring_package PASSED [100%]
============================= 1 passed in 19.02s ==============================
exit 0
```

**Failure class (this rerun):** **PASS_WITH_UV_ON_PATH** — prior `FileNotFoundError` / WinError 2 on `uv` is cleared when `%USERPROFILE%\.local\bin` is on `PATH`; scaffold product assertions unchanged from prior pip-classified MISSING_EXECUTABLE record.

## Outer phase b2 runner

The full phase **b2** campaign (`run_campaign_b_w2_alpha.py`) was **not** rerun in this turn. Only the single scaffold node above was remeasured.

## SCOPE

No `SCOPE_MILESTONE` or ledger change. Observation-only continuation record.