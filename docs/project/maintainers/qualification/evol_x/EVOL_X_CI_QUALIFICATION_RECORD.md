# EVOL-X — CI qualification record (C-CI-01/02)

**Status:** **EXACT BLOCKER — main workflow dispatch pending operator (OD-EVOL-10)**

| Field | Value |
|-------|-------|
| `workflow_def_sha_on_main` | *pending operator merge + dispatch* |
| `test_execution_sha` | *must equal TEST_EXECUTION_SHA @ W7 freeze (Batch D)* |
| `github_run_id` | — |
| `conclusion` | — |
| `artifact_name` | `evol-x-p3-l3-<TES>` |
| `junit_sha256` | — |
| `manifest_check` | — |
| `operator_authorization` | OD-EVOL-10 |

## Procedure (W6-04 exact-SHA)

1. Operator merges `evol-x-p3-l3-qualification` job definition to **`main`**; records **WORKFLOW_DEF_SHA**.
2. `workflow_dispatch` from **`main`** with input **`evol_test_execution_sha`** = 40-hex **TEST_EXECUTION_SHA** (never `github.sha`).
3. Runner checks out `development` history, then `git checkout "${TEST_EXECUTION_SHA}"` with `rev-parse` assertion — **test tree = TES**.
4. `uv sync --extra dev-ci --frozen`; `evol_x_vl3_manifest.py --check`; full `pytest tests/qualification/evol_x` → JUnit artifact.

**C-CI-02:** Qualification record **forbids** using `github.sha` or default-branch checkout alone as TEST_EXECUTION_SHA evidence; only dispatch input + step-2 assertion.

**Agent constraint:** no push to `main`; no dispatch without operator.

See also: [EVOL_X_CI_QUALIFICATION_PROCEDURE.md](./EVOL_X_CI_QUALIFICATION_PROCEDURE.md)
