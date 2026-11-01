# EVOL-X — CI qualification record (C-CI-01/02)

**Status:** **QUALIFIED (W6-04 exact-SHA CI run, conclusion success)** — scope: EVOL-X L3 CI dispatch only; **no FRZ-* promotion**.

| Field | Value |
|-------|-------|
| `workflow_def_sha_on_main` | blob `1eb6d2b9192a3e1b12db8daa2b2e391794394815` (`origin/main:.github/workflows/unit-tests.yml`); main commit `a00674d6ba6af9b5676c0341823cd23b162bbb99` (PR #380 merge, `evol-x-w6-04-workflow-only`, mergedAt 2026-11-01T15:16:41Z); job added in `890519cc1da3c05a8aa6eeed8304552ba0b44a86` |
| `test_execution_sha` | **`2b165516485154583a4118ba1ecb2e25549bb5d0`** (dispatch input `evol_test_execution_sha`; E-CORR-M09H-ORACLE; supersedes `dcd7ae6e…`) |
| `tech_sha` (active TECH publication) | **`1e175ff8f678d605aa841ae52362486a56ed006d`** (Mode A `--technical`; P03 bundle `434a31146af53d0e776b6c93b97ae7f2444b1f18`; M-09H NOT_MEASURED) |
| `workflow` / `profile` | `Regression gate` (id 250851568) / `evol_x_p3_l3`, job `evol-x-p3-l3-qualification`, ref `main` |
| `github_run_id` | **`41327408027`** |
| `run_url` | https://github.com/jakbuczarnecki/intergrax/actions/runs/41327408027 |
| `run_head_sha` (workflow def, not test tree) | `a00674d6ba6af9b5676c0341823cd23b162bbb99` |
| `conclusion` | **success** (job completed 2026-11-01T16:45:42Z) |
| `checkout_assertion` | step `Checkout TEST_EXECUTION_SHA` passed (`git rev-parse HEAD` == TES) |
| `tests` | **`288 passed in 365.96s`** (`tests/qualification/evol_x`); 0 failed / 0 skipped / 0 xfailed |
| `artifact_name` | `evol-x-p3-l3-2b165516485154583a4118ba1ecb2e25549bb5d0` (file `junit.xml`) |
| `junit_sha256` | `21dc041e196f9fb7aaa6eff9a50fea984728ef29ba92a407b58af8f8e13ce7c6` |
| `manifest_check` | PASS — `VL3 manifest OK (288 node ids)` (`evol_x_vl3_manifest.py --check`) |
| `operator_authorization` | OD-EVOL-10 |

## Procedure (W6-04 exact-SHA)

1. Operator merged `evol-x-p3-l3-qualification` job definition to **`main`** (PR #380); **WORKFLOW_DEF_SHA** recorded above.
2. `workflow_dispatch` from **`main`** with input **`evol_test_execution_sha`** = **`2b165516485154583a4118ba1ecb2e25549bb5d0`** (never empty; never `github.sha`).
3. Runner checks out `development` history, then `git checkout "${TEST_EXECUTION_SHA}"` with `rev-parse` assertion — **test tree = TES**.
4. `uv sync --extra dev-ci --frozen`; `evol_x_vl3_manifest.py --check`; full `pytest tests/qualification/evol_x` → JUnit artifact.

**C-CI-02:** Qualification record **forbids** using `github.sha` or default-branch checkout alone as TEST_EXECUTION_SHA evidence; only dispatch input + step-2 assertion.

**Note:** run `headSha` is the main workflow-definition commit; the tested tree is TES (in `development` history, not an ancestor of `main`). Local session artifacts: `.tmp/session/evol-x-w6-04-requal/` (gitignored).

## Historical (superseded TES — do not use for active qualification)

| Field | Value |
|-------|-------|
| `test_execution_sha` | `dcd7ae6e264d435994c61162e0d41aba25f71283` |
| `tech_sha` @ that era | `9f875fc62d1f6b2231b31c843ae0428a2d871031` |
| `github_run_id` | `41320511747` |
| `run_url` | https://github.com/jakbuczarnecki/intergrax/actions/runs/41320511747 |
| `tests` | `279 passed in 322.52s` |
| `manifest_check` | PASS — `VL3 manifest OK (279 node ids)` |
| `junit_sha256` | `96904f8568ff95fa619dbdf0fb4b53d033f4760d49fb2807d40f36c51cc79b30` |
| `artifact_name` | `evol-x-p3-l3-dcd7ae6e264d435994c61162e0d41aba25f71283` |
| `superseded_by` | run `41327408027` @ TES `2b165516485154583a4118ba1ecb2e25549bb5d0` (E-CORR-M09H-ORACLE) |

See also: [EVOL_X_CI_QUALIFICATION_PROCEDURE.md](./EVOL_X_CI_QUALIFICATION_PROCEDURE.md)
