# EVOL-X — CI qualification record (C-CI-01/02)

**Status:** **QUALIFIED (W6-04 exact-SHA CI run, conclusion success)** — scope: EVOL-X L3 CI dispatch only; **no FRZ-* promotion**, TES/TECH unchanged.

| Field | Value |
|-------|-------|
| `workflow_def_sha_on_main` | blob `1eb6d2b9192a3e1b12db8daa2b2e391794394815` (`origin/main:.github/workflows/unit-tests.yml`); main commit `a00674d6ba6af9b5676c0341823cd23b162bbb99` (PR #380 merge, `evol-x-w6-04-workflow-only`, mergedAt 2026-11-01T15:16:41Z); job added in `890519cc1da3c05a8aa6eeed8304552ba0b44a86` |
| `test_execution_sha` | **`dcd7ae6e264d435994c61162e0d41aba25f71283`** (dispatch input `evol_test_execution_sha`; supersedes `6fec9ec…`) |
| `tech_sha` (unchanged) | `9f875fc62d1f6b2231b31c843ae0428a2d871031` |
| `workflow` / `profile` | `Regression gate` (id 250851568) / `evol_x_p3_l3`, job `evol-x-p3-l3-qualification`, ref `main` |
| `github_run_id` | `41320511747` |
| `run_url` | https://github.com/jakbuczarnecki/intergrax/actions/runs/41320511747 |
| `run_head_sha` (workflow def, not test tree) | `a00674d6ba6af9b5676c0341823cd23b162bbb99` |
| `conclusion` | **success** (job completed 2026-11-01T15:24:07Z) |
| `checkout_assertion` | step `Checkout TEST_EXECUTION_SHA` passed (`git rev-parse HEAD` == TES) |
| `tests` | `279 passed in 322.52s` (`tests/qualification/evol_x`) |
| `artifact_name` | `evol-x-p3-l3-dcd7ae6e264d435994c61162e0d41aba25f71283` (file `junit.xml`) |
| `junit_sha256` | `96904f8568ff95fa619dbdf0fb4b53d033f4760d49fb2807d40f36c51cc79b30` |
| `manifest_check` | PASS — `VL3 manifest OK (279 node ids)` (`evol_x_vl3_manifest.py --check`) |
| `operator_authorization` | OD-EVOL-10 |

## Procedure (W6-04 exact-SHA)

1. Operator merged `evol-x-p3-l3-qualification` job definition to **`main`** (PR #380); **WORKFLOW_DEF_SHA** recorded above.
2. `workflow_dispatch` from **`main`** with input **`evol_test_execution_sha`** = **`dcd7ae6e264d435994c61162e0d41aba25f71283`** (never empty; never `github.sha`).
3. Runner checks out `development` history, then `git checkout "${TEST_EXECUTION_SHA}"` with `rev-parse` assertion — **test tree = TES**.
4. `uv sync --extra dev-ci --frozen`; `evol_x_vl3_manifest.py --check`; full `pytest tests/qualification/evol_x` → JUnit artifact.

**Invalidated:** earlier note "dispatch pending operator / dispatch @ `6fec9ec387089dfcee1039089ba5ad901fab39d6`" is superseded by run `41320511747` @ TES `dcd7ae6…`.

**C-CI-02:** Qualification record **forbids** using `github.sha` or default-branch checkout alone as TEST_EXECUTION_SHA evidence; only dispatch input + step-2 assertion.

**Note:** run `headSha` is the main workflow-definition commit; the tested tree is TES (in `development` history, not an ancestor of `main`). Local session artifacts: `.tmp/session/evol-x-w6-04/` (gitignored).

See also: [EVOL_X_CI_QUALIFICATION_PROCEDURE.md](./EVOL_X_CI_QUALIFICATION_PROCEDURE.md)
