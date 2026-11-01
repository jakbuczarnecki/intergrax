# EVOL-X — CI qualification record (C-CI-01/02)

**Status:** **QUALIFIED (W6-04 exact-SHA CI run, conclusion success)** — scope: EVOL-X L3 CI dispatch @ TES `3c4a641b3…`; **no FRZ-* promotion**.

## Active qualification (@ TES `3c4a641b3e48a13f437e830a0562e2a80b03aabb`)

| Field | Value |
|-------|-------|
| `workflow_def_sha_on_main` | blob `1eb6d2b9192a3e1b12db8daa2b2e391794394815` (`origin/main:.github/workflows/unit-tests.yml`); main commit `a00674d6ba6af9b5676c0341823cd23b162bbb99` (PR #380 merge) |
| `test_execution_sha` | **`3c4a641b3e48a13f437e830a0562e2a80b03aabb`** (dispatch input `evol_test_execution_sha`; OD-EVOL-13 / VL3 290 nodes) |
| `evidence_final_publication_sha` (scoped bundle @ separate commit) | **`bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1`** — not substituted for TES |
| `tech_sha` (historical TECH @ superseded TES) | **`1e175ff8f678d605aa841ae52362486a56ed006d`** @ TES `2b165516…` |
| `workflow` / `profile` | `Regression gate` (id 250851568) / `evol_x_p3_l3`, job `evol-x-p3-l3-qualification`, ref `main` |
| `github_run_id` | **`41337077334`** |
| `run_url` | https://github.com/jakbuczarnecki/intergrax/actions/runs/41337077334 |
| `run_head_sha` (workflow def on main, not test tree) | `a00674d6ba6af9b5676c0341823cd23b162bbb99` |
| `conclusion` | **success** (completed 2026-11-01T18:41:08Z) |
| `checkout_assertion` | step `Checkout TEST_EXECUTION_SHA` — `git rev-parse HEAD` == TES |
| `tests` | **`290 passed`** (`tests/qualification/evol_x`; 0 failed / 0 skipped / 0 xfailed per job log) |
| `artifact_name` | `evol-x-p3-l3-3c4a641b3e48a13f437e830a0562e2a80b03aabb` (`junit.xml`) |
| `junit_sha256` | `69c6a9ad9721b950008888dbf0d68852a35cc9e8b9df0ab5e3d895fabed15ea4` |
| `manifest_check` | PASS — `evol_x_vl3_manifest.py --check` (290 node ids) |
| `operator_authorization` | OD-EVOL-10; scoped FINAL semantics OD-EVOL-13 |
| `qualification_record_publication_note` | This file updated @ doc commit after run `41337077334` (see git log); does not alter TES |

## Procedure (W6-04 exact-SHA)

1. Workflow job on **`main`** with `workflow_dispatch`, `profile=evol_x_p3_l3`, `evol_test_execution_sha=3c4a641b3e48a13f437e830a0562e2a80b03aabb`.
2. Runner checks out `development`, then `git checkout "${TEST_EXECUTION_SHA}"` with equality assertion.
3. `uv sync --extra dev-ci --frozen`; `evol_x_vl3_manifest.py --check`; full `pytest tests/qualification/evol_x` → JUnit artifact.

**C-CI-02:** Forbid using `github.sha` alone as TEST_EXECUTION_SHA evidence.

**Session artifact (gitignored):** `.tmp/session/evol-x-w6-04-requal-3c4a641/junit.xml` (copy of uploaded artifact digest).

## Historical (superseded — do not use for active TES)

| Field | Value |
|-------|-------|
| TES | `2b165516485154583a4118ba1ecb2e25549bb5d0` |
| `github_run_id` | `41327408027` |
| `tests` | 288 passed |
| `junit_sha256` | `21dc041e196f9fb7aaa6eff9a50fea984728ef29ba92a407b58af8f8e13ce7c6` |
| `superseded_by` | run **`41337077334`** @ TES **`3c4a641b3…`** |

| Field | Value |
|-------|-------|
| TES | `dcd7ae6e264d435994c61162e0d41aba25f71283` |
| `github_run_id` | `41320511747` |
| `tests` | 279 passed |

See [EVOL_X_CI_QUALIFICATION_PROCEDURE.md](./EVOL_X_CI_QUALIFICATION_PROCEDURE.md).
