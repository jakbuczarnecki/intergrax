# EVOL-X — CI qualification procedure (C-CI-02)

Executable spec for job `evol-x-p3-l3-qualification` in `.github/workflows/unit-tests.yml`.

## Forbidden for qualification evidence

- Treating `github.sha` at dispatch time as TEST_EXECUTION_SHA.
- Asserting qualification from the first checkout ref alone (`development`) without the explicit `git checkout "${TEST_EXECUTION_SHA}"` step.

## Required auditor checks

| Check | Action |
|-------|--------|
| Run metadata | `gh run view <RUN_ID> --json headSha,conclusion,event` |
| Log | Step **Checkout TEST_EXECUTION_SHA** → `git rev-parse HEAD` equals dispatch input |
| Local replay | Clean checkout @ TES; `uv sync --extra dev-ci --frozen`; manifest `--check`; pytest |
| Mode B | `evol_x_p03_evidence.py verify-test-execution --certification --replay-junit …` @ TES vs archived outcomes |

Record outcomes in [EVOL_X_CI_QUALIFICATION_RECORD.md](./EVOL_X_CI_QUALIFICATION_RECORD.md).
