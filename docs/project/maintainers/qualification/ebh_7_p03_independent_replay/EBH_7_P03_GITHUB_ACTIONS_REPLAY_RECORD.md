# EBH-7 — Protocol 03 L3 GitHub Actions replay record

**Status:** agent-run CI replay (not B6 PASS). **Not** independent auditor sign-off.

## Frozen certification baseline

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA (qualification code identity) | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| EVIDENCE_PUBLICATION_SHA (P03 replay checkout HEAD) | `96069c09c2ca5a7a3a9586db336716c9430ca1d7` |

**Rule:** CI checks out `EVIDENCE_PUBLICATION_SHA` and asserts `git diff TEST..EVIDENCE -- tests/qualification/` is empty.

## Trigger

```powershell
gh workflow run unit-tests.yml --ref development -f profile=ebh7_p3_l3
```

Job: `ebh7-p03-l3-qualification` · Workflow file @ `github.workflow_sha` on each run.

## Run history

| Run ID | URL | Result | pytest exit | tests (junit) | notes |
| --- | --- | --- | --- | --- | --- |
| 41211553184 | https://github.com/jakbuczarnecki/intergrax/actions/runs/41211553184 | FAIL | — | — | evidence SHA unreachable on shallow checkout (fixed) |
| 41211640855 | https://github.com/jakbuczarnecki/intergrax/actions/runs/41211640855 | FAIL | — | — | bash `fi` typo in manifest gate (fixed) |
| 41211744930 | https://github.com/jakbuczarnecki/intergrax/actions/runs/41211744930 | FAIL | 1 | 194+1 fail | digest SSOT @ TEST-only checkout (superseded by replay-tree fix) |
| *latest green* | *see Actions* | *pending* | *pending* | *pending* | after replay @ EVIDENCE tree |

## Push smoke (not L3)

`push` → `ci-smoke` often **cancelled** ~10m (`timeout-minutes: 10`). Unrelated to `ebh7_p3_l3` dispatch.

## Non-claims

B6 PASS, `INDEPENDENT_ACCEPTANCE_BASELINE_SHA`, OD-EBH7-08, parent CLOSED, global FRZ PASS.
