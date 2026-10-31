# EBH-7 — Protocol 03 L3 GitHub Actions replay record

**Status:** populated after each `workflow_dispatch` profile `ebh7_p3_l3`. **Not** B6 PASS.

## Frozen certification baseline

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA (code tree) | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| EVIDENCE_PUBLICATION_SHA (MANIFEST profile) | `96069c09c2ca5a7a3a9586db336716c9430ca1d7` |

## Trigger (auditor / operator)

```powershell
gh workflow run unit-tests.yml --ref development -f profile=ebh7_p3_l3
# optional overrides (must match dossier unless corrective replay):
#   -f ebh7_test_execution_sha=4eb7eb1a89f82215d5356aa7d000198e432777b0
#   -f ebh7_evidence_publication_sha=96069c09c2ca5a7a3a9586db336716c9430ca1d7
```

Workflow job: `ebh7-p03-l3-qualification` in [`.github/workflows/unit-tests.yml`](../../../../.github/workflows/unit-tests.yml).

- Checkout **exact** `TEST_EXECUTION_SHA`.
- Resolve `pytest_profile.command` from `MANIFEST.json` at **EVIDENCE_PUBLICATION_SHA** (fail-closed if `TEST_EXECUTION_SHA` in manifest mismatches checkout).
- `uv sync --frozen --extra dev-ci`; single pytest invocation; artifacts under `build/qualification/ebh7-l3/`.

## Latest run

| Field | Value |
| --- | --- |
| GitHub run URL | *pending first green run* |
| `github.run_id` | *pending* |
| `github.workflow_sha` | *pending* |
| Checkout HEAD | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| Pytest exit | *pending* |
| JUnit tests / skipped | *pending* |
| Artifact name | `ebh7-p03-l3-4eb7eb1a89f82215d5356aa7d000198e432777b0` |

## Push smoke cancellations (context)

`push` to `development` runs `ci-smoke` (not L3). Recent runs show **cancelled** after ~10m — `ci-smoke` `timeout-minutes: 10` vs long suite. L3 replay uses **manual** `ebh7_p3_l3` dispatch (separate concurrency group; not cancelled by PR smoke).

## Non-claims

B6 PASS, `INDEPENDENT_ACCEPTANCE_BASELINE_SHA`, OD-EBH7-08, parent CLOSED, global FRZ PASS.
