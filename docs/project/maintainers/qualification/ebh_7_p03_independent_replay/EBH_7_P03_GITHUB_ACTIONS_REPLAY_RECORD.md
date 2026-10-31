# EBH-7 — Protocol 03 L3 GitHub Actions replay record

**Status:** agent-run CI replay (not B6 PASS).

## Frozen roles (certification baseline)

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA (qualification code) | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| EVIDENCE_PUBLICATION_SHA (MANIFEST authority) | `96069c09c2ca5a7a3a9586db336716c9430ca1d7` |
| P03 replay checkout HEAD (CI) | `04cb7aeece321d875b5e8731a26ae89091709d6c` |

CI: `git diff TEST..replay -- tests/qualification/` empty · runner digest reconcile before SSOT gates.

## Trigger

```powershell
gh workflow run unit-tests.yml --ref development -f profile=ebh7_p3_l3
```

Job: `ebh7-p03-l3-qualification` · Workflow: [`unit-tests.yml`](../../../../.github/workflows/unit-tests.yml).

## Canonical green run

| Field | Value |
| --- | --- |
| Run ID | *dispatch after `04cb7aeec`* |
| URL | https://github.com/jakbuczarnecki/intergrax/actions/workflows/unit-tests.yml |
| Checkout replay HEAD | `04cb7aeece321d875b5e8731a26ae89091709d6c` |
| Expected pytest | MANIFEST L3 profile · 195 tests · exit 0 |
| Artifact | `ebh7-p03-l3-4eb7eb1a89f82215d5356aa7d000198e432777b0` |

## Push smoke cancellations

`push` → `ci-smoke` often **cancelled** ~10m (timeout). Not L3. Use manual `ebh7_p3_l3`.

## Non-claims

B6 PASS, `INDEPENDENT_ACCEPTANCE_BASELINE_SHA`, OD-EBH7-08, parent CLOSED, global FRZ PASS.
