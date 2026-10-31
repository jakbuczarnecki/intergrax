# EBH-7 — Protocol 03 L3 GitHub Actions replay record

**Status:** agent-run CI replay **COMPLETED_PASS** on canonical run (not B6 PASS).

## Frozen roles (certification baseline)

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA (qualification code) | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| EVIDENCE_PUBLICATION_SHA (MANIFEST authority) | `96069c09c2ca5a7a3a9586db336716c9430ca1d7` |
| P03 replay checkout HEAD (CI default) | `04cb7aeece321d875b5e8731a26ae89091709d6c` |

CI gates: `git diff TEST..replay -- tests/qualification/` empty · runner digest reconcile before SSOT · MANIFEST profile from replay tree.

## Trigger

```powershell
gh workflow run unit-tests.yml --ref development -f profile=ebh7_p3_l3
```

## Canonical green run (2026-10-31)

| Field | Value |
| --- | --- |
| **Run ID** | `41212701986` |
| **URL** | https://github.com/jakbuczarnecki/intergrax/actions/runs/41212701986 |
| **Workflow ref SHA** | `090a62778` (dispatch @ `development`) |
| **Replay checkout HEAD** | `04cb7aeece321d875b5e8731a26ae89091709d6c` |
| **Pytest** | MANIFEST L3 profile · **195 passed** · 0 skip · exit **0** |
| **Artifact** | `ebh7-p03-l3-4eb7eb1a89f82215d5356aa7d000198e432777b0` |

F07: `test_ebh_5_dual_slug_rpl_e_operational_gates.py` (8) in profile. F09: covered via `tests/qualification/ebh_7/` in same invocation.

## Push smoke (not L3)

`push` → `ci-smoke` often **cancelled** ~10m (`timeout-minutes: 10`). Use manual `ebh7_p3_l3`.

## Non-claims

B6 PASS, `INDEPENDENT_ACCEPTANCE_BASELINE_SHA`, OD-EBH7-08, parent CLOSED, global FRZ PASS.
