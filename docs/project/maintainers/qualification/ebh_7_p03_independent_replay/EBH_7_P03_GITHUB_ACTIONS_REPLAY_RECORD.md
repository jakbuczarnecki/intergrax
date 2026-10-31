# EBH-7 — Protocol 03 L3 GitHub Actions replay record

**Status:** B5-corrected workflow · await canonical green run after push (supersedes run `41212701986`).

## Frozen roles (certification baseline)

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA (qualification code) | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| EVIDENCE_PUBLICATION_SHA (MANIFEST authority) | `96069c09c2ca5a7a3a9586db336716c9430ca1d7` |
| P03 replay checkout (default) | `3c745c2c0272ff17b8ac5c2975bde4369fea575d` (`github.sha` on `workflow_dispatch`) |

## B5 CI gate order (fail-closed)

1. SHA input validation  
2. Fetch → checkout replay HEAD  
3. Tracked clean tree  
4. Immutable git-object digests @ replay HEAD  
5. Executable tree identity (full diff + allowlist)  
6. MANIFEST pins + dossier digests @ HEAD  
7. Environment conformance → MANIFEST pytest profile → artifacts  

**No** pre-pytest digest reconcile in certified checkout.

## Trigger

```powershell
gh workflow run unit-tests.yml --ref development -f profile=ebh7_p3_l3
```

Optional: `-f ebh7_p03_replay_head_sha=<40-char>` for pinned replay audit.

## Superseded run (B5 defect)

| Field | Value |
| --- | --- |
| Run ID | `41212701986` |
| Defect | Digest reconcile mutated dossier before verify |
| Disposition | **Invalid for B5** — do not cite for independent B6 |

## Canonical B5 run

| Field | Value |
| --- | --- |
| **Run ID** | `41214574614` |
| **URL** | https://github.com/jakbuczarnecki/intergrax/actions/runs/41214574614 |
| **Workflow / dispatch SHA** | `210accb224aad276a472307c8fefe3798bd44f10` |
| **Replay checkout** | `210accb224aad276a472307c8fefe3798bd44f10` |
| **Pytest** | MANIFEST L3 profile · **195 passed** · exit **0** |
| **B5 gates** | Immutable git-objects + identity + replay-disk PASS (artifacts in run log) |

## Non-claims

B6 PASS, `INDEPENDENT_ACCEPTANCE_BASELINE_SHA`, OD-EBH7-08, parent CLOSED, global FRZ PASS.
