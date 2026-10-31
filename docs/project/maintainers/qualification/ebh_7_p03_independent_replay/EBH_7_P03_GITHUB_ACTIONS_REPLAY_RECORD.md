# EBH-7 — Protocol 03 L3 GitHub Actions replay record

**Status:** final B6 evidence reconciliation — workflow corrected (strict JUnit metadata + skip disposition verification); canonical final replay run is recorded in the *Final reconciliation run* section below once completed. Not a B6 PASS.

## Current roles

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA (qualification code) | `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` (= `EBH7_B6_FINAL_RECONCILIATION_IMPLEMENTATION_SHA`) |
| EVIDENCE_PUBLICATION_SHA (MANIFEST authority) | pinned in `MANIFEST.json` (`EVIDENCE_PUBLICATION_SHA`) |
| P03 replay checkout | `github.sha` of the `workflow_dispatch` (default) — recorded per run below |

Historical roles (provenance only): B6-F10 TEST `496016c0cf6cc5cf1f9a29d885677605dd0415d3` / EVIDENCE `06d46537f5f252dbb395e865a33ce6f2475bf7b1`; B5 TEST `4eb7eb1a89f82215d5356aa7d000198e432777b0` / EVIDENCE `96069c09c2ca5a7a3a9586db336716c9430ca1d7`.

## CI gate order (fail-closed)

1. SHA input validation
2. Fetch → checkout replay HEAD
3. Tracked clean tree
4. Immutable git-object digests @ replay HEAD
5. Executable tree identity (full diff + allowlist)
6. MANIFEST pins + dossier digests @ HEAD
7. Environment conformance → MANIFEST pytest profile (single invocation) → `junit.xml`
8. B6-F10 execution attestation (`f10-attestation.json`)
9. **Strict JUnit summary** (`junit-summary.json`) and **skip disposition verification** (`skip-disposition.json`)
10. `meta.txt` finalization (statistics copied from the strict summary; `unknown` + `junit_parse=FAIL` on any parse problem — never zero statistics) → artifact upload → exit status (pytest, F10 attestation, JUnit summary, skip disposition must all be 0)

**No** pre-pytest digest reconcile in the certified checkout.

## Trigger

```powershell
gh workflow run unit-tests.yml --ref development -f profile=ebh7_p3_l3
```

Optional: `-f ebh7_p03_replay_head_sha=<40-char>` for pinned replay audit.

## Final reconciliation run (current)

*Pending — filled after the `workflow_dispatch` replay (run ID, URL, workflow/replay SHA, JUnit totals read from `junit.xml`, F10 attestation, skip disposition result, job conclusion).*

## Historical runs (provenance — not current)

| Run | Checkout | TEST | Result | Disposition |
| --- | --- | --- | --- | --- |
| `41231430749` (B6-F10 certification) | `e6ff57b13848f70c964ab141cba5bd89b50e8315` | `496016c0cf6cc5cf1f9a29d885677605dd0415d3` | JUnit **590** tests: **581 passed / 9 skipped / 0 failed**; exit 0; F10 attestation `PASS nodes=56 problems=0`; executable identity PASS; immutable git objects PASS; job success | Valid F10 certification replay. Metadata defect: `meta.txt` `junit_tests=0` (+ `f07_log_lines` printed twice) — pytest exit code and JUnit were authoritative; fixed in TEST `dc18e36d9`. Skips: all nine dispositioned in `EBH_7_SKIP_DISPOSITION_REGISTER.md` against this run's `junit.xml` |
| `41227765200` | `c1dc4f838d172d7624ecdec306854ab652f4a348` | `74e2191dd4eaad544b2146f505fe691d459997b3` | 579 passed / 3 failed | Superseded — clean-checkout defects (B6-F10-P03-DIRTY-TREE), fixed in TEST `496016c0c` |
| `41214574614` (B5) | `210accb224aad276a472307c8fefe3798bd44f10` | `4eb7eb1a89f82215d5356aa7d000198e432777b0` | 195 passed; `tests/qualification/ebh_7/` shadowed by member-file arguments (B6-F10-P03-PROFILE-SHADOW) | Historical B5 only |
| `41212701986` | — | — | Digest reconcile mutated dossier before verify | **Invalid** — do not cite |

## Non-claims

B6 PASS, `INDEPENDENT_ACCEPTANCE_BASELINE_SHA`, OD-EBH7-08, parent CLOSED, global FRZ PASS.
