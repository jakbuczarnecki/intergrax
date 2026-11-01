# EBH-7 — Protocol 03 L3 GitHub Actions replay record

**Status:** final B6 evidence reconciliation — workflow corrected (strict JUnit metadata + skip disposition verification); canonical final replay run `41238620036` completed successfully (see *Final reconciliation run*). Not a B6 PASS.

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

| Field | Value |
| --- | --- |
| **Run ID** | `41238620036` |
| **URL** | https://github.com/jakbuczarnecki/intergrax/actions/runs/41238620036 |
| **Trigger** | `workflow_dispatch` · `profile=ebh7_p3_l3` (default SHA inputs = MANIFEST pins) |
| **TEST_EXECUTION_SHA** | `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` |
| **EVIDENCE_PUBLICATION_SHA** | `f30a90a14910afe8583081cab3b2e39480988c17` |
| **Workflow SHA / replay checkout** | `73965c2eb3533dfb11ba5c57697316377f956bdd` (= `actual_head_sha`) |
| **Executable tree identity** | PASS — `git diff --name-only TEST..replay` = 17 paths, all allowlisted (docs / `.github/` / B5 meta-gate files); no `intergrax/`, `agents/`, `applications/` change |
| **Immutable digests** | git-objects @ replay HEAD PASS · MANIFEST pins PASS · working-tree digests PASS · clean tracked tree PASS |
| **Pytest** | MANIFEST L3 profile (full `tests/qualification/ebh_7/`) · exit **0** · `625 passed, 9 skipped` |
| **JUnit (`junit.xml`, root `<testsuites>`, 1 suite)** | **634** tests = **625 passed · 0 failed · 0 errors · 9 skipped · 0 xfailed** — values read from the file by the strict parser (`meta.txt`: `junit_parse=PASS junit_tests=634 junit_passed=625 junit_failed=0 junit_errors=0 junit_skipped=9 junit_xfailed=0 junit_f07_operational_gate_nodes=8`); cross-checked with an independent raw XML read |
| **F10 execution attestation** | `f10_attestation=PASS nodes=56 problems=0` · C1–C5 PASS · disposition `EBH-7 — B6-F10 NORMATIVELY QUALIFIED / READY FOR INDEPENDENT B6 ACCEPTANCE` (computed; not B6 PASS) |
| **Skip disposition** | `skip_disposition=PASS skipped=9 registered=9 problems=0` — 0 unregistered, 0 stale; every evidence node PASS in this JUnit — [`EBH_7_SKIP_DISPOSITION_REGISTER.md`](EBH_7_SKIP_DISPOSITION_REGISTER.md) |
| **Job conclusion** | `ebh7-p03-l3-qualification` = success · run conclusion `success` |
| **Artifact** | `ebh7-p03-l3-dc18e36d971c93c8f631fbc86f73149d1b81b4d7` (retention 90 days) |

Artifact SHA-256 (as downloaded):

| File | SHA-256 |
| --- | --- |
| `junit.xml` | `15f1d0b891df996c5ef5c3866276ca9455e7f3ba1c22753249449b140bd8b158` |
| `meta.txt` | `cdcf9ebb1339f2959a5b02375daa8a44771dd9df229e0630b0b8a9284f93dcb8` |
| `junit-summary.json` | `e3538ae5c7bab895a17a45549a21e74cf89c44e22f4967ffde22ecec6092793e` |
| `skip-disposition.json` | `b0ae4d507eef951b5b0c1e193ed39d251ee82cf1297869701ef5042e2f206d3f` |
| `f10-attestation.json` | `8cbcdc47a050bd3451e7f4510f91555614ba5636ab833a17bc8036245e79c191` |
| `pytest.log` | `9ca786012c00af406ba84d7f4a9cbb4b62a094a88006c77d70369e64e116c663` |

Delta vs historical run `41231430749`: +44 tests (new gates `test_ebh_7_b6_junit_skip_disposition_gates.py`), same nine skips, `meta.txt` statistics now correct (`junit_tests=634`, previously `0`).

## Historical runs (provenance — not current)

| Run | Checkout | TEST | Result | Disposition |
| --- | --- | --- | --- | --- |
| `41231430749` (B6-F10 certification) | `e6ff57b13848f70c964ab141cba5bd89b50e8315` | `496016c0cf6cc5cf1f9a29d885677605dd0415d3` | JUnit **590** tests: **581 passed / 9 skipped / 0 failed**; exit 0; F10 attestation `PASS nodes=56 problems=0`; executable identity PASS; immutable git objects PASS; job success | Valid F10 certification replay. Metadata defect: `meta.txt` `junit_tests=0` (+ `f07_log_lines` printed twice) — pytest exit code and JUnit were authoritative; fixed in TEST `dc18e36d9`. Skips: all nine dispositioned in `EBH_7_SKIP_DISPOSITION_REGISTER.md` against this run's `junit.xml` |
| `41227765200` | `c1dc4f838d172d7624ecdec306854ab652f4a348` | `74e2191dd4eaad544b2146f505fe691d459997b3` | 579 passed / 3 failed | Superseded — clean-checkout defects (B6-F10-P03-DIRTY-TREE), fixed in TEST `496016c0c` |
| `41214574614` (B5) | `210accb224aad276a472307c8fefe3798bd44f10` | `4eb7eb1a89f82215d5356aa7d000198e432777b0` | 195 passed; `tests/qualification/ebh_7/` shadowed by member-file arguments (B6-F10-P03-PROFILE-SHADOW) | Historical B5 only |
| `41212701986` | — | — | Digest reconcile mutated dossier before verify | **Invalid** — do not cite |

## Non-claims

B6 PASS, `INDEPENDENT_ACCEPTANCE_BASELINE_SHA`, OD-EBH7-08, parent CLOSED, global FRZ PASS.
