# EBH-7 — Protocol 03 L3 GitHub Actions replay record

**Status:** PDC-X P7 final re-pin (T-IGX-041-J) — `TEST_EXECUTION_SHA` re-pinned to `d39a520dda965981a38f8e1a90ec5135ea5d508e`; the one L3 replay at this pin (run `43738431866`) **FAILED** (see *Re-pin replay*). Run `41238620036` (TEST `dc18e36d9`) is the previous final reconciliation replay and is historical. Not a B6 PASS.

## Current roles

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA (qualification code) | `d39a520dda965981a38f8e1a90ec5135ea5d508e` (PDC-X P7 final re-pin: `development` tip after the last P7 repair merge) |
| EVIDENCE_PUBLICATION_SHA (MANIFEST authority) | `d39a520dda965981a38f8e1a90ec5135ea5d508e` (= TEST_EXECUTION_SHA; pinned in `MANIFEST.json`) |
| Previous TEST_EXECUTION_SHA (historical) | `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` (= `EBH7_B6_FINAL_RECONCILIATION_IMPLEMENTATION_SHA`) |
| INDEPENDENT_ACCEPTANCE_BASELINE_SHA | `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` (independently accepted tree; not re-accepted at the re-pinned TEST) |
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

Optional: `-f ebh7_p03_replay_head_sha=<40-char>` for pinned replay audit. For the re-pin replay pass the SHA inputs explicitly (both equal the MANIFEST pins):

```powershell
gh workflow run unit-tests.yml --ref development -f profile=ebh7_p3_l3 -f ebh7_test_execution_sha=d39a520dda965981a38f8e1a90ec5135ea5d508e -f ebh7_evidence_publication_sha=d39a520dda965981a38f8e1a90ec5135ea5d508e
```

## Re-pin replay (current — FAILED, not a B6 PASS)

One `workflow_dispatch` of `unit-tests.yml` (`profile=ebh7_p3_l3`, TEST and EVIDENCE inputs `d39a520dda965981a38f8e1a90ec5135ea5d508e`) was made after the W1' PR merged. It failed. **This run is not a B6 PASS and not evidence of a green L3.**

| Field | Value |
| --- | --- |
| **Run ID** | `43738431866` |
| **URL** | https://github.com/jakbuczarnecki/intergrax/actions/runs/43738431866 |
| **Trigger** | `workflow_dispatch` · `profile=ebh7_p3_l3` · TEST and EVIDENCE inputs `d39a520dda965981a38f8e1a90ec5135ea5d508e` |
| **Workflow SHA / replay checkout** | `8115de877249e9a4a62d1c2390513690bc082cd2` (= `actual_head_sha`; the W1' merge, PR #586) |
| **Executable tree identity** | PASS — 13 changed paths between TEST and replay head, all allowlisted |
| **Immutable digests** | git-objects PASS · MANIFEST pins PASS · working-tree digests PASS · clean tracked tree PASS |
| **Pytest** | MANIFEST L3 profile · `2 failed, 720 passed, 9 skipped in 149.26s`; `meta.txt` records `pytest_exit_code=UNKNOWN` |
| **JUnit (`junit.xml`)** | not summarised: strict summary `junit_summary=FAIL` — `duplicate testcase (repeated statistics): tests/unit/runtime/architecture/test_ee_final_arch_owner_uniqueness.py::test_ee_final_arch_owner_matrix_documented`; `meta.txt` therefore has `junit_parse=FAIL` and every `junit_*` count `unknown` (by design, never zero) |
| **F10 execution attestation** | `f10_attestation=PASS nodes=70 problems=0` (exit 0); the computed disposition line of that artifact is not a B6 PASS |
| **Skip disposition** | `skip_disposition=FAIL` — not produced because the strict JUnit summary failed; `skip-disposition.json` is not part of the artifact |
| **Job conclusion** | `ebh7-p03-l3-qualification` = failure · run conclusion `failure` |
| **Artifact** | `ebh7-p03-l3-d39a520dda965981a38f8e1a90ec5135ea5d508e` |

Failures (pytest `FAILED` lines):

1. `tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py::test_f10_c4_041k_removing_the_entries_reopens_exactly_the_three_paths`
2. `tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py::test_f10_c4_041o_removing_the_entries_reopens_exactly_the_three_paths`

Cause: test defects (reported by the D1 session of the platform audit; the appended paths of the two tests are already inside the F10 range at the re-pinned `d39a520dd`), fixed in the test tree by T-IGX-041-Q (PR #593, merge `969cbd964`) after this run. The duplicate JUnit testcase comes from the MANIFEST command: it names the node `tests/unit/runtime/architecture/test_ee_final_arch_owner_uniqueness.py::test_ee_final_arch_owner_matrix_documented` and, separately, its whole file; the strict parser refuses repeated testcase statistics.

Operator condition (`DELEGATION-PDC-LOOP2`): no re-run of the replay and no new pin. This record therefore states the result as it is. The earlier replays below, including the P6 run `43576897070` (6 failed / 619 passed / 9 skipped, MANIFEST `historical_replays`), stay as recorded.

Artifact SHA-256 (as downloaded):

| File | SHA-256 |
| --- | --- |
| `junit.xml` | `b361b5f93600d1186d45077da43c6102a40acb7a5b979d4fd86c70081fc5104e` |
| `meta.txt` | `54b5475cd4632d6ae588c4c9f51b7ef9ca249b29d3c6698e007a71be99cf5246` |
| `junit-summary.json` | `56ec968a57265ea1105b2156eeb1a936bc402b5b959e573200ffb6b5762d8df4` |
| `f10-attestation.json` | `e480d4e6849c0ae812bf5fc4978697da544ba2f8a31ae725e508c7733b4babf7` |
| `pytest.log` | `7b7a4e71e70a3955fa8485a0eca964f7daf4ab96b3770fbd187a314d697a77e0` |

Raw artifacts are not committed; only SHA-256 values are recorded, as for run `41238620036`.

## Previous final reconciliation run (historical — superseded as current by the re-pin)

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
