# EBH-7 — Protocol 03 B5 integrity record

**Status:** B5 gates implemented · prior run `41212701986` **superseded** (pre-verify digest reconcile).

## B5-F01 (immutable evidence)

- Removed CI step **Reconcile P03 digests on runner** before integrity gates.
- Verification order: checkout replay HEAD → clean tracked tree → `git show` digest cross-check @ replay HEAD → MANIFEST pins → working-tree dossier read @ HEAD blobs.
- `scripts/qualification/render_ebh7_p03_artifact_digests.py` hashes **git objects @ HEAD** (isolated compare; does not mutate dossier in certified checkout).
- Formal digest republication: `ARTIFACT_DIGESTS.json` aligned to git-tracked blobs (appendix + ECAP checklist corrections).

## B5-F02 (executable identity)

- `verify_executable_tree_identity`: full-tree `git diff TEST_EXECUTION_SHA..replay` with allowlist (qualification docs, `.github/`, P03 B5 scripts/tests only).
- Certified code remains `4eb7eb1a89f82215d5356aa7d000198e432777b0`.

## B5 SHA roles (historical — B5 run)

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| EVIDENCE_PUBLICATION_SHA (MANIFEST pin) | `96069c09c2ca5a7a3a9586db336716c9430ca1d7` |
| P03 replay / workflow implementation | `210accb224aad276a472307c8fefe3798bd44f10` (canonical GHA run `41214574614`) |

## Final reconciliation — B5 zero regression (TEST `dc18e36d9`)

- **Mechanisms unchanged:** `scripts/qualification/verify_ebh7_p03_immutable_evidence.py` (immutable git-object digests before any use, full executable-tree identity + allowlist, MANIFEST pins, clean tracked tree, ancestry), `render_ebh7_p03_artifact_digests.py` and `attest_ebh7_f10_execution.py` are byte-identical to the B6-F10 certification tree `496016c0c`; no production code (`intergrax/`, `agents/`, `applications/`) changed.
- **Executable identity:** new TEST `dc18e36d9` differs from `496016c0c` only by qualification tooling/tests (strict JUnit parser, skip disposition register and gates, skip-equivalent bindings), allowlisted constants/B5 gate files and the workflow — see `EBH_7_SHA_LADDER.md`. Replay diff `TEST..replay HEAD` must list documentation / `.github/` / allowlisted files only (fail-closed otherwise).
- **Fail-closed on invalid MANIFEST / digest drift:** unchanged (`test_ebh_7_p03_b5_immutable_gates.py`).
- **Added fail-closed layer:** the workflow now also fails on unparsable/inconsistent `junit.xml` and on unregistered/stale skips or failing skip-evidence nodes (`junit-summary.exit` / `skip-disposition.exit`).
## L1 adversarial

`tests/qualification/ebh_7/test_ebh_7_p03_b5_immutable_gates.py` (7 tests).

## Non-claims

B6 PASS · OD-EBH7-08 · EBH-7 CLOSED.
