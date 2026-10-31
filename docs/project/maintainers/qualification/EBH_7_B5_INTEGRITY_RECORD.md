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

## SHA roles (unchanged)

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| EVIDENCE_PUBLICATION_SHA (MANIFEST pin) | `96069c09c2ca5a7a3a9586db336716c9430ca1d7` |
| P03 replay / workflow implementation | `3c745c2c0272ff17b8ac5c2975bde4369fea575d` (default `github.sha` on dispatch) |

## L1 adversarial

`tests/qualification/ebh_7/test_ebh_7_p03_b5_immutable_gates.py` (7 tests).

## Non-claims

B6 PASS · OD-EBH7-08 · EBH-7 CLOSED.
