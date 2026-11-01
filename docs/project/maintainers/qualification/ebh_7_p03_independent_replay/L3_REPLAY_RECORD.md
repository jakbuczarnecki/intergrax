# EBH-7 — L3 replay record

## Current roles (final B6 evidence reconciliation)

| Field | Value |
| --- | --- |
| EBH7_CLOSURE_IMPLEMENTATION_SHA | `d4937889fc0b07515649dbdddf602df2a4415489` (pre-B6 terminal closure — lineage) |
| EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA | `8afff322cb2525f0106c2bbccf2d44f682d3e0df` |
| EBH7_B5_TEST_EXECUTION_SHA (historical) | `4eb7eb1a89f82215d5356aa7d000198e432777b0` — 195 passed; `tests/qualification/ebh_7/` shadowed (F10-P03-PROFILE-SHADOW) |
| TEST_EXECUTION_SHA | `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` (= `EBH7_B6_FINAL_RECONCILIATION_IMPLEMENTATION_SHA`: strict JUnit metadata + skip disposition register; qualification tooling/tests only) |
| Profile | MANIFEST `pytest_profile` — full `tests/qualification/ebh_7/` + F10 regression nodes (unchanged; new gates are picked up by the directory argument) |
| **Authoritative L3 replay** | GHA run `41238620036` @ `73965c2eb3533dfb11ba5c57697316377f956bdd` (TEST `dc18e36d9`, EVIDENCE `f30a90a14`): JUnit **634** tests = **625 passed / 9 skipped / 0 failed / 0 errors / 0 xfailed**, exit 0; F10 attestation **PASS (56 nodes, 0 problems)**; skip disposition **PASS (9/9)**; strict JUnit metadata PASS — [`EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md`](EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md) |
| Skip dispositions | Nine individual `N/A — WITH EVIDENCE` records — [`EBH_7_SKIP_DISPOSITION_REGISTER.md`](EBH_7_SKIP_DISPOSITION_REGISTER.md); verified against the real JUnit by the workflow (`skip-disposition.json`) |

## Historical provenance (do not cite as current)

| Item | Value |
| --- | --- |
| B6-F10 certification TEST (historical) | `496016c0cf6cc5cf1f9a29d885677605dd0415d3` (B6-F10 computed certification + F10-derived fixes) |
| First B6-F10 GHA replay (superseded) | run `41227765200` @ `c1dc4f838d172d7624ecdec306854ab652f4a348` (TEST `74e2191dd`): 579 passed / 3 failed / 8 skipped; F10 attestation PASS — failures were clean-checkout defects (F10-P03-DIRTY-TREE) |
| B6-F10 certification GHA replay (historical) | run `41231430749` @ `e6ff57b13848f70c964ab141cba5bd89b50e8315` (TEST `496016c0c`): JUnit **590** tests = **581 passed / 9 skipped / 0 failed**; F10 attestation PASS (56 nodes, 0 problems). Metadata defect: `meta.txt` `junit_tests=0` (parser misread root `<testsuites>`) and `f07_log_lines=0` printed twice — corrected by the strict parser at TEST `dc18e36d9`. Accepted as historical evidence of the F10 certification; superseded as the current replay |
| Local pre-publication runs (historical, Windows) | replay HEAD `6558d396607bb3cc58c39c0a64435d23545ed0fe`: 575 passed / 8 skipped; replay HEAD `e6ff57b13848f70c964ab141cba5bd89b50e8315`: 582 passed / 8 skipped / 0 failed; log `.tmp/session/ebh7-f10-cert/l2-profile-C.log` (local only, not replayable evidence) |

**Semantic @ replay tree:** RPL **56/56 QUALIFIED**; dual-slug **28/28**; TEN **26 EXACT / 0 OPEN**; XDM **34 EXACT / 0 OPEN**; G01–G09 **0 OPEN_FINDING**.

Matrix: [`EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md`](EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md) · [`B6_EVIDENCE_CONSISTENCY_MATRIX.md`](B6_EVIDENCE_CONSISTENCY_MATRIX.md).
