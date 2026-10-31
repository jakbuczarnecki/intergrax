# EBH-7 — L3 replay record

| Field | Value |
| --- | --- |
| EBH7_CLOSURE_IMPLEMENTATION_SHA | `d4937889fc0b07515649dbdddf602df2a4415489` (pre-B6 terminal closure — lineage) |
| EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA | `8afff322cb2525f0106c2bbccf2d44f682d3e0df` |
| EBH7_B5_TEST_EXECUTION_SHA (historical) | `4eb7eb1a89f82215d5356aa7d000198e432777b0` — 195 passed; `tests/qualification/ebh_7/` shadowed (F10-P03-PROFILE-SHADOW) |
| TEST_EXECUTION_SHA | `74e2191dd4eaad544b2146f505fe691d459997b3` (B6-F10 computed certification + F10-derived fixes) |
| Profile | MANIFEST `pytest_profile` — full `tests/qualification/ebh_7/` + F10 regression nodes |
| Local pre-publication run | Windows, replay HEAD `6558d396607bb3cc58c39c0a64435d23545ed0fe`: 575 passed / 8 skipped; 7 documentation-sync failures fixed in this publication; F10 JUnit attestation PASS (56 nodes, 0 problems) |
| Log | `.tmp/session/ebh7-f10-cert/l2-profile-C.log` |
| Authoritative replay | GitHub Actions — [`EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md`](EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md) |

**Semantic @ replay tree:** RPL **56/56 QUALIFIED**; dual-slug **28/28**; TEN **26 EXACT / 0 OPEN**; XDM **34 EXACT / 0 OPEN**; G01–G09 **0 OPEN_FINDING**.

Matrix: [`EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md`](EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md) · [`B6_EVIDENCE_CONSISTENCY_MATRIX.md`](B6_EVIDENCE_CONSISTENCY_MATRIX.md).
