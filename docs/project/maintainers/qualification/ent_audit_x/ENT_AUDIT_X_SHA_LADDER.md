# ENT-AUDIT-X — SHA ladder (SSOT)

**PLAN_BASELINE_SHA (pre-plan HEAD):** `8a41d32043b89e555d7c18e052cf7022b8c94681`  
**PLAN_PUBLICATION_SHA (initial master plan):** `6328f0cf915ed1f4cee13acb5eb68f07a7438b8e`  
**Branch:** `development`  
**Authority:** [ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md](./ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md) (execution SSOT).

| Role | SHA | Notes |
| --- | --- | --- |
| PLAN_BASELINE_SHA | `8a41d32043b89e555d7c18e052cf7022b8c94681` | Last commit before ENT plan publication wave |
| PLAN_PUBLICATION_SHA | `6328f0cf915ed1f4cee13acb5eb68f07a7438b8e` | First publication: master plan + annexes (8 files) |
| PLAN_RECONCILIATION_SHA | `1656cb8659c0b5c8cac422f861856a59401f7ca1` | P-11/P-12/P-13 documentation corrections only |
| IMPLEMENTATION_SHA | `86832319b08b3837cfb4ae9c90919389f202cc37` | Tier-0 runner + manifest + CLI + qual L1/L2 |
| FINAL_PLAN_PUBLICATION_SHA | `0cee41a88274a53d14bb3558752d55bdde0d0b01` | W0–W6 B6 readiness wave landed on `development` |
| IMPLEMENTATION_SHA (ENT-AUDIT-X closure baseline) | `1fa5347a56c569fa62ac2c777ff3947d11e906b4` | Audited HEAD @ post-implementation closure master plan |
| IMPLEMENTATION_SHA (R0 trust wave) | `34e03425b2322f415a99a7cfd290fe0982bf6970` | OD verify, dry-run, evidence eligibility, readiness fail-closed, qual tests |
| POST_IMPLEMENTATION_FINAL_CLOSURE_PLAN_PUBLICATION_SHA | `f7243601dbeec93bf26d258c05f545bb1704c7eb` | Initial post-implementation closure SSOT |
| POST_IMPLEMENTATION_FINAL_CLOSURE_PLAN_PUBLICATION_SHA (87e185dd) | `87e185dd3bcce45f107149b4b3691c3ede614c8a` | Prior SSOT publish @ audited baseline |
| CURRENT_DOCUMENTATION_HEAD (pre SSOT repair commit) | `c6d600df56f3fedb85bcdce12ba508ea8274ccd4` | Doc pin only; no production delta after 87e185dd |
| IMPLEMENTATION_SHA (audited baseline @ current-HEAD SSOT) | `d4974efc255bf6ff6ca96e00fb084293ea685d66` | Wave 2 + ladder pin; **code baseline for Matrix C** |
| IMPLEMENTATION_SHA (pre-qual matrix marker + PRE_QUAL_PASS) | `90f6dfdb3e382fd5bfcf9a5df6a916bc8dcea23e` | BND-03 smoke excluded from ent_semantic_matrix; qual_wave PRE_QUAL_PASS |
| IMPLEMENTATION_SHA (C063 launchers + readiness hooks) | `530cae6e87e9b3ef8ab315324812077489c64eb9` | ent_launcher matrix, BAT/SH non-certifying FAST, quickstart |
| IMPLEMENTATION_SHA (R0 one-pass trust wave) | `4f87e5bb20fd8691033a8f9704d205dad18c48c6` | Evidence/VBN/baseline/launchers + qual harness (agent one-pass partial) |
| IMPLEMENTATION_SHA (wave 2 provenance + semantic acceptance) | `e0f64ed7fd57180efc9b93798a3595b9578497af` | Parent of audited baseline |
| POST_IMPLEMENTATION_SSOT_REPAIR_PUBLICATION_SHA | `8efac05b4ca18a31a754cf003851f5bf24725f76` | Matrix E + DAG + validator repair @ audited baseline |
| IMPLEMENTATION_SHA (Wave F FREEZE hardening + per-run jailed baseline) | `b8bb88c9db1060153ec0a6a29bbb79ba16e00078` | qual_wave certifying replay gate; ancestor of evidence work |
| IMPLEMENTATION_SHA (C075-S2 WBS reconcile + mechanical layers) | `8ae58734d117acf23223ddce0138f6e451941d34` | Matrix C/E/GAP-RC sync @ TES; strict CI/WBS mechanical gates |
| TEST_EXECUTION_SHA (historical) | `b8bb88c9db1060153ec0a6a29bbb79ba16e00078` | Wave F certifying — **superseded** by R6 row below |
| TEST_EXECUTION_SHA (final closure R6 @ HEAD) | `87203c1b6ce9786dce0523cca29b556106b6bc8d` | Certifying qual_wave PASS @ `.tmp/session/ent-audit-x-r6-certifying/` + B5 handoff `qualification_at_87203c1b…` |
| TEST_EXECUTION_SHA (historical — not current @ d4974efc) | `0cee41a88274a53d14bb3558752d55bdde0d0b01` | 164 passed — **do not** substitute for re-qual @ audited baseline |
| EVIDENCE_PUBLICATION_SHA (historical) | `894b0e1aff071d19636d23bea406b95dcffc79e1` | B5 @ `b8bb88c9…` — archived |
| EVIDENCE_PUBLICATION_SHA (final closure R7) | `d18f01f2291328a073dcbdfbe05489faf98013b7` | B5 handoff `qualification_at_87203c1b…` |
| INDEPENDENT_TECHNICAL_REAUDIT_SHA | `0e538269a7188075003b9ff2c7cce2545d8f1ccb` | Technical re-audit + regression gate `41804655934` — **not** formal B6 attestation |
| FORMAL_B6_PACKAGE_PREPARATION_SHA | `b14a409c9894c77257487364a2d42d63ec22e295` | ENT+EVOL formal preparatory packages — [ENT_EVOL_FORMAL_CLOSURE_PUBLICATION_EVIDENCE.md](../ENT_EVOL_FORMAL_CLOSURE_PUBLICATION_EVIDENCE.md) |
| B6_ACCEPTANCE_PUBLICATION_SHA | *unset* | Signed Protocol 03 ENT-AUDIT-X PASS (authorized independent auditor) |
| INTERNAL_CERTIFICATION_STATUS | **INTERNALLY CERTIFIED — OPERATOR ACCEPTED (TECHNICAL SCOPE)** | OD-IC-01 (2026-11-04); technical re-audit recommendation **≠** formal attestation — [record](../EVOL_ENT_INTERNAL_CERTIFICATION_RECORD_2026_11_04.md) |
| INTERNAL_CERTIFICATION_PUBLICATION_SHA | *see record path* | Resolve: `git log -1 --format=%H -- docs/project/maintainers/qualification/EVOL_ENT_INTERNAL_CERTIFICATION_RECORD_2026_11_04.md` |
| FORMAL_CLOSURE_PUBLICATION_SHA | *unset* | OD-ENT-08 + roadmap §5 |
| ACCEPTED_BASELINE_SHA | *unset* | Operator OD-ENT-01 after first green FREEZE |
| EVIDENCE_PUBLICATION_SHA | `300859537f12a1de7bf3d75fd685bc3a5c727183` | W8 B6 corrective B5 `qualification_at_89c76eb7…` (strict `verify_b5_publication_tree`) |
| TEST_EXECUTION_SHA | `89c76eb7b53a0a3ce6f75ab395cf4a2802638a91` | W8 B6 corrective certifying qual @ strict B5 verifier wave (supersedes R6/R7 canonical rows below) |
| EVIDENCE_PUBLICATION_SHA (pending row placeholder) | `d18f01f2291328a073dcbdfbe05489faf98013b7` | **Superseded** — see W8 row above |
| TEST_EXECUTION_SHA (historical — R6/R7) | `87203c1b6ce9786dce0523cca29b556106b6bc8d` | Preserved; B5 @ `d18f01f22` fails `verify_b5_publication_tree()` |
| EVIDENCE_PUBLICATION_SHA (historical — R7) | `d18f01f2291328a073dcbdfbe05489faf98013b7` | Preserved defect record; do not use for B6 evidence integrity |
| CURRENT_DOCUMENTATION_HEAD (R8 external gate packet) | `d3d9b62da165ce98236ad7ece51a0f4b750a67d2` | Dossier EVID pin + W0–W7 closure prep (doc/scripts only post-TES) |
| CURRENT_DOCUMENTATION_HEAD (W7 mechanical readiness) | `1bee524218d9185175f3e24c6e6f205565ff0569` | C075-S2 sync + canonical ladder pins + dossier CI @ HEAD |

## Lineage rules

1. Do **not** rewrite historical SHAs; append new rows for each documentation or evidence publication commit.
2. `merge-base --is-ancestor TEST_EXECUTION_SHA EVIDENCE_PUBLICATION_SHA` required before B6.
3. Tests execute @ `TEST_EXECUTION_SHA`; doc-only commits use manifest split (`qualification_sha` ≠ `audit_execution_sha`).
4. Reconciliation commits (plan fixes) do not substitute for TEST_EXECUTION_SHA.

## EVOL predecessor pins (orchestration only)

| Role | SHA |
| --- | --- |
| EVOL TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| EVOL EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |
| EVOL B6_ACCEPTANCE_PUBLICATION_SHA | **PENDING** |
