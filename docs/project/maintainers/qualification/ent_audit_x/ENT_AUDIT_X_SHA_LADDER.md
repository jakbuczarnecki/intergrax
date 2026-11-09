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
| INTERNAL_CERTIFICATION_PUBLICATION_SHA | `5c08e5e6c44d21998c76bd27ad1efab172440f45` | OD-IC-01 consolidated record + roadmap §5 |
| POST_IC_CONSOLIDATION_PUBLICATION_SHA | `81829712acc19ebeadb4773467b9c5642ca975e3` | Pre-admission + IC consolidation handoff — [`ARCH_FREEZE_PRE_ADMISSION_READINESS_REPORT.md`](../ARCH_FREEZE_PRE_ADMISSION_READINESS_REPORT.md) |
| FORMAL_CLOSURE_PUBLICATION_SHA | *unset* | OD-ENT-08 + roadmap §5 |
| ACCEPTED_BASELINE_SHA | *unset* | Operator OD-ENT-01 after first green FREEZE |
| EVIDENCE_PUBLICATION_SHA (historical — superseded by the TES 62d1c615 row below) | `300859537f12a1de7bf3d75fd685bc3a5c727183` | W8 B6 corrective B5 `qualification_at_89c76eb7…` (strict `verify_b5_publication_tree`) |
| TEST_EXECUTION_SHA (historical — superseded by the TES 62d1c615 row below) | `89c76eb7b53a0a3ce6f75ab395cf4a2802638a91` | W8 B6 corrective certifying qual @ strict B5 verifier wave (supersedes R6/R7 canonical rows below) |
| EVIDENCE_PUBLICATION_SHA (pending row placeholder) | `d18f01f2291328a073dcbdfbe05489faf98013b7` | **Superseded** — see W8 row above |
| TEST_EXECUTION_SHA (historical — R6/R7) | `87203c1b6ce9786dce0523cca29b556106b6bc8d` | Preserved; B5 @ `d18f01f22` fails `verify_b5_publication_tree()` |
| EVIDENCE_PUBLICATION_SHA (historical — R7) | `d18f01f2291328a073dcbdfbe05489faf98013b7` | Preserved defect record; do not use for B6 evidence integrity |
| CURRENT_DOCUMENTATION_HEAD (R8 external gate packet) | `d3d9b62da165ce98236ad7ece51a0f4b750a67d2` | Dossier EVID pin + W0–W7 closure prep (doc/scripts only post-TES) |
| CURRENT_DOCUMENTATION_HEAD (W7 mechanical readiness) | `1bee524218d9185175f3e24c6e6f205565ff0569` | C075-S2 sync + canonical ladder pins + dossier CI @ HEAD |
| ONE_PASS_CAMPAIGN_PLAN_BASELINE_SHA | `7de6b41f637df1d6949985d9dc567602a074e230` | ENT-AUDIT-X one-pass B0–B7 @ 2026-11-05 |
| INTERNAL_CLOSURE_STATUS | **INTERNALLY CLOSED — AUTOMATED INTERNAL AUDIT ACCEPTED — PUBLISHED** | Post-B4 campaign; not formal ECAP closure |
| POST_B4_CORRECTIVE_PUBLICATION_SHA | `477017f15b81a645d76a2be60a17380ee9d1c14a` | Matrix C 69/6; nightly doc-only CI reuse @ `162acea31…` |
| INTERNAL_CLOSURE_PUBLICATION_SHA | `53fd582b654ab1d45cae67a4a43da1be737734d2` | Terminal report + SSOT reconcile — Ref T-IGX-002-A |
| B3_SEMANTIC_ACCEPTANCE_HEAD_SHA | `d07fbde5745569ab543ff8a57a7397023f62ce59` | 88/88 `ENT_AUDIT_X_SEMANTIC=1` @ `.tmp/session/ent-audit-x-closure/b3-acceptance-semantic.log` |
| B4_MECHANICAL_READINESS_PUBLICATION_SHA | `162acea31833a6d40d3a5e8e79638db7e443ce3e` | Terminal report B3/B4; nightly CI **42015310879**; `mechanical_readiness_check` exit 0 |
| IIAC_POLICY_ACTIVATION_SHA | `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d` | IIAC-2026-11-05.1 IN_FORCE (operator OD-01); governs ENT formal closure - T-IGX-024-A |
| ENT_FORMAL_CAMPAIGN_PLAN_BASELINE_SHA | `4d54dfa9e485329d81d0e0fa72436867458f3426` | Formal campaign F0 baseline (`plan_baseline_sha` in the closure status JSON) - historical |
| ENT_FORMAL_MASTER_PLAN_PUBLICATION_SHA | `116fb6a2ae5a40605bb569ac7685b62eb0a42539` | Formal master plan publication (status JSON); Path A / human-auditor assumptions superseded by IIAC |
| ENT_IIAC_CLOSURE_PLAN_BASELINE_SHA | `ba07ebaa34d04633bd60510f68c18346b1ef160d` | `development` tip at the start of the IIAC closure campaign (T-IGX-024-A, W0) |
| TEST_EXECUTION_SHA (historical — superseded by R-FRESH row below) | `62d1c615855a3eaf569777a10e342aacc889a128` | IIAC loop-1 corrective TES (F-01/F-02/F-09/F-10/F-11 + test repairs #409/#411/#412); certifying qual wave PASS 10/10 stages, clean tree; nightly FULL run 42491342540 success on this SHA; supersedes the 89c76eb7 row - T-IGX-024-A |
| EVIDENCE_PUBLICATION_SHA (historical — superseded by R-FRESH row below) | `163ef0c03b3cbead556499ee81cb99d640665dce` | B5 `qualification_at_62d1c615855a3eaf569777a10e342aacc889a128` (strict `verify_b5_publication_tree`, byte-preserved via .gitattributes) - T-IGX-024-A |
| TEST_EXECUTION_SHA | `b5c2481d79f955fa5f483a9288aee8de4581f953` | R-FRESH current-HEAD requal after HIST product merge (12 product paths vs `62d1c615`); certifying FULL 85/85 + FREEZE bootstrap-candidate; local qual_wave PASS; nightly `workflow_dispatch` run `42593448899` success on branch `ent-r-fresh-tes` @ this SHA; supersedes loop-1 TES — T-IGX-024-B |
| EVIDENCE_PUBLICATION_SHA | `63ff80819da5a7324257ea598e342810c1838199` | B5 `qualification_at_b5c2481d79f955fa5f483a9288aee8de4581f953` (strict `verify_b5_publication_tree`) — T-IGX-024-B |
| B6_ACCEPTANCE_PUBLICATION_SHA | `add6a6ce2015ebaa4bc38b7fadf3936f100a1522` | IIAC W2-04 phase-1: independent audit PASS_SCOPED evidence publication (manifest + pass record + certification record); pin set in W2-05 — T-IGX-024-B |

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
| EVOL_IIAC_B6_ACCEPTANCE_SHA (supersedes the EVOL B6 row above) | `1dd769331506149a1ee2f411c2bc62e85e5bbf25` |
| EVOL_PARENT_CLOSURE_SHA | `be77b3a5fb97f10a3bd2df6ee038cb92aad77eca` |
| EVOL_ENT_HANDOFF_PUBLICATION_SHA | `7a4f279fd8c0986431941c77283d872ea5b0c1ba` |
