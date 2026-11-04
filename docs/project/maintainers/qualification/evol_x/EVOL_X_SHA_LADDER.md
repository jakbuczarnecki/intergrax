# EVOL-X — SHA ladder (SSOT, C-LADDER-01)

**Authority:** [EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md](../EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md) Part V-0.  
Root stub: [../EVOL_X_SHA_LADDER.md](../EVOL_X_SHA_LADDER.md) (pointer only).

| Role | SHA | Notes |
|------|-----|-------|
| PLAN_PUBLICATION_SHA | `b4fca4bc413a650b2e5a8dd8826669a9025864c5` | master plan v3 |
| W7 baseline (blockers doc) | `4393692a2e7447797c5e26408811e4b335c642f5` | pre-W7 A–D remediation |
| TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` | OD-EVOL-13; VL3 **290/290**; EVOL-M-09A contract + P03 profile; supersedes `2b165516485154583a4118ba1ecb2e25549bb5d0` |
| EVIDENCE_TECH_PUBLICATION_SHA | `1e175ff8f678d605aa841ae52362486a56ed006d` | Mode A `--technical`; P03 bundle @ `2b165516…`; M-09H NOT_MEASURED (historical TECH) |
| EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` | **AUTOMATION_FIRST_SCOPED** FINAL; M-09A MEASURED; M-09H NOT_MEASURED; OD-EVOL-13 |
| OD_EVOL_17_PUBLICATION_SHA | `7ab634dcf33bf4954a942b29bd573c9931617fb8` | OPERATOR APPROVED — LOCAL PATH A. Not a B6 acceptance SHA. |
| OD_EVOL_18_PUBLICATION_SHA | `a91b8fb653d87b4788b81beb40b4142bcaf91ef6` | **OD-EVOL-18** final operator technical acceptance. Not B6, not closure, not W7-06 pin. |
| B6_ACCEPTANCE_PUBLICATION_SHA | — | **PENDING.** OD-EVOL-17 selects local ECAP-§2.B Path A. Amendment `ECAP-P4-EVOL-SCOPE-AUTO-B6` stays `NOT_IN_FORCE`. OD-EVOL-11 `auditor_id` is not recorded. No Protocol 03 PASS. |
| INTERNAL_CERTIFICATION_STATUS | **INTERNALLY CERTIFIED — OPERATOR ACCEPTED (SCOPED)** | OD-IC-01 (2026-11-04); **not** formal B6; **not** FRZ promotion — [record](../EVOL_ENT_INTERNAL_CERTIFICATION_RECORD_2026_11_04.md) |
| INTERNAL_CERTIFICATION_PUBLICATION_SHA | `5c08e5e6c44d21998c76bd27ad1efab172440f45` | OD-IC-01 consolidated record + roadmap §5 |
| POST_IC_CONSOLIDATION_PUBLICATION_SHA | `81829712acc19ebeadb4773467b9c5642ca975e3` | ARCH-FREEZE pre-admission dossier + ladder reconciliation — [`ARCH_FREEZE_PRE_ADMISSION_READINESS_REPORT.md`](../ARCH_FREEZE_PRE_ADMISSION_READINESS_REPORT.md) |
| INTERNAL_AUDIT_FINAL_DOSSIER_PUBLICATION_SHA | `c452721855b03f1f68db59edd1dc58df2d6c5000` | Internal audit final dossier @ baseline `444ffb09…` — [`EVOL_X_INTERNAL_AUDIT_FINAL_DOSSIER_2026_11_04.md`](./EVOL_X_INTERNAL_AUDIT_FINAL_DOSSIER_2026_11_04.md); **not** B6; Ref T-IGX-011-A |
| R2_GLOBAL_AUDIT_RECONCILIATION_PUBLICATION_SHA | `167583ad1d33d9eb6e82589661e6d18aee15fa92` | R2 global audit reconciliation — orchestrator fix, canonical FRZ matrix, findings F-B6-AUD-01..03; parent `e719afe58352422b800daaaec308d412257af128` — [`EVOL_X_R2_GLOBAL_AUDIT_RECONCILIATION_FINAL.md`](./EVOL_X_R2_GLOBAL_AUDIT_RECONCILIATION_FINAL.md) |
| FORMAL_CLOSURE_CAMPAIGN_PUBLICATION_SHA | — | **PENDING** — terminal formal closure campaign docs; **does not** set B6 or FORMAL_CLOSURE SHAs — [`EVOL_X_FINAL_FORMAL_CLOSURE_CAMPAIGN_TERMINAL_REPORT.md`](./EVOL_X_FINAL_FORMAL_CLOSURE_CAMPAIGN_TERMINAL_REPORT.md) |
| PROTOCOL_03_AUTOMATED_AUDIT_PUBLICATION_SHA | `e719afe58352422b800daaaec308d412257af128` | Automated separate-agent audit record @ 2026-11-04 — **historical**; FRZ/orchestrator errata in R2 reconciliation doc |

## Historical W1 Documentation and Audit Lineage

**HISTORICAL — NOT CURRENT CERTIFYING EVIDENCE.** Preserves EVOL-X W1 documentation and independent-audit remediation chain from the pre–post-IC root ladder (`1b82324abe0f775086617f907870dd30dd11aade`). Does **not** supersede active `TEST_EXECUTION_SHA`, `EVIDENCE_FINAL_PUBLICATION_SHA`, B6, or internal-certification rows above.

### W1 commit chain (ancestor order)

| Order | Role | Full SHA | Subject (verified @ PIC-AUD-001 corrective) |
|-------|------|----------|---------------------------------------------|
| 0 | **START_HEAD** / authorized stage baseline | `5a4e922ffacabef0ee24eb676e5c8705c20ef4a1` | Pin EBH-7 FORMAL_CLOSURE_PUBLICATION_SHA and P03 identity allowlist. |
| 1 | **W1_DOCUMENTATION_SHA** | `9f9cc8b573d2629c140b3a58bf239e8a92d0b1d0` | docs(EVOL-X): publish W1 architecture freeze and qualification annexes |
| 2 | **W1_LADDER_UPDATE_SHA** | `cac5dfd91f8a98c1183e6864a93bc74ddc41b725` | docs(EVOL-X): record W1_DOCUMENTATION_SHA on ladder |
| 3 | **W1_RECONCILIATION_PUBLICATION_SHA** | `54e0de377644fe19156a2f8497efa6cd10d7fdd2` | docs(EVOL-X): W1 reconciliation — N-18/N-04, audit index, AF/OD register |
| 4 | **W1_AUDIT_REMEDIATION_SHA** | `e48cae037fc022da0a99da66314d17dd6abbe296` | docs(EVOL-X): W1 audit remediation F-01/F-02/F-03 |

**Independent audit snapshot (documentation):** pin **`e48cae037fc022da0a99da66314d17dd6abbe296`** (includes F-batch) or **`54e0de377644fe19156a2f8497efa6cd10d7fdd2`** (pre–F-batch tip) — record which in AF-8 worksheet. Index: [`EVOL_X_W1_INDEPENDENT_AUDIT_INDEX.md`](EVOL_X_W1_INDEPENDENT_AUDIT_INDEX.md).

**Ancestry:** `merge-base --is-ancestor` holds for each consecutive pair through **W1_AUDIT_REMEDIATION_SHA**; W1 chain is an ancestor of **IMPLEMENTATION_PARTIAL_SHA** and **PLAN_PUBLICATION_SHA** below.

### Implementation baseline (audit @ plan publication; not qualification evidence)

| Role | SHA | Notes |
|------|-----|--------|
| **IMPLEMENTATION_PARTIAL_SHA** | `8848857741f4b04e5b476dca3c47f7d0a60bf672` | Partial W2–W6 code on `development` at architecture audit; **not** TEST_EXECUTION_SHA |

### Plan publication (historical anchor; active pin in table above)

| Role | SHA | Notes |
|------|-----|--------|
| **PLAN_PUBLICATION_SHA** | `b4fca4bc413a650b2e5a8dd8826669a9025864c5` | Same object as **PLAN_PUBLICATION_SHA** in the active SSOT table — master plan v3 publication; not a second certifying pin. |

## Superseded (historical)

| Role | SHA | Notes |
|------|-----|-------|
| TEST_EXECUTION_SHA | `dcd7ae6e264d435994c61162e0d41aba25f71283` | W6-04 run `41320511747`; superseded by `2b165516485154583a4118ba1ecb2e25549bb5d0` (E-CORR-M09H-ORACLE) |
| EVIDENCE_TECH_PUBLICATION_SHA | `9f875fc62d1f6b2231b31c843ae0428a2d871031` | historical TECH @ `dcd7ae6e…` |
| TEST_EXECUTION_SHA | `cb407403c686f464a17d8f97e1ba0177c15c5e3b` | W7 VL3 243/243 (pre E-CORR-M09) |
| TEST_EXECUTION_SHA | `6fec9ec387089dfcee1039089ba5ad901fab39d6` | superseded by NEW_TES (F-EVOLX-P0-M09H-ATTRIBUTION-SLA-FALSE-PASS: v1 applied 900 s median to tool time); 260/260 historical |
| EVIDENCE_TECH_PUBLICATION_SHA | `43a44b492c77acdbaaf0ef632705915c9cc139f7` | superseded by NEW_TECH; technical bundle @ `6fec9ec387089dfcee1039089ba5ad901fab39d6` |
| EVIDENCE_TECH_PUBLICATION_SHA | `20a3bc63f3d34560e999905e16496a02087b75df` | technical bundle @ `cb407403` |

## W7 disposition

- **Technical evidence** never carries full certification PASS (no composite `EVOL-M-09`, no M-09H PASS).
- **Final evidence (human):** adds `evol_m09_human_drill.v2.json` @ same `test_execution_sha` (OD-EVOL-08-M1).
- **Final evidence (automation-first scoped):** `certification_profile=AUTOMATION_FIRST_SCOPED` + `evol_m09_automated_diagnostic_qualification.v1` + **EVOL-M-09A** MEASURED; **M-09H NOT_MEASURED** (OD-EVOL-13).
- **Mode B certification** @ TES: `--certification --replay-junit` mandatory.
- **OD-EVOL-18:** implementer/operator **technical work** on EVOL-X is **complete**; formal independent certification and closure SHAs remain **pending**.

## CI (W6-04)

**QUALIFIED** — run **`41337077334`** @ TEST_EXECUTION_SHA **`3c4a641b3e48a13f437e830a0562e2a80b03aabb`** (290/290, VL3 290 node ids). Historical: run `41327408027` @ `2b165516…` (288); `41320511747` @ `dcd7ae6e…` (279). See [EVOL_X_CI_QUALIFICATION_RECORD.md](./EVOL_X_CI_QUALIFICATION_RECORD.md).
