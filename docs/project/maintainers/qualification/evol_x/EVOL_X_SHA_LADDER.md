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
| B6_ACCEPTANCE_PUBLICATION_SHA | — | **PENDING.** OD-EVOL-17 (2026-11-02) selects local ECAP-§2.B Path A. Amendment `ECAP-P4-EVOL-SCOPE-AUTO-B6` stays `NOT_IN_FORCE`. OD-EVOL-11 `auditor_id` is not recorded. No Protocol 03 PASS. |

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

## CI (W6-04)

**QUALIFIED** — run **`41337077334`** @ TEST_EXECUTION_SHA **`3c4a641b3e48a13f437e830a0562e2a80b03aabb`** (290/290, VL3 290 node ids). Historical: run `41327408027` @ `2b165516…` (288); `41320511747` @ `dcd7ae6e…` (279). See [EVOL_X_CI_QUALIFICATION_RECORD.md](./EVOL_X_CI_QUALIFICATION_RECORD.md).
