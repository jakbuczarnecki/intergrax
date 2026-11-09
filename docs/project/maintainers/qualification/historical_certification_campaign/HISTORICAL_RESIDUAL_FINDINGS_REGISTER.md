# Historical campaign — tracked residual findings (HIST-owned disposition)

| Field | Value |
|---|---|
| Record date | 2026-11-09 |
| Campaign evidence SHA | `427646f41` (integration) · POST_ACTIVATION qualification `bf03550aa` |
| Rule | Residuals stay **OPEN** until a **future ticket** closes them; HIST does **not** mark them fixed in this campaign. |

## Register

| ID | Severity | Location | HIST disposition | FRZ touch | Next owner / gate |
|---|---|---|---|---|---|
| **HIST-F-BND-03B** | HIGH (tier boundary) | `intergrax/applications/_shared/product_manifest_registry.py` — dynamic Tier-0 imports of Tier-3 manifest modules | **TRACKED — OUT OF 033-A SCOPE**; formal **N/A for HIST implementation** with evidence: C4 `(b2)` seam applied; manifest registry explicitly excluded in plan §A.2 and P4 **PASS_SCOPED** | `FRZ-BND-03` remains **OPEN** (`p7_audit_verdict`: P4_PASS_SCOPED; `p7_residual_ids`: HIST-F-BND-03B) | Future product ticket (Operator / ARCH-FREEZE B7); not HIST P3–P8 |
| **HIST-F-DUR-01** | MEDIUM (qualification honesty) | Pin stores: hard-coded `is_durable=True` on KV and document backends | **TRACKED — OUT OF 032-A SCOPE**; formal **defer** per master plan §A.1 (A2 resolver only; durability module forbidden) | `FRZ-PRD-07` row carries `p7_residual_ids`: HIST-F-DUR-01; C3 **PASS** does not claim durability fix | Future SKILL-X / durability ticket; not 032-A |
| **C4 CLI identity fixture** | MEDIUM (env) | `tests/.../test_cli_verification_hardening.py` — `canonical_governance_identity_required` | **PASS_SCOPED boundary confirmed**: P4 documents 3 failed / pre-existing identity gap; **not** a 033-A regression proof | EBH / CLI portion of BND-03 scoped only | Optional env/fixture ticket; outside HIST mandate |

## Evidence pointers

- P4 product audit: [`audits/aud-product/AUDIT_REPORT.md`](audits/aud-product/AUDIT_REPORT.md) + [`findings.json`](audits/aud-product/findings.json) (`open_residuals`).
- FRZ matrix annotations: [`HISTORICAL_FRZ_85_FINAL_DISPOSITION_MATRIX.json`](HISTORICAL_FRZ_85_FINAL_DISPOSITION_MATRIX.json) (`p7_disposition_note` on BND-03 and PRD-07 rows).
- Master plan explicit residuals: [`HISTORICAL_CERTIFICATION_FINAL_MASTER_PLAN.md`](HISTORICAL_CERTIFICATION_FINAL_MASTER_PLAN.md) §A residuals list.
