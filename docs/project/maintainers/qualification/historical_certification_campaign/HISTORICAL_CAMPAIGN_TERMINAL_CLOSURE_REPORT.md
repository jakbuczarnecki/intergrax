# Historical campaign terminal closure report (P8)

| Field | Value |
|---|---|
| Verdict | **`HISTORICAL_CAMPAIGN_PARTIAL_EXACT_BLOCKERS`** |
| Record date | 2026-11-09 |
| START_HEAD (plan) | `fe09375177bd0fa0d9ebb661b7f9bf64e2f1ed25` |
| INTEGRATION_SHA | `427646f41de0df992cd54c0e157de8f36b560f39` |
| P4 audit publication | `468b17576b50597ebec0aaabfd54b79989655086` @ `hist-cert/aud-product-report` |
| P5 merge SHA | `3c91694c645ff0d97d104adf0c5a564b9a4c5b3c` (merged to `development`) |
| P5 activation effective SHA | `8045915a353bb04ea5a290f9ef8b6def33dee96d` (`IIAC_POLICY_ACTIVATION_RECORD`, IN_FORCE) |
| P6 replay SHA (authoritative) | `bf03550aa95ce239a63542916331c6a648617e33` — label **`POST_ACTIVATION`** |
| P6 diagnostic (superseded) | `427646f41` — `PRE_ACTIVATION_DIAGNOSTIC` lineage only |
| PUBLICATION_SHA | `ff172a1e004c1b9796e1034d53cd19f4a7f358c1` (`hist-cert/p8-certifier-closeout`; prior docs merge `b5c2481d7`) |
| Remote `origin/development` | **Pending push** — target tip `ff172a1e004c1b9796e1034d53cd19f4a7f358c1` |
| Operator P2 consent | [`delegations/DELEG-HIST-IIAC-P2-SUPERSESSION-2026-11-09.md`](delegations/DELEG-HIST-IIAC-P2-SUPERSESSION-2026-11-09.md) |
| Operator P8 cert commission | [`delegations/DELEG-HIST-P8-CERT-2026-11-09.md`](delegations/DELEG-HIST-P8-CERT-2026-11-09.md) |
| P8 final audit | [`audits/final-p8/AUDIT_REPORT.md`](audits/final-p8/AUDIT_REPORT.md) — **PASS** |
| P8 certifier | [`audits/cert-p8/CERT_DECISION.md`](audits/cert-p8/CERT_DECISION.md) — **ACCEPT** `HISTORICAL_CAMPAIGN_PARTIAL_EXACT_BLOCKERS` |
| P8-VERDICT | [`HISTORICAL_CAMPAIGN_P8_VERDICT.md`](HISTORICAL_CAMPAIGN_P8_VERDICT.md) |

## Phase status

| Phase | Task | Status | Evidence |
|---|---|---|---|
| P5 | P5-RECHECK | N/A | No ENT activation covering C1–C4 |
| P5 | P5-MERGE-PRODUCT | **DONE** | Merge `3c91694c6` on `development` |
| P5 | P5-AUDIT-IIAC | **PASS_SCOPED** (pre-activation package) | `iiac_evidence/HIST-P5-2026-11-09/` |
| P5 | P5-SIGN-ACT | **DONE** | `IIAC_OD_ACTIVATION_IIAC-OD-HIST-P5-2026-11-09.md` + record `IN_FORCE` |
| P5 | P5-VERIFY | **DONE** | `verify_iiac_policy_activation --require-in-force` exit **0** |
| P6 | P6-REPLAY | **DONE** | POST_ACTIVATION 595 passed, exit 0 — [`HISTORICAL_FINAL_QUALIFICATION_EVIDENCE.md`](HISTORICAL_FINAL_QUALIFICATION_EVIDENCE.md) |
| P7 | P7-RECALC | **DONE (draft)** | [`HISTORICAL_FRZ_85_FINAL_DISPOSITION_MATRIX.json`](HISTORICAL_FRZ_85_FINAL_DISPOSITION_MATRIX.json) |
| P7 | P7-PUB checklist | **DONE** | Scoped FRZ-BND-03 / FRZ-BND-04 evidence annotations only (certifier `hist-p8-cert`) |
| P8 | P8-AUD-FINAL | **PASS** | [`audits/final-p8/AUDIT_REPORT.md`](audits/final-p8/AUDIT_REPORT.md) |
| P8 | P8-CERT | **DONE** | [`audits/cert-p8/CERT_DECISION.md`](audits/cert-p8/CERT_DECISION.md) |
| P8 | P8-PUB merge | **DONE** | Docs package `b5c2481d7`; closeout branch merges certifier records |
| P8 | P8-VERDICT | **DONE** | [`HISTORICAL_CAMPAIGN_P8_VERDICT.md`](HISTORICAL_CAMPAIGN_P8_VERDICT.md) |

## Product batch verdicts (P4, unchanged)

| Lane | Verdict |
|---|---|
| C1 | PASS |
| C2 | PASS |
| C3 | PASS |
| C4 | PASS_SCOPED |

## Exact blockers (next action each)

| ID | Owner | Next action |
|---|---|---|
| ENT-REACTIVATION-CONSENT | ENT owner | Counter-sign agreement or accept shared sequencing |
| RPL-01..04 | ARCH-FREEZE | External acceptance |
| REG Pop E | ENT-AUDIT-X | Population E rollup |
| HIST-F-BND-03B | Future ticket | Manifest registry edge |
| HIST-F-DUR-01 | Tracked | Durability hard-code |

## Predecessor readiness (ARCH-FREEZE)

**Not satisfied** for full freeze. Scoped product corrective batch is audited, IIAC is IN_FORCE, and Dimension A replay is POST_ACTIVATION green; ENT/RPL gates block global promotion.

## Path B

PRD-01/02 and live obligations remain OPEN — [`HISTORICAL_FINAL_PATH_B_RESIDUAL_REGISTER.md`](HISTORICAL_FINAL_PATH_B_RESIDUAL_REGISTER.md).
