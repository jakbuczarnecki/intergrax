# HIST campaign P8-VERDICT (formal partial closure)

| Field | Value |
|---|---|
| Record date | 2026-11-09 |
| Campaign verdict | **`HISTORICAL_CAMPAIGN_PARTIAL_EXACT_BLOCKERS`** |
| Alternative not granted | `HISTORICAL_CAMPAIGN_CLOSED_VERIFIED` (§J not satisfied) |
| Certifier | `hist-p8-cert` — [`audits/cert-p8/CERT_DECISION.md`](audits/cert-p8/CERT_DECISION.md) |
| P8-AUD-FINAL | [`audits/final-p8/AUDIT_REPORT.md`](audits/final-p8/AUDIT_REPORT.md) — PASS |
| Operator commission | [`delegations/DELEG-HIST-P8-CERT-2026-11-09.md`](delegations/DELEG-HIST-P8-CERT-2026-11-09.md) |

## What this campaign closed (scoped, evidence on publication SHA)

- Historical corrective **product batch** C1–C4 audited (C1–C3 PASS, C4 PASS_SCOPED) at integration `427646f41`.
- **P5** merge, IIAC **IN_FORCE**, `verify_iiac_policy_activation --require-in-force` exit **0**.
- **P6** Dimension A replay **POST_ACTIVATION** (`bf03550aa`, 595 passed, exit 0).
- **P7** 85-row final disposition matrix published (draft accepted; no global FRZ promotion).
- **P8** dossier independent audit **PASS**; certifier acceptance of terminal label (this record).

## External exact blockers (one next action each)

| ID | Owner | Next action |
|---|---|---|
| ENT-REACTIVATION-CONSENT | ENT owner / Operator | ENT counter-sign shared reactivation agreement or accept documented shared sequencing |
| RPL-01..04 | ARCH-FREEZE | External independent acceptance (not campaign authority) |
| REG Population E | ENT-AUDIT-X | Population E rollup on an ancestor of the activated replay SHA |
| HIST-F-BND-03B | Future ticket | Product manifest registry edge (C4 PASS_SCOPED residual) |
| HIST-F-DUR-01 | Tracked | Durability hard-code (RC-DUR; not fixed in this batch) |

Path B live rows (PRD-01, PRD-02, live halves) remain **OPEN** — [`HISTORICAL_FINAL_PATH_B_RESIDUAL_REGISTER.md`](HISTORICAL_FINAL_PATH_B_RESIDUAL_REGISTER.md).

## Predecessor readiness (ARCH-FREEZE)

**Not satisfied** for full freeze. **Satisfied only for the scoped subset:** audited product corrective batch, POST_ACTIVATION replay, and IIAC activation on `development`; ENT / RPL / Population E block global promotion.

## Authority limits (repeated)

No ENT counter-signature claimed. No `CLOSED_VERIFIED` with open RPL or Path B live obligations. No FRZ checklist **Status** promotion to global PASS.
