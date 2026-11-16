# CERT_DECISION — HIST P8-CERT (campaign terminal label)

| Field | Value |
|---|---|
| Certifier role | `designated certifying session` / HIST P8-CERT |
| Session name | `hist-p8-cert` |
| Assurance | `process_enforced` (Cursor session; distinct from P8-AUD-FINAL auditor) |
| Independence | Not P8-AUD-FINAL session `a91e4f28-6c3d-4b8e-a0f5-2e7d9c1b4f63`; not product-batch implementer; writes **no** IIAC `operator @` activation line |
| Worktree | `D:/Projekty/_wt/intergrax/hist-p8-cert` |
| Dossier audit | [`../final-p8/AUDIT_REPORT.md`](../final-p8/AUDIT_REPORT.md) — **PASS** |
| Integration evidence | `427646f41de0df992cd54c0e157de8f36b560f39` (P4); P6 POST_ACTIVATION `bf03550aa95ce239a63542916331c6a648617e33` |
| Pre-closeout pin | `878e2e384e47cafffe39162f59e3971d83141783` (`origin/development` at certifier start) |

## Verdict

**ACCEPT** the campaign terminal label:

**`HISTORICAL_CAMPAIGN_PARTIAL_EXACT_BLOCKERS`**

This is **not** `HISTORICAL_CAMPAIGN_CLOSED_VERIFIED`. This acceptance is **not** FRZ promotion, **not** global architecture freeze, and **not** ENT formal closure.

## Evidence per check

1. **Delegation.** Operator consent 2026-11-09 transcribed in [`delegations/DELEG-HIST-P8-CERT-2026-11-09.md`](../../delegations/DELEG-HIST-P8-CERT-2026-11-09.md). Scope: P8-CERT + gated P7-PUB / P8-PUB / P8-VERDICT docs only. PASS (process_enforced transcription).
2. **P8-AUD-FINAL.** Independent auditor PASS on dossier; thirteen handoff checks per audit report; condition 10 (certifier) was OPEN at audit time and is satisfied by this session. PASS.
3. **Master plan §J map.** IIAC verifier green; C1–C4 audited; P6 POST_ACTIVATION exit 0; Path B OPEN; RPL / REG Population E external; ENT counter-sign absent. Legal label is **PARTIAL_EXACT_BLOCKERS** only. PASS.
4. **85-row matrix.** `HISTORICAL_FRZ_85_FINAL_DISPOSITION_MATRIX.json` present with 85 historical rows; no global PASS claimed. PASS.
5. **Role separation.** Certifier distinct from auditor session id above; no product-path commit by this session. PASS.

## Residual risks

- Delegation is MASTER transcription of Operator Polish chat; assurance `process_enforced`.
- P7-PUB checklist annotations cite HIST product-batch evidence; **Status column stays OPEN** (no FRZ promotion).
- ENT-REACTIVATION-CONSENT remains Operator P2 supersession only until ENT counter-sign.

## Action taken

Published this decision, [`HISTORICAL_CAMPAIGN_P8_VERDICT.md`](../../HISTORICAL_CAMPAIGN_P8_VERDICT.md), and terminal-report phase updates. Scoped P7-PUB annotations on FRZ-BND-03 / FRZ-BND-04 only (P7-ARCH).

## Approved

Approved: operator-delegate hist-p8-cert @ 2026-11-09T09:40:00+01:00 | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-P8-CERT-2026-11-09.md@8169b342c97677f6165e068ed61dd663eb563ba0

Accepted: HISTORICAL_CAMPAIGN_PARTIAL_EXACT_BLOCKERS @ 2026-11-09T09:40:00+01:00 | Certifier: hist-p8-cert | Audit: audits/final-p8/AUDIT_REPORT.md (PASS, auditor a91e4f28-6c3d-4b8e-a0f5-2e7d9c1b4f63)
