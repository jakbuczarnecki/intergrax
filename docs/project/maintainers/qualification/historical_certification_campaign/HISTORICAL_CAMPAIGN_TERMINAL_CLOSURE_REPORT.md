# Historical campaign terminal closure report (P8)

| Field | Value |
|---|---|
| Verdict | **`HISTORICAL_CAMPAIGN_PARTIAL_EXACT_BLOCKERS`** |
| Record date | 2026-11-09 |
| START_HEAD (plan) | `fe09375177bd0fa0d9ebb661b7f9bf64e2f1ed25` |
| INTEGRATION_SHA | `427646f41de0df992cd54c0e157de8f36b560f39` |
| P4 audit publication | `468b17576b50597ebec0aaabfd54b79989655086` @ `hist-cert/aud-product-report` |
| P5 merge SHA (local branch) | `3c91694c645ff0d97d104adf0c5a564b9a4c5b3c` @ `hist-cert/p5-product-merge` |
| P6 replay SHA | `427646f41` (label `PRE_ACTIVATION_DIAGNOSTIC`) |
| PUBLICATION_SHA | **TBD** — `hist-cert/final-docs` not merged to `origin/development` |
| Operator P2 consent | [`delegations/DELEG-HIST-IIAC-P2-SUPERSESSION-2026-11-09.md`](delegations/DELEG-HIST-IIAC-P2-SUPERSESSION-2026-11-09.md) |

## Phase status

| Phase | Task | Status | Evidence |
|---|---|---|---|
| P5 | P5-RECHECK | N/A | No ENT activation covering C1–C4 |
| P5 | P5-MERGE-PRODUCT | **DONE (local)** | Merge commit `3c91694c6`; remote `development` still `f9e3a932a` until push/PR |
| P5 | P5-AUDIT-IIAC | **PASS_SCOPED** | `iiac_evidence/hist-p5-2026-11-09/` on `hist-cert/p5-product-merge` |
| P5 | P5-SIGN-ACT | **BLOCKED** | Literal `operator @` activation required |
| P5 | P5-VERIFY | **BLOCKED** | `verify_iiac_policy_activation` exit **1** (ACT-N11, 12 paths) |
| P6 | P6-REPLAY | **DONE (diagnostic)** | 595 passed, exit 0 — [`HISTORICAL_FINAL_QUALIFICATION_EVIDENCE.md`](HISTORICAL_FINAL_QUALIFICATION_EVIDENCE.md) |
| P7 | P7-RECALC | **DONE (draft)** | [`HISTORICAL_FRZ_85_FINAL_DISPOSITION_MATRIX.json`](HISTORICAL_FRZ_85_FINAL_DISPOSITION_MATRIX.json) |
| P7 | P7-PUB checklist | **BLOCKED** | P5-VERIFY + P8 certifier |
| P8 | P8-AUD-FINAL | **HANDOFF** | [`audits/final-p8/P8_FINAL_AUDIT_HANDOFF.md`](audits/final-p8/P8_FINAL_AUDIT_HANDOFF.md) |
| P8 | P8-CERT | **BLOCKED** | Needs fresh certifier after audit |
| P8 | P8-PUB merge | **BLOCKED** | Docs branch push only |

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
| IIAC-ACTIVATION | Operator | Publish new activation + `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <time>` |
| ENT-REACTIVATION-CONSENT | ENT owner | Counter-sign agreement or accept shared sequencing |
| REMOTE-MERGE | Operator / CI | Merge `hist-cert/p5-product-merge` to `development` with required CI green |
| P8-AUDIT | Independent auditor | Execute handoff; verdict on dossier |
| P8-CERT | Certifier | Accept terminal label after audit |
| RPL-01..04 | ARCH-FREEZE | External acceptance |
| REG Pop E | ENT-AUDIT-X | Population E rollup |
| HIST-F-BND-03B | Future ticket | Manifest registry edge |
| HIST-F-DUR-01 | Tracked | Durability hard-code |

## Predecessor readiness (ARCH-FREEZE)

**Not satisfied** for full freeze. Scoped product corrective batch is audited and replay-diagnosed; IIAC and ENT gates block global promotion.

## Path B

PRD-01/02 and live obligations remain OPEN — [`HISTORICAL_FINAL_PATH_B_RESIDUAL_REGISTER.md`](HISTORICAL_FINAL_PATH_B_RESIDUAL_REGISTER.md).
