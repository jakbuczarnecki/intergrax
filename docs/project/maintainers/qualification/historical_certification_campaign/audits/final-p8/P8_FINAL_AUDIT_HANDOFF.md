# P8-AUD-FINAL handoff

| Field | Value |
|---|---|
| Required | Fresh final auditor session (distinct from implementer and P4/P5 auditors) |
| Package branch | `hist-cert/final-docs` (tip after orchestrator publication commit) |
| Verdict target | PASS only if brief §8 conditions hold; else dossier FAIL with exact gaps |

## Pre-orchestrator checklist (for incoming auditor)

| # | Condition | Orchestrator pre-check |
|---|---|---|
| 1 | 85 FRZ IDs in `HISTORICAL_FRZ_85_FINAL_DISPOSITION_MATRIX.json` | Built from PR408 baseline + P3–P6 deltas |
| 2 | No global PASS claimed | Terminal label `PARTIAL_EXACT_BLOCKERS` |
| 3 | No stale SHA in evidence pointers | Integration `427646f41`, merge `3c91694c6`, P6 `427646f41` |
| 4 | Historical evidence files not mutated in place | New successor files only |
| 5 | Path B not upgraded | `HISTORICAL_FINAL_PATH_B_RESIDUAL_REGISTER.md` |
| 6 | ENT formal results untouched | Yes |
| 7 | Gates not weakened | P4 PASS / C4 PASS_SCOPED unchanged |
| 8 | IIAC verifier | **RED** until P5-SIGN-ACT (expected) |
| 9 | Remote `development` | **Not** at merge SHA until Operator merge/push of `hist-cert/p5-product-merge` |
| 10 | Independent certifier | **BLOCKED** — needs P8-AUD-FINAL PASS first |
| 11 | ENT consent | Still absent |
| 12 | RPL / Population E | Unresolved by design |
| 13 | Session IDs | Disclosed as `process_enforced` where applicable |

Assurance level: incoming audit should record `process_enforced` if subagent-backed.
