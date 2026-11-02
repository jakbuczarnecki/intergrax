# A3 — Adversarial, contract and safety audit

**Agent:** A3 · **Profile:** `AUTOMATION_FIRST_SCOPED` (OD-EVOL-13)

## Profile isolation

| Check | Status | Evidence |
|-------|--------|----------|
| `MANIFEST.certification_profile` | **PASS** | `AUTOMATION_FIRST_SCOPED` @ FINAL |
| Human drill artifact absent | **PASS** | No `evol_m09_human_drill.v2.json` in `artifact_paths` |
| M-09H not conflated with M-09A | **PASS** | `evol_metrics.v1.json`: `EVOL-M-09H` = **NOT_MEASURED**; `EVOL-M-09A` = **MEASURED** PASS |
| HUMAN_FULL false PASS path | **PASS** | Negative M-09H cases in VL3 remain for **HUMAN_FULL** bundles; scoped FINAL does not claim human attribution |
| `--technical` on scoped FINAL | **PASS (guard)** | B6 package STOP rule; TECH lineage `1e175ff8…` separate from active FINAL |

## Fail-closed / machine evidence

| Check | Status | Evidence |
|-------|--------|----------|
| C1–C3 automated diagnostic JSON | **PASS** | `m09_automated_diagnostic/C1..C3.json` + `evol_m09_automated_diagnostic_qualification.v1.json` (`qualification_policy_id=OD-EVOL-13`) |
| AC-AUTO / adversarial negatives | **PASS** | VL3 gate tests @ TES (including M-09 negative parametrization for human path — N/A attribution for scoped FINAL) |
| Operator decision binding | **PASS** | [EVOL_X_OPERATOR_DECISION_REGISTER_W1.md](../EVOL_X_OPERATOR_DECISION_REGISTER_W1.md) OD-EVOL-13 **OPERATOR APPROVED** |

## FRZ-EVOL-07 / production safety

| Claim | Allowed | Evidence |
|-------|---------|----------|
| Qualification-layer deploy assessor (S7 / DM) | **Scoped behavioral PASS** | `evol_frz_evidence.v1.json` row FRZ-EVOL-07 + footnote |
| Production `ActivationService` rollback | **NOT CLAIMED** | OD-EVOL-05b Option 1; Part X master plan |
| W4-04 production port | **NOT AUTHORIZED** | E3 disposition |

## Disposition

**A3 — TECHNICAL PASS** (scoped contract honored; no human M-09H PASS; no production rollback claim).
