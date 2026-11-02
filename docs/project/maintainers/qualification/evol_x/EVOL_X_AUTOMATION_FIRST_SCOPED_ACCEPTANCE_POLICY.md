# EVOL-X — Automation-first scoped acceptance (APPROVED)

**Status:** **OPERATOR APPROVED** @ 2026-11-01 · **OD-EVOL-13**

**Decision ID:** **OD-EVOL-13**

## Verbatim authorization

«**OPERATOR APPROVED — EVOL-X AUTOMATION-FIRST SCOPED ACCEPTANCE.** For **scoped EVOL-X closure only**, a correctly qualified automated C1–C3 diagnostic evaluation (`EVOL-M-09A`, artifact `evol_m09_automated_diagnostic_qualification.v1`, `certification_profile=AUTOMATION_FIRST_SCOPED`) is the accepted substitute for the **interactive M-09H human drill**. This does **not** mean the human drill passed. **`EVOL-M-09H` remains NOT_MEASURED** unless a separate genuine human evaluation is performed. Human attribution SLA (median ≤900 s) applies **only** to human `attribution_seconds` under **OD-EVOL-08-M1**; it MUST NOT be applied to machine tool timing. Independent **B6** acceptance is still required; implementers/agents MUST NOT self-certify.»

## Approved scope

- P03 Mode A **`AUTOMATION_FIRST_SCOPED`** FINAL evidence path.
- Metric **`EVOL-M-09A`** (distinct from **M-09H**).
- Typed artifact **`evol_m09_automated_diagnostic_qualification.v1`** with **`qualification_policy_id=OD-EVOL-13`**.

## Not approved

- Fabricated human worksheets, `attribution_seconds`, or `evol_m09_human_drill.v2` without human execution.
- Automatic **B6** sign-off or **OD-EVOL-11** auditor impersonation.
- Global **FRZ-*** promotion beyond scoped EVOL-X evidence rules.
- Production rollback hardening or unrelated roadmap stages.

## Relationship to prior decisions

- **OD-EVOL-08 / OD-EVOL-08-M1:** preserved; human drill semantics unchanged for **`HUMAN_FULL`** profile.
- **OD-EVOL-11:** independent auditor still required before formal B6 acceptance.

Supersedes the non-binding draft in [EVOL_X_E2_PROPOSED_M09H_AUTOMATION_SUBSTITUTION_AMENDMENT.md](./e2_m09h_human_drill_prep/EVOL_X_E2_PROPOSED_M09H_AUTOMATION_SUBSTITUTION_AMENDMENT.md) for implementation purposes only (draft header retained as historical).
