# OD-EBH5-08 — Formal parent closure signature record

| Field | Value |
| --- | --- |
| **Decision ID** | OD-EBH5-08 |
| **Action** | Publish **`EBH-5 — CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)`** in roadmap §5 and stage table |
| **Status** | **OPEN — READY FOR OPERATOR SIGNATURE** |
| **Prerequisite** | **OD-EBH5-07** = **`INDEPENDENT AUDIT PASS — SCOPED`** @ `33ecf19cc020b112d9e586c87b49e819202f5d6d` (**satisfied 2026-10-28**) |

## Verification (2026-10-28 publication batch)

| Check | Result |
| --- | --- |
| Canonical operator phrase in [`EBH_5_OPERATOR_DECISION_REGISTER.md`](EBH_5_OPERATOR_DECISION_REGISTER.md) | **NOT FOUND** |
| Prior operator publication (`OPERATOR APPROVED — EBH-5 SAC FREEZE`, OD-EBH5-04*) | **Present** — does **not** substitute for OD-EBH5-08 |
| Agent instruction to “bring EBH-5 to closure” | **Insufficient** — ECAP requires explicit closure publication authority |

**Conclusion:** Do **not** record OD-EBH5-08 as **APPROVED** or publish parent **CLOSED** until operator emits the required phrase below.

## Required operator wording (canonical)

```text
OPERATOR APPROVED — EBH-5 FORMAL PARENT CLOSURE PUBLICATION (SCOPED)
```

Optional scope anchor (recommended in same decision message):

```text
INDEPENDENT_ACCEPTANCE_BASELINE_SHA = 33ecf19cc020b112d9e586c87b49e819202f5d6d
Scoped limitations per EBH_5_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md — no global FRZ PASS.
```

## On signature — publication obligations

1. Set OD-EBH5-08 **APPROVED** with date and exact phrase in operator register.
2. Update roadmap §3.0.8, stage row **EBH-5**, and §5 ledger to **`CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)`**.
3. Finalize [`EBH_5_PARENT_FORMAL_CLOSURE_RECONCILIATION.md`](EBH_5_PARENT_FORMAL_CLOSURE_RECONCILIATION.md) with **DOCUMENTATION_PUBLICATION_SHA**.
4. Set **Next mandatory stage:** **`EBH-6`** (eligible — **not** entered until operator opens wave).

## Preserved OPEN after closure (scoped)

- **51** matrix rows **NOT YET QUALIFIED**
- **Global FRZ-*** unchanged (**new global FRZ PASS = 0**)
- **EBH-6** mandatory hardcoding re-audit
- **PROD-Q** live slice where applicable
