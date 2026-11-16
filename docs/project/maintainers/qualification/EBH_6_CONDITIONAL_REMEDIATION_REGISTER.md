# EBH-6 — Conditional Remediation Register (E04)

> **Currency note (EQO-A remediation, as of 2026-11-16).** Historical register pinned at baseline `202239f726dd5e3fe23bf1a68cafa61c1eb9fca7`; its trigger table (L8-L14) is unchanged. For current state read the EBH-6 records at `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`: `OD-EBH6-01` APPROVED - 2026-10-29 and `OD-EBH6-Q2D2` APPROVED - PATH B - 2026-10-29 (`EBH_6_OPERATOR_DECISION_REGISTER.md` L6, L10), the E02-OP-01 ratification (same register L11) and the roadmap row L491 (label `CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)`, quoted from the roadmap, not re-asserted here). Read the rows "Q2-D2 Path B - Not triggered (operator OPEN)" and "APP-TYP category B - all A pending ratification" as the state at the pinned baseline only. Line references (L<n>) are to the files at that SHA, before this note was added.

| Field | Value |
| --- | --- |
| **Baseline SHA** | `202239f726dd5e3fe23bf1a68cafa61c1eb9fca7` |
| **Wave status** | **NO mandatory E04 production delta** @ baseline |

| Trigger | Status @ baseline | Task | Authorization |
| --- | --- | --- | --- |
| Q2-D2 Path B | Not triggered (operator **OPEN**) | E04-Q2D2-01 | **OD-EBH6-01** |
| APP-TYP category **B** | None proposed (all **A** pending ratification) | E04-APP-* | **OD-EBH6-01** |
| E01 independent INSUFFICIENT | Not known until E01-PRE-01 | E04-BND-* | Qual-first |
| PROHIBITED > 0 | **0** @ baseline | — | — |
| Doc/register drift | Addressed Wave 0 | E04-SYNC-01 | Agent |

**Agent STOP:** no production changes without **OD-EBH6-01**. No remediation without proven trigger.
