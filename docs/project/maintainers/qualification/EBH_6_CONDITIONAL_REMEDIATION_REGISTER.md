# EBH-6 — Conditional Remediation Register (E04)

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
