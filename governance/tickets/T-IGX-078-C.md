# T-IGX-078-C - Nexus typed exposure payload and colocated drain (FRZ-TYP-01, FRZ-TYP-03, FRZ-RPL-04)

## Owner

AF-CLOSE, lane K1-B (WORKLIST row WL-K1-B-02). Opened by GA.2R (T-IGX-078-B) as an ownership ticket.

## Scope

Product fix and tests for three RW CODE_FIX rows that no open ticket owned:

| Row | Defect to remove | Planned negative tests |
|---|---|---|
| FRZ-TYP-01 | object-typed decision exposure payload on the Nexus path; the typing ratchet is above its frozen counts | NT-WF-B01..B06 |
| FRZ-TYP-03 | the same payload, seen as an untyped seam between runtime modules | NT-WF-B01..B06 |
| FRZ-RPL-04 | the colocated drain adapter does not satisfy the drain protocol | NT-WF-C01..C05 |

The ticket restores the frozen-contract typing ratchet to 357 / 11 / 10 with one baseline under `governance/` (residual of T-IGX-051-G).

The candidate write set is listed under this ticket in `docs/project/maintainers/qualification/arch_freeze_program/rw_plan.v1.json`. It is indicative. This ticket grants no write authority: a plan with an explicit write set and its approval are required first.

## Open item

Inverting the EBH-6 Q2-D2 gate, which the typed seam depends on, is reserved to the Operator (question Q-WF-04). The DW0-03 paths do not cover these Nexus files. See `docs/project/maintainers/qualification/arch_freeze_program/continuation/RED_FIRST_GA2R_OPERATOR_ITEMS.md`, item 3.

## Plan

No plan exists yet.

## Out of scope

Every product or test edit until a plan is approved.

## Lifecycle state

plan_pending
