# DELEGATION-PDC-WAVE-A — Operator commission for plan-approval signing (wave A)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-WAVE-A |
| **Stage** | PDC-X |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (summarized in `governance/tickets/README.md` §Approved); precedent `DELEGATION-PDC-P1-P3.md` |
| **Date** | 2026-11-12 |

## Commission (transcription; English rendering of the Operator's Polish chat messages)

1. On the PDC-X progress board request, the Operator wrote that tasks which can run in parallel may be delegated to subagents.
2. After PRs #464 and #465 were merged, the Operator wrote: **"You can start wave A."**

Wave A is defined on the board as three parts:

- PDC-P4 implementation design;
- the red-gate diagnosis (read-only);
- the PDC-P5-DOC documentation reconciliations.

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` followed by `Delegation: <this file>@<commit sha>`, in **T-IGX-036-G** and **T-IGX-036-H** only.
- The delegate writes these lines only after independently confirming three things:
  - the plan digest matches the plan;
  - the plan is consistent with the PDC-X master plan and the locked ADRs;
  - the plan touches no product code.
- **Not delegated:** OD signatures, ADR changes, IIAC activation, merges, FRZ promotion, and any other ticket.

## Separation

- The implementer session, and any implementer subagent it runs for G or H, never writes these lines.
- The delegate session implements neither ticket.
