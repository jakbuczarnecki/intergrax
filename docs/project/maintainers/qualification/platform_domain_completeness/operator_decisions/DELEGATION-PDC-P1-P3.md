# DELEGATION-PDC-P1-P3 — Operator commission for plan-approval signing (PDC-P1..P3)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-P1-P3 |
| **Stage** | PDC-X |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved): a commission given in chat is sufficient; the delegate session must be new, dedicated to the task, and not the implementer |
| **Date** | 2026-11-12 |

## Commission (transcription)

In the PDC-X campaign session the implementer proposed to run PDC-P1 through PDC-P3 iteratively, with a dedicated delegate session writing the plan-approval lines. The Operator answered in chat, in English: **"Yes, run P1–P3"**.

## Scope of the delegation

- The delegate may write `Approved: operator-delegate @ <ISO8601>` followed by `Delegation: <this file>@<commit sha>` in **T-IGX-036-B**, **T-IGX-036-C** and **T-IGX-036-D** only.
- It may do so only after independently confirming three things:
  - the recorded `Plan digest:` matches `scripts/governance/plan_digest.py` for the attached plan;
  - each plan is consistent with the PDC-X master plan (`PLAN_T-IGX-036-A.md`);
  - the plan's scope is documentation, governance and scripts/tests only, with no product code.
- **Not delegated:**
  - any OD-PDC-* signature;
  - ADR acceptance;
  - IIAC activation;
  - merges;
  - FRZ promotion;
  - any ticket outside the three listed.

## Separation

- The implementer session that runs P1–P3 does not write these lines.
- The delegate session does not implement any of the three tickets.
