# DELEGATION-PDC-WAVE-B — Operator commission for plan-approval signing (wave B)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-WAVE-B |
| **Stage** | PDC-X |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); precedents `DELEGATION-PDC-P1-P3.md` and `DELEGATION-PDC-WAVE-A.md` |
| **Date** | 2026-11-12 |

## Commission (transcription)

1. The implementer asked whether to start wave B, the repair of 12 stale gates that touches tests only. The Operator answered in chat, in English: **"Yes, start Wave B"**.
2. An automated safety check blocked the implementer's first attempt to write this record. The implementer then asked the Operator how ticket T-IGX-036-I should be approved, offering three options: the Operator writes the line, a delegation record, or a transcription. The Operator chose **"Utwórz rekord delegacji"** (English rendering: "Create the delegation record").

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` followed by `Delegation: <this file>@<commit sha>`, in **T-IGX-036-I** only.
- The delegate writes these lines only after confirming all of the following:
  - the plan digest matches;
  - the plan matches batch B-REG-T in `PDC_IMPLEMENTATION_WBS.md` and the diagnosis in `PDC_REG_DIAGNOSIS.md`;
  - the plan changes no product code;
  - the plan does not weaken any governance invariant.
- **Not delegated:**
  - OD signatures;
  - IIAC activation;
  - merges;
  - FRZ promotion;
  - the `ollama` regression batch B-REG-P;
  - any other ticket.

## Separation

- The implementer session and its implementer subagents never write these lines.
- The delegate session does not implement T-IGX-036-I.
