# DELEGATION-PDC-WAVE-3A — Operator commission for plan-approval signing (wave 3a product batch B-D6)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-WAVE-3A |
| **Stage** | PDC-X |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); precedents `DELEGATION-PDC-WAVE-1.md` and `DELEGATION-PDC-WAVE-2.md` |
| **Date** | 2026-11-12 |

## Commission (transcription)

1. The Operator held B-D6 until the end of wave 2, in line with the dependency DAG (`PDC_DEPENDENCY_DAG.md`: B-D2 and B-D5 precede B-D6). Chat answer, English rendering: **"D6 will wait for the end of wave 2"** (Polish original: "D6 poczeka na koniec Fali 2").
2. Wave 2 is implemented (draft PRs #489 for B-D2 and #488 for B-D5, not merged). The Operator then answered in chat, in English rendering: **"Start D6"** (original wording in Polish: "Start D6").
3. The Operator's earlier decisions still bind this wave: product code is implemented in isolated worktrees and merged only after MCP-SERVER-X P8 is published (OD-PDC-SEQ-01), and the `ollama` regression (B-REG-P) is excluded.

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` followed by `Delegation: <this file>@<commit sha>`, in **T-IGX-036-P** only.
- The delegate writes these lines only after confirming all of the following:
  - the plan digest matches;
  - the plan matches batch B-D6 in `PDC_IMPLEMENTATION_WBS.md` (the write set equals the WBS write set, every requirement row is covered) and the Accepted ADR-PDC-06 with OD-PDC-SCOPE-07 = A;
  - the write set stays inside the batch's own lane and writes under the shared hotspot `intergrax/applications/_shared/` only the four files named in the WBS;
  - the plan creates no new policy engine, identity, trust or audit authority beside the existing owner;
  - the plan does not weaken any governance invariant and states that the PR is not merged before MCP-SERVER-X P8.
- **Not delegated:**
  - OD signatures;
  - IIAC activation (PDC-P7-00);
  - merges, including any merge before MCP-SERVER-X P8;
  - FRZ promotion;
  - the `ollama` regression batch B-REG-P;
  - tickets M, N, O, Q..S and any other ticket.

## Separation

- The implementer session and its implementer subagents never write these lines.
- The delegate session does not implement T-IGX-036-P.
