# DELEGATION-PDC-WAVE-3B — Operator commission for plan-approval signing (wave 3b product batch B-D8)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-WAVE-3B |
| **Stage** | PDC-X |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); precedents `DELEGATION-PDC-WAVE-1.md`, `DELEGATION-PDC-WAVE-2.md` and `DELEGATION-PDC-WAVE-3A.md` |
| **Date** | 2026-11-13 |

## Commission (transcription)

1. Wave 3a (B-D6) is implemented as draft PR #494 (not merged). The dependency DAG (`PDC_DEPENDENCY_DAG.md`) places B-D8 after B-D6. The implementer asked whether to start B-D8 and said its plan would be signed by a delegate session. The Operator answered in chat, in English rendering: **"Start D8"** (original wording in Polish: "Start D8").
2. The Operator's earlier decisions still bind this wave: product code is implemented in isolated worktrees and merged only after MCP-SERVER-X P8 is published (OD-PDC-SEQ-01), and the `ollama` regression (B-REG-P) is excluded.

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` followed by `Delegation: <this file>@<commit sha>`, in **T-IGX-036-Q** only.
- The delegate writes these lines only after confirming all of the following:
  - the plan digest matches;
  - the plan matches batch B-D8 in `PDC_IMPLEMENTATION_WBS.md` (the write set equals the WBS write set, all 17 requirement rows are covered) and the Accepted ADR-PDC-08 with OD-PDC-SCOPE-04 = A;
  - the write set stays inside the batch's own lane; under the shared hotspots `intergrax/contracts/` and `intergrax/applications/_shared/` only the files named in the WBS are written;
  - the plan creates no global storage engine, no security logic inside stores and no new registry, scheduler or authority beside the existing owner;
  - the plan does not weaken any governance invariant and states that the PR is not merged before MCP-SERVER-X P8.
- **Not delegated:**
  - OD signatures;
  - IIAC activation (PDC-P7-00);
  - merges, including any merge before MCP-SERVER-X P8;
  - FRZ promotion;
  - the `ollama` regression batch B-REG-P;
  - tickets M..P, R, S and any other ticket.

## Separation

- The implementer session and its implementer subagents never write these lines.
- The delegate session does not implement T-IGX-036-Q.
