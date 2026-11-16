# DELEGATION-PDC-WAVE-3C — Operator commission for plan-approval signing (wave 3c product batch B-D4)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-WAVE-3C |
| **Stage** | PDC-X |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); precedents `DELEGATION-PDC-WAVE-1.md`, `DELEGATION-PDC-WAVE-2.md`, `DELEGATION-PDC-WAVE-3A.md` and `DELEGATION-PDC-WAVE-3B.md` |
| **Date** | 2026-11-13 |

## Commission (transcription)

1. The dependency DAG (`PDC_DEPENDENCY_DAG.md`) places B-D4 after B-D7 and B-D8. The implementer proposed preparing and signing the B-D4 plan (docs only) while B-D8 is still being implemented, and rehearsing the integration of the draft branches without merging. The Operator answered in chat, in English rendering: **"Start both"** (original wording in Polish: "Start obu").
2. Only the **plan** is commissioned now. The implementation of B-D4 starts after B-D8 is implemented, because the two batches share the hotspot `intergrax/applications/_shared/` and the WBS allows one batch per wave on a hotspot.
3. The Operator's earlier decisions still bind this wave: product code is implemented in isolated worktrees and merged only after MCP-SERVER-X P8 is published (OD-PDC-SEQ-01), and the `ollama` regression (B-REG-P) is excluded.

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` followed by `Delegation: <this file>@<commit sha>`, in **T-IGX-036-R** only.
- The delegate writes these lines only after confirming all of the following:
  - the plan digest matches;
  - the plan matches batch B-D4 in `PDC_IMPLEMENTATION_WBS.md` (the write set equals the WBS write set, all 9 requirement rows are covered; read the whole section) and the Accepted ADR-PDC-04 with OD-PDC-SCOPE-02 = A;
  - the write set stays inside the batch's own lane; under the shared hotspot `intergrax/applications/_shared/` only the two files named in the WBS are written;
  - the plan creates no hosting-owned metrics backend, no second supervisor or composition root and no new authority beside the existing owner;
  - the plan does not weaken any governance invariant and states that the PR is not merged before MCP-SERVER-X P8.
- **Not delegated:**
  - OD signatures;
  - IIAC activation (PDC-P7-00);
  - merges, including any merge before MCP-SERVER-X P8;
  - FRZ promotion;
  - the `ollama` regression batch B-REG-P;
  - tickets M..Q, S and any other ticket.

## Separation

- The implementer session and its implementer subagents never write these lines.
- The delegate session does not implement T-IGX-036-R.
