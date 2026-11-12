# DELEGATION-PDC-WAVE-1 — Operator commission for plan-approval signing (wave 1 product batches)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-WAVE-1 |
| **Stage** | PDC-X |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); precedents `DELEGATION-PDC-WAVE-A.md` and `DELEGATION-PDC-WAVE-B.md` |
| **Date** | 2026-11-12 |

## Commission (transcription)

1. The implementer announced wave 1 of PDC-P5 (batches B-D1, B-D3, B-D7, tickets T-IGX-036-J, -K, -L) and stated that each plan would be signed by a delegate session, not by the implementer. The Operator answered in chat, in English rendering: **"Start Wave 1"** (original wording in Polish: "Start Fali 1").
2. The Operator's earlier decisions still bind this wave: product code is implemented in isolated worktrees and merged only after MCP-SERVER-X P8 is published (OD-PDC-SEQ-01), and the `ollama` regression (B-REG-P) is excluded.

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` followed by `Delegation: <this file>@<commit sha>`, in **T-IGX-036-J**, **T-IGX-036-K** and **T-IGX-036-L** only.
- The delegate writes these lines only after confirming all of the following, per ticket:
  - the plan digest matches;
  - the plan matches the corresponding batch (B-D1, B-D3, B-D7) in `PDC_IMPLEMENTATION_WBS.md` and the Accepted ADR (ADR-PDC-01, -03, -07);
  - the write set is the one listed in the WBS and stays inside the batch's own lane (no writes in `intergrax/applications/_shared/` or other batches' packages);
  - the plan creates no new registry, scheduler or authority beside the existing owner;
  - the plan does not weaken any governance invariant and states that the PR is not merged before MCP-SERVER-X P8.
- **Not delegated:**
  - OD signatures;
  - IIAC activation (PDC-P7-00);
  - merges, including any merge before MCP-SERVER-X P8;
  - FRZ promotion;
  - the `ollama` regression batch B-REG-P;
  - tickets M..S and any other ticket.

## Separation

- The implementer session and its implementer subagents never write these lines.
- The delegate session does not implement any of the three tickets.
