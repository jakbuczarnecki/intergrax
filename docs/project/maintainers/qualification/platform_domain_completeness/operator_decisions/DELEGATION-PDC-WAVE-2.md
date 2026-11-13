# DELEGATION-PDC-WAVE-2 — Operator commission for plan-approval signing (wave 2 product batches)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-WAVE-2 |
| **Stage** | PDC-X |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); precedents `DELEGATION-PDC-WAVE-A.md`, `DELEGATION-PDC-WAVE-B.md` and `DELEGATION-PDC-WAVE-1.md` |
| **Date** | 2026-11-12 |

## Commission (transcription)

1. The implementer proposed starting the D2, D5 and D6 batches together and said each plan would be signed by a delegate session. The Operator answered in chat, in English rendering: **"Start D2+D5+D6"**.
2. On re-reading the locked dependency DAG (`PDC_DEPENDENCY_DAG.md`: B-D2 and B-D5 precede B-D6, and B-D5 and B-D6 both write under the shared hotspot `intergrax/applications/_shared/`), the implementer corrected its earlier advice and started only **B-D2 (T-IGX-036-N)** and **B-D5 (T-IGX-036-O)**. The Operator has not decided to override the DAG for **B-D6 (T-IGX-036-P)**, so no plan for it exists yet and it is not part of this delegation.
3. The Operator's earlier decisions still bind this wave: product code is implemented in isolated worktrees and merged only after MCP-SERVER-X P8 is published (OD-PDC-SEQ-01), and the `ollama` regression (B-REG-P) is excluded.

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` followed by `Delegation: <this file>@<commit sha>`, in **T-IGX-036-N** and **T-IGX-036-O** only.
- The delegate writes these lines only after confirming all of the following, per ticket:
  - the plan digest matches;
  - the plan matches the corresponding batch (B-D2, B-D5) in `PDC_IMPLEMENTATION_WBS.md` and the Accepted ADR (ADR-PDC-02 and OD-PDC-SCOPE-06 = A, ADR-PDC-05);
  - the write set is the one listed in the WBS and stays inside the batch's own lane;
  - the plan creates no new registry, scheduler, engine or authority beside the existing owner;
  - the plan does not weaken any governance invariant and states that the PR is not merged before MCP-SERVER-X P8.
- **Not delegated:**
  - OD signatures;
  - IIAC activation (PDC-P7-00);
  - merges, including any merge before MCP-SERVER-X P8;
  - FRZ promotion;
  - the `ollama` regression batch B-REG-P;
  - tickets M, P..S and any other ticket.

## Separation

- The implementer session and its implementer subagents never write these lines.
- The delegate session does not implement either ticket.

## Amendment 1 — write set of B-D2 (T-IGX-036-N)

- **Operator decision** (chat, English rendering of the Polish original "Tak, rozszerz"): **"Yes, extend"**, in answer to the question whether the write set of B-D2 may be extended to wire RAG source freshness (PDC-D2-R005) end to end. The implementer had delivered the contract in `intergrax/rag/governance/` only, because retrieval and ingest were outside the write set.
- **Extension** (exactly three files, additive and optional): `intergrax/rag/ingest/ingest_pipeline.py`, `intergrax/rag/retrieval/retrieval_result.py`, `intergrax/rag/retrieval/retrieval_service.py`. No other batch writes these paths (`pdc_x_wbs.py --check`: no write conflicts).
- **Effect on the signature:** the plan digest of T-IGX-036-N changed, so the earlier `Approved:` line no longer applies. The ticket returns to `plan_pending`.
- **Delegation:** the delegate session may re-sign **T-IGX-036-N only**, under the same checks as above, additionally confirming that the plan's write set equals the amended WBS write set. T-IGX-036-O is unchanged and keeps its approval.
- **Not delegated** stays as listed above.
