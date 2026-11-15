# DECISION-PDC-F56 — Operator decision on the repair of finding F-56 (`roadmap_replay_x` scoped path check)

| Field | Value |
| --- | --- |
| **Record ID** | DECISION-PDC-F56 |
| **Stage** | PDC-X (closing wave) |
| **Finding** | F-56 of the independent audit (addendum 4), assigned to PDC-X for repair |
| **Date** | 2026-11-14 |
| **Ticket** | `governance/tickets/T-IGX-040-C.md` |

## Question put to the Operator (summary)

`tests/qualification/roadmap_replay_x/test_roadmap_replay_x_closure_counter_integrity.py::test_b6_03_scoped_disposition_valid_at_head` fails on `development` (`83d939095`) with `scoped_path_not_in_prod_diff:intergrax/integrations/providers/embedding_provider/ollama/manifest.py`. The file was added after the DUP-X pin and deleted in T-IGX-036-Z, so its net diff against the pin is empty, but the frozen PRE-ENT list still names it. Options: loosen the check minimally, or edit the frozen list.

## Decision (transcription)

The Operator answered in chat on 2026-11-14, in English rendering: **"Yes, loosen it"** (original wording in Polish: "Tak, poluzuj"). The orchestrator session relayed the decision with this precision (English rendering of the orchestrator's brief, not a verbatim Operator quote): loosen the check minimally. A scoped path that is not in the production diff is acceptable only when the file exists neither at HEAD nor at the pin (net-zero add and delete). Every other case — the path exists at HEAD but is not in the diff, a deleted path that existed at the pin, a wrong qualification head, a missing disposition — is still reported. The frozen PRE-ENT list data is not edited.

## Scope of the decision

- It authorizes the one change described in `governance/tickets/plans/PLAN_T-IGX-040-C.md` (one branch of `scoped_prod_delta_disposition_violations` plus new pinned tests) and nothing else.
- It is a gate-semantics change. It does **not** delegate the signing of the plan: the signature of `PLAN_T-IGX-040-C.md` is the Operator's, requested through the orchestrator; no `DELEGATION-PDC-*` record covers it.
- It adds no authority, changes no product path, no IIAC document and no activation record, and it is not a certification verdict.
- The change waits for the unfreeze of `development` (merge of the MCP-RC joint rebind PR); the EBH-7 identity gate is expected to stay red until the campaign's final re-pin, which the orchestrator has accepted.
