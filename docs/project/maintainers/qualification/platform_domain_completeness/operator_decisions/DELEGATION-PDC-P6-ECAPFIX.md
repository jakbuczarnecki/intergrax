# DELEGATION-PDC-P6-ECAPFIX — Operator commission for ticket T-IGX-036-X (ECAP VBN-REACHABILITY harness order)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-P6-ECAPFIX |
| **Stage** | PDC-X P6 |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); `DELEGATION-PDC-STANDING.md`; `DELEGATION-PDC-P6-REPIN.md` |
| **Date** | 2026-11-14 |

## Commission (transcription)

1. The P6 acceptance record (`PDC_P6_ACCEPTANCE_RECORD.md`, PR 532) shows ECAP FULL BLOCKED by one check, `VBN-REACHABILITY`: two `ebh_5` package C v2 cases fail because the harness registers the embedding provider of `ollama` on a cleared catalog, while batch B-REG-P (PR 505) made that registration augment the existing model-serving row. The orchestrator asked the Operator, in chat, whether to open a ticket with a test-only fix of the registration order in the harness. The Operator answered in English rendering: **"Yes, a test-only ticket"** (original wording in Polish: "Tak, test-only ticket").
2. A product change was offered as the alternative and not chosen.

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` and `Delegation: <this file>@<commit sha>` in **T-IGX-036-X** only, after confirming: the plan digest matches; the plan matches batch B-P6-F in `PDC_IMPLEMENTATION_WBS.md` (write set equal); no write path lies under `intergrax/`, `agents/`, `applications/`, `intergrax/contracts/` or `intergrax/applications/_shared/`; the plan adds no new registry, scheduler, engine, trust, policy or audit authority; no gate is weakened (the fix mirrors the real registration order and adds a negative test); the plan states that only known red classes (IIAC activation) are tolerated at merge.
- The implementer merges the batch PR under `DELEGATION-PDC-STANDING.md` authorization 3.
- Write-set amendments of T-IGX-036-X are **not** delegated: any need to touch a product file or another `ebh_5` module returns to the Operator.

## Not delegated (unchanged)

OD signatures; FRZ promotion and the PDC-X roadmap closure; IIAC activation lines other than as stated in `DELEGATION-PDC-STANDING.md`; lifting C10/C15; edits to hotspot paths not named in the plan.

## Separation

The implementer session and its implementer subagents never write `Approved:` lines. A delegate session does not implement the ticket it signs.
