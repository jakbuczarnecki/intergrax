# DELEGATION-PDC-P6-MODEL — Operator commission for ticket T-IGX-036-U (model binding)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-P6-MODEL |
| **Stage** | PDC-X P6 |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); `DELEGATION-PDC-STANDING.md`; `DELEGATION-PDC-P6-REPAIR.md` |
| **Date** | 2026-11-13 |

## Commission (transcription)

1. The B-P6 report (T-IGX-036-S) recorded XD-01 findings 9 (an activated agent revision pins no model profile) and 10 (no required-model-features declaration under `intergrax/contracts/agent_contract_meta.py`, and no production caller of `LLMAdapterRegistry.require_capabilities`). The implementer left them out of T-IGX-036-T because they add functional scope and touch a hotspot, and asked the Operator what to do. The Operator answered in chat, in English rendering: **"A separate ticket"** (original wording in Polish: "Osobny ticket").

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` and `Delegation: <this file>@<commit sha>` in **T-IGX-036-U** only, after confirming: the plan digest matches; the plan matches batch B-P6-M in `PDC_IMPLEMENTATION_WBS.md` (write set equal); the hotspot paths in the write set are exactly `intergrax/contracts/agent_contract_meta.py` and `intergrax/applications/_shared/llm_resolver.py` and each change there is additive with a safe default; the plan adds no new registry, scheduler, engine, trust, policy or audit authority; no gate is weakened; the plan states that only known red classes (IIAC activation) are tolerated at merge.
- The implementer merges the batch PR under `DELEGATION-PDC-STANDING.md` authorization 3.
- Write-set amendments of T-IGX-036-U are **not** delegated: any further hotspot path returns to the Operator.

## Not delegated (unchanged)

OD signatures; FRZ promotion and the PDC-X roadmap closure; IIAC activation lines other than as stated in `DELEGATION-PDC-STANDING.md`; lifting C10/C15; edits to hotspot paths not named in the plan.

## Separation

The implementer session and its implementer subagents never write `Approved:` lines. A delegate session does not implement the ticket it signs.
