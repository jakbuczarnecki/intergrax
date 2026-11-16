# DELEGATION-PDC-P6-IMPACT — Operator commission for ticket T-IGX-036-V (ENT-AUDIT impact-map coverage)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-P6-IMPACT |
| **Stage** | PDC-X P6 |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); `DELEGATION-PDC-STANDING.md`; `DELEGATION-PDC-P6-REPAIR.md` |
| **Date** | 2026-11-13 |

## Commission (transcription)

1. The implementer of T-IGX-036-T reported that XD finding 8 (the ENT-AUDIT impact map covers none of the PDC-touched product modules) cannot be repaired inside that batch: `manifest.py` requires exactly 85 checks, `impact_map.py` derives prefixes only from entrypoints and a fixed table, and the regeneration script has `--emit-ssot`, not `--write`. The orchestrator asked the Operator whether item 15 becomes a separate ticket V. The Operator answered in chat, in English rendering: **"Yes, a separate ticket"** (original wording in Polish: "Tak, osobny ticket").

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` and `Delegation: <this file>@<commit sha>` in **T-IGX-036-V** only, after confirming: the plan digest matches; the plan matches batch B-P6-G in `PDC_IMPLEMENTATION_WBS.md` (write set equal); no write path lies under `intergrax/contracts/` or `intergrax/applications/_shared/`; the manifest and `manifest.py` are not in the write set; the plan adds no new check, registry, scheduler, engine, trust, policy or audit authority; no gate is weakened (coverage may only grow); the plan states that only known red classes (IIAC activation) are tolerated at merge.
- The implementer merges the batch PR under `DELEGATION-PDC-STANDING.md` authorization 3.
- Write-set amendments of T-IGX-036-V are **not** delegated: any need to touch the manifest, `manifest.py` or `changed_since.py` returns to the Operator.

## Not delegated (unchanged)

OD signatures; FRZ promotion and the PDC-X roadmap closure; IIAC activation lines other than as stated in `DELEGATION-PDC-STANDING.md`; lifting C10/C15; edits to hotspot paths not named in the plan.

## Separation

The implementer session and its implementer subagents never write `Approved:` lines. A delegate session does not implement the ticket it signs.
