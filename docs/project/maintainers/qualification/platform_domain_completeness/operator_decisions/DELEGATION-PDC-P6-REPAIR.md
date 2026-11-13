# DELEGATION-PDC-P6-REPAIR — Operator commission for ticket T-IGX-036-T (integration repairs)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-P6-REPAIR |
| **Stage** | PDC-X P6 |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); `DELEGATION-PDC-STANDING.md` |
| **Date** | 2026-11-13 |

## Commission (transcription)

1. All P5 batches are merged on `development`. The implementer reported that the integration repairs found in the lanes (for example `tzdata`, D3 capability flags, D2 freshness, D8 composition wiring) need product changes outside every lane write set, and that standing authorization 4 covers only plans M and S. The implementer asked for consent to a separate ticket **T-IGX-036-T**. The Operator answered in chat, in English rendering: **"Consent to ticket T, go on"** (original wording in Polish: "Zgoda na ticket T, jedź dalej").

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` and `Delegation: <this file>@<commit sha>` in **T-IGX-036-T** only, after confirming: the plan digest matches; the plan matches batch B-P6-R in `PDC_IMPLEMENTATION_WBS.md` (write set equal); the plan adds no new registry, scheduler, engine, trust, policy or audit authority; no gate is weakened; the plan states that only known red classes (IIAC activation) are tolerated at merge.
- **Write-set amendments of T-IGX-036-T** may be re-signed by a delegate when all of these hold: each added path is needed to close a finding of an XD run or of a lane report; the path is not under `intergrax/contracts/` or `intergrax/applications/_shared/` unless the plan already names that path; no gate is weakened (a gate may be strengthened or given one classification entry); no new authority and no new functional scope. Anything else returns to the Operator.
- The implementer merges the batch PR under `DELEGATION-PDC-STANDING.md` authorization 3.

## Not delegated (unchanged)

OD signatures; FRZ promotion and the PDC-X roadmap closure; IIAC activation lines other than as stated in `DELEGATION-PDC-STANDING.md`; lifting C10/C15; edits to hotspot paths not named in the plan.

## Separation

The implementer session and its implementer subagents never write `Approved:` lines. A delegate session does not implement the ticket it signs.
