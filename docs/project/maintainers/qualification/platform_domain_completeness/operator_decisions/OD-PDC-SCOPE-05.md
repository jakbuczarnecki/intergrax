# OD-PDC-SCOPE-05 — Recurring scheduling (interval/cron, time zones, misfire) — implement or authorize exclusion

| Field | Value |
| --- | --- |
| **Decision ID** | OD-PDC-SCOPE-05 |
| **Stage** | PDC-X |
| **Status** | **APPROVED — Option A** |
| **Requirements** | PDC-D7-R006, PDC-D7-R007, PDC-D7-R008 |
| **Gate** | PDC-P3 (architecture lock) |

## Question

SCHED-01 certified one-shot delayed resume only; recurrence has no owner. Is recurrence part of the frozen platform contract?

## Recommended option (A)

**Implement** per ADR-PDC-07 (recurrence materializes one-shot occurrences on the existing SchedulerLedger; no second engine). Recommended: the campaign mandate lists recurrence, time-zone and misfire semantics as mandatory audit items, and scenario work after freeze would otherwise invent scheduler logic locally.

## Alternate option (B)

Authorize exclusion: the platform freezes with one-shot scheduling only; recurrence is a post-freeze extension behind the same ledger.

## Operator record

Approved: operator @ 2026-11-12T07:11:25+01:00

**Selected variant:** Option A (recommended).

Provenance: the Operator answered the P3-gate question "Accept the recommended option (A) for all P3-gate ODs?" in chat with the words "Accept all A". This record was transcribed on the Operator's instruction under T-IGX-036-E.
