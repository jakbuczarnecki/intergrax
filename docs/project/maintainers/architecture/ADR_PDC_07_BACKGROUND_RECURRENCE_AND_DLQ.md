# ADR-PDC-07 — Recurrence as one-shot occurrences on the scheduler ledger; terminal dead-letter semantics

| Field | Value |
| --- | --- |
| **Document ID** | ADR-PDC-07 |
| **Status** | **Accepted** (OD-PDC-ADR-01, 2026-11-12) |
| **Campaign** | PDC-X P3 (T-IGX-036-D) |
| **Domain** | PDC-D7 Background & Scheduling |
| **Requirements** | PDC-D7-R006..R008 (subject to OD-PDC-SCOPE-05), PDC-D7-R013, PDC-D7-R016 — see [`PDC_DOMAIN_REQUIREMENTS_MATRIX.md`](../qualification/platform_domain_completeness/PDC_DOMAIN_REQUIREMENTS_MATRIX.md) |
| **FRZ** | FRZ-EXE-06, FRZ-EXE-07, FRZ-REL-07, FRZ-REL-09, FRZ-TRC-12 |

## Context

- SCHED-01 certified one-shot delayed resume only; cron/interval/calendar/DST were NOT APPLICABLE and no stage owns recurrence.
- No DLQ exists anywhere in `intergrax/queueing`; enqueue carries only tenant_id, run_id and idempotency_key.
- GAP_AUDIT §16: Execution Engine (UER) FROZEN — extend via boundary only (PDC-P1-F05).

## Decision (proposed)

- If OD-PDC-SCOPE-05 selects implementation: a `RecurrenceRule` (interval or cron, IANA time zone, misfire policy `skip | fire_once | catch_up(n)`) materializes the next **one-shot** `ScheduledResume` on the existing `SchedulerLedger` after each occurrence. No second scheduler, no execution engine change; each occurrence is admitted through canonical execution like any scheduled resume.
- Dead-letter is a terminal task state in the queue contract (`DEAD_LETTERED` with typed reason and attempt count) reached when the retry budget is exhausted; providers may map it to native DLQs.
- The enqueue envelope carries canonical execution identity (attempt/execution id, principal, trace correlation) through `TaskRequest`.

- **OD-PDC-SCOPE-05 = A (implement):** recurrence is implemented as specified above. The conditional wording ("If OD-PDC-SCOPE-05 selects implementation") is now satisfied.

## Alternatives rejected

- Adopting a provider scheduler (Celery beat, etc.) as platform authority.
- A second task identity authority on the queue side.

## Verification

`tests/qualification/pdc_x/d7/`: DST transition, misfire policies, duplicate-fire fencing across two processes, retry-budget exhaustion to dead-letter, identity on enqueue.

## Status note

Accepted by OD-PDC-ADR-01 (Operator, 2026-11-12, recorded under T-IGX-036-E). This authorizes PDC-P4 design. Product merges still follow OD-PDC-SEQ-01 (merge after MCP-SERVER-X P8), which is pending.
