# HARNESS-RESIDUAL / SCHED-01 — Current-HEAD Scheduling Enterprise Semantics Recertification

**Status:** READY FOR AUDIT (Cursor session evidence — not closure)  
**Baseline START_HEAD / AUDITED_HEAD:** `3a9b58e6fa1e14ddc9786bd9a447a1d76552d196` (`development`)  
**Scope:** One-shot delayed resume + human-timeout ledger path; UTC instant scheduling only.

## Stage reconciliation

| Workstream | State |
| ---------- | ----- |
| EBH-4 | CLOSED (independently accepted) |
| HARNESS-W7 / HOST-01 | CLOSED @ `5d5442dae7769671a7532a973428c8203b492ec6` |
| HARNESS-W8 / BG-01 | CLOSED @ `3a9b58e6fa1e14ddc9786bd9a447a1d76552d196` (BG qualification record) |
| **SCHED-01** | **Current residual Harness workstream** (**this record**) |
| HARNESS-FINAL | NOT ENTERED (blocked until SCHED-01 independent acceptance + residual reconciliation) |
| CE-01 / PLUG-01 / ART-01 | NOT assumed closed |

**Why current residual = SCHED-01:** Canonical Harness SSOT order HOST-01 → BG-01 → SCHED-01. BG-01/W8 independently accepted on baseline SHA; scheduling semantics are the next unresolved frozen-scope Harness slice before HARNESS-FINAL.

## Scheduling graph (current HEAD)

```text
ScheduledResume
      ↓
ScheduledResumePersistence (claim_due / complete_claim / cancel)
      ↓
LongRunningScheduler.tick(now=...)
      ↓
HostTaskResumeExecutor
      ↓
HostTaskExecutionPort.execute(...)
      ↓
Governance / execution admission
      ↓
Execution Engine
      ↓
private Nexus
```

**WHEN-only:** scheduler evaluates due time, claims occurrence, invokes resume executor.  
**Not scheduler-owned:** authorization, ExecutionId minting, Nexus/backend choice, execution terminal truth.

Production harness path: `wire_harness_host_long_running_scheduler` → `wire_long_running_scheduler_with_host_execution` → `HostTaskResumeExecutor(host_execution)`.

## Closed-world scheduling inventory (§6)

| Concern | Contract / type | Semantic owner | Persistence owner | Execution owner | Current path | Status |
| ------- | --------------- | -------------- | ----------------- | --------------- | ------------ | ------ |
| schedule creation | `ScheduledResume`, `LongRunningScheduler.schedule_resume` | LongRunningScheduler API | `ScheduledResumePersistence.schedule` | n/a | `scheduler.schedule_resume` → store | CURRENT |
| due detection | `claim_due(before_utc_iso=...)` | LongRunningScheduler | `ScheduledResumePersistence` | n/a | `_process_due_schedules` | CURRENT |
| claim | `ScheduledResumeClaim` | persistence contract | provider `claim_due` atomic | n/a | store `claim_due` | CURRENT |
| lease | `owner_id`, `lease_expires_at_utc` on entry | persistence + claim | store row | n/a | claim sets lease | CURRENT |
| fence | `fence` on `ScheduledResume` | persistence contract | `complete_claim` validates | n/a | SCHED-Q6 tests | CURRENT |
| cancellation | `ScheduledResumePersistence.cancel` | scheduler/store contract | store | n/a | PENDING cancel; RUNNING reject | CURRENT |
| misfire | one-shot overdue | LongRunningScheduler poll | durable PENDING row | n/a | fire once on first due poll | CURRENT |
| uncertain claim | `ScheduledResumeStatus.UNCERTAIN` | persistence policy | lease expiry on RUNNING | n/a | no auto re-execute | CURRENT |
| resume dispatch | `TaskResumeExecutor` | HostTaskResumeExecutor | n/a | `HostTaskExecutionPort` | `_execute_resume` → port | CURRENT |
| terminal handling | schedule COMPLETED vs execution terminal | scheduler completes *occurrence* | `complete_claim` | `ExecutionTerminalService` (optional gate) | `_can_resume_checkpoint` | CURRENT |
| custom store | `ScheduledResumePersistence` SPI | scheduler core | plugin impl | n/a | `_MemoryScheduleStore` in SCHED-Q10 | CURRENT |

## Occurrence identity

- `ScheduledResume.schedule_id` = one logical one-shot occurrence (stable through write/read/claim/complete).
- Not `ExecutionId` / `RunId` / `AttemptId` / `TaskId`.
- Resume identity from checkpoint via `execution_identity_from_checkpoint` in `HostTaskResumeExecutor`.

## SCHED-Q1..Q15

| Q | Verdict | Evidence |
| --- | ------- | -------- |
| SCHED-Q1 | PASS | `test_sched_q1_due_occurrence_invokes_host_execution_port` |
| SCHED-Q2 | PASS | `test_sched_q2_schedule_survives_store_reopen` |
| SCHED-Q3 | PASS | `test_sched_q3_tenant_mismatch_blocks_resume` |
| SCHED-Q4 | PASS | `test_sched_q4_schedule_id_stable_across_reads` |
| SCHED-Q5 | PASS | `test_atomic_due_claim_exactly_one_winner`, `test_two_schedulers_single_resume_call` |
| SCHED-Q6 | PASS | `test_fenced_completion_rejected`, `test_active_claim_blocks_second_owner` |
| SCHED-Q7 | PASS | `test_sched_q7_misfire_late_due_fires_once` |
| SCHED-Q8 | PASS | `test_sched_q8_scheduler_retry_preserves_checkpoint_identity` |
| SCHED-Q9 | PASS | `test_expired_running_becomes_uncertain_no_second_resume` |
| SCHED-Q10 | PASS | `test_sched_q10_custom_schedule_store_provider` |
| SCHED-Q11 | PASS | `test_sched_q11_fake_clock_boundaries` |
| SCHED-Q12 | PASS | `test_sched_q12_cancelled_pending_not_dispatched`, `test_cancel_active_running_rejected` |
| SCHED-Q13 | PASS | `test_sched_q13_production_wiring_resumes_through_host_task_execution_port` |
| SCHED-Q14 | PASS | `test_sched_q14_scheduling_core_import_layer_gate` |
| SCHED-Q15 | PASS | `test_sched_q15_no_alternate_execution_engine_in_scheduler_core` |

Catalog: `tests/qualification/sched_01/catalog.py`.

## Tenant 16Q (scoped SCHED evidence)

| # | Result | Evidence |
| --- | ------ | -------- |
| 1 | PASS | `ScheduledResume.tenant_id` required field |
| 2 | PASS | SQLite/reference store persists tenant on schedule row |
| 3 | PASS | `TaskCheckpoint.tenant_id` on paused checkpoint |
| 4 | PASS | `get_by_token(task_id, entry.tenant_id, resume_token)` before resume |
| 5 | PASS | SCHED-Q3 adversarial mismatch → tick 0, port not awaited |
| 6 | PASS | due scan keyed on persisted entry, not recomputed tenant |
| 7 | PASS | claim tests — tenant unchanged on entry |
| 8 | PASS | fence completion does not alter tenant |
| 9 | PASS | UNCERTAIN path preserves entry tenant (PCM test) |
| 10 | PASS | no default/global tenant fallback in scheduler resume path |
| 11 | PASS | custom in-memory store preserves tenant (Q10) |
| 12 | PASS | cancel by schedule_id; active RUNNING cancel rejected |
| 13 | PASS | Q2 store reopen retains tenant |
| 14 | PASS | Q1/Q13 port.execute receives task with checkpoint tenant |
| 15 | PASS | scheduler does not widen tenant authority |
| 16 | PASS | `test_sched_q3_tenant_mismatch_blocks_resume` |

No global TEN PASS claimed.

## Typing audit (semantic seams)

| Finding | Semantic or storage/transport? | Evidence | Action |
| ------- | ------------------------------ | -------- | ------ |
| `resume_metadata: Dict[str, Any]` on `ScheduledResume` | storage/transport metadata | `scheduled_resume.py` | acceptable |
| `UnifiedTaskResumeExecutor` legacy adapter | compatibility seam (not production wiring) | `scheduler.py` | documented; Q15 gate excludes production alternate engine |
| Pyright `ledger=checkpoint_store` union | static checker limitation | `wiring.py` | pre-existing; runtime `TaskCheckpointPersistence` implements `SchedulerLedger` |
| No `cast`/`type: ignore` in scheduler/wiring semantic paths | — | grep | none required |

## Replaceability

`ScheduledResumePersistence` remains the extensibility seam; `SQLiteTaskCheckpointStore` is reference/default, not semantic contract (SCHED-Q10).

## Recovery / durability (scoped)

- Schedule row durable across store reopen (SCHED-Q2).
- Atomic claim + fence + lease expiry → UNCERTAIN (Q5–Q9).
- Checkpoint truth owned by long-running checkpoint subsystem; scheduler references by token.
- **Not claimed:** full platform DR, backup/restore, universal multi-node provider qualification (PROD-Q / STATE-X).

## Current limitation

SCHED-01 supports **one-shot delayed resume** and human-timeout ledger actions only.  
**NOT APPLICABLE TO CURRENT FROZEN SCHEDULING CONTRACT:** cron, interval, calendar recurrence, DST/timezone calendar semantics.

## Distributed claims

Reference durable provider proves atomic cross-instance claim when sharing persistence; custom providers must implement `claim_due` + `complete_claim` fence contract. No universal distributed scheduling claim.

## Tests (session)

| Run | Command | Result |
| --- | ------- | ------ |
| 1 | `uv run pytest -p no:xdist tests/qualification/sched_01/ -q` | **14 passed** |
| 2 | `uv run pytest -p no:xdist tests/unit/runtime/long_running/test_pcm_scheduler_integrity.py -q` | **20 passed** |
| 3 | `uv run pytest -p no:xdist tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py tests/unit/runtime/execution/test_p0c6_terminal_outcome_convergence.py tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py tests/unit/applications/test_task_control_governed_resume.py -q` | **10 failed, 95 passed** — failures are stale harness (`NexusLoop._finish_task` missing `runtime_event_metric_scope`) and embedded mandatory-suite subprocess drift; **not SCHED scheduling defects** |
| 4 | `uv run pytest -p no:xdist tests/unit/runtime/architecture/test_ue_11gp_production_host_execution_gate.py tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py tests/unit/runtime/architecture/test_ebh_2f_r1_host_execution_port_replaceability.py -q` | **15 passed** |

## Pyright

```text
uv run pyright intergrax/runtime/long_running/scheduler.py intergrax/runtime/long_running/scheduled_resume.py intergrax/runtime/long_running/scheduler_claim.py intergrax/runtime/long_running/persistence_contract.py intergrax/runtime/long_running/wiring.py intergrax/applications/_shared/harness_host_auxiliary_wiring.py
```

**2 errors** (pre-existing structural typing: `harness_host_auxiliary_wiring` evidence port vs `RuntimeEventPersistence`; `wiring.py` `TaskCheckpointPersistence` vs `SchedulerLedger` nominal). **0 new SCHED semantic errors introduced in this session.**

## FRZ contribution (evidence only — no global PASS delta)

Scoped contribution toward checklist IDs listed in task §42; **global FRZ PASS delta = 0**, **new FRZ-TEN PASS delta = 0**.

## Tracked freeze debt (non-blocking for SCHED-01)

| Item | Owner |
| ---- | ----- |
| `test_npsc42_orchestration_backend_access_confined_to_allowlist` stale gate | HARNESS-FINAL / QUAL-X |
| Run #3 p0c6 / lineage / embedded mandatory-suite drift | QUAL-X / harness test maintenance |
| Universal provider / DR / cron | PROD-Q / STATE-X / future contract |

## Unresolved findings

| ID | Class |
| -- | ----- |
| — | **IN-SCOPE BLOCKER = 0** |

## Recommendation

**SCHED-01 = READY FOR AUDIT**  
**HARNESS-FINAL = NOT ENTERED**  
**NEXT** = independent SCHED-01 audit acceptance, then current-HEAD Harness residual reconciliation (CE-01, PLUG-01, ART-01, etc. not auto-closed).

**production changed files = 0** (recertification-first).
