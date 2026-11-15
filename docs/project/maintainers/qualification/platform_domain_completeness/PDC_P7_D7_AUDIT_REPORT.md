# PDC-X P7 - IA-D7 independent audit report (D7 Background and Scheduling)

| Field | Value |
| --- | --- |
| **Stage** | PDC-X / P7 / D7 |
| **Auditor** | IA-D7 - independent auditor session (clean session; never implemented, signed or certified anything of D7; no product, test, script or governance edit) |
| **Audited SHA** | `c08efe46a74a20595468d8d8bf06eab2dca5f573` (= `origin/development` at audit start; worktree HEAD verified equal) |
| **Policy** | `IIAC-2026-11-05.1`, OD-PDC-IIAC-SCOPE Option A (verdicts PASS / PASS_SCOPED / PARTIAL / FAIL) |
| **Scope** | Rows `PDC-D7-R001..R023` of `PDC_DOMAIN_REQUIREMENTS_MATRIX.json`, ADR-PDC-07, batch B-D7 (T-IGX-036-L), ownership rows of `PDC_OWNERSHIP_MATRIX.md` |
| **Overall verdict** | **PARTIAL** (IIAC manifest value `CONDITIONAL`) |
| **Findings** | 3 HIGH, 10 MEDIUM, 7 LOW, 3 INFO |
| **Row verdicts** | 10 PASS, 1 PASS_SCOPED, 11 PARTIAL, 1 FAIL (23 rows) |
| **Evidence** | `docs/project/maintainers/qualification/iiac_evidence/PDC-X/D7/` |

This report is an audit, not a certification. It claims no FRZ change, closure or production readiness. Repairs belong to a later ticket of the owning domain implemented by a different session; a fresh auditor then re-audits (max 2 loops).

## 1. Method

1. Read OD-PDC-IIAC-SCOPE, the 23 D7 matrix rows, ownership rows, ADR-PDC-07, the B-D7 plan (T-IGX-036-L) and the P7-00 dossier/addenda; the implementer's commit (`52f35a41f`) and checkpoints were used only as leads.
2. Verified the D7 product and test paths are byte-identical between the implementation commit `52f35a41f` and the audited SHA (`git diff --stat` empty for `intergrax/queueing`, `intergrax/background_tasks`, `intergrax/runtime/long_running`, `intergrax/runtime/background_execution`, `tests/qualification/pdc_x/d7`). Everything after it on those paths is outside D7.
3. Read the owner code in full: `recurrence.py`, `scheduler.py`, `scheduler_claim.py`, `scheduled_resume.py`, `store.py` (schedule/claim/complete/cancel), `task_queue.py`, `document_store_task_queue.py`, `colocated_worker.py`, `dispatcher.py`, `execution.py`, `retry_policy.py`, `worker_runtime.py`, `reentry_admission.py`, `required_audit_evidence.py`, `bootstrap.py`, `admission_wiring.py`, both registries, `wiring.py`, `scheduled_resume_metadata.py`, and the D7 sections of the two BACKGROUND_TASKS documents.
4. Ran the cited tests, one pytest command at a time with narrow paths (all outputs under `logs/`, worktree-local `.venv` created by `uv run`; `tzdata` 2025c present):

| Command | Result |
| --- | --- |
| `uv run pytest tests/qualification/pdc_x/d7 -q` | 52 passed (28.6 s), process exited normally |
| `uv run pytest tests/qualification/bg_01 tests/qualification/sched_01 -q` | 36 passed (= bg_01 22 + sched_01 14, matches the matrix evidence) |
| `uv run pytest tests/qualification/pdc_x/xd/test_xd_05_host_background.py tests/qualification/pdc_x/xd/test_xd_06_background_security.py -q` | 9 passed |

5. Adversarial probes (six scripts, `adversarial_scripts/`, read-only against product code, temporary SQLite/in-memory stores): tenant collision, crash and failure of the recurrence chain, batch abort, non-UTC `run_at_utc`, DST gap/fall-back/half-hour zones, unsatisfiable and overflowing rules, terminal-state overwrite, head-of-line blocking, two-instance double claim, tenant isolation of handles, recurrence smuggled through metadata with catch-up cost, occurrence behaviour after the checkpoint advances. Layer-boundary scan (`intergrax/` must not import `agents/` or `applications/`): no violation in the D7 paths.
6. Verdict rule used: **PASS** = the claim holds and no MEDIUM/HIGH finding attaches (a LOW that falsifies the row's own deliverable also yields PARTIAL); **PASS_SCOPED** = holds inside an Operator-authorized exclusion (C10/C15 freeze); **PARTIAL** = holds in the tested envelope but a MEDIUM/HIGH defect or gap attaches; **FAIL** = the claimed capability is absent or false at the audited SHA.

## 2. Per-row verdicts

| Row | Requirement | Verdict | Evidence |
| --- | --- | --- | --- |
| PDC-D7-R001 | Task definition and queue contract | **PASS** | task_queue.py read in full; bg_01 22 passed; DEAD_LETTERED/TaskExecutionIdentity are additive (existing providers unchanged, test_existing_provider_shape...). LOW: F-D7-14. |
| PDC-D7-R002 | Worker admission through canonical execution | **PASS** | reentry_admission.py read in full; used by dispatcher, DocumentStore worker and WorkerRuntime; bg_01 + d7 pass. LOW: F-D7-14 (terminal ack). |
| PDC-D7-R003 | Durable dispatch | **PASS** | dispatcher.py read; bg_01 passes. Celery identity/DLQ is declared debt and not audited. Related MEDIUM defects sit on the DocumentStore queue, not on the dispatcher (F-D7-07/08 recorded under R010/R009). |
| PDC-D7-R004 | One-shot delayed execution | **PARTIAL** | sched_01 14 passed, but F-D7-01 (HIGH, batch abort) and F-D7-10 (run_at_utc). |
| PDC-D7-R005 | Queue-level delay/ETA owned by SchedulerLedger (N/A) | **PASS** | TaskRequest has no delay/ETA field (task_queue.py); available_at_epoch is internal retry back-off only; ownership matrix row agrees. N/A confirmed. |
| PDC-D7-R006 | Recurrence as occurrences on SchedulerLedger | **PARTIAL** | No second scheduler/engine (verified), 52 d7 tests pass; F-D7-02 (HIGH), F-D7-04, F-D7-05, F-D7-11, F-D7-13. |
| PDC-D7-R007 | Time-zone and DST semantics | **PASS** | Gap/fall-back/wildcard/Lord Howe probes behave as specified; tests pass. INFO F-D7-21, LOW F-D7-18. |
| PDC-D7-R008 | Misfire / catch-up policy | **PARTIAL** | skip/fire_once/catch_up(n) behave as specified and degrade to skip past MAX_MISFIRE_SCAN; F-D7-11 (unbounded cost), F-D7-02. |
| PDC-D7-R009 | Aggregate retry budget | **PARTIAL** | Budget exhaustion to DEAD_LETTERED works in the DocumentStore path (tests); no aggregate budget across layers, not composed (F-D7-05), F-D7-06, F-D7-08, LOW F-D7-15. |
| PDC-D7-R010 | Idempotency and duplicate handling | **PARTIAL** | Per-tenant key scoping and execution-layer idempotency hold (bg_01, probe P12); F-D7-07 (enqueue race, double claim), worker uses idempotency_store=None (F-D7-23). |
| PDC-D7-R011 | At-least-once delivery documented | **PASS** | BACKGROUND_TASKS.md B.2.2 exists and is accurate for what it states; F-D7-12 covers staleness of neighbouring sections. |
| PDC-D7-R012 | Lease/fencing across processes | **PASS_SCOPED** | Two-OS-process SQLite proof reproduced (test_d7_fencing_two_process, exactly-once + stale fence rejected). Scoped to SQLite: PostgreSQL/multi-node excluded by the C10/C15 freeze. Adjacent defects F-D7-01/02 do not falsify the fencing proof itself. |
| PDC-D7-R013 | Dead-letter / terminal-failed semantics | **PARTIAL** | DEAD_LETTERED with typed reason + attempts reached via the DocumentStore path; not composed (F-D7-05), not sticky (F-D7-06), silent (F-D7-17). Celery/Kafka/Rabbit DLQ is declared debt. |
| PDC-D7-R014 | Graceful worker shutdown with host drain | **FAIL** | No product implementation (F-D7-03, HIGH); proof relies on a test-side adapter; F-D7-20. |
| PDC-D7-R015 | Crash recovery and resume | **PARTIAL** | Crash after claim is at-most-once (UNCERTAIN, proven); recurrence chain does not survive it (F-D7-02), batch abort (F-D7-01), no UNCERTAIN reconciler. |
| PDC-D7-R016 | Full execution identity on enqueue | **PARTIAL** | Quadruple adopted on the DocumentStore path (tests); no producer, principal/trace unused, partial ignored (F-D7-09). |
| PDC-D7-R017 | Tenant scope of background work | **PASS** | bootstrap.py tenant resolution read; per-tenant partitions verified (probe P12: foreign handle gets PENDING/None/False/[]); bg_01 passes. INFO: task_tenant_id cross-check has no production caller. |
| PDC-D7-R018 | Governance of background work | **PASS** | admission_wiring.py durable-authority validation and required-audit-evidence-before-handler ordering read and covered by bg_01; INFO: durability flags of stores are store-reported (not re-audited). |
| PDC-D7-R019 | Observability of background work | **PASS** | Required causal evidence is persisted before the handler (code + bg_01). LOW F-D7-17 (retry/dead-letter silent). |
| PDC-D7-R020 | Operational control (list/cancel/purge) | **PASS** | list/cancel present and tenant-isolated in the DocumentStore queue (cancel only PENDING); purge is provider-dependent. LOW F-D7-17, F-D7-19. |
| PDC-D7-R021 | Horizontal scaling boundary documented | **PARTIAL** | Section exists but is stale (multi-process proof 'planned') and omits the single-instance limit (F-D7-12, F-D7-07). |
| PDC-D7-R022 | TaskRegistry/WorkerRuntime layered over TaskExecutionRegistry; ownership recorded | **PARTIAL** | Layering confirmed (TaskRegistry.bind_execution_registry); the matrix cell for the composition root is wrong (F-D7-16). |
| PDC-D7-R023 | Plan statuses reconciled; doc registered in the runtime hub | **PARTIAL** | Hub registration present (intergrax_runtime_architecture.md line 99); the plan's A.1 table now states recurrence/DLQ/identity 'Not implemented - Open' (F-D7-12). |

## 3. Findings summary

| ID | Severity | Rows | Finding |
| --- | --- | --- | --- |
| F-D7-01 | HIGH | R004, R006, R012, R015 | One failing resume aborts the whole claimed batch; the sibling schedules are stranded and then lost |
| F-D7-02 | HIGH | R006, R008, R015 | The recurrence chain ends silently on a crash, on lease expiry, on a transient store error or after UNCERTAIN |
| F-D7-03 | HIGH | R014 | Graceful worker shutdown with host drain has no product implementation; the closure proof uses a test-side adapter |
| F-D7-04 | MEDIUM | R006 | Occurrence identity has no tenant component; a collision silently drops the other tenant's chain (dossier F-03, confirmed) |
| F-D7-05 | MEDIUM | R006, R007, R008, R009, R013 | Recurrence, the retry budget and dead-lettering are not reachable from any production composition |
| F-D7-06 | MEDIUM | R013, R009 | Terminal queue states are not sticky: a late completion overwrites DEAD_LETTERED and a late retry re-opens SUCCEEDED |
| F-D7-07 | MEDIUM | R003, R010, R021 | DocumentStoreTaskQueue exclusion is a per-instance threading.RLock; two queue instances (or processes) over one store double-claim and race on idempotent enqueue |
| F-D7-08 | MEDIUM | R003, R009 | Retry back-off rows block fresh work: claim_pending inspects only the first max(4*limit, 32) pending-index rows |
| F-D7-09 | MEDIUM | R016 | Execution identity on the enqueue envelope is carried but not adopted: no producer sets it, principal and trace are never consumed, partial identity is ignored |
| F-D7-10 | MEDIUM | R004, R007 | ScheduledResume.run_at_utc is not validated or normalized; the store compares ISO strings lexicographically |
| F-D7-11 | MEDIUM | R006, R008 | Recurrence is unbounded and can be minted through plain resume_metadata; catch-up drain is super-linear |
| F-D7-12 | MEDIUM | R011, R021, R023 | Architecture and plan documents still state that recurrence, DLQ, identity on enqueue and multi-process fencing are not implemented or planned |
| F-D7-13 | MEDIUM | R006 | Occurrences become silent completed no-ops once the target checkpoint is no longer the resumable latest |
| F-D7-14 | LOW | R001, R002 | Consumers of the queue status do not treat DEAD_LETTERED uniformly; a terminal duplicate is acknowledged as SUCCEEDED |
| F-D7-15 | LOW | R009 | RetryPolicy.max_retry_window_seconds raises TypeError for max_backoff_seconds=None, which validate() accepts |
| F-D7-16 | LOW | R022 | Ownership record names the wrong composition root for task handler registration; the two registries diverge on duplicates |
| F-D7-17 | LOW | R013, R019, R020 | Retry and dead-letter transitions are silent, an idempotency key pins a dead-lettered task, and the DocumentStore queue cannot purge |
| F-D7-18 | LOW | R006, R007 | Unsatisfiable cron and overflowing intervals are accepted; errors escape as OverflowError |
| F-D7-19 | LOW | R020 | ScheduledResumePersistence.cancel(schedule_id) is not tenant-scoped, and recurrence has no list/cancel surface |
| F-D7-20 | LOW | R014 | DocumentStoreTaskWorker.stop() abandons a still-running thread and a later start() can run two consumers |
| F-D7-21 | INFO | R007 | DST gap semantics: the docstring says 'first valid instant after the gap', the code shifts by the gap length |
| F-D7-22 | INFO | R006, R012, R013, R016 | Test-quality notes: weak or missing negative tests |
| F-D7-23 | INFO | R009, R010 | Re-verification of routed items and nearby observations |

## 4. Findings

### F-D7-01 - HIGH - One failing resume aborts the whole claimed batch; the sibling schedules are stranded and then lost

- **Rows:** PDC-D7-R004, PDC-D7-R006, PDC-D7-R012, PDC-D7-R015
- **Verification:** PROBE_CONFIRMED + CODE_READ
- **Evidence:** scheduler.py:_process_due_schedules (try/finally per claim, no except) lets any exception from _process_claimed_schedule/_execute_resume (resume_executor raising, StaleClaimError from complete_claim after lease expiry) escape the for-loop. The remaining claims of the same claim_due batch stay RUNNING; the next claim_due converts them to UNCERTAIN (store.py claim_due), which nothing ever re-claims. The loop and the finally were present for the one-shot path before B-D7 (pre-existing SCHED-01 shape); B-D7 added the finally but not an except.
- **Reproduction:** probe_batch.py: 3 due recurring schedules of one tenant, the first resume raises. tick() raises, the other two occurrences end 'uncertain' without ever calling the executor and their chains have no pending row (logs/probe_batch.out). The batch is not tenant-scoped (claim_due ranks by run_at only), so one tenant's failing resume strands another tenant's due schedules.
- **Impact:** Silent, permanent loss of independent due schedules (up to 100 per batch) and of their recurrence chains; cross-tenant blast radius for a failure of one tenant's task. Not covered by any d7/bg_01/sched_01 test.
- **Suggested repair scope:** Isolate each claim (catch per claim, release or fail the claim explicitly, continue the batch); decide and test the StaleClaimError path; add a negative test with a raising executor and a multi-claim batch.
- **Owner path:** `intergrax/runtime/long_running/scheduler.py`

### F-D7-02 - HIGH - The recurrence chain ends silently on a crash, on lease expiry, on a transient store error or after UNCERTAIN

- **Rows:** PDC-D7-R006, PDC-D7-R008, PDC-D7-R015
- **Verification:** PROBE_CONFIRMED
- **Evidence:** Occurrence N+1 is materialized inline AFTER occurrence N finished (scheduler.py _materialize_next_occurrence in a finally; recurrence.py materialize_next). (a) Crash/kill while N is RUNNING: the lease expires, N becomes UNCERTAIN (store.py claim_due), nobody materializes N+1 and there is no reconciler for UNCERTAIN rows or for rules. (b) A crash between complete_claim and materialize_next, or any exception in materialize_next (logged only, scheduler.py:~222), ends the chain with no retry. (c) The finally also runs for dead tasks (checkpoint missing / not resumable): rules without end_utc keep creating rows for a terminal task (dossier F-14, confirmed).
- **Reproduction:** probe_sched.py P2: claim by a dying owner, survivor ticks later -> only the 'uncertain' row exists, no pending successor. P2b: schedule() raises once during materialization -> two completed rows, no pending successor (logs/probe_sched.out). The d7 crash test (test_d7_fencing_two_process.py) covers a one-shot only.
- **Impact:** A recurring schedule dies permanently and silently after any restart or deploy that overlaps an occurrence (the window is the whole resume execution). This contradicts the crash-recovery claim of R015 for recurrence.
- **Suggested repair scope:** Materialize the successor before or atomically with the claim (or in the same transaction as the occurrence state), or run a reconciler that restores a pending successor for every live rule; surface materialization failures (retry/alert) instead of logging; stop materializing for terminal tasks; add chain-crash tests.
- **Owner path:** `intergrax/runtime/long_running/scheduler.py, recurrence.py`

### F-D7-03 - HIGH - Graceful worker shutdown with host drain has no product implementation; the closure proof uses a test-side adapter

- **Rows:** PDC-D7-R014
- **Verification:** CODE_READ + GREP + TEST_RUN
- **Evidence:** intergrax/hosting/shutdown.py defines the HostedApplicationActiveWorkController port; no class in intergrax/, agents/ or applications/ implements stop_intake/active_work_count/wait_for_idle/cancel_active_work (grep). WorkerRuntime (the semantic owner) has no stop/drain API. DocumentStoreTaskWorker.stop() joins with a timeout, does not wait for or requeue in-flight tasks, and sets _thread=None even when the join timed out (code-read: a later start() then runs recover_interrupted_running over the still-running old thread's task and starts a second consumer). tests/qualification/pdc_x/xd/test_xd_05_host_background.py states the gap in its own docstring ('the product ships the port but no implementation over WorkerRuntime').
- **Reproduction:** grep for the four port methods; run test_xd_05 (passes, 9 passed with xd_06) - it passes only through the _WorkerActiveWork class defined inside the test.
- **Impact:** The row's requirement (worker drains in-flight work, leases released/fenced on shutdown) is not delivered by the platform; hosts must write their own controller and no worker releases its RUNNING tasks.
- **Suggested repair scope:** Ship a worker-side HostedApplicationActiveWorkController (WorkerRuntime and DocumentStoreTaskWorker), make stop() drain or requeue RUNNING tasks within a bound, and re-point XD-05 at the product class.
- **Owner path:** `intergrax/background_tasks/worker_runtime.py, intergrax/queueing/providers/document_store/colocated_worker.py`

### F-D7-04 - MEDIUM - Occurrence identity has no tenant component; a collision silently drops the other tenant's chain (dossier F-03, confirmed)

- **Rows:** PDC-D7-R006
- **Verification:** PROBE_CONFIRMED
- **Evidence:** recurrence.py occurrence_schedule_id(rule_id, occurrence) -> 'rec_<rule_id>_<UTC stamp>' is the global scheduled_resumes PRIMARY KEY; _materialize swallows ScheduledResumeScheduleConflictError and returns None; RecurrenceMaterializer.start() returns None for 'exists', 'no occurrence' and 'conflict' alike.
- **Reproduction:** probe_sched.py P1: two tenants with rule_id 'daily-report' and the same start: tenant-a gets the row, tenant-b start() returns None and has no row.
- **Impact:** A tenant can pre-claim or collide with another tenant's rule_id/instant (availability, existence oracle); the second chain never starts and nothing reports it.
- **Suggested repair scope:** Qualify the id by tenant (and task), return a typed result/raise on conflict that is not the same rule, add a cross-tenant negative test.
- **Owner path:** `intergrax/runtime/long_running/recurrence.py`

### F-D7-05 - MEDIUM - Recurrence, the retry budget and dead-lettering are not reachable from any production composition

- **Rows:** PDC-D7-R006, PDC-D7-R007, PDC-D7-R008, PDC-D7-R009, PDC-D7-R013
- **Verification:** GREP + CODE_READ
- **Evidence:** wire_long_running_scheduler_with_host_execution (wiring.py) has no recurrence parameter and RecurrenceRule/RecurrenceMaterializer are not exported from intergrax.runtime.long_running; the three DocumentStoreTaskWorker construction sites (vendor_knowledge/sync_runtime.py, local_workspace_application/workspaces/sync_runtime.py x2) pass no retry_policy, so production workers never retry, never consume a budget and never dead-letter (recover_interrupted_running(max_retries=None) -> FAILED).
- **Reproduction:** grep 'recurrence=' and 'retry_policy=' over intergrax/agents/applications outside tests.
- **Impact:** The D7 capabilities exist only as library code exercised by tests; a host using the sanctioned wiring cannot enable them. The ownership matrix names wiring.py as composition root of 'Delayed / recurring execution'.
- **Suggested repair scope:** Expose recurrence (and a rule-registration API) through the composition root, wire a RetryPolicy into the DocumentStoreTaskWorker composition, add a host-level test; or record an explicit scope decision.
- **Owner path:** `intergrax/runtime/long_running/wiring.py, intergrax/queueing/providers/document_store/colocated_worker.py composition sites`

### F-D7-06 - MEDIUM - Terminal queue states are not sticky: a late completion overwrites DEAD_LETTERED and a late retry re-opens SUCCEEDED

- **Rows:** PDC-D7-R013, PDC-D7-R009
- **Verification:** PROBE_CONFIRMED
- **Evidence:** document_store_task_queue.py mark_succeeded/mark_failed/mark_dead_lettered/retry_or_dead_letter/_settle_against_budget load the record and write without checking the current status or attempt token.
- **Reproduction:** probe_queue.py P5: DEAD_LETTERED -> mark_succeeded -> SUCCEEDED (stale 'dead_letter' field remains in the row); P5b: SUCCEEDED -> retry_or_dead_letter -> PENDING.
- **Impact:** A stale or duplicate worker flips a terminal outcome (including a dead-letter) and can re-run finished work; the contract calls DEAD_LETTERED terminal.
- **Suggested repair scope:** Guard every transition on the expected status (and delivery attempt), reject transitions out of a terminal status, add negative tests.
- **Owner path:** `intergrax/queueing/providers/document_store/document_store_task_queue.py`

### F-D7-07 - MEDIUM - DocumentStoreTaskQueue exclusion is a per-instance threading.RLock; two queue instances (or processes) over one store double-claim and race on idempotent enqueue

- **Rows:** PDC-D7-R003, PDC-D7-R010, PDC-D7-R021
- **Verification:** PROBE_CONFIRMED
- **Evidence:** claim_pending and enqueue read-modify-write the DocumentStore under self._lock only; the ConditionalDocumentStore (compare-and-set) contract is not used. sync_runtime.py builds more than one DocumentStoreTaskQueue over the same store (placeholder worker queue and message-bus queue).
- **Reproduction:** probe_queue_race.py: 2 queue instances over one in-memory store claim 40 tasks, 37 are claimed by both (logs/probe_queue_race.out). Enqueue idempotency lookup + put is the same TOCTOU.
- **Impact:** Duplicate handler invocation across instances or hosts sharing a store; the scaling-boundary doc does not state this limit of the durable local provider.
- **Suggested repair scope:** Use conditional writes (CAS) for claim/enqueue or document and enforce single-instance ownership; add a two-instance test; state the limit in BACKGROUND_TASKS.md B.2.3.
- **Owner path:** `intergrax/queueing/providers/document_store/document_store_task_queue.py`

### F-D7-08 - MEDIUM - Retry back-off rows block fresh work: claim_pending inspects only the first max(4*limit, 32) pending-index rows

- **Rows:** PDC-D7-R003, PDC-D7-R009
- **Verification:** PROBE_CONFIRMED
- **Evidence:** claim_pending queries the pending index with limit=max(limit*4,32) and skips rows whose available_at is in the future; they stay in the index, so 32+ backed-off rows hide every later ready task.
- **Reproduction:** probe_queue.py P9: 40 tasks failed with a 1 h back-off, then a fresh task: claim_pending(limit=4) returns [].
- **Impact:** Liveness loss for a tenant-wide or global queue for the duration of the back-off.
- **Suggested repair scope:** Index by available_at or move backed-off rows out of the scan window; add a head-of-line test.
- **Owner path:** `intergrax/queueing/providers/document_store/document_store_task_queue.py`

### F-D7-09 - MEDIUM - Execution identity on the enqueue envelope is carried but not adopted: no producer sets it, principal and trace are never consumed, partial identity is ignored

- **Rows:** PDC-D7-R016
- **Verification:** GREP + CODE_READ
- **Evidence:** No production TaskRequest(...) site passes identity=; QueuedHostTaskExecutionAdapter, message_bus service, retention_sweep, vendor sync_jobs and the LKW handler all enqueue without it. The worker only seeds the TaskId/RunId/AttemptId/ExecutionId quadruple (colocated_worker._seed_carried_identity); principal_id and trace_id are validated and stored but read nowhere; a partial identity returns silently; only the DocumentStore provider persists identity (Celery/broker enqueue drop it - declared debt).
- **Reproduction:** grep 'identity=' / 'principal_id' / 'trace_id' in intergrax/queueing, background_tasks, background_execution; test_d7_enqueue_identity.py exercises only the quadruple through the DocumentStore path.
- **Impact:** The documented ADOPTION GAP (full identity on enqueue) is not closed in practice: ADR text names principal and trace correlation (FRZ-TRC-12) that reach neither execution nor evidence.
- **Suggested repair scope:** Populate identity from the canonical authority at the real enqueue sites, carry principal/trace into the execution identity or causal evidence, reject partial identity, add an end-to-end test from a real producer.
- **Owner path:** `intergrax/queueing/contracts/task_queue.py, intergrax/queueing/providers/document_store/colocated_worker.py, intergrax/runtime/task/queued_host_task_execution_adapter.py`

### F-D7-10 - MEDIUM - ScheduledResume.run_at_utc is not validated or normalized; the store compares ISO strings lexicographically

- **Rows:** PDC-D7-R004, PDC-D7-R007
- **Verification:** PROBE_CONFIRMED
- **Evidence:** ScheduledResume (scheduled_resume.py) has no validator for run_at_utc; store.py/PG claim_due and list_due use 'run_at_utc <= ?' on text, and tick() does not convert a non-UTC aware 'now' (_ensure_utc).
- **Reproduction:** probe_sched.py P3 at 08:30Z: '2026-12-01T05:00:00-05:00' (=10:00Z) fires 90 min EARLY, '2026-12-01T10:00:00+02:00' (=08:00Z) does not fire, '' fires at once, 'not-a-time' never fires.
- **Impact:** Delayed work can run hours early or never, with no error, from a field named *_utc.
- **Suggested repair scope:** Validate and normalize to UTC at ScheduledResume construction/persistence; reject non-ISO; convert 'now' to UTC in tick; add tests.
- **Owner path:** `intergrax/runtime/long_running/scheduled_resume.py, store.py, postgresql_task_checkpoint_store.py`

### F-D7-11 - MEDIUM - Recurrence is unbounded and can be minted through plain resume_metadata; catch-up drain is super-linear

- **Rows:** PDC-D7-R006, PDC-D7-R008
- **Verification:** PROBE_CONFIRMED
- **Evidence:** validate_scheduled_resume_metadata does not forbid 'recurrence_rule'/'recurrence_occurrence_utc', so any caller of LongRunningScheduler.schedule_resume can start a chain without RecurrenceMaterializer.start; RecurrenceRule accepts interval_seconds=1 and catch_up(n) with no upper bound (only MAX_MISFIRE_SCAN=10000 bounds the scan); every fired occurrence re-scans the missed range.
- **Reproduction:** probe_meta.py: a recurrence rule smuggled via resume_metadata, interval 1 s, catch_up(5000): 50 s late -> 2.0 s, 100 s -> 6.2 s, 200 s -> 14.9 s for the drain (a 3600 s backlog did not finish in 6 min and was stopped).
- **Impact:** Unbounded resume storms and CPU cost after downtime; an authority-bearing key (it mints future executions) bypasses the SX-F13 metadata gate.
- **Suggested repair scope:** Add the keys to the forbidden set or gate creation on the materializer, set a minimum interval and a catch_up cap, plan the missed range once per tick.
- **Owner path:** `intergrax/runtime/long_running/scheduled_resume_metadata.py, recurrence.py`

### F-D7-12 - MEDIUM - Architecture and plan documents still state that recurrence, DLQ, identity on enqueue and multi-process fencing are not implemented or planned

- **Rows:** PDC-D7-R011, PDC-D7-R021, PDC-D7-R023
- **Verification:** CODE_READ + GIT_LOG
- **Evidence:** docs/project/architecture/BACKGROUND_TASKS.md B.2.1 ('not cron/interval/calendar in core', DST 'NOT APPLICABLE'), B.2.3 ('multi-process fencing proof is planned'), 'ADOPTION GAP' and docs/project/maintainers/plans/BACKGROUND_TASKS.md A.1 ('Not implemented ... Open - PDC-P5-D7') were written by P5-DOC before B-D7 (commit 0e92fd812 at 07:38, B-D7 52f35a41f at 11:58) and not updated; the text also omits the F-D7-07 limit of the durable local provider and the fact that no host composes the new capabilities.
- **Reproduction:** git log --format='%h %ad' -- the two files vs 52f35a41f; read the cited sections.
- **Impact:** Operators reading the canonical docs get the opposite of the code state (FRZ-DOC-01 drift).
- **Suggested repair scope:** Reconcile both documents with the delivered state and its limits (including F-D7-05/07), after the repairs.
- **Owner path:** `docs/project/architecture/BACKGROUND_TASKS.md, docs/project/maintainers/plans/BACKGROUND_TASKS.md`

### F-D7-13 - MEDIUM - Occurrences become silent completed no-ops once the target checkpoint is no longer the resumable latest

- **Rows:** PDC-D7-R006
- **Verification:** PROBE_CONFIRMED (skip path)
- **Evidence:** Each occurrence resumes one (task_id, resume_token). _process_claimed_schedule completes the claim without executing when the token's checkpoint is not the latest / not resumable (complete_claim then continue). The d7 fixture paused_checkpoint() is engineered to 'stay resumable across repeated resumes' and the executor is a counting stub; no test drives a real resume.
- **Reproduction:** probe_repause.py: after the first fire a newer checkpoint is saved (what a re-pause does): occurrences 2 and 3 fire 0 times and end 'completed' while the chain keeps materializing rows. A real execution engine run was NOT performed (limitation).
- **Impact:** A recurring job executes at most once for a normal task while the ledger shows completed occurrences.
- **Suggested repair scope:** Define and test the recurrence target semantics end to end (what is resumed by occurrence 2..n), and record a skipped occurrence as skipped, not completed.
- **Owner path:** `intergrax/runtime/long_running/scheduler.py, tests/qualification/pdc_x/d7/_support.py`

### F-D7-14 - LOW - Consumers of the queue status do not treat DEAD_LETTERED uniformly; a terminal duplicate is acknowledged as SUCCEEDED

- **Rows:** PDC-D7-R001, PDC-D7-R002
- **Verification:** CODE_READ
- **Evidence:** applications/local_workspace_application/serving/background_task_proof_routes.py:195 computes 'completed' from {SUCCEEDED, FAILED}; colocated_worker.drain_once and WorkerRuntime ack TERMINAL_ALREADY_RECORDED as SUCCEEDED without looking at the recorded terminal outcome.
- **Reproduction:** code-read.
- **Impact:** Polling clients never see a dead-lettered task complete; a failed/cancelled terminal can show as SUCCEEDED in the queue.
- **Suggested repair scope:** Use is_terminal_task_status in consumers; map the recorded terminal outcome.
- **Owner path:** `applications/local_workspace_application/serving/background_task_proof_routes.py, colocated_worker.py, worker_runtime.py`

### F-D7-15 - LOW - RetryPolicy.max_retry_window_seconds raises TypeError for max_backoff_seconds=None, which validate() accepts

- **Rows:** PDC-D7-R009
- **Verification:** PROBE_CONFIRMED
- **Evidence:** retry_policy.py min(backoff, self.max_backoff_seconds); used by queueing/worker_bootstrap.py:55 and runtime/task/worker_bootstrap.py:174.
- **Reproduction:** probe_queue.py P6: TypeError: '<' not supported between 'NoneType' and 'float'.
- **Impact:** Worker bootstrap crashes with an uninformative error for a valid policy (fail-closed but unclear).
- **Suggested repair scope:** Treat None as unbounded or reject it in validate(); test it.
- **Owner path:** `intergrax/queueing/worker/retry_policy.py`

### F-D7-16 - LOW - Ownership record names the wrong composition root for task handler registration; the two registries diverge on duplicates

- **Rows:** PDC-D7-R022
- **Verification:** CODE_READ
- **Evidence:** PDC_OWNERSHIP_MATRIX.md D7 row: composition root 'queueing/registry.py' is TaskQueueProviderRegistry (queue provider registry). Handler binding happens in applications/local_workspace_application/host/background_worker_factory.py via TaskRegistry.bind_execution_registry. TaskRegistry.register overwrites a duplicate name, TaskExecutionRegistry.register raises.
- **Reproduction:** read intergrax/queueing/registry.py and background_tasks/registry.py.
- **Impact:** The recorded ownership is inaccurate in one of two cells; duplicate-name behaviour differs between the layers.
- **Suggested repair scope:** Correct the matrix cell; make TaskRegistry.register reject duplicates.
- **Owner path:** `docs/.../PDC_OWNERSHIP_MATRIX.md, intergrax/background_tasks/registry.py`

### F-D7-17 - LOW - Retry and dead-letter transitions are silent, an idempotency key pins a dead-lettered task, and the DocumentStore queue cannot purge

- **Rows:** PDC-D7-R013, PDC-D7-R019, PDC-D7-R020
- **Verification:** PROBE_CONFIRMED + CODE_READ
- **Evidence:** DocumentStoreTaskWorker emits no TaskEvent/log/metric for retry or dead-letter (TaskEventName has no dead-lettered member though the architecture doc lists task.dead_lettered); enqueue returns the dead-lettered handle for the same idempotency key (probe P10) with no redrive; DocumentStoreTaskQueue does not override purge_completed (default 0, indistinguishable from 'nothing to purge') so terminal and idempotency rows grow.
- **Reproduction:** probe_queue.py P10, P13; grep of document_store provider for logger/emit.
- **Impact:** No operator signal or lifecycle for poison tasks.
- **Suggested repair scope:** Emit a typed dead-letter event, add redrive/purge and a documented idempotency-key policy.
- **Owner path:** `intergrax/queueing/providers/document_store/, intergrax/background_tasks/events.py`

### F-D7-18 - LOW - Unsatisfiable cron and overflowing intervals are accepted; errors escape as OverflowError

- **Rows:** PDC-D7-R006, PDC-D7-R007
- **Verification:** PROBE_CONFIRMED
- **Evidence:** RecurrenceRule validation accepts '0 0 30 2 *' (first_occurrence None, start() returns None silently) and interval_seconds=10**15 (next_after raises OverflowError, not RecurrenceError).
- **Reproduction:** probe_sched.py P4.
- **Impact:** Silent never-firing rule; the scheduler only logs the OverflowError.
- **Suggested repair scope:** Reject unsatisfiable rules and cap interval/horizon at validation.
- **Owner path:** `intergrax/runtime/long_running/recurrence.py`

### F-D7-19 - LOW - ScheduledResumePersistence.cancel(schedule_id) is not tenant-scoped, and recurrence has no list/cancel surface

- **Rows:** PDC-D7-R020
- **Verification:** CODE_READ
- **Evidence:** store.py/PG cancel(schedule_id) takes only the id; recurrence ids are deterministic and guessable (rec_<rule>_<stamp>); cancelling the pending occurrence is the only way to stop a chain.
- **Reproduction:** code-read.
- **Impact:** A caller holding the port can cancel another tenant's pending occurrence; no tenant-scoped control.
- **Suggested repair scope:** Add tenant to cancel/list and a rule-level cancel.
- **Owner path:** `intergrax/runtime/long_running/scheduled_resume.py, store.py`

### F-D7-20 - LOW - DocumentStoreTaskWorker.stop() abandons a still-running thread and a later start() can run two consumers

- **Rows:** PDC-D7-R014
- **Verification:** CODE_READ
- **Evidence:** colocated_worker.py stop(): join(timeout) then self._thread = None regardless of is_alive().
- **Reproduction:** code-read (not reproduced).
- **Impact:** Double consumption after a slow shutdown/restart in one process.
- **Suggested repair scope:** Keep the thread handle until it exits or fail stop() when the join times out.
- **Owner path:** `intergrax/queueing/providers/document_store/colocated_worker.py`

### F-D7-21 - INFO - DST gap semantics: the docstring says 'first valid instant after the gap', the code shifts by the gap length

- **Rows:** PDC-D7-R007
- **Verification:** PROBE_CONFIRMED
- **Evidence:** recurrence.py module docstring vs _day_instants: '30 2 * * *' in America/New_York on 2026-03-08 fires at 03:30 EDT (not 03:00); '0 2' fires at 03:00. Fall-back fixed hour fires once at the first pass; wildcard hours follow elapsed time.
- **Reproduction:** probe_sched.py P7.
- **Impact:** Documentation precision only; behaviour is deterministic and tested.
- **Suggested repair scope:** Reword the docstring/ADR to 'wall time + gap length'.
- **Owner path:** `intergrax/runtime/long_running/recurrence.py`

### F-D7-22 - INFO - Test-quality notes: weak or missing negative tests

- **Rows:** PDC-D7-R006, PDC-D7-R012, PDC-D7-R013, PDC-D7-R016
- **Verification:** CODE_READ
- **Evidence:** test_queue_side_mints_no_identity_of_its_own greps source for 'mint_'; the crash-after-claim test covers a one-shot only; no test for tenant collision, batch abort, terminal stickiness, head-of-line, non-UTC run_at, production composition, principal/trace, or a real resume across occurrences; XD-05/XD-06 use test-side adapters (their own docstrings report the missing product pieces).
- **Reproduction:** read tests/qualification/pdc_x/d7 and xd_05/xd_06.
- **Impact:** Defects F-D7-01..13 pass all 52 tests.
- **Suggested repair scope:** Add the negative tests listed in F-D7-01..13 with the repairs.
- **Owner path:** `tests/qualification/pdc_x/d7/`

### F-D7-23 - INFO - Re-verification of routed items and nearby observations

- **Rows:** PDC-D7-R009, PDC-D7-R010
- **Verification:** TEST_RUN + GREP
- **Evidence:** Dossier F-03 and F-14 confirmed (F-D7-04, F-D7-02). F-13 (D4): colocated_worker passes idempotency_store=None, so budgeted redelivery re-invokes handlers (documented in BACKGROUND_TASKS.md B.2.2: handlers must be idempotent); not re-audited as a D4 row. XD-06 reports that no composition wires the credential broker into the worker path (D6). The pdc_x pytest process exited normally for d7 and xd_05/xd_06 (debt R-05 not observed there; xd_03 not run). No import from agents/ or applications/ in the D7 intergrax/ paths (scan).
- **Reproduction:** logs/*.log; grep.
- **Impact:** None.
- **Suggested repair scope:** None.
- **Owner path:** `-`


## 5. Routed items re-verified

| Source | Item | Result at c08efe46a |
| --- | --- | --- |
| P7-00 dossier F-03 (MEDIUM, IA-D7) | Occurrence id has no tenant; conflict swallowed | **Confirmed** - F-D7-04 (probe P1) |
| P7-00 dossier F-14 (LOW, IA-D7) | Materialization in `finally` for dead tasks; failure only logged | **Confirmed and escalated** - F-D7-02 (the chain is lost, not only logged) |
| P7-00 dossier F-13 (LOW, IA-D4) | Budgeted redelivery with `idempotency_store=None` | Confirmed by code read (colocated_worker); documented in B.2.2; D4 owns it (F-D7-23) |
| P7-00 dossier F-34 (INFO) | `tzdata` absent in an unsynced `.venv` | Not observed: a fresh `uv run` environment contains tzdata 2025c |
| P7-00 dossier F-09 / debt R-05 | `pdc_x` pytest process not exiting | Not observed for d7 and xd_05/xd_06 (exit 0); xd_03 not run |
| P7-00 dossier addenda 1-4 | Any further D7 item | None routed to IA-D7 beyond F-03 and F-14 (grep of dossier and addenda) |

## 6. Overall verdict

**PARTIAL.** The delivered pieces work inside the envelope the 52 d7 tests, bg_01, sched_01 and XD-05/06 exercise: recurrence materializes deterministic one-shot occurrences on the existing ledger with sound DST and misfire semantics, DEAD_LETTERED is a typed terminal state reached through the budget, the carried identity is adopted on the DocumentStore path, and two-process SQLite fencing holds. Three HIGH defects remain: one failing resume strands the sibling claims of its batch (F-D7-01), the recurrence chain is lost on crash, lease expiry or a transient store error (F-D7-02), and graceful worker shutdown has no product implementation (F-D7-03, R014 FAIL). The capabilities are also not composed anywhere in production (F-D7-05), and the canonical documents contradict the code (F-D7-12). The all-green test suite does not exercise any of these failure paths (F-D7-22).

Repair scope for the owning domain (ticket of D7, different session): F-D7-01, 02, 03 first; then F-D7-04..13; LOW/INFO with the same batch or a doc batch. Requalification: fresh IA-D7 re-audit at the repair SHA.

## 7. Limitations (what this audit did NOT cover)

- **Declared exceptions and debt (not audited as defects):** Celery/Kafka/RabbitMQ identity and DLQ, strict-host `plugin_provenance`, `ent_audit_x` `launcher_05` FREEZE timeout, `dist_scale_x` and Docker/PostgreSQL scope (C10/C15 freeze, not lifted). PostgreSQL scheduler/checkpoint store code was read only for schedule-conflict parity, never executed; no Docker, Testcontainers, broker, Mongo or real object store.
- **Known reds not re-reported:** EBH-7 identity gate (until the final re-pin), five `roadmap_replay_x` tests (`historical_evidence_invalidated_by_later_changes` / `unscoped_prod_delta`). They were not run here and are not touched by D7 paths.
- Tests run: only the three commands above. Not run: `tests/unit`, the full `tests/qualification` tree, EBH replays, ENT-AUDIT, CI contexts, `igx_gov ticket`, the P7-00 baseline suite, any live-provider or network test, `ruff`/mypy.
- A real Execution Engine resume across recurring occurrences was **not** performed (F-D7-13 is a probe of the skip path with a synthetic checkpoint advance). The `HostTaskResumeExecutor` path and the human-timeout ledger (`claim_action`) were read, not stress-tested.
- Concurrency and crash behaviour beyond the probes (SQLite on Windows only; the multi-instance queue race used an in-memory store with an artificial 0.5 ms write delay to widen the window; true multi-process document-store access was not tested).
- Not re-audited: D4 host lifecycle, D6 credential admission (XD-06 test-side port), D1..D3/D5/D8, the Celery dispatcher's retry/rate-limit logic beyond reading, `identity_persistence`/`identity_dual_read`/`identity_record_codec` internals, `broker_*` providers, MCP-RC, FRZ coverage and SHA-ladder consistency (IA-GLOBAL/P8).
- Mutation testing was not performed; test adequacy is judged by reading and by the probes that pass through the all-green suite.
- `certifier_session_id`, `evidence_publication_sha` and `certification_publication_sha` in the manifest are placeholders for the certifier/publisher. No certification, FRZ promotion, closure or production-readiness claim is made.

## 8. Independence and provenance

IA-D7 is a clean session that did not implement B-D7 or any PDC-X batch, did not sign or certify D7 items and did not fix anything it audited. Product/test/script/governance paths are unchanged in this branch (docs only). Commit authorship cannot establish session independence (all PDC-X commits share one git author); independence rests on session separation as stated by the orchestrator. Evidence index: `iiac_evidence/PDC-X/D7/` (`audit_scope.json`, `iiac_audit_d7_findings.json`, `iiac_audit_session_manifest.PDC-X.d7.json`, `test_matrix.txt`, `EVIDENCE_MANIFEST.txt`, `logs/`, `adversarial_scripts/`).
