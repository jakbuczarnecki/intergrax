# PLAN T-IGX-051-E — RW train node T-AF2-RW-RECOVERY: background and recovery (drain controller, wait_for_idle, CAS recovery, durable CAS, UNCERTAIN reconciler, capped retry, producer identity)

Draft, 2026-11-17. Nothing was run and nothing was changed outside `.tmp`. No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this plan. Evidence notation `path@H:line`; `H` = `2727a2c2364a97492202b931375772a400279a9b` (`origin/development` fetched 2026-11-17; its product tree equals AUDIT_BASELINE_2 `7960ae7f6`, `git diff --stat 7960ae7f6 origin/development -- intergrax agents applications` is empty). Every write-set product path was also compared between `H` and the train branch `origin/arch-freeze/rw-train` (`05699b633`, PT-WA merged): no difference, so the line numbers below hold on the train. Labels such as "PARTIAL", "PASS_SCOPED", "FAIL" are quoted from the cited registers and are not statements of this plan.

## 1. Purpose and findings

Train node `T-AF2-RW-RECOVERY` (RW-PLAN v13 table T12, lane 3; ticket `T-IGX-051-E`). The ticket carries the background and recovery remediation of the PDC residual package (plan of record section E, row "REMEDIATION — background / recovery"). It repairs; it claims no closure. Closure of each ID is an independent acceptance at the freeze SHA.

| ID | What the register says (quoted, shortened) | What this ticket does | What it cannot close alone |
|---|---|---|---|
| EX-L-06 | F-D7-20: `DocumentStoreTaskWorker.stop()` abandons a running thread; a later `start()` can run two consumers (`PDC_EXCEPTION_REGISTER.md@H:58`) | Part A: `stop()` no longer forgets a live thread; `start()` refuses while the old consumer lives | the worker-side drain controller (Part C) |
| EX-L-03 | `recover_interrupted_running` is not compare-and-set; repair cannot tell a crash from a write in flight (`@H:55`) | Part B: per-row compare-and-set on a conditional store | a worker lease or heartbeat on queue rows (register: "commissioned"; not in this write set); over a plain `DocumentStore` the module stays single-instance by contract (`document_store_task_queue.py@H:10-14`) |
| EX-D7-01 (F-D7-03 HIGH, F-D7-20) / PDC-D7-R014 | no class implements `HostedApplicationActiveWorkController`; `WorkerRuntime` has no stop/drain API; XD-05 passes only through a test-side adapter (`@H:168`) | Part C: a product controller over `WorkerRuntime` and the document-store worker | re-pointing XD-05 at the product class is a test edit outside A1 (question Q-RC-05) |
| EX-D4-01 (F-D4-02) | no product implementation of `wait_for_idle`; `_build_engine_factory` builds the engine without the seams (`@H:129`) | Part C: the runner seam and a runner-level drain test | wiring of a concrete controller into each product host composition (outside A1; question Q-RC-03) |
| EX-D1-07 (F-D1-11, G-08) | the durable CAS neutralizes `revision_state` and `activated_at` and reduces bindings to slug references on both sides (`@H:81`) | Part D: the comparison covers lifecycle fields and the full bindings | a lossy descriptor decode, if it is the cause of the slug reduction, lives in a file outside A1 (question Q-RC-02) |
| PDC-D7-R015 | "no UNCERTAIN reconciler; the plan's double-execution check has no recorded result" (PDC residual file, row PDC-D7-R015) | Part E: a reconciler module with a recorded per-entry result | the stores that hold UNCERTAIN rows (`store.py`, `postgresql_task_checkpoint_store.py`) are outside A1 (question Q-RC-01); the row stays PARTIAL until a store implements the source port |
| G-20 | a permanently failing read or materialization is retried every tick with no cap (`PDC_TO_ARCH_FREEZE_RESIDUAL_QUALIFICATION.md` row EX-L-10; RW-PLAN A-08) | Part F: a bounded release count per schedule, then a recorded skip | the count is process-local (a durable count needs a store field outside A1) |
| EX-D7-03 (F-D7-09) | "no producer sets `identity=`" (`@H:170`); DS-EXC option: durable producers set an identity | Part G: the two A1 sites carry `TaskRequest.identity` | 3 further constructor sites are outside every train ticket (question Q-RC-04) |

Binding scope: RW-PLAN v13 appendix A1 block `T-AF2-RW-RECOVERY (T-IGX-051-E)` (12 product paths, 5 tests, 0 conditional) with the negative-test rows NT-ROW-109..116, the regression row of section (e) and the overlap pair `PT-FC2 × T-AF2-RW-RECOVERY` of section (b). Nothing is added to or removed from that block (INPUT file, differences: none). Items that look out of scope are in section 14 as questions.

Operator and orchestrator inputs that bind this plan: DS-EXC dispositions of EX-L-06, EX-D7-01, EX-D4-01, EX-L-03, EX-D1-07 as REMEDIATION and EX-D7-03 as the DS-EXC option "durable producers set an identity" (plan of record section E; RW-PLAN rows `:737-:870`); OD-AF-11 ratchet (this ticket edits nothing under `intergrax/contracts/`, section 10); OD-AF-22 (approval route: one wave packet approved by the Operator, the `Approved:` block written by the signing delegate, never by this draft or the implementer); Q-RW-19 does not concern this ticket.

## 2. Design rules applied to every part

1. No new `Any`, `object` annotation, `getattr`, `hasattr`, `setattr`, `importlib` or `__import__` dispatch on a new port or dataclass; ports are `Protocol` classes with concrete signatures.
2. Fail closed. A refused state returns or raises; it never degrades to "best effort". A plain (non-conditional) `DocumentStore` keeps its documented single-instance contract and the code says so.
3. No edit of an existing test. The five new test files are the only test write paths. A pinned existing test that turns red is a STOP (sections 6 and 10), not an edit.
4. Tier boundaries: `intergrax/background_tasks/` must not import `intergrax.hosting`; the controller satisfies the hosting `Protocol` structurally and the test imports the hosting `Protocol` to prove it. `intergrax/queueing/` must not import `intergrax.background_tasks`. `intergrax/` imports nothing from `agents/` or `applications/` (the `worker_handler.py` edit is Tier-3 and may import `intergrax`).
5. Behaviour changes that can surprise a caller are named in the part that makes them (Part A: `start()` after an unfinished `stop()`; Part F: a recorded skip after the cap).
6. Identity is carried, never minted here (the `TaskExecutionIdentity` docstring, `task_queue.py@H:84-94`, ADR-PDC-07).

## 3. Part A — EX-L-06: no second consumer after `stop()` (firm: `colocated_worker.py`)

Facts. `stop()` sets the stop event, joins for `timeout_seconds` and then sets `self._thread = None` whether or not the thread is still alive (`colocated_worker.py@H:103-108`). `start()` returns early only when `self._thread` is not None and alive (`:80-82`), otherwise it calls `self._stop.clear()` (`:95`) and starts a new thread (`:96-101`). After a timed-out `stop()`, the old thread is still inside `drain_once` (`:110-160`), `self._thread` is None, and `start()` clears the very event the old thread waits on, so the old loop (`:207-210`) keeps running next to the new one: two consumers.

Design.
- `stop()` keeps the thread reference when the join times out and returns `bool` (True = the consumer has ended and the reference was dropped). The existing positional/keyword signature (`*, timeout_seconds: float = 10.0`) is unchanged; callers that ignore the return value keep working.
- `start()` raises `DocumentStoreTaskWorkerStillRunning` (a new `RuntimeError` subclass defined in the same module; no new file) when a previous consumer is still alive after a `stop()`. A `start()` on a live, non-stopping worker stays a no-op (`:81-82`).
- Each `start()` creates a new `threading.Event` for its loop and passes it to `_run_loop`, so a late thread can never observe a cleared event of a later run. This is a belt-and-braces measure: with the refusal above there is no later run while the old thread lives, but the event hygiene removes the shared-state hazard itself.
- In-flight accounting for Part C lives here as three small methods (`stop_intake()` is the existing `_stop.set()` path exposed as a method that does not join; `in_flight_count()` counts claimed rows being handled inside `drain_once`; `wait_idle(timeout_seconds) -> bool`). `drain_once` increments before the first claimed row and decrements in a `finally` per row. These are plain methods on the class; the class does not import `background_tasks`.

Behaviour change to name: a caller that calls `start()` after a `stop()` that timed out used to get a second consumer; it now gets the exception. The two product compositions that start the worker (`intergrax/runtime/vendor_knowledge/sync_runtime.py` and `applications/local_workspace_application/workspaces/sync_runtime.py`, both outside A1) call `start()`/`stop()`; they are regression scope (section 10), not write scope.

## 4. Part B — EX-L-03: compare-and-set recovery (firm: `document_store_task_queue.py`)

Facts. `recover_interrupted_running` reads the running index (`:595`), loads each row (`:606`), and writes the settled row with an unconditional `put` through `_put` (`:616`, `:622`), then deletes the index row (`:623`). The module already owns a compare-and-set helper, `_cas_update` (`:547-577`), used by `retry_or_dead_letter` (`:456`) and `_settled_fields` (`:478-506`). On a `ConditionalDocumentStore` the compare is `replace_if_match`; on a plain store it is a `put` (`:572-574`).

Design.
- Replace the load-then-put by one `_cas_update(tenant, task_id, mutate)` per index row. `mutate` returns None (reject) unless the row is still `RUNNING`; for a row that is not `RUNNING` any more (a concurrent worker finished it, another instance already recovered it) the row is left untouched and not reported as interrupted.
- The settle step (budgeted: `_settled_fields` with `max_retries`; unbudgeted: `FAILED`) moves inside `mutate`, so the write is compared to the row that was read in the same attempt.
- The running-index row is deleted only after the CAS result is `applied` or the row is observed in a non-RUNNING state, never after a lost race on a still-RUNNING row (a lost race re-reads, as `_cas_update` already does).
- The returned list contains only rows whose CAS was applied. Over a plain `DocumentStore` the result is the old single-instance behaviour; the docstring says that and does not claim multi-instance safety. A crash versus a slow writer is still not distinguishable by timestamp alone: this part prevents the overwrite, it does not add a lease (register: "commissioned", not in this ticket).
- `colocated_worker.start()` (`:84-86`) is unchanged in signature and now receives only applied rows.

## 5. Part C — EX-D7-01 and EX-D4-01: drain controller and `wait_for_idle` (firm: `background_tasks/*`, `colocated_worker.py`, `hosting/runner.py`, `hosting/engine/engine.py`, `hosting/shutdown.py`, `queue_worker_wiring.py`)

Facts. The hosting port is `HostedApplicationActiveWorkController` with `stop_intake`, `active_work_count`, `wait_for_idle`, `cancel_active_work` (`shutdown.py@H:38-48`). The engine carries `active_work_controller` (`engine.py@H:111`, default None) and `flush_services` (`:112`) and hands both to `HostedApplicationShutdownExecutor` (`:706-712`). The executor runs STOP_INTAKE, then DRAIN through `wait_for_idle` and CANCEL through `cancel_active_work` according to the strategy (`shutdown.py@H:240-257`, `:264-358`); with no controller each phase is recorded `SKIPPED`. `_build_engine_factory` constructs `HostedApplicationEngine(...)` without either seam (`runner.py@H:236-247`). `WorkerRuntime.process_request` (`worker_runtime.py@H:110`) has no counter, no intake gate and no wait method (the file has no `stop`, `drain`, lock or thread token). The test-side adapter that XD-05 uses is `_WorkerActiveWork` (`tests/qualification/pdc_x/xd/test_xd_05_host_background.py@H:99`; the module docstring at `:11` says the product ships the port but no implementation).

Design.
1. `intergrax/background_tasks/worker_runtime.py`: `WorkerRuntime` gets an intake gate and an in-flight counter guarded by one `threading.Condition`: `stop_intake()`, `in_flight_count()`, `wait_idle(timeout_seconds) -> bool`. `process_request` checks the gate first and raises `WorkerIntakeClosedError` (new, in the same file) before any state write or event, increments on entry and decrements in a `finally`. A request refused at the gate performs no side effect, so it can be redelivered after restart without a double execution.
2. New module `intergrax/background_tasks/active_work.py` (under the A1 glob `intergrax/background_tasks/*`; the glob matches 6 files at `H`, this is a 7th): a small `Protocol` `ActiveWorkSource` (`stop_intake`, `in_flight_count`, `wait_idle`) and the class `BackgroundActiveWorkController(sources: Sequence[ActiveWorkSource], cancel_grace_seconds: float)` implementing the four hosting methods: `stop_intake` closes every source's intake; `active_work_count` sums `in_flight_count`; `wait_for_idle` awaits `wait_idle` of each source in a thread (`asyncio.to_thread`), without a timeout of its own (the executor bounds the phase through `run_bounded_phase`, `shutdown.py@H:131-145`); `cancel_active_work` is cooperative: it keeps intake closed and waits `cancel_grace_seconds`, it cannot kill a synchronous handler thread, so `active_work_count()` may stay above zero and the executor records `forced` (this is stated in the docstring; the plan does not claim preemption). `WorkerRuntime` and `DocumentStoreTaskWorker` satisfy `ActiveWorkSource` structurally. No import from `intergrax.hosting`.
3. `intergrax/hosting/runner.py`: `_RunnerFactories` gets one field with a default (`create_active_work_controller: Callable[[], HostedApplicationActiveWorkController | None]`, default returns None) and `run_hosted_application` gets a keyword `active_work_controller_factory` that replaces it, in the shape of the existing `event_publisher_factory` (`runner.py@H:297-317`). `_build_engine_factory` passes the controller to every engine instance it builds (`:236-247`). Defaults keep every existing caller and test working. The optional `flush_services` seam of the same register row is added the same way (default empty tuple) if question Q-RC-06 is answered yes (draft default yes; it adds no behaviour without a caller).
4. `intergrax/hosting/engine/engine.py` and `intergrax/hosting/shutdown.py` are firm paths of A1. The engine and the executor already call the controller (facts above). No behaviour edit is planned. The red-first test NT-RW-REC-04 drives the real engine through the runner seam; if it shows that a controller whose `wait_for_idle` completes while `active_work_count()` is still above zero is reported as drained, the implementer repairs that in `shutdown.py:309-317` (it is already handled: `completed` requires a zero count) or reports that no edit was needed. An unused firm path is allowed; an edit outside the A1 paths is not.
5. `intergrax/applications/_shared/queue_worker_wiring.py` (serialized after PT-FC2, section 12): `QueueWorkerWiring` (`:31-36`) gets one optional field `active_work_sources: tuple[ActiveWorkSource, ...] = ()` and docstring text saying that the inline and eager-Celery paths (`:61-67`, `:69-106`) have no queue consumer to drain and leave it empty. The edit is decided after PT-FC2 has merged and its final shape of this file is read; if PT-FC2 leaves no room, the field is the only change. The source-scan gates named in the PT-FC2 plan (`production_mode: bool`, `production_mode=production_mode`, `HostTaskExecutionRunAdapter`, `QueuedHostTaskExecutionAdapter`, `host_execution: HostTaskExecutionPort`, no `task_runner:`) must stay green. The document-store worker compositions (`vendor_knowledge/sync_runtime.py`, `local_workspace_application/workspaces/sync_runtime.py`) are outside A1; until a follow-up wires them, a product host composes the controller only through the new seam (question Q-RC-03). EX-D4-01 therefore has a product class, a runner seam and a runner-level drain test, which is the closing condition the register states (`@H:129`: "a product active-work adapter ... exists, then a runner seam plus a runner-level drain test").

## 6. Part D — EX-D1-07: full durable compare-and-set (firm: `durable_registry_projection_input_store.py`)

Facts. `_in_persisted_representation` replaces `revision_state` with `VALIDATED` and `activated_at` with None on both sides (`durable_registry_projection_input_store.py@H:40-42`) and round-trips the manifest through JSON (`:43-46`); `register` compares the two normalized fingerprints (`:112-118`). The fingerprint itself covers `runtime_revision`, `effective_roster`, `manifest`, the artifact digest, the factory authorities and the build context (`registry_projection.py@H:218-228`). The comment at `:109-111` says the descriptor decodes integration bindings to slug references.

Design.
- Compare the lifecycle fields as they are: remove the neutralization of `revision_state` and `activated_at`. A different value on the durable side than on the registered bundle is a concurrent change and raises `RegistryProjectionError` (the existing error type, `:118`).
- Keep the manifest JSON round-trip (a representation normalization that does not drop content) and compare the bindings in full. Whether the full-bindings requirement can be met inside this file depends on where the slug reduction happens: the store file only normalizes the manifest; if the slug reduction is in the descriptor decode (`registry_projection_descriptor.py`, not in A1), the implementer stops and reports (question Q-RC-02) instead of re-weakening the compare or editing the descriptor.
- The compare stays a registration-time check under the existing lock discipline (`:119-127`); the in-process fingerprint conflict check (`:89-94`, `:119-124`) is unchanged.
- The existing pin `tests/qualification/pdc_x/p8_repair/test_r6_durable_projection_slug.py` (a different-slug refusal) and `tests/unit/applications/test_prod_q_r1_r7_d3_production_restart_reconciliation.py` are the regression nodes: they must stay green. If a legitimate lifecycle progression between "inputs built while VALIDATED" and "durable authority ACTIVE" turns the second one red, that is a STOP and a finding, because the neutralization was introduced for exactly that flow (docstring `:33-38`); the plan does not decide which side is authoritative for lifecycle progress (question Q-RC-02, draft default: full compare, stop on the red).

## 7. Parts E, F — PDC-D7-R015 and G-20: scheduler (firm: `long_running/scheduler.py`, new `long_running/uncertain_reconciler.py`)

Facts. A lease-expired `RUNNING` schedule row becomes `UNCERTAIN` inside `claim_due` (`store.py@H:504-515`), a malformed `PENDING` row is quarantined as `UNCERTAIN` (`:526-540`), and the scheduler ledger turns a lease-expired `STARTED` action into `UNCERTAIN` (`:703-790`). The status exists in the model (`scheduled_resume.py@H:62-69`). Nothing reads `UNCERTAIN` rows afterwards: the persistence port has no listing by status and no resolve transition (`scheduled_resume.py@H:97-141`). In the scheduler, two paths release a claim for retry with no counter: an eligibility read that raised for a recurring schedule (`scheduler.py@H:211-220`) and a materialization failure other than `ValueError` or `OverflowError` (`:309-315`); both end in `_release_claim` (`:318-328`). `OverflowError` is already mapped to `skip_claim` (`:302-308`).

Part E design (new module, no store edit).
- `UncertainScheduleSource` (`Protocol`): `list_uncertain(limit) -> Sequence[ScheduledResume]` and `resolve_uncertain(schedule_id, *, expected_fence, outcome) -> bool` (a fenced transition; False = lost the race, nothing changed).
- `UncertainOutcome` (enum): `EXECUTION_PROVEN` (the terminal authority holds a terminal record for the task: the row is closed as completed, never redriven), `NOT_STARTED_PROVEN` (no attempt and no terminal record exist: eligible for one redrive), `UNPROVEN` (evidence is missing: the row stays UNCERTAIN, a recorded result says so, an operator decides). Draft default policy is fail-closed: automatic redrive happens only on `NOT_STARTED_PROVEN`; every other state is left for a human.
- `UncertainReconciler(source, evidence)` with `reconcile_once(now) -> tuple[ReconciliationRecord, ...]`. `evidence` is a `Protocol` over the existing terminal authority (`ExecutionTerminalService.get_terminal_record(tenant_id=..., task_id=...)` as already read at `scheduler.py@H:267`) and the attempt lifecycle read; no new authority is created. Each record holds schedule id, outcome, whether the fenced resolve applied, and the double-execution check result (attempts observed for the task, must be at most 1). The records are returned and logged under one stable code; persisting them is the caller's choice (no new store).
- "Exactly once": the resolve is fenced, a second `reconcile_once` over the same row finds it already resolved and returns no record for it.
- `LongRunningScheduler.__init__` gets an optional `uncertain_reconciler` (default None: unchanged behaviour). `tick` (`scheduler.py@H:129-143`) runs the reconciler as a third isolated phase, like the two existing ones (a failure is collected and re-raised after the others ran).
- Limit stated in the module docstring and in section 13 (R6): the SQLite and PostgreSQL stores do not implement `UncertainScheduleSource` today and cannot be made to do so inside the A1 write set (question Q-RC-01). The unit test uses an in-memory source. PDC-D7-R015 therefore stays PARTIAL for a product host until a store implements the port.

Part F design (cap, G-20).
- `LongRunningScheduler.__init__` gets `max_claim_release_attempts: int = 5` (validated: at least 1). A process-local `dict[str, int]` counts consecutive releases per `schedule_id`.
- `_release_claim` is wrapped: after the cap is exceeded the scheduler does not release again; it calls `skip_claim` (the claim ends `SKIPPED`, nothing executed), logs one error with a stable code (`scheduled_resume_release_cap_reached`) and drops the counter. The counter is also dropped when the claim completes or is skipped by any other path, so the dict is bounded by claims in flight.
- Behaviour change to name: a recurrence chain whose eligibility read or materialization fails permanently now ends after the cap instead of retrying forever. This is the G-20 intent; the alternative (park in `UNCERTAIN`) is not available without a store edit (question Q-RC-01). G-02 (a transient eligibility failure must not end the chain, `:211`) stays true: the counter only reaches the cap on consecutive failures of the same schedule. The count is lost on process restart (stated, not hidden).
- `OverflowError` mapping stays as it is at `:302-308`; the test pins it (control, green at `H`).

## 8. Part G — EX-D7-03: producer identity (firm: `retention_sweep.py`, `worker_handler.py`)

Facts. Of 6 non-doc `TaskRequest(` constructor sites at `H`, only the document-store decode path passes `identity=` (`document_store_task_queue.py@H:630-636`); the five others are `worker_handler.py:99`, `retention_sweep.py:266`, `queued_host_task_execution_adapter.py:57`, `vendor_knowledge/sync_jobs.py:235` and `tools/providers/message_bus/service.py:50` (`closed-world/RESULT.md` section 2.3, re-listed with `git grep` at `H`). Two of them are inside A1.

Design (limited to the A1 sites).
- `retention_sweep.py:250-272`: `memory_retention_sweep_task_request` gets a keyword `identity: TaskExecutionIdentity | None = None` and passes it to `TaskRequest`. The builder has no product caller at `H` (`git grep` finds only its own `__all__` entry), so this is a signature adoption; a producer that enqueues with a minted identity passes it through. The sweep payload still carries no authority (`MemoryRetentionSweepPayload` docstring, `retention_sweep.py@H:229`).
- `worker_handler.py:99`: the request built inside the handler passes `identity=TaskExecutionIdentity(task_id=..., run_id=..., attempt_id=..., execution_id=...)` taken from the `execution_identity` the handler already receives and validates against the tenant and run (`:76-90`); no id is minted.
- The 3 sites outside A1 are not edited. NT-RW-REC-08 pins the adopted sites and lists the 3 open sites by path in the test as an explicit residual tuple, so EX-D7-03 is visibly not complete (question Q-RC-04).

## Write set

Branch rule: product commits of this ticket go only to branch `arch-freeze/rw-train`. Counts: 12 firm product paths (1 glob that holds one new file, 1 new, 10 edited), 5 firm test paths (all new), 0 conditional. The block equals RW-PLAN v13 appendix A1 `T-AF2-RW-RECOVERY (T-IGX-051-E)`; every path exists at `H` or is new with an existing parent directory (verified, FACT_CHECK F-30..F-31).

### Product paths (ACT-N11), edited — firm

```
intergrax/background_tasks/*
intergrax/queueing/providers/document_store/document_store_task_queue.py
intergrax/queueing/providers/document_store/colocated_worker.py
intergrax/hosting/engine/engine.py
intergrax/hosting/shutdown.py
intergrax/hosting/runner.py
intergrax/runtime/long_running/scheduler.py
intergrax/applications/_shared/durable_registry_projection_input_store.py
intergrax/applications/_shared/queue_worker_wiring.py
applications/local_workspace_application/background_ingest/worker_handler.py
intergrax/memory/retention_sweep.py
```

`intergrax/background_tasks/*` expands to the 6 files at `H` (`__init__.py`, `definition.py`, `events.py`, `registry.py`, `state_store.py`, `worker_runtime.py`) plus the new `active_work.py` (Part C). Expected edits: `worker_runtime.py` and the new module; the other five stay untouched unless a probe shows a need. `engine.py` and `shutdown.py` may end with no edit (Part C item 4).

### Product paths (ACT-N11), new — firm

```
intergrax/runtime/long_running/uncertain_reconciler.py
```

### Product paths, conditional

None.

### Tests, new — firm (the only test write paths)

```
tests/unit/queueing/test_rw_recovery_no_double_execution.py
tests/unit/hosting/test_rw_recovery_drain_wait_for_idle.py
tests/unit/runtime/long_running/test_rw_recovery_uncertain_reconciler.py
tests/unit/applications/test_rw_recovery_durable_cas_projection.py
tests/unit/queueing/test_rw_recovery_producer_identity.py
```

### Docs

None. No edit of gate scripts, `.github/`, `pyproject.toml`, `uv.lock`, the roadmap, the FRZ checklist, IIAC or PDC-X records, `intergrax/contracts/**`, any existing test, `store.py`, `scheduled_resume.py`, `postgresql_task_checkpoint_store.py` or `registry_projection_descriptor.py`.

## Negative tests

One row per RW-PLAN negative-test row (NT-ROW-109..116 = NT-RW-REC-01..08). Basis letters are RW-PLAN's (B = behaviour red at the pre-fix SHA, S = structural). Every test below must fail at the pre-fix SHA (the train head before this ticket's first product commit); a test that fails there only because a new API is absent counts, because the absence is the finding (EX-D7-01, PDC-D7-R015). Where a row also has a leg that is green at `H`, that leg is a separate control test and is marked C; it does not count as red-first.

| ID | RW row | Basis | Setup | Expected after the fix | Why red at the pre-fix SHA | Node id |
|---|---|---|---|---|---|---|
| NT-RW-REC-01 | NT-ROW-109 (EX-L-06) | B | in-memory conditional store, one task whose handler blocks on an event; `start()`, wait for the claim, `stop(timeout_seconds=0.05)` | `stop` returns False and the consumer thread stays referenced; `start()` raises `DocumentStoreTaskWorkerStillRunning`; the handler has run once; after release `stop` returns True and a new `start()` works | at `H` `stop` sets `_thread = None` (`colocated_worker.py:108`) and `start()` clears the stop event (`:95`) and starts a second consumer | `tests/unit/queueing/test_rw_recovery_no_double_execution.py::test_rw_rec_01_stop_then_start_never_two_consumers` |
| NT-RW-REC-02 | NT-ROW-110 (EX-L-03) | B | `DocumentStoreTaskQueue` over `InMemoryDocumentStore` (a `ConditionalDocumentStore`); one RUNNING row; a store subclass completes the task (`mark_succeeded`) between the recovery read and the recovery write | the row stays `SUCCEEDED`; it is not in the returned interrupted list; the index row is gone; a second recovery returns an empty list | at `H` the write is an unconditional `put` (`document_store_task_queue.py:616`, `:622`) and overwrites the terminal row | `tests/unit/queueing/test_rw_recovery_no_double_execution.py::test_rw_rec_02_recovery_is_compare_and_set` |
| NT-RW-REC-03 | NT-ROW-111 (EX-D7-01) | B | a real `WorkerRuntime` with a blocking handler; `BackgroundActiveWorkController([runtime])`; also a restart variant: after `stop_intake` a second `process_request` | `isinstance(controller, HostedApplicationActiveWorkController)`; `active_work_count()` is 1 while the handler runs; `wait_for_idle` returns only after release; the post-stop request raises `WorkerIntakeClosedError` with no state write; the handler ran once across the stop and a restart over the same idempotency store | the module `intergrax.background_tasks.active_work` and the intake API do not exist at `H` (import error is the failure) | `tests/unit/hosting/test_rw_recovery_drain_wait_for_idle.py::test_rw_rec_03_product_controller_drains_and_refuses_new_work` |
| NT-RW-REC-04 | NT-ROW-112 (EX-D4-01) | B | the real hosted engine through `run_hosted_application(profile, active_work_controller_factory=...)` following the pattern of `tests/unit/hosting/test_runner.py` and XD-05; a handler in flight, a stop request under `DRAIN_THEN_CANCEL` | the DRAIN phase record is `COMPLETED` with `active_work_after == 0`, recorded before the runtime-stop phase; the controller's `wait_for_idle` was called by the engine | at `H` `run_hosted_application` has no such keyword and `_build_engine_factory` builds the engine without a controller (`runner.py:236-247`) | `tests/unit/hosting/test_rw_recovery_drain_wait_for_idle.py::test_rw_rec_04_engine_waits_for_idle_through_runner_seam` |
| NT-RW-REC-05 | NT-ROW-113 (EX-D1-07) | B | `DescriptorBackedRegistryProjectionInputStore` over in-memory revision and descriptor stores; a durable descriptor exists; `register` a bundle that differs only in `revision_state`, then only in `activated_at`, then in an integration binding beyond its slug | each registration raises `RegistryProjectionError`; an identical bundle registers; a different slug still raises (control, C: green at `H`) | at `H` `_in_persisted_representation` neutralizes the two lifecycle fields (`:40-42`), so the first two registrations are accepted | `tests/unit/applications/test_rw_recovery_durable_cas_projection.py::test_rw_rec_05_lifecycle_and_bindings_are_compared` |
| NT-RW-REC-06 | NT-ROW-114 (PDC-D7-R015) | S | in-memory `UncertainScheduleSource` with three UNCERTAIN entries (terminal record present, nothing recorded, partial evidence); `UncertainReconciler.reconcile_once` twice | one record per entry with the three outcomes; the double-execution check result is recorded (attempts seen, at most 1 or a violation flag); the fenced resolve applies once; the second pass returns no record for resolved entries; no entry is redriven unless `NOT_STARTED_PROVEN` | at `H` no reconciler module exists | `tests/unit/runtime/long_running/test_rw_recovery_uncertain_reconciler.py::test_rw_rec_06_uncertain_entries_reconciled_exactly_once` |
| NT-RW-REC-07 | NT-ROW-115 (G-20) | B | a `LongRunningScheduler` with a fake schedule store; (a) `materialize_next` raising `RuntimeError` on every tick; (b) a recurring schedule whose eligibility read raises on every tick; (c, control C) `materialize_next` raising `OverflowError` | (a) and (b): at most `max_claim_release_attempts` releases, then one `skip_claim` and one error log with the stable code; (c): `skip_claim` at once, no release | at `H` (a) and (b) release the claim on every tick without bound (`scheduler.py:219`, `:314`); (c) is already mapped (`:302-308`) and is green at `H` | `tests/unit/runtime/long_running/test_rw_recovery_uncertain_reconciler.py::test_rw_rec_07_permanent_failure_release_is_capped` (+ `::test_rw_rec_07c_overflow_is_mapped_to_skip` as control) |
| NT-RW-REC-08 | NT-ROW-116 (EX-D7-03) | S | AST over the three adopted constructor sites (`worker_handler.py`, `retention_sweep.py`, the document-store decode path) plus a behavioural check: the retention builder returns a request whose `identity` equals the one passed | all adopted sites pass `identity=`; the 3 sites outside A1 are named in one explicit residual tuple (`queued_host_task_execution_adapter.py`, `vendor_knowledge/sync_jobs.py`, `tools/providers/message_bus/service.py`) | at `H` `worker_handler.py:99` and `retention_sweep.py:266` pass no identity | `tests/unit/queueing/test_rw_recovery_producer_identity.py::test_rw_rec_08_adopted_enqueue_sites_pass_identity` |

Totals: 8 NT rows, 9 test functions (NT-RW-REC-07 has a control; the rest are one function each; NT-RW-REC-01 and -03 may split their control legs into separate functions, which does not change the row count). Red-first at the pre-fix SHA: 8 of 8 rows (REC-07 through its cap legs); the controls (REC-05 different-slug leg, REC-07c) are green there by design and are labelled.

## 10. Regression scope and test commands

One process at a time, one `uv run pytest` per verification step with several paths, under the lock `D:/Projekty/_wt/_locks/pytest.lock` (create the directory with `mkdir`; the creator removes it with `rmdir`, also on failure), logs under `.tmp/session/arch-freeze/rw-recovery-impl/`, `-p no:cacheprovider`. No broad or diagnostic suite, no Docker, no vendor integration (C10/C15 freeze; `intergrax-env-vendor-qualification`).

| Step | Scope | Purpose |
|---|---|---|
| S0 red-first | the five new test files on a detached worktree at the pre-change train commit | 8 NT rows red as designed, controls green; log kept for the auditor |
| S1 RW-PLAN section (e) row | `tests/unit/queueing`; `tests/unit/hosting`; `tests/unit/background_tasks`; `tests/unit/runtime/long_running`; `tests/qualification/pdc_x/xd` | the regression scope RW-PLAN binds for this ticket (queueing, hosting, background tasks, long-running, XD pairs) |
| S2 new tests | the five new files after the change | all green |
| S3 static | `uv run ruff check` and `uv run pyright` on the changed Python files | typing and style |

Additional regression nodes proposed by this plan (read-only runs; they are not in the RW-PLAN section (e) row, so they need orchestrator acceptance, question Q-RC-07). Each file exists at `H` and exercises a changed path: `tests/qualification/pdc_x/d7/test_d7_dead_letter.py` and `tests/unit/queueing/test_document_store_task_queue.py` (recovery); `tests/qualification/pdc_x/d7/test_d7_enqueue_identity.py` (producer identity); `tests/qualification/pdc_x/d7/test_d7_recurrence.py`, `tests/qualification/pdc_x/d7_repair/test_s1_batch_isolation.py`, `test_s2_chain_survival.py`, `tests/qualification/pdc_x/p8_repair/test_r2_scheduler_eligibility.py` (scheduler); `tests/qualification/pdc_x/p8_repair/test_r6_durable_projection_slug.py` and `tests/unit/applications/test_prod_q_r1_r7_d3_production_restart_reconciliation.py` (durable store); the five `applications/local_workspace_application/tests/workspaces/` files that name `colocated_worker` (worker start/stop). No pass is claimed for any of them by this plan.

Typing ratchet (OD-AF-11): this ticket edits nothing under `intergrax/contracts/`. The wave-packet contract check (`rw-plan/work/typ05/measure_typ05.py` plus `typ_measure.py`) is therefore expected to report nothing new; if the implementer finds a need to touch `intergrax/contracts/`, that is a STOP.

## 11. Order of work

0. Precondition: PT-FC2 has merged into `arch-freeze/rw-train` (section 12) and the approval block of this ticket exists; read `queue_worker_wiring.py` as PT-FC2 left it.
1. S0: write the five test files, run them red-first at the pre-change commit (one process).
2. Part A (`colocated_worker.py`) and Part B (`document_store_task_queue.py`): NT-RW-REC-01, -02 green.
3. Part C: `worker_runtime.py`, `active_work.py`, `runner.py`, `queue_worker_wiring.py`: NT-RW-REC-03, -04 green. `engine.py` and `shutdown.py` only if the probe demands.
4. Part D: `durable_registry_projection_input_store.py`: NT-RW-REC-05 green, or STOP per section 6.
5. Parts E and F: `uncertain_reconciler.py`, `scheduler.py`: NT-RW-REC-06, -07 green.
6. Part G: `retention_sweep.py`, `worker_handler.py`: NT-RW-REC-08 green.
7. S1, S3, then the pre-push gate `uv run python -m scripts.governance.igx_gov ticket`. Commits carry the trailer `Ref: governance/tickets/T-IGX-051-E.md`; no AI-tool attribution anywhere.

## 12. Overlaps and merge order

Lane 3 (RW-PLAN section c). Predecessors: PT-WA (merged into the train, `05699b633`), PT-WB (in progress at drafting), PT-FC2 (file overlap on `intergrax/applications/_shared/queue_worker_wiring.py`; hard precedence "PT-FC2 -> T-AF2-RW-RECOVERY: plan Z: serialized with PT-FC2 on queue wiring"). **This ticket is serialized after PT-FC2 on the queue wiring and must not start its `queue_worker_wiring.py` edit before PT-FC2 has merged into the train.** No other pair of the 19 overlapping pairs lists this ticket. The three constructor sites outside A1 (Part G), `store.py`, `scheduled_resume.py`, `registry_projection_descriptor.py` and the two `sync_runtime.py` compositions are not paths of any train ticket (search of `RW_PLAN.md` appendix A1 and `rw_plan.v1.json`: no match). `intergrax/runtime/long_running/postgresql_task_checkpoint_store.py` is a conditional path of PT-WB (`RW_PLAN.md:1054`, block `PT-WB (T-IGX-043-H)`): if Q-RC-01 is answered "add the store files", that file would create a new pair with PT-WB and the overlap matrix and lane order must be re-derived by the orchestrator.

ACT-N11 / IIAC: every product path above breaks the IIAC activation until the single train-level re-pin; this ticket does not perform it and must not merge outside the wave (project memory and the same statement in the PT-WA ticket).

## 13. What closes each ID, risks, rollback

Closure conditions (none is claimed here): an independent acceptance at the freeze SHA with the section 10 results and mutation checks of the auditor's design. PDC-D7-R015 and EX-D1-07 have explicit residuals (Q-RC-01, Q-RC-02).

Risks.
- R1 `start()` after an unfinished `stop()` now raises (Part A). Mitigation: the two product compositions are in regression scope.
- R2 The CAS recovery on a plain `DocumentStore` is not safer than before (Part B); the docstring says so.
- R3 Counting in `WorkerRuntime.process_request` sits on the hot path of every background task (Part C). Mitigation: one short lock around a counter, no I/O under the lock.
- R4 `_RunnerFactories` and the runner signature gain defaulted members; a test that builds `_RunnerFactories` positionally would break. Mitigation: the new field is last with a default; `tests/unit/hosting/test_runner.py` is in S1.
- R5 Full lifecycle comparison (Part D) can reject a legitimate flow (section 6). Mitigation: STOP rule, no silent weakening.
- R6 The reconciler cannot reach UNCERTAIN rows of the real stores (Q-RC-01); the Part E test proves the algorithm over an in-memory source only.
- R7 The cap ends a recurrence chain after a permanent failure; the alternative needs a store edit (Q-RC-01).
- R8 ACT-N11 (section 12).

Rollback: the ticket is a set of commits on `arch-freeze/rw-train` before the FCS merge; a revert of the merge commit of the ticket PR restores the previous behaviour; no data migration, no store schema change.

## 14. Open questions with draft defaults

| ID | Question | Draft default |
|---|---|---|
| Q-RC-01 | PDC-D7-R015 and G-20 need a listing/resolve port on the schedule stores (`store.py`, `postgresql_task_checkpoint_store.py` (a PT-WB conditional path), `scheduled_resume.py`; all outside A1). Add them as conditional paths, or keep A1 and accept the residual? | keep A1; the reconciler defines the port, the stores are a follow-up; PDC-D7-R015 stays PARTIAL for product hosts |
| Q-RC-02 | EX-D1-07: if the slug reduction is in `registry_projection_descriptor.py` (outside A1), may it be a conditional path? Which side is authoritative for lifecycle progress if the full compare breaks the production flow? | stop and report; no weakening; no descriptor edit |
| Q-RC-03 | Wiring the controller into the document-store worker compositions (`vendor_knowledge/sync_runtime.py`, `local_workspace_application/workspaces/sync_runtime.py`) is outside A1. A follow-up, or conditional paths? | follow-up; this ticket ships class, runner seam and runner-level test |
| Q-RC-04 | EX-D7-03 and NT-ROW-116 say "every durable enqueue site", but A1 contains 2 of the 5 open sites; 3 are in no train ticket. Add the 3 as conditional paths (`queued_host_task_execution_adapter.py`, `vendor_knowledge/sync_jobs.py`, `tools/providers/message_bus/service.py`) or narrow the row to the A1 sites? | narrow to the A1 sites with the residual tuple; EX-D7-03 not complete until ruled |
| Q-RC-05 | Re-point XD-05 (`tests/qualification/pdc_x/xd/test_xd_05_host_background.py`) at the product controller as the register asks; it is not in A1. | out of this ticket; the test-side adapter keeps working |
| Q-RC-06 | Include the optional `flush_services` seam in the runner next to the controller? | yes, default empty |
| Q-RC-07 | Accept the extra regression nodes listed in section 10? | yes |
| Q-RC-08 | Reconciler policy: automatic redrive only on `NOT_STARTED_PROVEN`; everything else left UNCERTAIN with a recorded result. | as drafted |
| Q-RC-09 | After the cap (Part F) the claim is skipped (chain ends). Alternative: keep the claim and alarm. | skip, with the stable error code |
| Q-RC-10 | RW-PLAN row NT-ROW-115 lists "OverflowError is mapped" as a red-first expectation; at `H` it is already mapped (`scheduler.py:302`). This plan keeps it as a control, not a red-first leg. | as drafted |

## 15. Authorization requests

1. The approval of this ticket by the OD-AF-22 route (wave packet approved by the Operator once per wave; the delegate writes the three-line block `Approved: operator-delegate @ <ISO>`, `Delegation: <path>@<sha40>`, `Delegate session: <id>`). This draft writes no approval.
2. Product-path authorization for the 12 firm product paths and the new `uncertain_reconciler.py`, bound by the train branch rule.
3. Orchestrator rulings on Q-RC-01..Q-RC-07 before the wave packet is approved (Q-RC-01, -02, -04 change the write set if answered the other way).
4. No Docker, no vendor integration, no production configuration change (operator freeze); one pytest process at a time under the shared lock.
