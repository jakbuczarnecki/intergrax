# Background Tasks - Platform Architecture

**Status:** Target platform architecture; **frozen Execution Engine** is sole legal execution authority — Background Tasks is a **consumer** of execution identity and admission contracts (not a parallel runtime).  
**Plan (1:1):** [`plan/BACKGROUND_TASKS.md`](../maintainers/plans/BACKGROUND_TASKS.md)
**Hub:** [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md)
**Generalizes:** LKW.4 background ingest proof ([`applications/local_workspace_application/docs/ARCHITECTURE.md`](../../../applications/local_workspace_application/docs/ARCHITECTURE.md) §8.7)
**Last updated:** 2026-10-02 — **BG-01** current-HEAD recertification (`tests/qualification/bg_01/`, BG-Q1..Q15 green on session HEAD; READY FOR AUDIT — not CLOSED)

| Axis | Status | Meaning |
|------|--------|---------|
| **Foundation / contracts** | **CURRENT** | `TaskQueue` / `MessageBus`, `TaskRequest`, `BackgroundTaskHandler`, `TaskQueueProviderRegistry`, worker intake + BG-EXEC identity/reentry |
| **Execution integration** | **COMPLETE** | All active production **host-task / agent harness** background execution converges through `NexusWorkerRuntime` → `HostTaskExecutionPort` (queue dispatch via `QueuedHostTaskExecutionAdapter` + typed `decode_host_task_result_payload`); provider workloads documented below are out of canonical execution semantics |
| **Production qualification** | **PARTIAL** | LKW.4E; **BG-01** current-HEAD recertification evidence [`HARNESS_W8_BG_01_CURRENT_HEAD_RECERTIFICATION.md`](../maintainers/qualification/HARNESS_W8_BG_01_CURRENT_HEAD_RECERTIFICATION.md) (gates BG-Q1..Q15; READY FOR AUDIT); **SCHED-01** long-running delayed resume (`tests/qualification/sched_01/`, gates SCHED-Q1..Q15); not universal multi-tenant production qualification |

---

## A. Status / scope

This document defines the **target platform architecture** for background task registration, enqueueing, vendor dispatch, worker execution, status/result retrieval, event/pub-sub notifications, logging, metrics, tracing, and programmer extension points.

| Statement | Meaning |
|-----------|---------|
| **Target architecture** | Direction for platform implementation; core components exist with convergence work remaining |
| **Generalizes LKW.4** | LKW background ingest is the first proof workload, not a bespoke queue design |
| **Not production yet** | No claim of universal production maturity; `TaskRegistry`, `WorkerRuntime`, and `TaskEvent` foundations **exist** in code |
| **Execution authority** | **Execution Engine** (frozen) owns runtime identity, attempt semantics, and canonical lifecycle — see [ownership matrix](#b1-background-vs-execution-ownership) |
| **LKW.4E proof** | Must use **real platform components** and a **real local MessageBus provider** in the proof stack (for example RabbitMQ in Docker); mocks, fake queues, and in-memory-only bypasses are **not** platform proof |

Future platform code should converge on **TaskRegistry + WorkerRuntime + TaskEvent lifecycle**. Applications and agents must not invent application-owned queue systems.

---

## B. Core principle

```text
Background execution = consumer of frozen Execution Engine contracts.

enqueue / schedule / transport
        ↓
background intake (worker / broker consumer)
        ↓
Execution-owned identity + re-entry admission
        ↓
handler dispatch (TaskHandler or host-task Execution path)
        ↓
meaningful executable work under Execution Engine policy
```

**Implications:**

- Kafka, RabbitMQ, SQS, Temporal, and similar backends **store and deliver** `TaskRequest` messages. They do **not** invoke Python handlers or business logic directly.
- **Queue / scheduler providers** implement [`TaskQueue`](../../../intergrax/queueing/contracts/task_queue.py) (alias [`MessageBus`](../../../intergrax/integrations/contracts/message_bus.py)) — not application code.
- **WorkerRuntime** (and broker worker bases) are **transport dispatch mechanics**: consume messages, resolve `task_name`, run **execution-owned** background intake (`admit_background_execution_reentry`, required causal evidence), then dispatch handlers. WorkerRuntime is **not** canonical execution authority.
- Applications (Tier-3) and agents (Tier-2) **enqueue** through platform APIs/tools (`message_bus.enqueue`, future `background_tasks.enqueue`). They do **not** import vendor SDKs.
- Handler code is **registered ahead of time**; the queue carries `task_name` + validated payload bytes, **never arbitrary serialized executable code**.

### B.1 Background vs execution ownership

| Concern | Owner |
|---------|--------|
| Background task definition / registration | **Background Tasks** (`TaskDefinition`, `TaskRegistry`) |
| Queue transport | **Queue provider** (`TaskQueue` / `MessageBus` implementation) |
| Scheduling / enqueue control plane | **Background control plane** (enqueue APIs, pause/cancel **requests**) |
| Runtime identity (`TaskId`, `RunId`, `AttemptId`, `ExecutionId`, `EventId`) | **Execution Engine** |
| Attempt semantics (active attempt, terminal gate) | **Execution Engine** (`AttemptLifecycleService`, execution terminal) |
| Canonical execution lifecycle | **Execution Engine** |
| Worker transport lifecycle (consume, ack, lease, redelivery) | **Queue provider + worker adapter** |
| Governance / budget on admitted work | **Governance / Budget** (enforced on Execution paths) |
| Retry **transport** policy (visibility, DLQ, broker retry) | **Provider / control plane** |
| Canonical retry / whole-run attempt boundaries | **Execution Engine** (UEA §10) |

**Control plane ≠ execution plane:** background APIs may schedule, enqueue, request pause/cancel, and expose status; **Execution Engine** decides canonical lifecycle transitions per contracts and policy.

**Transport identity** (`task_id` / broker message id, delivery id, lease id, provider operation id) may exist in parallel with runtime identity — **Transport Identity ≠ Runtime Identity** (no semantic aliasing).

### B.2 Canonical background pipeline (contract-first)

```text
TaskDefinition
    ↓
TaskRegistry (platform catalog)
    ↓
TaskRequest (enqueue envelope)
    ↓
TaskQueue / MessageBus provider adapter
    ↓
worker intake (WorkerRuntime, BrokerWorkerBase, …)
    ↓
admit_background_execution_reentry + required causal evidence (BG-EXEC)
    ↓
execute_logical_task (registered TaskHandler) OR host-task Execution path (NexusWorkerRuntime)
```

Host-task / agent workloads dispatched through `NexusWorkerRuntime` re-enter **host task execution** ports wired to the frozen Execution stack; logical `TaskHandler` paths use the same identity admission boundary before handler code runs.

### B.2.1 Long-running delayed resume scheduling (SCHED-01)

**Scope:** `LongRunningScheduler` + `ScheduledResume` / `ScheduledResumePersistence` — **when** to resume paused checkpoints and human-timeout ledger actions, not a second Execution Engine.

```text
ScheduledResume (platform schedule definition)
    ↓
ScheduledResumePersistence (store contract; SQLite provider in default harness)
    ↓
LongRunningScheduler.tick / poll loop (Clock via explicit `now` or SystemTimeProvider)
    ↓
claim_due + fence / lease (SchedulerLedger for human-timeout path)
    ↓
HostTaskResumeExecutor → HostTaskExecutionPort.execute(resume_checkpoint=…)
    ↓
frozen Execution Engine
```

| Topic | Supported @ SCHED-01 |
|-------|----------------------|
| Trigger types | **One-shot** delayed resume (`run_at_utc`: an aware ISO-8601 timestamp, validated and normalized to UTC at construction; naive, empty and non-ISO values are rejected); human-timeout ledger actions; **recurrence** (fixed interval or 5-field cron, IANA time zone) expressed as a chain of one-shot occurrences (see *Recurrence* below) |
| Misfire (overdue one-shot) | **Fire once** on next poll when status is PENDING (no unbounded catch-up storm). A recurrence rule additionally carries a misfire policy `skip` / `fire_once` / `catch_up(n)` (n at most `MAX_CATCH_UP`); a missed range longer than `MAX_MISFIRE_SCAN` degrades to `skip` |
| Timezone | Every instant is stored and compared in **UTC**; `tick(now=…)` converts a non-UTC aware `now` to UTC. Cron rules are evaluated on the wall clock of their IANA zone with deterministic DST rules: a fixed-hour time inside a spring-forward gap fires once, shifted by the gap (wall time plus gap length, so `30 2` fires at 03:30 in a one-hour gap), a repeated fall-back hour fires once, a wildcard-hour rule follows elapsed time; interval rules are elapsed time and DST-immune |
| Occurrence identity | `schedule_id` per `ScheduledResume` row. Recurrence occurrences use a deterministic id qualified by tenant, task, `rule_id` and instant (tenant and task as fixed-length digests, so two tenants never collide); restarting a rule is idempotent |
| Multi-instance coordination | **QUALIFIED** for shared `ScheduledResumePersistence` implementations that provide atomic `claim_due` + fencing (SCHED-Q5/Q6; proof uses two in-process scheduler instances over one SQLite store) |
| Distributed multi-node deployment | **Provider-dependent**; not universally qualified by SCHED-01 (no independent multi-process / arbitrary distributed backend proof in this qualification slice) |
| Retry ownership | Scheduler poll/claim retry **≠** queue transport retry **≠** execution attempt retry (UEA / AttemptLifecycleService) |

**Recurrence (PDC-X P7 repair, T-IGX-041-G).** `RecurrenceRule` / `RecurrenceMaterializer` (`intergrax/runtime/long_running/recurrence.py`) add no scheduler of their own: each occurrence is an ordinary one-shot `ScheduledResume`, claimed, fenced and admitted like any other.

- The successor is persisted **before** its predecessor executes, so a crash or lease expiry while an occurrence runs (the predecessor ends `UNCERTAIN`, at-most-once) does not end the chain. If persisting the successor fails with a store error the claim is released back to `PENDING` and retried at the next tick; an invalid persisted rule skips the occurrence without executing it. The same applies to the eligibility reads of a recurring claim (target checkpoint, latest checkpoint, resume gates): an exception releases the claim for retry instead of leaving it `RUNNING` without a successor, and an unavailable execution terminal authority (`ExecutionTerminalError`, which both resume gates report as "not resumable") is retried, not recorded as `SKIPPED`; only a real terminal record or a missing or non-resumable target ends the chain. A permanently failing read is retried at every tick (no cap).
- A rule targets exactly one `(task_id, resume_token)`. When that checkpoint is superseded, terminal, missing or not resumable the occurrence is recorded `SKIPPED` (not `completed`), the chain ends explicitly with a WARNING and no successor is created. Retargeting a rule to a newer checkpoint is **not** implemented. One-shot entries keep their previous behaviour.
- Rules are bounded (`MIN_INTERVAL_SECONDS`, `MAX_INTERVAL_SECONDS`, `MAX_CATCH_UP`); a cron expression without any occurrence is rejected. `recurrence_rule` and `recurrence_occurrence_utc` are rejected in plain `resume_metadata` of `LongRunningScheduler.schedule_resume`, and persistence accepts them only on a self-consistent occurrence row (valid rule, matching tenant, task and token, a real occurrence instant, canonical `schedule_id`).
- With the successor created before execution, a second scheduler process may claim occurrence N+1 while occurrence N is still running (run time longer than the interval); no overlap guard exists. Occurrences of one rule resume the same checkpoint with the same deterministic run and attempt identity, so a long-running overlap is bounded only by the host execution admission.
- A scheduler tick processes every claim of its batch in isolation: one failing resume (or a lost lease at completion) does not strand the sibling claims. The first failure is still re-raised after the batch and after both tick phases (due schedules, human timeouts) ran. A failing resume leaves its own claim `RUNNING` until the lease expires, then `UNCERTAIN` (at-most-once, never auto-retried).
- **NOT WIRED:** `wire_long_running_scheduler_with_host_execution` has no recurrence parameter and `RecurrenceMaterializer` is not exported from `intergrax.runtime.long_running`; no host composes recurrence today (declared exception F-D7-05). Rows written before the tenant-qualified id (`rec_<rule>_<stamp>`) would not be recognised as the same occurrence; none exist in a production host.

**Multi-instance vs distributed:** one process may host multiple scheduler instances; SCHED-01 proves safe coordination when they share persistence with contract-grade claim/fence semantics. Universal multi-node correctness for every production store provider is **not** claimed here — custom `ScheduledResumePersistence` plugins (SCHED-Q10) prove replaceability, not distributed concurrency unless the implementation documents atomic claim + fence behavior.

Production harness wiring: `wire_harness_host_long_running_scheduler` → `wire_long_running_scheduler_with_host_execution` (see `intergrax/runtime/long_running/wiring.py`). Qualification: `tests/qualification/sched_01/`.

[`AGENT_DISTRIBUTION.md`](AGENT_DISTRIBUTION.md) is package installation/activation only. [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md) may constrain worker capacity but does not own Execution identity.

Aligned with [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md) §11, §20, **UEA-INV-011**, **UEA-INV-021**.

### B.2.2 Delivery semantics (at-least-once)

*(PDC-X reconciliation PDC-D7-R011, 2026-11-12)*

Queue delivery is **at-least-once**: a provider may deliver the same `TaskRequest` more than once (broker redelivery, visibility timeout, transport retry). Duplicates are therefore possible and are handled as follows:

- `TaskRequest.idempotency_key` identifies the logical task; with an idempotency store, a duplicate delivery of the same logical task results in a **single handler invocation** (BG-Q5).
- Transport retry is separate from execution attempts; redelivery does not reconcile or mint a new attempt (BG-Q6).
- Redelivery after a terminal outcome resolves to a safe `TERMINAL_ALREADY_RECORDED` disposition with no fresh side effect (BG-Q7).
- Over `DocumentStoreTaskQueue` the terminal states `SUCCEEDED`, `FAILED` and `DEAD_LETTERED` are **sticky**: a late or duplicate completion, failure or retry cannot overwrite them or re-open the task.
- **Retry budget and dead-letter (library capability, NOT wired):** `DocumentStoreTaskWorker(retry_policy=…)` settles a failed delivery against a `RetryPolicy` (back-off, then terminal `DEAD_LETTERED` with a typed `DeadLetterReason`). The three production construction sites (`vendor_knowledge/sync_runtime.py`, `local_workspace_application/workspaces/sync_runtime.py` twice) pass no `retry_policy`, so production workers neither retry nor dead-letter (declared exception F-D7-05). Retry and dead-letter transitions emit no dedicated event, and an idempotency key keeps resolving to a dead-lettered task.
- Certified by HARNESS-W8 / BG-01: [`HARNESS_W8_BG_01_CURRENT_HEAD_RECERTIFICATION.md`](../maintainers/qualification/HARNESS_W8_BG_01_CURRENT_HEAD_RECERTIFICATION.md).
- **Handlers must be idempotent for external effects.** Platform dedup covers handler invocation, not effects a handler issues to external systems before a crash; see [`EXTERNAL_EFFECT_CONTRACTS.md`](EXTERNAL_EFFECT_CONTRACTS.md).

### B.2.3 Horizontal scaling boundary

*(PDC-X reconciliation PDC-D7-R021, 2026-11-12)*

- Multi-node correctness **depends on the queue provider** (delivery, visibility and ack semantics) and on the `ScheduledResumePersistence` provider (atomic `claim_due` + fence); the platform does not claim universal distributed correctness.
- Scheduler fencing is proven for **two scheduler instances over one SQLite store in one process** (SCHED-01, SCHED-Q5/Q6: [`HARNESS_RESIDUAL_SCHED_01_CURRENT_HEAD_RECERTIFICATION.md`](../maintainers/qualification/HARNESS_RESIDUAL_SCHED_01_CURRENT_HEAD_RECERTIFICATION.md)) and, since PDC-X B-D7, for **two OS processes over one on-disk SQLite ledger** (`tests/qualification/pdc_x/d7/test_d7_fencing_two_process.py`: every due occurrence fires exactly once, a crash after claim is never re-fired and a stale completion is rejected). The proof is **scoped to SQLite**; PostgreSQL and other multi-node stores are excluded by the C10/C15 freeze and are not proven.
- `DocumentStoreTaskQueue` coordination: over a `ConditionalDocumentStore` (the in-memory, MongoDB, PostgreSQL and Cassandra document-store adapters) claims and state transitions are compare-and-set and an idempotency key is reserved with `put_if_absent`, so several queue instances over one store do not claim a task twice or create two tasks for one key (proven with two instances over the in-memory conditional store, `tests/qualification/pdc_x/d7_repair/`). Over a plain `DocumentStore` the only exclusion is a per-instance lock and the queue is **single-instance**; the platform does not check which mode a composition got. Retry back-off rows wait in a delayed index and are promoted when due, so they do not block fresh work. `recover_interrupted_running` (startup recovery of the co-located worker) is **not** compare-and-set and treats every `RUNNING` row of the store as interrupted, so recovery is a single-instance operation even over a `ConditionalDocumentStore`: the queue stores no owner or lease on `RUNNING` rows.
- The certified scale envelope is the scoped DIST-SCALE-X envelope ([`DIST_SCALE_X_CAPACITY_ENVELOPE.md`](../maintainers/qualification/DIST_SCALE_X_CAPACITY_ENVELOPE.md), [`DIST_SCALE_X_INTERIM_SCOPED_QUALIFICATION_RECORD.md`](../maintainers/qualification/DIST_SCALE_X_INTERIM_SCOPED_QUALIFICATION_RECORD.md)); nothing outside it is qualified.
- Declared exceptions (PDC-X P7, Operator option B): **graceful worker shutdown with host drain is not implemented** (no `HostedApplicationActiveWorkController` over `WorkerRuntime` or `DocumentStoreTaskWorker`; `DocumentStoreTaskWorker.stop()` can abandon a still-running thread; the XD-05 proof uses a test-side adapter) - reopen when the worker-side drain controller is commissioned; recurrence and the retry budget are **not composed** in any host (F-D7-05) - reopen when a host composition enabling them is commissioned. Decision record: [`ADR_PDC_07_BACKGROUND_RECURRENCE_AND_DLQ.md`](../maintainers/architecture/ADR_PDC_07_BACKGROUND_RECURRENCE_AND_DLQ.md).

### Canonical background execution identity (BG-EXEC-1 / BG-EXEC-2)

All supported background execution paths use the platform-owned canonical background execution bootstrap (`bootstrap_background_execution` / `resolve_background_execution` in `intergrax/runtime/background_execution/bootstrap.py`). Applications and scenarios do **not** mint or own runtime execution identity.

#### CURRENT IMPLEMENTATION

**CURRENT (as-built at worker intake):** identity persistence + re-entry admission — factual behavior, separate from frozen UEA whole-run retry semantics.

- `BackgroundExecutionIdentityPersistence` maps `BackgroundTransportExecutionRef` (tenant + provider + transport_task_id) → durable `TaskId`, `RunId`, `AttemptId`, `ExecutionId` (first `store_if_absent`; subsequent loads return the same quadruple for the same transport execution).
- `admit_background_execution_reentry` reconciles the **active** `AttemptId` via `AttemptLifecycleService` and blocks handler execution when execution terminal authority already recorded the run — **without** minting a new attempt on transport redelivery alone.
- `TaskRequest.run_id` and broker `run_id` remain **transport queue correlation** only (status/events indexing) — not canonical `RunId`.

```text
background transport (TaskRequest / broker message / Celery request)
       ↓
BackgroundTransportExecutionRef (tenant + provider + transport_task_id)
       ↓
resolve_background_execution → stable TaskId + RunId + AttemptId + ExecutionId (per transport ref)
       ↓
admit_background_execution_reentry → active AttemptId + terminal gate
       ↓
admit_background_execution_handler (required causal evidence)
       ↓
execute_logical_task / NexusWorkerRuntime.run_task
```

| Field | Owner at worker boundary (CURRENT) |
|-------|-------------------------------------|
| `TaskId` | Execution-owned identity persistence keyed by transport ref |
| `RunId` | Same — **not** `TaskRequest.run_id` |
| `AttemptId` | Persisted at first transport mapping; active attempt from `AttemptLifecycleService` — **not** queue delivery counter / Celery `request.retries` |
| `ExecutionId` | Persisted with transport mapping (v2 identity records) |
| `tenant_id` | Validated single scope; mismatch fails closed |

Stable `TaskId`/`RunId` across process restart and concurrent workers requires atomic identity persistence: `DistributedKVStore.compare_and_set` or `ConditionalDocumentStore.put_if_absent`. Generic `DocumentStore` without conditional create is rejected at composition; there is no process-local fallback.

`TaskRequest.run_id` and broker message `run_id` remain **transport queue correlation** for status/events indexing; they are not canonical runtime `RunId`.

#### Production handler classification (BG-01-R1)

| Handler / logical task | Class | Execution boundary | Notes |
|------------------------|-------|--------------------|-------|
| `nexus.task.v2` (`NexusWorkerRuntime`) | **HOST TASK PIPELINE** | `HostTaskExecutionPort` via `QueuedHostTaskExecutionAdapter` / Celery dispatcher | Canonical agent harness execution |
| `lkw.background_ingest.v1` | **HOST TASK PIPELINE** | `HostTaskExecution.execute` through ingest runner | LKW.4E proof; same execution stack as foreground host tasks |
| `lkw.managed_workspace_sync.v1`, knowledge ingestion, vendor-knowledge sync tasks | **PROVIDER-SPECIFIC WORKLOAD** | `execute_logical_task` + intake admission only | Document/workspace sync and ingestion — not agent harness execution |
| Dispatcher / broker housekeeping, compensation maintenance | **INTERNAL PLATFORM JOB** | Intake + handler; no business model execution | Transport and reliability mechanics |
| Test/eval Celery stacks | **TEST/EVAL ONLY** | Same as production wiring with eager backends | Gate and integration proofs |

**ACTIVE EXECUTION BYPASSES:** none for production host-task / agent workloads. Remaining `TaskHandler` registrations are provider workloads or platform jobs by design.

**Result decoding owner:** `intergrax/queueing/worker/result_codec.py` (`decode_host_task_result_payload`) — `QueuedHostTaskExecutionAdapter` injects an optional decoder callable; default is the codec (no duck typing on worker boundaries).

#### TARGET / ADOPTION GAPS

- **Frozen rule (UEA §10):** queue redelivery **≠** automatically new logical execution; provider retry **≠** automatically new `AttemptId`. A **new** `AttemptId` requires an explicit whole-Run retry boundary.
- **ADOPTION GAP:** the transport envelope **can** carry full runtime identity on enqueue (`TaskRequest.identity`: `TaskId`, `RunId`, `AttemptId`, `ExecutionId`, principal, trace) and the `DocumentStoreTaskQueue` persists it and `DocumentStoreTaskWorker` seeds the quadruple at intake, but **no production producer sets it**, principal and trace are carried and never consumed, a partial identity is ignored, and the Celery and broker providers drop it (declared exception F-D7-09). The worker side still stabilizes identity at intake.

### Required audit evidence admission (BG-EXEC-3)

Intergrax distinguishes **optional telemetry** from **required audit evidence**. Failure to persist evidence required to establish an execution boundary fails closed before business execution begins.

| Class | Examples | Failure semantics |
|-------|----------|-------------------|
| **Optional observability** | metrics export, vendor export, supplemental telemetry, non-critical subscribers | best-effort / degraded |
| **Required audit evidence** | `TRANSPORT_TASK_TRIGGERED_EXECUTION` linking `BackgroundTransportExecutionRef` → `BackgroundExecutionIdentity` | fail-closed before handler |

Required ordering at the worker admission boundary (DIAG-1I writer integration):

```text
transport task received
       ↓
BackgroundExecutionIdentity established (bootstrap)
       ↓
required causal evidence persisted (admit_background_execution_handler)
       ↓
execute_logical_task / handler
```

If required evidence persistence fails: handler invocation count = 0, no business side effects, failure propagates through existing worker/reliability handling (`RequiredAuditEvidencePersistenceError` → `FailureClass.DEPENDENCY_ERROR`). No new recovery engine.

`RuntimeEventBus` best-effort persistence is **not** the admission mechanism for required transport→execution causal evidence.

**Writer integration: DONE** for supported background execution paths (`BrokerWorkerBase`, `WorkerRuntime`, Celery `intergrax.execute`, `DocumentStoreTaskWorker`). Ordering: transport → identity bootstrap → `admit_background_execution_handler` (required `TRANSPORT_TASK_TRIGGERED_EXECUTION`) → `execute_logical_task`. Fail-closed on persistence failure.

**CURRENT implementation:** transport redelivery reloads persisted runtime identity; new causal evidence may be recorded per delivery without minting new `TaskId`/`RunId`/`AttemptId`/`ExecutionId` for the same transport ref. Whole-run retry boundaries remain Execution-owned (see UER hub lifecycle).

**TARGET:** full convergence of handler dispatch with frozen Execution admission for all background task types; enqueue-side transport envelopes carry canonical identity where product hosts require it.

`RuntimeEventBus` persistence remains separate execution truth; observability exporter remains optional.

Entry points that invoke the bootstrap: `BrokerWorkerBase.process_message`, `WorkerRuntime.process_request`, Celery `intergrax.execute` dispatcher, and `DocumentStoreTaskWorker`.

---

## C. Main concepts

| Concept | Role |
|---------|------|
| **TaskDefinition / JobDefinition** | Declarative registration of a background task type: `task_name`, payload schema, handler reference, policy, required capabilities/tools/integrations |
| **TaskRegistry** | Platform catalog (`intergrax.background_tasks.registry.TaskRegistry`) mapping `task_name` → `TaskDefinition`; composes into `TaskExecutionRegistry` with **`BackgroundTaskHandler` Protocol** handlers — **EXTENSIBILITY_GAP:** no separate `TaskRegistry` port ABC; replace via composition at host bootstrap |
| **TaskRequest** | Immutable enqueue envelope: `tenant_id`, `run_id`, `task_name`, `payload` (bytes), optional `idempotency_key`, `priority` - see [`intergrax/queueing/contracts/task_queue.py`](../../../intergrax/queueing/contracts/task_queue.py) |
| **TaskHandle** | Opaque handle returned after enqueue: `task_id`, `provider`, optional `tenant_id` |
| **TaskResult** | Final outcome: `status`, optional `output` (bytes), `error_message`, `attempts` |
| **TaskStatus** | Lifecycle enum: `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED` (current contract; extended states may map to events) |
| **WorkerRuntime** | Platform worker **dispatch** component: consumes `TaskRequest`, runs execution-owned intake, invokes handlers or host-task worker paths, stores task status/result, acks/retry/DLQ at transport layer — **not** runtime identity or lifecycle authority |
| **TaskHandler** | Developer-implemented function bound to a `TaskDefinition`; receives decoded payload + execution context; returns `TaskResult` through platform contracts |

All supported background handlers implement one canonical platform handler contract (`BackgroundTaskHandler`) and receive `BackgroundExecutionIdentity` explicitly through `execute_logical_task`.
| **TaskEvent** | Lifecycle/progress fact emitted on an event channel (separate from work transport) |
| **MessageBus / TaskQueue** | Tier-0 **transport contract** (ABC) for enqueue, status, result, list, cancel, purge — [`TaskQueue`](../../../intergrax/queueing/contracts/task_queue.py); [`MessageBus`](../../../intergrax/integrations/contracts/message_bus.py) aliases `TaskQueue`; [`TaskQueueProviderRegistry`](../../../intergrax/queueing/registry.py) registers provider implementations |
| **Provider adapter** | `TaskQueue` / `MessageBus` implementation for a vendor (Redis, RabbitMQ, document-store queue, in-memory test doubles); serializes `TaskRequest`, does not execute handlers |
| **TaskObserver / subscription** | Consumer of `TaskEvent` stream or pull APIs for status/result; may be app UI, agent loop, workflow, notification bridge, observability backend |
| **TaskPolicy** | Retry, timeout, concurrency, rate limit, dead-letter, cancellation, idempotency, and tenant-scoping rules attached to a `TaskDefinition` |
| **TaskTrace / lifecycle timeline** | Correlated observability view across enqueue → dispatch → execution → result, keyed by `run_id`, `task_id`, `correlation_id` |

---

## D. Programmer model

A developer defines a custom background task through platform contracts - **not** by putting code on a queue.

### Steps

1. **Define payload schema** - typed model (e.g. Pydantic) serialized to JSON/protobuf bytes in `TaskRequest.payload`.
2. **Define `task_name`** - stable, namespaced string (e.g. `lkw.background_ingest.v1`, `acme.report.export.v1`).
3. **Implement handler** - decode payload, execute domain logic through platform tools/integrations/runtime; no vendor SDK imports.
4. **Declare task policy** - retries, timeout, concurrency, idempotency semantics, dead-letter behavior.
5. **Declare required tools/integrations/capabilities** - e.g. `local.workspace.index`, `rag.ingest`, `relational_store.query`.
6. **Register `TaskDefinition` in `TaskRegistry`** - wires `task_name` → schema + handler + policy (host bootstrap or application factory).
7. **Enqueue from app/agent** - `message_bus.enqueue` or future `background_tasks.enqueue` with encoded payload.
8. **Observe status/result/events** - pull (`get_status`, `get_result`, `list_tasks`) and/or subscribe to `TaskEvent` stream.

### Example task types (illustrative)

| `task_name` (example) | Purpose |
|-----------------------|---------|
| `platform.log.emit.v1` | Structured diagnostic log batch |
| `acme.repo.scan.v1` | Scan repository or container image |
| `lkw.background_ingest.v1` | Rebuild / increment vector index |
| `acme.db.fetch.v1` | Fetch data from relational store |
| `acme.report.export.v1` | Generate and store export artifact |
| `acme.crm.sync.v1` | Sync external system |
| `acme.cache.clear.v1` | Clear namespaced cache |
| `acme.embeddings.batch.v1` | Run batch embeddings |

**Critical rule:** the queue carries `TaskRequest { task_name, payload }`. **WorkerRuntime** resolves `task_name` to **pre-registered handler code**. Payload is data only - paths, IDs, options - not executable logic.

---

## E. End-to-end lifecycle

```text
1. Developer registers TaskDefinition in TaskRegistry
2. Application/agent calls background_tasks.enqueue or message_bus.enqueue
3. Platform validates task_name, payload schema, permissions, tenant scope, idempotency
4. Platform creates TaskRequest
5. MessageBus provider adapter serializes and sends TaskRequest to vendor
6. Vendor stores/transports message
7. WorkerRuntime / broker worker receives message (provider-specific consumption model)
8. `admit_background_execution_reentry` + required causal evidence (execution-owned)
9. WorkerRuntime resolves task_name in TaskRegistry
10. Handler executes through `execute_logical_task` or host-task Execution path with `BackgroundExecutionIdentity`
11. Handler emits progress/logs/events/artifacts if needed
12. WorkerRuntime stores TaskResult
13. WorkerRuntime acks/commits vendor message (or schedules retry / dead-letter)
14. Application/agent retrieves result via pull or observes lifecycle via events
```

### Layer diagram

```text
┌─────────────────┐     enqueue      ┌──────────────────┐
│ App / Agent     │ ───────────────► │ Platform enqueue │
│ (Tier-2/3)      │                  │ + validation     │
└─────────────────┘                  └────────┬─────────┘
                                            │ TaskRequest
                                            ▼
┌─────────────────┐   transport    ┌──────────────────┐
│ Vendor          │ ◄────────────► │ Provider adapter │
│ Kafka/SQS/…     │                │ (MessageBus)     │
└─────────────────┘                └────────┬─────────┘
                                            │ TaskRequest
                                            ▼
                                  ┌──────────────────┐
                                  │ WorkerRuntime    │
                                  │ + TaskRegistry   │
                                  └────────┬─────────┘
                                            │ invoke
                                            ▼
                                  ┌──────────────────┐
                                  │ TaskHandler      │
                                  │ (registered)     │
                                  └────────┬─────────┘
                                            │ TaskResult + TaskEvents
                                            ▼
┌─────────────────┐   pull/events  ┌──────────────────┐
│ Observers       │ ◄──────────────│ Result store +   │
│ UI/agent/obs    │                │ EventBus         │
└─────────────────┘                └──────────────────┘
```

---

## F. Who invokes the task, when, and by whom

| Layer | Invokes business logic? | Responsibility |
|-------|-------------------------|----------------|
| **Kafka / RabbitMQ / SQS / etc.** | **No** | Deliver messages; persistence; partitioning; lease/visibility |
| **Provider adapter** | **No** | Serialize/deserialize `TaskRequest`; map to vendor APIs |
| **WorkerRuntime** | **No** (dispatch only) | Consume message, execution-owned intake, dispatch handler — not execution authority |
| **TaskHandler** | **Yes** (domain code) | Developer logic inside platform contracts after intake admission |
| **Execution Engine** | **Yes** (authority) | Identity, attempt, terminal, and canonical lifecycle on admitted paths |

**Timing** depends on provider consumption model:

| Provider style | When worker receives work |
|----------------|---------------------------|
| Kafka / RabbitMQ | Consumer receives when message available on subscribed topic/queue |
| SQS | Polling / long-poll; visibility timeout / lease model |
| Temporal | Workflow/activity worker model |
| **Local message bus provider (LKW proof stack)** | Async enqueue → broker/queue → worker consumer (for example RabbitMQ in local Docker) |

The platform hides provider-specific mechanics behind **TaskQueue/MessageBus + WorkerRuntime** contracts.

---

## G. Pull model

Pull inspection is the **mandatory baseline** - works for all providers and supports deterministic agent/workflow checking.

| API | Purpose |
|-----|---------|
| `get_status(TaskHandle)` | Current `TaskStatus` |
| `get_result(TaskHandle)` | Final `TaskResult` when completed |
| `list_tasks(tenant_id, …)` | Recent tasks for tenant with optional status filter |
| `cancel(TaskHandle)` | Request cancellation (provider/handler dependent) |
| `purge_completed(tenant_id, …)` | Retention cleanup for completed task records |

**Current tool surface:** `message_bus.get_status`, `message_bus.get_result`, `message_bus.list_tasks`, `message_bus.cancel`, `message_bus.purge_completed` - see [`intergrax/tools/providers/message_bus/service.py`](../../../intergrax/tools/providers/message_bus/service.py).

**Semantics:**

- Enqueue returns `TaskHandle` immediately; execution is asynchronous. LKW.4E platform proof must observe this async lifecycle - synchronous in-process bypass is not sufficient.
- Pull APIs are provider-neutral; callers pass `task_id` + `provider` (+ `tenant_id` when required).
- Agents waiting on background work should prefer pull + events rather than blocking vendor consumers.

---

## H. Event / pub-sub model

**Work transport and lifecycle events are separate channels.**

| Channel | Carries |
|---------|---------|
| **TaskQueue / MessageBus** | Commands / work requests (`TaskRequest`) |
| **TaskEvent / EventBus** | Facts / lifecycle / progress |

### Required lifecycle events (target)

| Event | Meaning |
|-------|---------|
| `task.registered` | `TaskDefinition` registered in `TaskRegistry` |
| `task.enqueue_requested` | Enqueue API invoked |
| `task.enqueued` | `TaskRequest` accepted; `TaskHandle` issued |
| `task.dispatch_requested` | WorkerRuntime scheduled consumption |
| `task.dispatched` | Message handed to worker |
| `task.started` | Handler execution began |
| `task.progress` | Optional fractional or staged progress |
| `task.tool_call_started` | Handler invoked a platform tool |
| `task.tool_call_completed` | Tool call finished |
| `task.succeeded` | Handler completed successfully |
| `task.failed` | Handler or runtime failure |
| `task.cancelled` | Cancellation applied |
| `task.result_stored` | `TaskResult` persisted |
| `task.acknowledged` | Vendor message acked/committed |
| `task.dead_lettered` | Exhausted retries; moved to DLQ |

### Typical consumers

- Application UI (progress, completion)
- Agent waiting/polling loop
- Workflow orchestrator
- Notification channel bridge (Slack, email - **observers only**)
- Observability backend (traces, metrics)
- Audit / debug tooling

---

## I. Pub-sub / notification options

| Mode | Role |
|------|------|
| **Pull status/result** | Baseline; always available |
| **Event subscription / stream** | Real-time lifecycle for UIs and orchestrators |
| **Callback / webhook** | Optional push to external HTTP endpoint on terminal states |
| **`notification_channel`** | Optional human-facing notify (Slack, email) - **not** core execution |
| **HITL prompt** | Optional human decision gate - observer to `TaskEvent` or terminal status |

Slack, email, and similar integrations **subscribe** to `TaskEvent` or poll final status. They do **not** execute background tasks.

---

## J. Logging, metrics, tracing

### Minimum trace fields (target)

| Field | Notes |
|-------|-------|
| `tenant_id` | Isolation boundary |
| `user_id` | When available from enqueue context |
| `run_id` | Correlation with Nexus/runtime run |
| `task_id` | Provider task identifier |
| `task_name` | Registered task type |
| `correlation_id` | Cross-service correlation |
| `idempotency_key` | Dedup key |
| `provider` | Message bus provider slug |
| `worker_id` | Worker instance |
| `handler_id` | Registered handler reference |
| `attempt` | Current attempt number |
| `status` | Lifecycle status |
| `duration_ms` | End-to-end or phase duration |
| `queue_latency_ms` | Enqueue → worker start |
| `execution_latency_ms` | Handler execution only |
| `retry_count` | Retries so far |
| `error_class` | Exception type (sanitized) |
| `error_message` | Sanitized message |
| `tool_calls` | Count of platform tool invocations |
| `input_size` / `output_size` | Byte sizes, sanitized summaries only |

### Metrics (target)

| Metric | Type |
|--------|------|
| `tasks_enqueued_total` | Counter |
| `tasks_started_total` | Counter |
| `tasks_succeeded_total` | Counter |
| `tasks_failed_total` | Counter |
| `task_queue_latency_ms` | Histogram |
| `task_execution_duration_ms` | Histogram |
| `task_retries_total` | Counter |
| `task_dead_letters_total` | Counter |
| `task_inflight` | Gauge |
| `task_result_size_bytes` | Histogram |

### Logs

- Structured lifecycle logs at enqueue, dispatch, start, progress, complete, fail, ack, dead-letter.
- **No raw payload content by default** - log `task_name`, sizes, hashes, redacted summaries.
- Payload redaction follows platform observability rules; secrets and document bodies must not appear in logs.

---

## K. Debug / inspection surface

Operators and developers should be able to answer:

| Question | Evidence |
|----------|----------|
| Was task type registered? | `TaskRegistry` / `task.registered` event |
| Was enqueue accepted? | `task.enqueued` event + `TaskHandle` |
| Which provider got the task? | `TaskHandle.provider` |
| What `TaskRequest` was created? | Audit record (redacted payload summary) |
| Was task deduplicated by idempotency? | Enqueue response + idempotency audit |
| Did worker receive it? | `task.dispatched` / consumer offset |
| Which handler was selected? | `handler_id` from registry resolution |
| Which attempt is this? | `attempt` / `retry_count` |
| What tools/integrations did handler call? | `task.tool_call_*` events + trace |
| Did it succeed/fail/cancel? | `TaskStatus` + terminal events |
| Where is the result? | `get_result(TaskHandle)` / result store |
| Which events were emitted? | Event stream / audit log |
| Which trace/run contains full timeline? | `run_id` + observability spine |

---

## L. Security / governance

| Rule | Requirement |
|------|-------------|
| **Tenant isolation** | `tenant_id` on every `TaskRequest`; handler context scoped to tenant |
| **Payload schema validation** | Reject unknown fields / invalid payloads at enqueue and worker decode |
| **Idempotency** | Stable `idempotency_key`; platform dedup where provider supports |
| **Allowlisted `task_name`** | Only registered tasks may be enqueued |
| **No arbitrary code from queue** | Payload is data; handler code is pre-registered |
| **Permissions / capabilities** | Handler declares required capabilities; runtime enforces tool/integration allowlists |
| **Rate limits / concurrency** | `TaskPolicy` per task type and per tenant |
| **Cancellation** | Cooperative cancel signal through runtime; follows Execution Tree semantics - transport cancel must not create a second canonical cancellation tree |
| **Retry / dead-letter** | Policy-driven; terminal failure → `task.dead_lettered` |
| **Payload redaction** | Logs, events, and traces redact sensitive payload fields |
| **No vendor SDK in apps/agents** | Tier-2/3 use `message_bus.*` tools only |

---

## M. Relationship to existing contracts

| Artifact | Location | Role today |
|----------|----------|------------|
| `TaskQueue` | [`intergrax/queueing/contracts/task_queue.py`](../../../intergrax/queueing/contracts/task_queue.py) | Current transport + pull contract |
| `TaskRequest`, `TaskHandle`, `TaskResult`, `TaskStatus` | Same | Enqueue and inspection envelopes |
| `MessageBus` | [`intergrax/integrations/contracts/message_bus.py`](../../../intergrax/integrations/contracts/message_bus.py) | Alias for `TaskQueue` |
| `MessageBusIntegrationContract` | [`intergrax/runtime/integrations/categories/messaging.py`](../../../intergrax/runtime/integrations/categories/messaging.py) | Provider category contract |
| `message_bus.*` tools | [`intergrax/tools/providers/message_bus`](../../../intergrax/tools/providers/message_bus) | Provider-neutral enqueue and pull surface |
| LKW proof workload | [`applications/local_workspace_application/background_ingest`](../../../applications/local_workspace_application/background_ingest) | First `TaskDefinition` proof |

**Proven pluggable seams (CURRENT):**

| Seam | Contract | Notes |
|------|----------|-------|
| Queue transport | `TaskQueue` ABC + `TaskQueueProviderRegistry` | Provider-neutral enqueue/pull |
| Handler invocation | `BackgroundTaskHandler` Protocol + `TaskExecutionRegistry` | Per `task_name` handler binding |
| Messaging integration category | `MessageBusIntegrationContract` | Tier-0 integration catalog |

**EXTENSIBILITY_GAP (report only):** formal `TaskRegistry` port; unified worker provider port beyond concrete `WorkerRuntime` / broker bases.

**Evolution:**

- Existing `TaskQueue` is the **current base** for transport and pull.
- **TaskRegistry**, **WorkerRuntime**, and **TaskEvent** **build on** this contract — they do not replace it with an application-specific design.
- **LKW.4E** must align with this architecture; it must **not** invent an LKW-only queue/worker stack.

---

## N. LKW.4 proof mapping

LKW background ingest is one concrete **TaskDefinition** in the target model:

| Target concept | LKW.4 implementation |
|----------------|----------------------|
| `task_name` | `lkw.background_ingest.v1` |
| `payload_schema` | `LkwBackgroundIngestJob` ([`contracts.py`](../../../applications/local_workspace_application/background_ingest/contracts.py)) |
| Enqueue helper | `enqueue_background_ingest_job` ([`enqueue.py`](../../../applications/local_workspace_application/background_ingest/enqueue.py)) |
| `TaskHandler` | `handle_background_ingest_task_request` ([`handler.py`](../../../applications/local_workspace_application/background_ingest/handler.py)) |
| Required capability | `local.workspace.index` via `local_indexer` agent path |
| Proof verification | `local.workspace.search` evidence after index completes |
| Idempotency | `background_ingest_idempotency_key(job)` |

```text
TaskDefinition:
  task_name = lkw.background_ingest.v1
  payload_schema = LkwBackgroundIngestJob
  handler = handle_background_ingest_task_request
  capability = local.workspace.index
  proof verification = local.workspace.search evidence
```

---

## O. Non-goals

Explicitly **out of scope** for this architecture document and the BG-TASKS track unless separately planned:

- Implementing production **WorkerRuntime** in this doc task
- Implementing Kafka/RabbitMQ/SQS client logic in applications
- File watcher (LKW.7)
- Scheduler
- Slack notify (LKW.6b)
- OS daemon (LKW.6)
- Arbitrary code serialization on the queue
- Cloud-managed vendor backends (SQS, Service Bus, Pub/Sub, etc.) in LKW.4E first pass - a **local** broker/provider in the proof stack is required
- Mocks, fake queues, in-memory-only bypasses, and unit-test-only handler invocation as LKW.4E platform proof

---

## Related documents

- [`plan/BACKGROUND_TASKS.md`](../maintainers/plans/BACKGROUND_TASKS.md) - implementation phases
- [`architecture/INTEGRATIONS.md`](INTEGRATIONS.md) - `message_bus` provider category
- [`applications/local_workspace_application/docs/ARCHITECTURE.md`](../../../applications/local_workspace_application/docs/ARCHITECTURE.md) §8.7 - LKW.4 product architecture
- [`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) §6 - LKW.4 task schedule
