# Background Tasks — Platform Architecture

**Status:** Target platform architecture (not production implementation yet)  
**Plan (1:1):** [`plan/BACKGROUND_TASKS.md`](../maintainers/plans/BACKGROUND_TASKS.md)
**Hub:** [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md)
**Generalizes:** LKW.4 background ingest proof ([`applications/local_workspace_application/docs/ARCHITECTURE.md`](../../../applications/local_workspace_application/docs/ARCHITECTURE.md) §8.7)
**Last updated:** 2026-08-25 — **BG-EXEC-3** required audit evidence admission semantics

---

## A. Status / scope

This document defines the **target platform architecture** for background task registration, enqueueing, vendor dispatch, worker execution, status/result retrieval, event/pub-sub notifications, logging, metrics, tracing, and programmer extension points.

| Statement | Meaning |
|-----------|---------|
| **Target architecture** | Direction for platform implementation; not all components exist in code yet |
| **Generalizes LKW.4** | LKW background ingest is the first proof workload, not a bespoke queue design |
| **Not production yet** | No claim that `TaskRegistry`, `WorkerRuntime`, or `TaskEvent` are fully implemented |
| **LKW.4E proof** | Must use **real platform components** and a **real local MessageBus provider** in the proof stack (for example RabbitMQ in Docker); mocks, fake queues, and in-memory-only bypasses are **not** platform proof |

Future platform code should converge on **TaskRegistry + WorkerRuntime + TaskEvent lifecycle**. Applications and agents must not invent application-owned queue systems.

---

## B. Core principle

```text
Vendors transport work messages.
The platform worker runtime executes code.
Applications and agents enqueue registered tasks.
Handlers contain developer custom logic but run through platform contracts.
```

**Implications:**

- Kafka, RabbitMQ, SQS, Temporal, and similar backends **store and deliver** `TaskRequest` messages. They do **not** invoke Python handlers or business logic directly.
- **WorkerRuntime** (platform) receives messages, resolves `task_name` in **TaskRegistry**, and invokes the registered **TaskHandler** inside a platform execution context.
- Applications (Tier-3) and agents (Tier-2) **enqueue** work through platform APIs/tools (`message_bus.enqueue`, future `background_tasks.enqueue`). They do **not** import vendor SDKs.
- Handler code is **registered ahead of time**; the queue carries `task_name` + validated payload bytes, **never arbitrary serialized executable code**.

### Canonical background execution identity (BG-EXEC-1 / BG-EXEC-2)

All supported background execution paths use the platform-owned canonical background execution bootstrap (`bootstrap_background_execution` / `resolve_background_execution` in `intergrax/runtime/background_execution/bootstrap.py`). Applications and scenarios do **not** mint or own runtime execution identity.

Redelivery/retry of the same transport task preserves canonical `TaskId` and `RunId` while each actual execution attempt receives a new `AttemptId`. This behavior is provider-neutral and owned by the central background execution mechanism.

```text
background transport (TaskRequest / broker message / Celery request)
       ↓
BackgroundTransportExecutionRef (tenant + provider + transport_task_id)
       ↓
BackgroundExecutionIdentityPersistence.resolve_or_create → stable TaskId + RunId
       ↓
bootstrap_background_execution → mint new AttemptId
       ↓
execute_logical_task / NexusWorkerRuntime.run_task
       ↓
runtime (TaskId, RunId, AttemptId)
```

| Field | Owner at worker boundary |
|-------|--------------------------|
| `TaskId` | Central identity persistence (`resolve_or_create`) keyed by transport ref |
| `RunId` | Central identity persistence — **not** `TaskRequest.run_id`; stable across retry/redelivery |
| `AttemptId` | Central bootstrap (mint per actual execution entry) — **not** Celery `request.retries` |
| `tenant_id` | Validated single scope; mismatch fails closed |

Stable `TaskId`/`RunId` across process restart and concurrent workers requires atomic identity persistence: `DistributedKVStore.compare_and_set` or `ConditionalDocumentStore.put_if_absent`. Generic `DocumentStore` without conditional create is rejected at composition; there is no process-local fallback.

`TaskRequest.run_id` and broker message `run_id` remain **transport queue correlation** for status/events indexing; they are not canonical runtime `RunId`.

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

**Writer integration: NOT YET** (DIAG-1I). Platform contract: `intergrax/runtime/background_execution/required_audit_evidence.py`.

Entry points that invoke the bootstrap: `BrokerWorkerBase.process_message`, `WorkerRuntime.process_request`, Celery `intergrax.execute` dispatcher, and `DocumentStoreTaskWorker`.

---

## C. Main concepts

| Concept | Role |
|---------|------|
| **TaskDefinition / JobDefinition** | Declarative registration of a background task type: `task_name`, payload schema, handler reference, policy, required capabilities/tools/integrations |
| **TaskRegistry** | Platform catalog mapping `task_name` → `TaskDefinition`; source of truth for what tasks may be enqueued and executed |
| **TaskRequest** | Immutable enqueue envelope: `tenant_id`, `run_id`, `task_name`, `payload` (bytes), optional `idempotency_key`, `priority` — see [`intergrax/queueing/contracts/task_queue.py`](../../../intergrax/queueing/contracts/task_queue.py) |
| **TaskHandle** | Opaque handle returned after enqueue: `task_id`, `provider`, optional `tenant_id` |
| **TaskResult** | Final outcome: `status`, optional `output` (bytes), `error_message`, `attempts` |
| **TaskStatus** | Lifecycle enum: `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED` (current contract; extended states may map to events) |
| **WorkerRuntime** | Platform process/component that consumes `TaskRequest` messages, resolves handlers, runs execution context, stores results, acks/retry/dead-letters vendor messages |
| **TaskHandler** | Developer-implemented function bound to a `TaskDefinition`; receives decoded payload + execution context; returns `TaskResult` through platform contracts |

All supported background handlers implement one canonical platform handler contract (`BackgroundTaskHandler`) and receive `BackgroundExecutionIdentity` explicitly through `execute_logical_task`.
| **TaskEvent** | Lifecycle/progress fact emitted on an event channel (separate from work transport) |
| **MessageBus / TaskQueue** | Tier-0 transport contract for enqueue, status, result, list, cancel, purge — [`MessageBus`](../../../intergrax/integrations/contracts/message_bus.py) aliases `TaskQueue` |
| **Provider adapter** | Integration implementing `MessageBus` for a vendor (e.g. Kafka publish/consume, SQS poll/lease); serializes/deserializes `TaskRequest`, does not execute handlers |
| **TaskObserver / subscription** | Consumer of `TaskEvent` stream or pull APIs for status/result; may be app UI, agent loop, workflow, notification bridge, observability backend |
| **TaskPolicy** | Retry, timeout, concurrency, rate limit, dead-letter, cancellation, idempotency, and tenant-scoping rules attached to a `TaskDefinition` |
| **TaskTrace / lifecycle timeline** | Correlated observability view across enqueue → dispatch → execution → result, keyed by `run_id`, `task_id`, `correlation_id` |

---

## D. Programmer model

A developer defines a custom background task through platform contracts — **not** by putting code on a queue.

### Steps

1. **Define payload schema** — typed model (e.g. Pydantic) serialized to JSON/protobuf bytes in `TaskRequest.payload`.
2. **Define `task_name`** — stable, namespaced string (e.g. `lkw.background_ingest.v1`, `acme.report.export.v1`).
3. **Implement handler** — decode payload, execute domain logic through platform tools/integrations/runtime; no vendor SDK imports.
4. **Declare task policy** — retries, timeout, concurrency, idempotency semantics, dead-letter behavior.
5. **Declare required tools/integrations/capabilities** — e.g. `local.workspace.index`, `rag.ingest`, `relational_store.query`.
6. **Register `TaskDefinition` in `TaskRegistry`** — wires `task_name` → schema + handler + policy (host bootstrap or application factory).
7. **Enqueue from app/agent** — `message_bus.enqueue` or future `background_tasks.enqueue` with encoded payload.
8. **Observe status/result/events** — pull (`get_status`, `get_result`, `list_tasks`) and/or subscribe to `TaskEvent` stream.

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

**Critical rule:** the queue carries `TaskRequest { task_name, payload }`. **WorkerRuntime** resolves `task_name` to **pre-registered handler code**. Payload is data only — paths, IDs, options — not executable logic.

---

## E. End-to-end lifecycle

```text
1. Developer registers TaskDefinition in TaskRegistry
2. Application/agent calls background_tasks.enqueue or message_bus.enqueue
3. Platform validates task_name, payload schema, permissions, tenant scope, idempotency
4. Platform creates TaskRequest
5. MessageBus provider adapter serializes and sends TaskRequest to vendor
6. Vendor stores/transports message
7. WorkerRuntime receives message (provider-specific consumption model)
8. WorkerRuntime resolves task_name in TaskRegistry
9. WorkerRuntime creates execution context (tools, integrations, trace, tenant scope)
10. Handler executes developer custom code through platform tools/integrations/runtime
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
| **WorkerRuntime** | **Yes** (orchestrates) | Consume message, resolve handler, run execution context |
| **TaskHandler** | **Yes** (domain code) | Developer logic inside platform contracts |

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

Pull inspection is the **mandatory baseline** — works for all providers and supports deterministic agent/workflow checking.

| API | Purpose |
|-----|---------|
| `get_status(TaskHandle)` | Current `TaskStatus` |
| `get_result(TaskHandle)` | Final `TaskResult` when completed |
| `list_tasks(tenant_id, …)` | Recent tasks for tenant with optional status filter |
| `cancel(TaskHandle)` | Request cancellation (provider/handler dependent) |
| `purge_completed(tenant_id, …)` | Retention cleanup for completed task records |

**Current tool surface:** `message_bus.get_status`, `message_bus.get_result`, `message_bus.list_tasks`, `message_bus.cancel`, `message_bus.purge_completed` — see [`intergrax/tools/providers/message_bus/service.py`](../../../intergrax/tools/providers/message_bus/service.py).

**Semantics:**

- Enqueue returns `TaskHandle` immediately; execution is asynchronous. LKW.4E platform proof must observe this async lifecycle — synchronous in-process bypass is not sufficient.
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
- Notification channel bridge (Slack, email — **observers only**)
- Observability backend (traces, metrics)
- Audit / debug tooling

---

## I. Pub-sub / notification options

| Mode | Role |
|------|------|
| **Pull status/result** | Baseline; always available |
| **Event subscription / stream** | Real-time lifecycle for UIs and orchestrators |
| **Callback / webhook** | Optional push to external HTTP endpoint on terminal states |
| **`notification_channel`** | Optional human-facing notify (Slack, email) — **not** core execution |
| **HITL prompt** | Optional human decision gate — observer to `TaskEvent` or terminal status |

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
- **No raw payload content by default** — log `task_name`, sizes, hashes, redacted summaries.
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
| **Cancellation** | Cooperative cancel signal through runtime |
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

**Evolution:**

- Existing `TaskQueue` is the **current base** for transport and pull.
- Future **TaskRegistry**, **WorkerRuntime**, and **TaskEvent** model **build on** this contract — they do not replace it with an application-specific design.
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
- Cloud-managed vendor backends (SQS, Service Bus, Pub/Sub, etc.) in LKW.4E first pass — a **local** broker/provider in the proof stack is required
- Mocks, fake queues, in-memory-only bypasses, and unit-test-only handler invocation as LKW.4E platform proof

---

## Related documents

- [`plan/BACKGROUND_TASKS.md`](../maintainers/plans/BACKGROUND_TASKS.md) — implementation phases
- [`architecture/INTEGRATIONS.md`](INTEGRATIONS.md) — `message_bus` provider category
- [`applications/local_workspace_application/docs/ARCHITECTURE.md`](../../../applications/local_workspace_application/docs/ARCHITECTURE.md) §8.7 — LKW.4 product architecture
- [`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) §6 — LKW.4 task schedule
