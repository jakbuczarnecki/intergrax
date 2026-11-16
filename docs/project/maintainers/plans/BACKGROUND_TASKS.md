# Background Tasks - Implementation Plan

**Architecture (1:1):** [`architecture/BACKGROUND_TASKS.md`](../../architecture/BACKGROUND_TASKS.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Proof consumer:** LKW.4 ([`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) §6)
**Last updated:** 2026-08-26 - **UE-DOC-0.7** distributed Execution Boundary admission target (architecture hub)

> **UE-DOC-0.7:** Worker transport must re-enter canonical Execution Boundary with same runtime identity and required causal evidence - see architecture §B distributed execution target. BG-TASKS phases unchanged.

---

## A. Status

| Item | Value |
|------|-------|
| **Track status** | Planned - architecture defined |
| **Architecture** | [`architecture/BACKGROUND_TASKS.md`](../../architecture/BACKGROUND_TASKS.md) - **BG-TASKS-ARCH-1 closed** |
| **Current proof consumer** | LKW.4 background ingest |
| **Next LKW step** | LKW.4E live proof (after LKW.4E-ARCH-1) |

This plan schedules platform implementation phases. **No production WorkerRuntime ships in the architecture-only milestone.**

### A.1 Status reconciliation (PDC-X)

*(PDC-X reconciliation PDC-D7-R023, 2026-11-12.)* The "Planned" track status above and the unmarked BG-TASKS-1..9 phases are historical and predate the code and certifications below. They are kept unchanged; this table records what exists now.

| Phase / item | What exists in code | Certification |
|--------------|---------------------|---------------|
| BG-TASKS-1 TaskDefinition / TaskRegistry | [`definition.py`](../../../../intergrax/background_tasks/definition.py) (`TaskDefinition`), [`registry.py`](../../../../intergrax/background_tasks/registry.py) (`TaskRegistry`) | Not separately certified |
| BG-TASKS-2 WorkerRuntime contract | [`worker_runtime.py`](../../../../intergrax/background_tasks/worker_runtime.py) (`WorkerRuntime`) | Not separately certified |
| BG-TASKS-3 TaskEvent model | [`events.py`](../../../../intergrax/background_tasks/events.py) (`TaskEvent`, `TaskEventName`) | Not separately certified |
| BG-TASKS-4 pull inspection (status/result) | [`state_store.py`](../../../../intergrax/background_tasks/state_store.py) (`TaskStateStore`, `TaskResultStore`) | Not separately certified |
| Background execution identity, idempotent single invocation, redelivery and terminal safety | canonical background bootstrap + host task port | **BG-01 CLOSED** - [`HARNESS_W8_BG_01_CURRENT_HEAD_RECERTIFICATION.md`](../qualification/HARNESS_W8_BG_01_CURRENT_HEAD_RECERTIFICATION.md) |
| One-shot delayed resume scheduling, two-scheduler claim/fence | `intergrax/runtime/long_running/` | **SCHED-01 CLOSED** - [`HARNESS_RESIDUAL_SCHED_01_CURRENT_HEAD_RECERTIFICATION.md`](../qualification/HARNESS_RESIDUAL_SCHED_01_CURRENT_HEAD_RECERTIFICATION.md) |
| Recurrence (interval/cron/time zone/misfire) as one-shot occurrences on the scheduler ledger | `intergrax/runtime/long_running/recurrence.py` and `scheduler.py`; successor persisted before the occurrence executes, tenant-qualified occurrence id, UTC-normalized `run_at_utc`, bounded rules | Delivered by PDC-X B-D7 and repaired by T-IGX-041-G; **library capability, NOT wired** into any host composition (declared exception F-D7-05); audit IA-D7 PARTIAL, re-audit pending |
| Retry budget and dead-letter (DLQ) on the durable queue | `DocumentStoreTaskQueue` / `DocumentStoreTaskWorker(retry_policy=…)`, terminal `DEAD_LETTERED` with a typed reason, sticky terminal states, delayed back-off index | Delivered by PDC-X B-D7 and repaired by T-IGX-041-G; **NOT wired** (no production `retry_policy`, F-D7-05); Celery / Kafka / RabbitMQ DLQ remains declared debt |
| Full execution identity on enqueue | `TaskRequest.identity` persisted by the DocumentStore provider; quadruple seeded by the worker | **Partly adopted:** no production producer, principal and trace unused (declared exception F-D7-09) |
| Multi-process fencing | `tests/qualification/pdc_x/d7/test_d7_fencing_two_process.py` | Proven for two OS processes over one SQLite ledger; PostgreSQL / multi-node excluded by the C10/C15 freeze |
| Graceful worker shutdown with host drain | **Not implemented** (port only; test-side adapter in XD-05) | **Declared exception** (F-D7-03): reopen when the worker-side drain controller is commissioned |

BG-TASKS-5..9 are not reconciled by this note; their status stays as written in §B.

---

## B. Phases

### BG-TASKS-1 - TaskDefinition and TaskRegistry contract

**Scope:** Platform registration model for background task types.

| Deliverable | Detail |
|-------------|--------|
| `TaskDefinition` | `task_name`, `payload_schema`, handler reference, `TaskPolicy`, required capabilities/tools/integrations |
| `TaskRegistry` | Register, lookup, list; reject duplicate `task_name` |
| Validation hooks | Schema validation at register and enqueue time |
| Extension point | Application/host bootstrap registers domain `TaskDefinition` entries |

**Depends on:** existing `TaskRequest.task_name` field  
**Blocks:** BG-TASKS-2, BG-TASKS-7

---

### BG-TASKS-2 - WorkerRuntime contract

**Scope:** Platform worker execution loop contract (not vendor-specific).

| Deliverable | Detail |
|-------------|--------|
| Receive `TaskRequest` | From provider consumer or local harness |
| Resolve `TaskDefinition` | `task_name` → handler + policy |
| Validate payload | Against registered schema + tenant scope |
| Create execution context | Tools, integrations, trace, capabilities |
| Invoke handler | Async/sync boundary defined |
| Return/store `TaskResult` | Align with existing `TaskResult` contract |

**Depends on:** BG-TASKS-1  
**Blocks:** BG-TASKS-7, BG-TASKS-9

---

### BG-TASKS-3 - TaskEvent lifecycle model

**Scope:** Event schema separate from work transport.

| Deliverable | Detail |
|-------------|--------|
| `TaskEvent` schema | Typed lifecycle events (see architecture §H) |
| Emission points | Enqueue, dispatch, start, progress, tool calls, terminal states |
| Redaction policy | No raw payload in events by default |
| Correlation | `run_id`, `task_id`, `correlation_id`, `tenant_id` |

**Depends on:** BG-TASKS-2 (partial), observability spine alignment  
**Blocks:** BG-TASKS-5, BG-TASKS-6

---

### BG-TASKS-4 - Pull inspection surface

**Scope:** Align pull APIs with registry + result store semantics.

| Deliverable | Detail |
|-------------|--------|
| `get_status` / `get_result` | Consistent across providers and local harness |
| `list_tasks` | Tenant-scoped listing with normalized `TaskSummary` |
| Result store semantics | When `get_result` returns `None` vs completed |
| Handle normalization | `TaskHandle` fields stable across providers |

**Depends on:** existing `message_bus.*` tools  
**Status:** Partial - current `TaskQueue` contract exists; registry-aware validation planned

---

### BG-TASKS-5 - Event / pub-sub observation surface

**Scope:** Subscribe/watch task lifecycle beyond pull.

| Deliverable | Detail |
|-------------|--------|
| Event subscription API | Stream or poll `TaskEvent` by `task_id` / `tenant_id` |
| Callback/webhook | Optional push on terminal states |
| Notification channel bridge | Optional mapping to `notification_channel` (Slack, email) |

**Depends on:** BG-TASKS-3  
**Note:** Notification channels are observers, not executors

---

### BG-TASKS-6 - Observability integration

**Scope:** Logs, metrics, traces for background task lifecycle.

| Deliverable | Detail |
|-------------|--------|
| Structured lifecycle logs | Enqueue through ack |
| Metrics | Counters/histograms per architecture §J |
| Trace timeline | `run_id` / `task_id` correlation in observability spine |
| Redaction | Payload-safe diagnostic summaries |

**Depends on:** BG-TASKS-3, [`plan/OBSERVABILITY.md`](OBSERVABILITY.md)

---

### BG-TASKS-7 - Local message bus provider for platform proof

**Scope:** Real local broker/queue in the LKW proof stack - **not** mocks, fake queues, or in-memory-only bypasses.

| Deliverable | Detail |
|-------------|--------|
| Local `MessageBus` provider wiring | Configured `message_bus` integration in proof stack (for example RabbitMQ in Docker) |
| Async enqueue / consume | `message_bus.enqueue` returns before handler completes; worker consumes from broker |
| WorkerRuntime proof mode | Resolves registry, runs handler, stores result |
| LKW.4E consumer | Background ingest live proof via public reviewer path ([`LKW_PLATFORM_PROOF.md`](../../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md) Step 8) |

**Depends on:** BG-TASKS-1, BG-TASKS-2 (minimal contract)  
**Out of scope:** Cloud-managed vendor backends (SQS, Service Bus, Pub/Sub) in LKW.4E first pass

---

### BG-TASKS-8 - Provider portability hardening

**Scope:** Vendor adapters behind `MessageBus` contract.

| Deliverable | Detail |
|-------------|--------|
| Adapter conformance | Kafka, RabbitMQ, SQS, Temporal examples |
| Conformance tests | Enqueue, status, result, idempotency behavior |
| Boundary enforcement | No application/agent vendor SDK imports |

**Depends on:** BG-TASKS-4, existing provider integrations (e.g. Kafka pilot)  
**Note:** LKW.4 does not require all providers

---

### BG-TASKS-9 - Production worker process

**Scope:** Long-running worker deployment model.

| Deliverable | Detail |
|-------------|--------|
| Consumer/polling loop | Provider-specific consumption hidden behind runtime |
| Ack / retry / dead-letter | Policy-driven |
| Concurrency | Per-task and per-tenant limits |
| Shutdown / cancellation | Graceful drain + cooperative cancel |

**Depends on:** BG-TASKS-2, BG-TASKS-8

---

## C. LKW mapping

| LKW task | Maps to |
|----------|---------|
| **LKW.4A** | Payload schema (`LkwBackgroundIngestJob`) → `TaskDefinition.payload_schema` |
| **LKW.4B** / **LKW.4B-PROP-1** | `message_bus.*` tool exposure guardrails |
| **LKW.4C** | Enqueue helper → platform `message_bus.enqueue` |
| **LKW.4D** | Handler contract → `TaskHandler` + `WorkerRuntime` invocation |
| **LKW.4E-ARCH-1** | Platform background task architecture ([`architecture/BACKGROUND_TASKS.md`](../../architecture/BACKGROUND_TASKS.md)) - **this milestone** |
| **LKW.4E** | Live platform proof with real local MessageBus provider (BG-TASKS-7) - enqueue → async worker → search evidence |
| **LKW.4F** | Proof closeout and plan alignment |

---

## D. Acceptance criteria (future implementation)

When BG-TASKS phases are implemented, the platform must satisfy:

1. App or agent can **enqueue a registered task** by `task_name` + validated payload.
2. Worker resolves `task_name` through **TaskRegistry** - unknown tasks rejected.
3. Vendor adapter **only transports** `TaskRequest` messages - no handler execution in provider code.
4. Handler runs through **platform execution context** (tools, integrations, capabilities).
5. **Status and result** available via pull (`get_status`, `get_result`, `list_tasks`).
6. **Lifecycle events** emitted on `TaskEvent` channel for major stages.
7. **Trace timeline** inspectable by `run_id` / `task_id`.
8. **Metrics** emitted per architecture §J.
9. **No raw payload content** leaked in logs, events, or default traces.
10. **Tenant isolation** and **idempotency** enforced at enqueue and execution.

---

## Related documents

- [`architecture/BACKGROUND_TASKS.md`](../../architecture/BACKGROUND_TASKS.md)
- [`architecture/INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md) - `message_bus` category
- [`plan/INTEGRATIONS.md`](INTEGRATIONS.md)
- [`applications/local_workspace_application/docs/ARCHITECTURE.md`](../../../../applications/local_workspace_application/docs/ARCHITECTURE.md) §8.7
- [`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) §6
