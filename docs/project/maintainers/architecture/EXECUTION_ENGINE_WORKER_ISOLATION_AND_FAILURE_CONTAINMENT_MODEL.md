# Execution Engine — Worker Isolation & Failure Containment Model (EE-B1.3)

**Status:** Certified baseline on `development` (EE-B1.3).  
**Predecessors:** EE-B1.1 failure semantics; EE-B1.2 capacity admission; W1 concurrent execution work.

## 1. Purpose

Define how **parallel execution work units** (workers) fail, stall, time out, or cancel **without** cascading to siblings, leaking root capacity, duplicating logical work, or creating second retry/recovery/scheduling owners.

Canonical path unchanged:

```text
Intent → Governance → Capacity Admission → ExecutionRuntime → ExecutionBoundary
      → StrategyExecutionRouter → Nexus / ChildExecutionRunner → worker / tool execution
```

## 2. ETAP 0 — Worker / concurrency inventory

| Mechanizm | Lokalizacja | Owner | Failure semantics | Reuse / Unrelated / Risk |
| --------- | ----------- | ----- | ----------------- | ------------------------ |
| Bounded concurrent work (strict) | `concurrent_execution_work.py` | Execution reliability | First failure → shutdown + cancel siblings → re-raise | **Reuse (canonical isolation primitive)** |
| Bounded concurrent work (resilient) | `concurrent_execution_work.py` | Execution reliability | Per-unit `ConcurrentExecutionWorkOutcome` | **Reuse (canonical resilient isolation)** |
| Work port seam | `execution_work_port.py` | Execution | Delegates to host/Nexus | **Reuse** |
| Root capacity slots | `local_execution_capacity_admission.py` | EE-B1.2 admission | acquire/release once per root | **Reuse (orthogonal layer)** |
| Graph parallel batch | `graph_executor.py` | Nexus scheduling | `asyncio.gather` without `return_exceptions` — batch fail-fast | **Unrelated (Nexus graph semantics)** |
| Fan-out bounds | `bounded_multi_agent_fanout.py` | NPSC-5B | partial recovery plane | **Unrelated (no duplicate limiter)** |
| Child execution | `child.py` | ChildExecutionRunner | typed boundary + evidence | **Reuse (lineage preserved)** |
| Failure classification | EE-B1.1 classifier | Execution reliability | observe/classify only | **Reuse** |
| Retry | `ExecutionAttemptRetryService` | Retry subsystem | post-failure authority | **Unrelated (must not move)** |
| Recovery | NPSC-5E plane | Recovery admission | resume/checkpoint | **Unrelated (must not move)** |

## 3. Canonical worker failure containment owner

**Answer — who owns worker failure containment between parallel in-process work units?**

**`intergrax/runtime/execution/concurrent_execution_work.py`** — Execution-owned **EXECUTION RELIABILITY** primitive:

- `execute_concurrent_execution_work` — **strict** (fail-fast, optional sibling cancellation)
- `execute_concurrent_execution_work_resilient` — **resilient** (typed per-unit isolation)

No `WorkerRuntime`, `WorkerSupervisor`, or alternate pool.

```text
ExecutionRuntime
      ↓
bounded execution work (ConcurrentExecutionWorkPolicy)
      ↓
┌──────────┬──────────┬──────────┐
│ Worker A │ Worker B │ Worker C │
└──────────┴──────────┴──────────┘
     ↓fail       ↓ok        ↓ok
     ↓
typed isolated outcome (resilient) OR propagated exception (strict)
     ↓
retry/recovery only through canonical owners
```

## 4. Strict vs resilient semantics

| Mode | API | Sibling on peer failure | Observable outcome |
|------|-----|-------------------------|-------------------|
| **Strict** | `execute_concurrent_execution_work` | Cancelled (fail-fast) | Single exception after drain |
| **Resilient** | `execute_concurrent_execution_work_resilient` | Continue | `ConcurrentExecutionWorkOutcome` per index |

Callers that need all-or-nothing MUST choose strict explicitly. Resilient MUST NOT swallow failures (`FAILED` carries `error`).

## 5. Failure classification model

Reuse EE-B1.1:

- `ExecutionFailureClassifier` / `ExecutionFailureSemanticCategory`

Worker isolation **may** classify upstream failures for projection; it **must not** loop retry internally.

| Outcome class | Typical source | Containment |
|---------------|----------------|-------------|
| SUCCESS | Normal return | N/A |
| EXCEPTION | `Exception` from port | Typed `FAILED` (resilient) or raise (strict) |
| TIMEOUT | `TimeoutError` / bounded `wait_for` at port/boundary | Same as exception; not mixed with cancel |
| CANCELLATION | `asyncio.CancelledError` | Propagates; resilient host cancels worker tasks |
| DEPENDENCY_FAILURE | Classifier `DEPENDENCY_FAILURE` | Typed decision; no provider retry in work pool |
| WORKER_CRASH / ABORT | Process/thread abort (adapter) | Mapped to failure outcome; recovery via NPSC-5E only |

Unknown side-effect state: **UNKNOWN / UNKNOWN_UNSAFE** — no silent retry in the work pool.

## 6. Timeout semantics

| Layer | Owner | Notes |
|-------|-------|-------|
| Execution / worker operation | Port or delegate boundary | Bounded waits on business work |
| Capacity wait | `LocalExecutionCapacityAdmission` | `ExecutionCapacityAdmissionTimeoutError` (DEFER) |
| Dependency | Bulkhead / tool adapters | Separate from concurrent work pool |
| Global runtime | None added by EE-B1.3 | No blanket runtime timeout |

Timeout may trigger cancellation internally but MUST surface as typed failure, not ambiguous cancel.

## 7. Cancellation semantics

- Root cancel → `ExecutionRuntime` lifecycle + cooperative cancellation ports (W4-A).
- Resilient concurrent host: cancel outer task → worker tasks cancelled; `CancelledError` propagates.
- Strict concurrent host: failure sets `shutdown` and cancels peer worker tasks.
- Shield: capacity `permit.release` uses shielded release (EE-B1.2) — does not shield entire execution body.

## 8. Capacity interaction (EE-B1.2)

Root slot held for entire `ExecutionRuntime.execute` regardless of internal worker mix:

```text
worker completion paths (success / exception / timeout / cancel)
        ↓
root delegate returns or raises
        ↓
ExecutionRuntime finally → permit.release exactly once
```

Concurrent work pool does **not** acquire root slots per worker. Saturation at N active roots is independent of `max_concurrency` inside one root.

## 9. Retry & recovery separation

| Concern | Owner |
|---------|-------|
| Worker pool retry | **Forbidden** |
| Attempt retry | `ExecutionAttemptRetryService` |
| Checkpoint / resume | NPSC-5E Recovery Plane |

## 10. Child execution behavior

Child runs via `ChildExecutionRunner` / boundary — not through concurrent work pool root acquisition.

- Child failure → typed boundary outcome + evidence (NPSC-5F).
- Fan-out partial failure → NPSC-5E R3 partial recovery; not reimplemented in EE-B1.3.
- Child failure MUST NOT crash process when domain-handled.

## 11. Fan-out & GraphExecutor

- **NPSC-5B** owns fan-out concurrency caps.
- **GraphExecutor** parallel batches use Nexus gather semantics (graph-level fail-fast unless graph policy says otherwise). EE-B1.3 does not change GraphExecutor scheduling ownership.

## 12. Long-running work

Checkpoint, worker restart, and recovery handoff remain **NPSC-5E** + long-running modules. EE-B1.3 documents in-process isolation only.

## 13. Local vs distributed guarantees

Qualified EE-B1.3 guarantees are **in-process asyncio worker isolation** plus **process-local root capacity**. No claim of cross-process worker isolation unless an adapter provides it.

## 14. Identity invariant

Worker failure containment MUST NOT mint `RunId`, `ExecutionId`, or `AttemptId` outside `AttemptLifecycleService`.

## 15. Duplicate execution audit

The concurrent work pool invokes `port.execute` **at most once per queued index** per host call. No hidden retry loop. Duplicate logical work requires retry/recovery authority upstream.

## 16. Observability & evidence

Failure outcomes remain observable via existing runtime events and mandatory evidence paths (NPSC-5F). No second worker event bus.

## 17. Shutdown interaction

EE-B1.1 phase order:

```text
STOP_ACCEPTING → DRAIN → FLUSH EVIDENCE → PERSIST FINAL STATE → TERMINATE WORKERS
```

During drain, active workers may complete or abort; failures stay contained; no new admission. Capacity release races use idempotent permit release (EE-B1.2).

## 18. Deadlock analysis

| Risk | Mitigation |
|------|------------|
| Capacity lock + worker lock | Root capacity lock not held during port.execute |
| Strict fail-fast cancel | Workers exit on shutdown flag; bounded worker count |
| Nested child + root slot | Child does not acquire root port (EE-B1.2) |
| Evidence flush vs worker | Persistence resilience (EE-B1.1); no new cycle |

Lock ordering: admission `Condition` → permit release guard; business work runs outside admission critical sections.

## 19. Multi-tenant note

Process-local capacity and concurrent pools are **not tenant-partitioned** unless composition injects separate admission instances.
