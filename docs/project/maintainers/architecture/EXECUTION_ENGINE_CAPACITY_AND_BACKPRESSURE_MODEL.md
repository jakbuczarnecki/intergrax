# Execution Engine — Capacity & Backpressure Model (EE-B1.2)

**Status:** Certified baseline on `development` (EE-B1.2).  
**Predecessor:** W1-A process-local root admission; EE-B1.1 failure semantics.

## 1. Purpose

Define a single, auditable model for **execution capacity admission** and **backpressure** on the canonical execution path without a second scheduler, runtime, or hidden queue.

## 2. Capacity owner

| Concern | Canonical owner | Location |
|--------|-----------------|----------|
| Root execution slot admission | `ExecutionCapacityAdmissionPort` + default `LocalExecutionCapacityAdmission` | `intergrax/contracts/execution_capacity_admission.py`, `intergrax/runtime/execution/local_execution_capacity_admission.py` |
| Wiring into lifecycle | `ExecutionRuntime.execute` (acquire before delegate, release in `finally`) | `intergrax/runtime/execution/runtime.py` |
| Host composition | Optional port injected at composition root | `host_task.py`, recovery handoff |
| Typed decision preview (no slot) | `ExecutionCapacityEvaluator` / `assess_root_execution_capacity` | `intergrax/contracts/execution_capacity/` |

**Answer — who owns execution capacity admission?**  
The **Execution admission plane** owns root process-local slots via `ExecutionCapacityAdmissionPort`. Nexus fan-out bounds, graph executor semaphores, and tool thread pools are **separate layers** (see §12).

## 3. Backpressure owner

Backpressure at root is expressed by the admission port:

- `ExecutionCapacityExceededError` → **REJECT** (overload_mode `REJECT`)
- `ExecutionCapacityAdmissionTimeoutError` → **DEFER** bounded wait exhausted (`WAIT_WITH_TIMEOUT`)

No unbounded buffering: saturated REJECT does not enqueue work inside the capacity module.

## 4. Capacity vs budget

| Plane | Question |
|-------|----------|
| **Capacity** | May this root execution enter the runtime now? |
| **Execution budget** | How much work may this run consume after admission? |

`RunBudget` / ledger is created **after** capacity permit is held inside `ExecutionRuntime` — orthogonal concerns.

## 5. Capacity vs governance

Governance permission (`ExecutionAdmissionHook`, policy engines upstream) may **ALLOW** while capacity **REJECT**s. Permission does not imply physical slots.

Preferred ordering on the canonical path:

```text
Governance / lineage / hooks (ExecutionBoundary)
        ↓
ExecutionRuntime.execute
        ↓
ExecutionCapacityAdmissionPort.acquire (optional)
        ↓
delegate / Nexus / tools
```

## 6. Capacity vs retry

Capacity rejection and admission timeout are **pre-admission**. They must not mint `AttemptId` or invoke `ExecutionAttemptRetryService`. Execution failure retry applies only after a started attempt.

## 7. Local vs distributed semantics

Current qualified implementation is **LOCAL PROCESS GLOBAL CAPACITY** (single `_LocalRootCapacityState` per `LocalExecutionCapacityAdmission` instance). Distributed quota is deferred; `ExecutionCapacityAdmissionPort` remains async for future adapters.

## 8. Admission state machine

```text
REQUEST
  ↓
CAPACITY CHECK (port.acquire or evaluator preview)
  ├── ALLOW  → ADMIT → EXECUTE → RELEASE (permit.release, exactly once)
  ├── DEFER  → bounded wait OR return DEFERRED (timeout → typed error)
  └── REJECT → RETURN REJECTED (ExecutionCapacityExceededError)
```

Preview API (`assess_root_execution_capacity`) mirrors decisions without acquiring.

## 9. Saturation behavior

When `active_root_executions >= capacity_limit`:

- `REJECT` → immediate `ExecutionCapacityExceededError`; delegate not started.
- `WAIT_WITH_TIMEOUT` → wait on condition with `asyncio.wait_for`; on timeout → `ExecutionCapacityAdmissionTimeoutError` (DEFER semantics, no slot leak).

## 10. Release semantics

`ExecutionRuntime.execute` releases the permit in `finally` after success, exception, or cancellation propagation. `LocalExecutionCapacityAdmission` permit release is idempotent and shields against cancellation during release.

## 11. Cancellation behavior

Cancelled root tasks must still release slots; W1-A/W4-A qualification covers shielded release.

## 12. Layering (fan-out & Nexus)

```text
Global / host root capacity (W1-A, optional)
    ↓
Nexus bounded scheduling (MAX_FAN_OUT_*, GraphExecutor caps)
    ↓
Execution-level budget (RunBudget ledger)
```

NPSC-5B fan-out limits remain owned by `bounded_multi_agent_fanout` — capacity policy does not duplicate them.

## 13. Nested child execution

`ChildExecutionRunner` does **not** acquire root `ExecutionCapacityAdmissionPort` slots (child budget via ledger). Parent holding the last root slot while awaiting a child does not consume a second root slot for the child — **no root-slot nested deadlock** for the qualified model. Child concurrency is bounded by parent budget and Nexus/graph caps, not the root admission counter.

## 14. Recovery interaction

`handoff_task_resume_recovery_start` may acquire the same port before `ExecutionRuntime` when wired — single permit handed via `held_root_capacity_permit` to avoid double admission.

## 15. Shutdown

`ExecutionRuntimeShutdownPhase.STOP_ACCEPTING_NEW_WORK` (EE-B1.1) is enforced at host/runtime orchestration; capacity port does not admit new roots once the host stops scheduling. Composition must not bypass admission for new starts.

## 16. Observability

Reuse `RuntimeEvent` / existing observability planes. Capacity does not introduce a parallel metrics engine. Saturation is visible via admission errors and optional host metrics.

## 17. Failure semantics

Capacity infrastructure failures map to typed errors (`ExecutionCapacityExceededError`, `ExecutionCapacityAdmissionTimeoutError`). **Fail-closed:** unknown distributed capacity must not default to ALLOW (future adapters).

## 18. Security

Assessment context counters must come from platform state (`_LocalRootCapacityState`), not from caller-supplied “active count” on admission requests.

## 19. Inventory reference (ETAP 0)

See EE-B1.2 qualification record for the full mechanism table (`local_execution_capacity_admission`, Nexus semaphores, fan-out bounds, recovery admission, dependency bulkheads).
