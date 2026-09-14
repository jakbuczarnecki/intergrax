# Execution Engine — Chaos Engineering & Fault Injection Model (EE-B2)

**Status:** Qualification baseline on `development` (EE-B2).  
**Predecessors:** EE-B1.1 failure semantics; EE-B1.2 capacity; EE-B1.3 worker isolation; NPSC-5E recovery; NPSC-5F evidence; W5-H1 observability.

## 1. Purpose

Prove through **deterministic, controlled fault injection** that the Execution Engine preserves ownership, fail-closed semantics, evidence integrity, capacity correctness, worker isolation, and recovery correctness when subsystems fail — **without** introducing a second reliability runtime.

```text
FAULT INJECTION → typed containment → no ownership bypass → no duplicate execution
→ no identity corruption → no evidence corruption → retry/recovery via canonical owners only
→ deterministic terminal state
```

## 2. Chaos scope

| Level | Scope | Examples |
| ----- | ----- | -------- |
| **L1** | Component fault | Worker exception, dependency port failure, append failure |
| **L2** | Boundary fault | Capacity reject, mandatory evidence fail-closed, stale checkpoint |
| **L3** | Compound | Worker failure + OTLP outage; capacity hold + cancel; execution + mandatory evidence failure |

## 3. Allowed injection surfaces

| Fault surface | Existing abstraction | Inject point | Expected owner | Deterministic? |
| ------------- | -------------------- | ------------ | -------------- | -------------- |
| Execution worker | `ExecutionWorkPort` | Faulting port / `DeterministicWorkerFaultPort` | `concurrent_execution_work` | Yes |
| Dependency / tool | `ExecutionWorkPort` / tool gateway | `ConnectionError` / bounded timeout at port | EE-B1.1 classifier + boundary | Yes |
| Persistence | `RuntimeEventPersistence` | `FailOnAppendPersistence` | Evidence plane + `runtime_persistence_resilience` | Yes |
| Evidence mandatory | Event bus `record` | Fail-on-append | Mandatory → FAIL_CLOSED | Yes |
| Checkpoint | `SQLiteTaskCheckpointStore` | Stale `expected_revision` | Long-running coordinator / NPSC-5E | Yes |
| Capacity | `LocalExecutionCapacityAdmission` | REJECT / acquire-release | EE-B1.2 | Yes |
| Child execution | Resilient concurrent work | Per-label fault port | Same as worker isolation | Yes |
| Recovery | Checkpoint revision | `StaleCheckpointWriteError` | NPSC-5E | Yes |
| Observability export | `ObservabilityExporter` | Failing exporter adapter | W5-H1 export policy | Yes |
| Cancellation | Host task + `PhaseGate` | `asyncio` cancel at barrier | Execution runtime / concurrent host | Yes |
| Timeout | Port `wait_for` | Bounded timeout | Worker isolation | Yes |
| Shutdown | Phase order contract | Cancel during simulated drain | EE-B1.1 shutdown phases | Yes |

## 4. Prohibited injection techniques

- No `ChaosRuntime`, `ChaosScheduler`, `FaultRecoveryEngine`, or second retry/recovery/evidence owners in production.
- No `getattr` / `setattr` / `hasattr` or private field mutation for faults (`testing_support/chaos`).
- No `random` chaos, unbounded `sleep`, or network toxiproxy in EE-B2 scope.
- No production import of `testing_support.chaos`.

## 5. Deterministic fault model

- `FailOnCall(call_number=N)` — fail on N-th invocation.
- `PhaseGate` — block until explicit `release()`.
- Typed exceptions at ports (worker, dependency, persistence).
- Compound scenarios document **primary failure**, **secondary failure**, and **terminal authority**.

## 6. Ownership matrix (unchanged)

| Concern | Owner |
| ------- | ----- |
| Worker containment | `concurrent_execution_work` |
| Capacity | `LocalExecutionCapacityAdmission` (EE-B1.2) |
| Mandatory evidence | `RuntimeEventPersistence` + fail-closed resilience |
| Recovery / resume | NPSC-5E recovery plane |
| Retry | `ExecutionAttemptRetryService` |
| OTLP export | Observability export policy (non-authoritative for execution result) |

## 7. Identity & lineage

Chaos helpers **must not mint** execution identity. Tests assert stable `RunId` / `ExecutionId` / `AttemptId` via existing mint helpers in test orchestration only.

Child fan-out partial failure preserves sibling successes; recovery replay of successful siblings is forbidden (NPSC-5E R3 regression referenced).

## 8. Retry ≠ recovery

Chaos qualification reuses frozen NPSC-5E and EE-B1 contracts: retry decisions ≠ checkpoint resume ≠ partial fan-out recovery.

## 9. Shutdown chaos

Canonical order: `STOP_ACCEPTING_NEW_WORK` → `DRAIN_ACTIVE_EXECUTIONS` → `FLUSH_REQUIRED_EVIDENCE` → `PERSIST_FINAL_STATE` → `TERMINATE_WORKERS`. Drain cancellation with contained worker fault must not crash the runtime.

## 10. Observability vs evidence

External export failure is **observability-local**. Mandatory evidence append failure remains **FAIL_CLOSED** and is not equivalent to OTLP outage.

## 11. Test-only boundary

Reusable helpers live under `testing_support/chaos/`. Qualification tests under `tests/unit/runtime/architecture/test_ee_b2_*.py`.

## 12. Chaos matrix (summary)

| Scenario | Fault point | Expected outcome | Forbidden outcome |
| -------- | ----------- | ---------------- | ----------------- |
| Worker failure | Port raise | FAILED + siblings OK (resilient) | Duplicate invocation |
| Dependency | ConnectionError | DEPENDENCY_FAILURE classification | Hidden pool retry |
| Timeout | Bounded `wait_for` | Typed TimeoutError | Infinite wait |
| Capacity | 2 slots + 3rd | REJECT / DEFER per policy | Hidden queue |
| Cancel | Host cancel | CancelledError + permit release | Capacity leak |
| Mandatory evidence | Append fail | MandatoryEvidencePersistenceError | False success |
| Checkpoint | Stale writer | StaleCheckpointWriteError | Corrupt resume |
| Recovery interrupt | Stale write during recovery path | Fail closed | Sealed attempt reopen |
| Child partial | B fails, A/C ok | Preserved successes | Replay A/C |
| OTLP | Export raise | Event persisted | Execution result flipped |
| Compound | Worker + OTLP | Worker authority | Export masks worker |
| Shutdown | Cancel during drain | Contained cancel | Runtime crash |
