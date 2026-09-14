# Execution Engine — Reliability & Failure Semantics Model (EE-B1.1)

**Classification:** `MAINTAINER_CERTIFICATION`  
**Status:** `CERTIFIED` (EE-B1.1 foundation on `development`)  
**Audience:** Maintainers, enterprise qualification, architecture gates  

**Semantic authority:** [`EXECUTION_ENGINE_OWNERSHIP_MODEL.md`](EXECUTION_ENGINE_OWNERSHIP_MODEL.md) (EE-A1). This document adds **reliability ownership and failure semantics** without changing the canonical execution stack.

**Related:** [`EXECUTION_ENGINE.md`](EXECUTION_ENGINE.md) · EE-B1 production scale certification (follow-on).

---

## 1. Purpose

Define how the Execution Engine behaves under failure, resource degradation, persistence errors, worker faults, and infrastructure limits — **without** a second execution loop, scheduler, retry engine, or recovery bypass.

Canonical stack (unchanged):

```text
ExecutionRuntime → ExecutionBoundary → StrategyExecutionRouter
        → Nexus / ChildExecutionRunner → RuntimeToolInvoker
```

---

## 2. ETAP 0 — Reliability inventory (audit baseline)

| Obszar | Aktualny owner | Ryzyko |
| --- | --- | --- |
| Execution lifecycle | `ExecutionRuntime` (`runtime/execution/runtime.py`) | Niskie — single root owner (EE-A1) |
| Worker lifecycle | `concurrent_execution_work` + host admission; long-running via `LongRunningScheduler` | Średnie — resilient vs strict concurrent modes must stay explicit |
| Persistence failure (evidence) | `RuntimeEventPersistence` + `runtime_persistence_resilience` | Średnie — mandatory fail-closed enforced at port |
| Persistence failure (checkpoint) | `LongRunningCoordinator` / `TaskCheckpointPersistence` | Średnie — integrity → no resume-with-corruption |
| Retry failure | `ExecutionAttemptRetryService` + `execution/retry/*` | Niskie — decisions only; execution re-enters runtime |
| Recovery failure | Recovery Plane (NPSC-5E) | Niskie — no agent-local recovery loops |
| Shutdown | Host lifecycle (`intergrax/hosting/shutdown.py`) + runtime drain hooks | Średnie — EE-B1.1 documents runtime phase order |
| Backpressure | `local_execution_capacity_admission`, broker/worker admission | Średnie — tenant-agnostic local slots (W2 inventory) |

---

## 3. Reliability ownership

| Capability | Owner |
| --- | --- |
| execution lifecycle | `ExecutionRuntime` |
| attempt lifecycle | `AttemptLifecycleService` |
| retry decisions | Retry subsystem (`execution/retry`, `ExecutionAttemptRetryService`) |
| checkpoint durability | `LongRunningCoordinator` |
| evidence durability | `RuntimeEventPersistence` |
| recovery | Recovery Plane (NPSC-5E) |
| governance decision | Governance Plane (NPSC-4.2) |

---

## 4. Failure classification contract

**Contract:** `intergrax/contracts/execution_reliability/failure_classification_contract.py`  
**Default classifier:** `intergrax/runtime/execution/reliability/default_failure_classifier.py`

Semantic categories (provider-neutral):

| Category | Meaning |
| --- | --- |
| `TRANSIENT` | Safe to consider retry via retry subsystem |
| `PERMANENT` | Non-retryable execution fault |
| `POLICY_BLOCKED` | Governance / policy denial |
| `RESOURCE_EXHAUSTED` | Budget / capacity exhaustion |
| `DEPENDENCY_FAILURE` | External dependency fault |
| `UNKNOWN` | Uncertain; may carry unknown side-effect flag |

Retry projection uses existing `ExecutionFailureClassification` (`intergrax/contracts/execution_retry.py`) — **no second retry taxonomy**.

---

## 5. Persistence failure semantics

**Contract:** `intergrax/contracts/execution_reliability/persistence_failure_semantics.py`  
**Runtime enforcement (evidence):** `intergrax/runtime/events/runtime_persistence_resilience.py`

| Surface | Owner | On failure |
| --- | --- | --- |
| checkpoint save | Long-running checkpoint port | Integrity → fail closed; no resume if corrupted |
| event append | `RuntimeEventPersistence` | Mandatory evidence → fail closed; no success without evidence |
| state persistence | Decision / terminal stores | Production composition validates durable stores |
| lineage persistence | `ExecutionLineagePersistence` | Propagate error; no silent drop |

Policies: `FAIL_CLOSED`, `RETRY`, `DEGRADE`, `ESCALATE` — mapped to `PersistenceReliabilityDisposition` via `resolve_persistence_failure_policy`. Mandatory durability **never** degrades to continue-without-evidence.

---

## 6. Worker failure semantics

| Scenario | Expected behavior |
| --- | --- |
| A — one worker fails, others continue | `execute_concurrent_execution_work_resilient` isolates per-unit failure; no global crash |
| B — worker crash during execution | Identity mint remains authority-owned; retry creates new attempt; lineage preserved (EE-A2) |

Forbidden: duplicate execution from remint, hidden worker execution loops, local recovery outside Recovery Plane.

---

## 7. Graceful shutdown contract

**Contract:** `intergrax/contracts/execution_reliability/shutdown_contract.py`

Ordered phases:

1. `STOP_ACCEPTING_NEW_WORK`
2. `DRAIN_ACTIVE_EXECUTIONS`
3. `FLUSH_REQUIRED_EVIDENCE`
4. `PERSIST_FINAL_STATE`
5. `TERMINATE_WORKERS`

Forbidden: kill without cleanup, mandatory evidence loss, reopening sealed attempts.

---

## 8. Architecture verification (EE-B1.1)

Gate tests: `tests/unit/runtime/architecture/test_ee_b1_1_*.py`

Forbidden in production:

- Second `ExecutionRuntime` lifecycle owner
- Second orchestration scheduler for execution
- Second retry engine
- Second checkpoint durability owner bypassing `LongRunningCoordinator`

---

## 9. Certification gates

| Gate | Test module |
| --- | --- |
| EE-B1.1 foundation | `test_ee_b1_1_failure_semantics_certification.py` |
| Worker isolation | `test_ee_b1_1_worker_failure_isolation.py` |
| Shutdown contract | `test_ee_b1_1_shutdown_contract.py` |
| Persistence contract | `test_ee_b1_1_persistence_failure_contract.py` |

Frozen regressions: EE-A1, EE-A2 (+ H1–H3), NPSC-4.2, NPSC-5E Final, NPSC-5F R1–R4, EE-FINAL-02 evidence plane qualification.
