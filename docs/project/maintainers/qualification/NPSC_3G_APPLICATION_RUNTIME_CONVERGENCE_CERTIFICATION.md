# NPSC-3G — Application Runtime Execution Convergence Certification

**Status:** `CERTIFIED`

**Verdict:** **PASS**

**Date:** 2026-09-08

**Branch:** `development`

**Predecessor:** NPSC-3F (harness runtime convergence), NPSC-3C FINAL (execution engine freeze)

**Task:** NPSC-3G — converge fastapi_core run/queue execution to canonical `HostTaskExecutionPort`

---

## 1. Before

```text
HTTP POST /runs / queue dispatch
        |
        v
NexusTaskExecutionAdapter / QueuedNexusExecutionAdapter
        |
        v
UnifiedTaskRunner.run_task()
        |
        v
execute_root_task()  (root bypass of HostTaskExecutionPort)
```

Tier-3 hosts **legal_application** and **dispute_sim_application** composed `DefaultRunService` with `NexusTaskExecutionAdapter`. Queue workers used `NexusWorkerRuntime` → `UnifiedTaskRunner`.

---

## 2. After

```text
HTTP POST /runs / queue dispatch / Celery worker
        |
        v
HostTaskExecutionRunAdapter / QueuedHostTaskExecutionAdapter
        |
        v
HostTaskExecutionExecutor
        |
        v
HostTaskExecutionPort
        |
        v
ExecutionRuntime (frozen NPSC-3C owner)
        |
        v
StrategyExecutionRouter → Nexus orchestration
```

---

## 3. Migration matrix

| Path | Action | Status |
| ---- | ------ | ------ |
| `HostTaskExecutionRunAdapter` | new thin fastapi_core adapter | **CONVERGED** |
| `QueuedHostTaskExecutionAdapter` | queue dispatch adapter | **CONVERGED** |
| `NexusWorkerRuntime` | worker uses `build_host_task_execution` | **CONVERGED** |
| `wire_optional_queue_execution` | `host_execution` injection | **CONVERGED** |
| `legal_application/host/factory.py` | run service via host execution | **CONVERGED** |
| `dispute_sim_application/host/factory.py` | queue-only run service wiring | **CONVERGED** |
| `NexusTaskExecutionAdapter` | legacy inline adapter | **RETIRED** |
| `QueuedNexusExecutionAdapter` | legacy queued adapter | **RETIRED** |
| Interaction / task-control / scheduler (NPSC-3F) | already canonical | unchanged |

---

## 4. Ownership proof

| Concern | Owner | NPSC-3G result |
| ------- | ----- | -------------- |
| Lifecycle | `ExecutionRuntime` | unchanged — sole owner |
| Identity mint | `identity_authority.py` | no new mint sites in run/queue adapters |
| Strategy selection | `StrategyExecutionRouter` | unchanged |
| Nexus | orchestration backend only | no root lifecycle ownership added |
| Run/queue adapters | thin boundary only | no lifecycle, identity, or strategy selection |

---

## 5. Identity proof

Post-migration scan targets:

| Area | `mint_` / `bind_active_execution_identity` |
| ---- | ------------------------------------------ |
| `intergrax/fastapi_core/` | none |
| `host_task_execution_run_adapter.py` | none |
| `queued_host_task_execution_adapter.py` | none |
| `queue_worker_wiring.py` | none |
| `nexus_worker_execution.py` (worker) | none |

Allowed owners remain `identity_authority.py` and `ExecutionBoundary`.

---

## 6. Regression evidence

Gate module: `tests/unit/applications/architecture/test_npsc3g_application_runtime_convergence_gate.py`

Additional:

- `tests/unit/runtime/task/test_host_task_execution_run_adapter.py`
- `tests/integration/applications/test_unified_execution_entry_j2.py`
- `tests/integration/applications/test_unified_execution_entry_j3.py`
- NPSC-3C / NPSC-3E / NPSC-3F frozen gates

---

## 7. Remaining debt

| Item | Classification | Notes |
| ---- | -------------- | ----- |
| `UnifiedTaskRunner` | internal primitive | scheduler legacy API / eval paths; not application run root |
| `mount_harness_task_routes` | legacy test API | NPSC-3F documented |
| `build_task_runner_with_enricher` | task_control helper | retained for legacy harness_task_routes only |
| Celery logical task name `nexus.task.v2` | transport label | handler now routes through host execution |

---

## 8. Definition of Done

- [x] `NexusTaskExecutionAdapter` retired
- [x] All application run/queue execution enters `HostTaskExecutionPort`
- [x] `ExecutionRuntime` remains only lifecycle owner
- [x] `identity_authority` remains only mint owner
- [x] NPSC-3G gate PASS
- [x] Application regression PASS

---

## 9. Certification

NPSC-3G completes application runtime execution convergence without modifying frozen NPSC-3C execution engine ownership, without new identity owners, and without granting Nexus lifecycle ownership.
