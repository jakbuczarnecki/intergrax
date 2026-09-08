# NPSC-3F — Runtime Convergence Certification

**Status:** `CERTIFIED`

**Verdict:** **PASS**

**Date:** 2026-09-08

**Branch:** `development`

**START HEAD:** `de2af369dfd71eb062c63904a229e3cdc5a55001`

**Predecessor:** NPSC-3E (debug/lab convergence), NPSC-3C FINAL (execution engine freeze)

**Task:** NPSC-3F — converge remaining harness scheduler/task-control/runtime paths to canonical `ExecutionRuntime`

---

## 1. Before

Historical harness execution paths after NPSC-3E:

```text
Harness host factory / scaffold
        |
        +--> build_harness_host_task_runner → UnifiedTaskRunner
        +--> wire_long_running_scheduler(task_runner=...)
        +--> wire_harness_task_control(task_runner=...)
        +--> mount_harness_task_routes(task_runner=...)
        |
        v
    execute_root_task / Nexus (bypassing HostTaskExecutionPort boundary)
```

Lab harness FastAPI (`intergrax/harness/lab_fastapi.py`) still constructed `UnifiedTaskRunner(nexus_loop)` for `/run`.

---

## 2. After

```text
Runtime entry (HTTP / scheduler / task-control / lab harness)
        |
        v
TaskExecutor (HostTaskExecutionExecutor)
        |
        v
HostTaskExecutionPort
        |
        v
HostTaskExecution
        |
        v
ExecutionRuntime (frozen NPSC-3C owner)
        |
        v
StrategyExecutionRouter → Nexus orchestration
```

---

## 3. Ownership proof

| Concern | Owner | NPSC-3F result |
| ------- | ----- | -------------- |
| Lifecycle | `ExecutionRuntime` | unchanged — sole owner |
| Identity mint | `identity_authority.py` | no new mint sites in harness convergence |
| Strategy selection | `StrategyExecutionRouter` | unchanged |
| Nexus | orchestration backend only | no root lifecycle ownership added |
| Host boundary | `HostTaskExecution` | scheduler resume, task-control, lab harness `/run` |

---

## 4. Migrated paths

| Surface | Migration | Status |
| ------- | --------- | ------ |
| `task_control_wiring.wire_harness_task_control` | `host_execution` + `mount_canonical_harness_task_routes` | **CONVERGED** |
| Tier-3 host factories (legal, LKW, research, …) | `wire_harness_host_long_running_scheduler` + canonical task control | **CONVERGED** |
| Scaffold templates (`new_application*`) | canonical scheduler/task-control wiring | **CONVERGED** |
| `intergrax/harness/lab_fastapi.py` | `HostTaskExecutionExecutor` | **CONVERGED** |
| `build_harness_host_task_runner` | removed from harness auxiliary wiring | **RETIRED** |

---

## 5. Exceptions (documented)

| Item | Classification | Owner | Retirement plan |
| ---- | -------------- | ----- | --------------- |
| `resolve_harness_host_nexus_loop_legacy` | harness compat seam | platform harness | retain for plugin/observability wiring only |
| `mount_harness_task_routes` | legacy test/direct-mount API | task control | unit tests only; production uses `mount_canonical_harness_task_routes` |
| `wire_long_running_scheduler` | legacy scheduler API | long_running | definition retained; harness hosts use `wire_harness_host_long_running_scheduler` |
| `build_task_runner_with_enricher` + `NexusTaskExecutionAdapter` | fastapi_core run dispatch | queue/run service | classified non-root harness execution; uses thin `execute_root_task` adapter, not lifecycle owner |
| `UnifiedTaskRunner` | scheduler/eval thin adapter module | runtime.task | allowed as primitive; forbidden as harness composition root |

---

## 6. Test evidence

Gate module: `tests/unit/applications/architecture/test_npsc3f_runtime_convergence_gate.py`

Frozen regression: execution, interactions, background_execution, contracts, NPSC-3B/3C/3E, UE-11GP, identity authority gates.

---

## 7. Certification

NPSC-3F completes harness runtime convergence without redesigning `ExecutionRuntime`, without new identity owners, and without granting Nexus lifecycle ownership.
