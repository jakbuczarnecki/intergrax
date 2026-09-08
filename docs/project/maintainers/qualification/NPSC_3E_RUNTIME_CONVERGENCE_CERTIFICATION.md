# NPSC-3E — Debug/Lab Runtime Convergence Certification

**Status:** `CERTIFIED`

**Verdict:** **PASS**

**Date:** 2026-09-08

**Branch:** `development`

**START HEAD:** `3b24a2958af5cf5ee6feddf3f40170da28336edb`

**Predecessor:** NPSC-3C FINAL (execution engine freeze)

**Task:** NPSC-3E — converge debug/lab/harness auxiliary execution paths to canonical `ExecutionRuntime`

---

## 1. Migration scope

| Surface | Before | After | Status |
| ------- | ------ | ----- | ------ |
| `intergrax/debug/app.py` | mixed Nexus wiring | `HostTaskExecutionExecutor` → `HostTaskExecution` | **CONVERGED** |
| `intergrax/debug/hitl_service.py` | `execute_root_task` + ad-hoc `NexusLoop` | `HostTaskExecutionPort.execute` | **CONVERGED** |
| `applications/lab_application/host/factory.py` | `UnifiedTaskRunner` scheduler/task-control | `wire_harness_host_*` canonical helpers | **CONVERGED** |
| `intergrax/lab/organization_worker.py` | already canonical intake | unchanged canonical path | **CONVERGED** |
| `scenario_runtime_baseline.execute_scenario_task` | `UnifiedTaskRunner` + `mint_run_id` | `build_host_task_execution` | **CONVERGED** |

---

## 2. Before / after architecture

```text
Before (debug/lab residual):
  debug HITL / lab scheduler / harness task routes
        |
        +--> UnifiedTaskRunner / execute_root_task

After (NPSC-3E):
  debug / lab / harness auxiliary
        |
        v
  TaskExecutor (HostTaskExecutionExecutor)
        |
        v
  HostTaskExecutionPort
        |
        v
  ExecutionRuntime (frozen NPSC-3C owner)
```

---

## 3. Ownership proof

| Concern | Owner | NPSC-3E result |
| ------- | ----- | -------------- |
| Lifecycle | `ExecutionRuntime` | unchanged — sole owner |
| Identity mint | `identity_authority.py` | no new mint sites in debug/lab |
| Strategy selection | `StrategyExecutionRouter` | unchanged |
| Nexus | orchestration backend only | no root lifecycle ownership added |
| Host boundary | `HostTaskExecution` | debug/lab intake + HITL + lab scheduler/task-control |

---

## 4. Exception list (harness-only, documented)

| Item | Classification | Notes |
| ---- | -------------- | ----- |
| `resolve_harness_host_nexus_loop_legacy` | harness compat | plugin/observability wiring only |
| `build_harness_host_task_runner` → `UnifiedTaskRunner` | harness legacy helper | retained for non-lab harness hosts pending isolated retirement |
| `wire_long_running_scheduler(task_runner=...)` | harness legacy API | lab uses `wire_harness_host_long_running_scheduler` |
| `mount_harness_task_routes(task_runner=...)` | harness legacy API | lab uses `mount_canonical_harness_task_routes` |

---

## 5. Test evidence

| Suite | Result | Log |
| ----- | ------ | --- |
| NPSC-3E gates | **PASS** (5) | `.tmp/session/NPSC-3E/phase2-convergence.log` |
| Execution regression (`execution/`, `interactions/`, `background_execution/`) | **752 passed**, 1 skipped (celery) | `.tmp/session/NPSC-3E/phase3-execution-regression.log` |
| Frozen gates (NPSC-3B, 3C-A..D, 3E, F-R1, UE-11GP, NPSC-2) | **54 passed** | `.tmp/session/NPSC-3E/phase4-frozen-gates-rerun.log` |

Gate module: `tests/unit/applications/architecture/test_npsc3e_runtime_convergence_gate.py`

---

## 6. Residual debt

- Tier-3 harness hosts other than lab still compose scheduler/task-control through `UnifiedTaskRunner` helpers (outside NPSC-3E scope; no production root bypass).
- `applications/lab_application/docker/runtime-context/` vendored snapshot is excluded from gate scans (not authoritative source).
- Celery-backed background admission test skipped when `celery` extra not installed (environment parity item).

---

## 7. Certification

NPSC-3E converges debug and lab execution surfaces to the frozen canonical engine without redesigning `ExecutionRuntime`, without new identity owners, and without Nexus lifecycle ownership changes.
