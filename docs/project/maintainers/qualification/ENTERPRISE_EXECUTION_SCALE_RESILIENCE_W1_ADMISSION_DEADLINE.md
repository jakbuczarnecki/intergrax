# Enterprise Execution Scale & Resilience — W1 Admission & Deadline

**Task:** Enterprise Scale & Resilience/W1 — Execution Admission, Bounded Concurrent Work & Global Deadline Propagation  
**Status:** B+C complete; **W1-A** root capacity lifecycle complete (production wiring optional — see CONFIGURATION OWNERSHIP).

## Baseline

| Field | Value |
|-------|--------|
| START_HEAD | `63c7ff83f89c2906072ab47fc9eddec2ac22473e` |
| START_ORIGIN | `63c7ff83f89c2906072ab47fc9eddec2ac22473e` |
| Branch | `development` |

Architecture: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md`](../architecture/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md).  
P0: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_P0_INVENTORY.md`](ENTERPRISE_EXECUTION_SCALE_RESILIENCE_P0_INVENTORY.md).  
W0: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W0_GUARDRAILS.md`](ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W0_GUARDRAILS.md).

## ROOT CAPACITY OWNER

| Field | Value |
|-------|--------|
| Owner | `ExecutionRuntime` — acquire before delegate, `finally` release |
| Hook unchanged | `ExecutionAdmissionHook.admit()` remains pre-execution validation only |
| Scope | **Root execution only** — not child executions, not Nexus scheduling |
| Semantics | **Process-local** bounded concurrency — not fair tenant scheduler |

## CONTRACT

| Artifact | Location |
|----------|----------|
| Request | `ExecutionCapacityAdmissionRequest` |
| Policy | `ExecutionCapacityPolicy` (`max_concurrent_root_executions >= 1`) |
| Overload | `ExecutionCapacityOverloadMode`: `REJECT`, `WAIT_WITH_TIMEOUT` |
| Port | `ExecutionCapacityAdmissionPort.acquire()` |
| Permit | `ExecutionCapacityPermit.release()` (async; idempotent local impl) |
| Errors | `ExecutionCapacityExceededError`, `ExecutionCapacityAdmissionTimeoutError` |

Module: `intergrax/contracts/execution_capacity_admission.py`.

## LOCAL IMPLEMENTATION

| Field | Value |
|-------|--------|
| Class | `LocalExecutionCapacityAdmission` |
| Module | `intergrax/runtime/execution/local_execution_capacity_admission.py` |
| Mechanism | `asyncio.Semaphore` (private — not part of public contract) |
| Default when omitted | `execution_capacity_admission=None` on `ExecutionRuntime` — legacy / test backward compatible |

## OVERLOAD POLICY

| Mode | Behavior |
|------|----------|
| `REJECT` | Fail fast with `ExecutionCapacityExceededError` when no immediate slot |
| `WAIT_WITH_TIMEOUT` | Requires `wait_timeout_seconds > 0`; timeout → `ExecutionCapacityAdmissionTimeoutError` |

No `WAIT_FOREVER`. No platform-owned unlimited admission backlog — waiter population bounded by external request transport when waiting.

## RELEASE SEMANTICS

Single outer `finally` on `ExecutionRuntime.execute`: `await permit.release()` after success, exception, or cancellation. Local permit: idempotent second `release()` is a no-op (no semaphore over-release).

## CANCELLATION SEMANTICS

Waiting for permit: cancellation propagates `CancelledError` without acquiring. Holding permit: cancellation during delegate still runs `finally` release.

## EXCEPTION SEMANTICS

Delegate exceptions propagate after `finally` release; slot not leaked.

## CONFIGURATION OWNERSHIP

Explicit `ExecutionCapacityPolicy` required to construct `LocalExecutionCapacityAdmission`. No global production default limit in platform code. Canonical production composition may inject the port when an Execution-owned configuration seam exists; until then inject via `ExecutionRuntime(execution_capacity_admission=...)`.

## KNOWN LIMITATIONS

- Process-local only — `N workers × M` ≠ global `M`.
- No strict FIFO fairness guarantee (`asyncio.Semaphore`).
- Root capacity independent of `GraphExecutor` `max_parallel_nodes` / `max_inflight_nodes`.
- Global deadline unchanged — applies only when root `RunBudget.max_wall_time_seconds` set.

## DISTRIBUTED ADMISSION DEFERRED

Future: Redis / DB / cluster coordinator implementing `ExecutionCapacityAdmissionPort` without changing `ExecutionRuntime`.

## CONCURRENT WORK

| Field | Value |
|-------|--------|
| Status | COMPLETE |
| Owner | `intergrax/runtime/execution/concurrent_execution_work.py` |
| Bound | `ConcurrentExecutionWorkPolicy.max_concurrency` (≥1, caller-supplied; no platform default) |
| Default | **None** — policy is required at every public call site |
| Algorithm | Worker pool + index queue; at most `min(max_concurrency, len(requests))` active `port.execute()` |
| Ordering | Input order preserved in result tuples |
| Strict | First failure fails operation; in-flight workers cancelled |
| Resilient | Per-item `ConcurrentExecutionWorkOutcome` |
| Cancellation | Parent cancel propagates (`CancelledError`) |
| Production caller | Composition injects `concurrent_work_policy` into `execute_council_deliberation` / parallel proposal hosts |

## GLOBAL DEADLINE

| Field | Value |
|-------|--------|
| Status | COMPLETE (propagation gap closed) |
| Canonical owner | Root `RunBudget.max_wall_time_seconds` → absolute `global_deadline_monotonic` at `bind_root_execution_budget` |
| Public peek | `peek_active_execution_global_deadline_monotonic()` |
| Child inheritance | Child budget bind copies parent deadline (no reset) |
| Propagation path | Active execution budget → `NexusGraphRunner._transition_attempt_for_retry` → `ExecutionRetryEligibilityRequest` |
| R1 eligibility | Unchanged — policy decides retry vs `global_deadline_exceeded` |
| Deadline reset on retry | **NO** — same absolute timestamp for execution scope |

## REUSED COMPONENTS

- `ExecutionRetryEligibilityRequest` / `evaluate_execution_retry_eligibility`
- `ExecutionRuntime` + `bind_root_execution_budget`
- `ExecutionAttemptRetryService` / `GraphRunner` retry seam
- Fan-out platform max (64) is **independent** from `ConcurrentExecutionWorkPolicy` (W1-B decoupling)

## NEW CONTRACTS

- `intergrax/contracts/concurrent_execution_work.py` — `ConcurrentExecutionWorkPolicy` (explicit injection; no platform ceiling)
- `intergrax/contracts/execution_capacity_admission.py` — W1-A root capacity port/policy/permit
- `ActiveExecutionBudgetState.global_deadline_monotonic` + `peek_active_execution_global_deadline_monotonic()`

## DEFERRED TO W2

Provider/tool bulkheads, distributed admission, retry concurrency per provider, tenant fairness.

## TEST EVIDENCE

```text
uv run pytest tests/unit/runtime/execution/test_enterprise_scale_resilience_w1_a_root_capacity_admission.py -q
uv run pytest tests/unit/runtime/execution/test_enterprise_scale_resilience_w1.py -q
uv run pytest tests/unit/runtime/execution/test_concurrent_execution_work.py -q
uv run pytest tests/unit/runtime/architecture/test_enterprise_scale_resilience_p0_inventory.py -q
uv run pytest tests/unit/applications/test_host_execution_capacity_guardrails.py -q
uv run pytest tests/unit/runtime/nexus/orchestration/test_graph_runner_attempt_lifecycle.py -q
uv run pytest tests/unit/runtime/execution/test_execution_runtime.py -q
```
