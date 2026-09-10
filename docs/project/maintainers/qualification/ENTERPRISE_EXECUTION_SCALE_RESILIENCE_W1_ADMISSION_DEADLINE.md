# Enterprise Execution Scale & Resilience — W1 Admission & Deadline

**Task:** Enterprise Scale & Resilience/W1 — Execution Admission, Bounded Concurrent Work & Global Deadline Propagation  
**Status:** PARTIAL / ARCHITECTURAL DECISION REQUIRED (root admission deferred; B+C complete)

## Baseline

| Field | Value |
|-------|--------|
| START_HEAD | `63c7ff83f89c2906072ab47fc9eddec2ac22473e` |
| START_ORIGIN | `63c7ff83f89c2906072ab47fc9eddec2ac22473e` |
| Branch | `development` |

Architecture: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md`](../architecture/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md).  
P0: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_P0_INVENTORY.md`](ENTERPRISE_EXECUTION_SCALE_RESILIENCE_P0_INVENTORY.md).  
W0: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W0_GUARDRAILS.md`](ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W0_GUARDRAILS.md).

## ROOT ADMISSION

| Field | Value |
|-------|--------|
| Status | **ARCHITECTURAL DECISION REQUIRED** — not implemented |
| Canonical owner | `ExecutionRuntime` / `ExecutionBoundary` (unchanged) |
| Reused contract | `ExecutionAdmissionHook.admit()` — pre-start only; insufficient for capacity lifecycle |
| Missing capability | Typed acquire → run → finally release permit/lease spanning whole root Execution |
| Overload behavior | Not owned — no silent queue |
| Release semantics | N/A until lifecycle contract exists |
| Scope | W1 requires process-local only when implemented |
| Distributed | Deferred — `N workers × M ≠ global M` |

## CONCURRENT WORK

| Field | Value |
|-------|--------|
| Status | COMPLETE |
| Owner | `intergrax/runtime/execution/concurrent_execution_work.py` |
| Bound | `ConcurrentExecutionWorkPolicy.max_concurrency` (1…64, contract-owned) |
| Default | `DEFAULT_CONCURRENT_EXECUTION_WORK_POLICY` (64) — explicit typed default, not magic in module |
| Algorithm | Worker pool + index queue; at most `max_concurrency` active `port.execute()` |
| Ordering | Input order preserved in result tuples |
| Strict | First failure fails operation; in-flight workers cancelled |
| Resilient | Per-item `ConcurrentExecutionWorkOutcome` |
| Cancellation | Parent cancel propagates (`CancelledError`) |
| Production caller | `council_deliberation.py` passes participant-scoped policy |

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
- Fan-out platform max (64) as contract ceiling for concurrent work policy

## NEW CONTRACTS

- `intergrax/contracts/concurrent_execution_work.py` — `ConcurrentExecutionWorkPolicy`, `MAX_CONCURRENT_EXECUTION_WORK`
- `ActiveExecutionBudgetState.global_deadline_monotonic` + `peek_active_execution_global_deadline_monotonic()`

## DEFERRED TO W2

Provider/tool bulkheads, distributed admission, retry concurrency per provider, tenant fairness.

## KNOWN LIMITATIONS

- Root execution admission still unbounded until lifecycle permit/lease ADR.
- Global deadline applies only when `RunBudget.max_wall_time_seconds` is set at root bind; unset → field omitted (R1 fail-open for deadline check, unchanged).
- Process-local concurrent work bound only.

## TEST EVIDENCE

```text
uv run pytest tests/unit/runtime/execution/test_enterprise_scale_resilience_w1.py -q
uv run pytest tests/unit/runtime/execution/test_concurrent_execution_work.py -q
uv run pytest tests/unit/runtime/architecture/test_enterprise_scale_resilience_p0_inventory.py -q
uv run pytest tests/unit/applications/test_host_execution_capacity_guardrails.py -q
uv run pytest tests/unit/runtime/nexus/orchestration/test_graph_runner_attempt_lifecycle.py -q
```
