# Enterprise Execution Scale & Resilience — W1 Admission & Deadline

**Task:** Enterprise Scale & Resilience/W1 Final — Admission, Concurrent Work & Global Deadline Qualification
**Status:** **W1 FINAL: PASS**

## Baseline

| Field | Value |
|-------|--------|
| START_HEAD (qualification) | `3dbcc3405519dc79e1f1648c36c1539b7c788ca9` |
| START_ORIGIN | `3dbcc3405519dc79e1f1648c36c1539b7c788ca9` |
| FINAL_HEAD | `3dbcc3405519dc79e1f1648c36c1539b7c788ca9` |
| FINAL_ORIGIN | `3dbcc3405519dc79e1f1648c36c1539b7c788ca9` |
| Branch | `development` |

Historical anchors (ancestors of FINAL_HEAD; code on HEAD is authoritative): W1-A `48854e41…`, W1-B decoupling `be7e5741…`, W1-B strict failure `0da3fd32…`.

Architecture: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md`](../architecture/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md).  
P0: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_P0_INVENTORY.md`](ENTERPRISE_EXECUTION_SCALE_RESILIENCE_P0_INVENTORY.md).  
W0: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W0_GUARDRAILS.md`](ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W0_GUARDRAILS.md).

## W1 scope (what W1 provides)

```text
process-local root capacity (optional port)
bounded explicit concurrent execution work (per-call policy)
absolute monotonic global deadline → R1 retry eligibility (no reset on retry)
```

W1 does **not** provide: cluster-global admission, tenant fairness, provider bulkheads, durable admission queues, distributed rate limits.

## ROOT CAPACITY OWNER

| Field | Value |
|-------|--------|
| Owner | `ExecutionRuntime` — acquire before delegate, `finally` release |
| Hook unchanged | `ExecutionAdmissionHook.admit()` remains pre-execution validation only |
| Scope | **Root execution only** — not child executions, not Nexus scheduling |
| Semantics | **PROCESS-LOCAL** bounded concurrency — not fair tenant scheduler |

## CONTRACT (W1-A)

| Artifact | Location |
|----------|----------|
| Request | `ExecutionCapacityAdmissionRequest` |
| Policy | `ExecutionCapacityPolicy` (`max_concurrent_root_executions >= 1`) |
| Overload | `ExecutionCapacityOverloadMode`: `REJECT`, `WAIT_WITH_TIMEOUT` |
| Port | `ExecutionCapacityAdmissionPort.acquire()` |
| Permit | `ExecutionCapacityPermit.release()` (async; idempotent local impl) |
| Errors | `ExecutionCapacityExceededError`, `ExecutionCapacityAdmissionTimeoutError` |

Module: `intergrax/contracts/execution_capacity_admission.py`. Typed contracts only — no `dict[str, Any]`, raw semaphore injection, or dynamic callables on the port surface.

## LOCAL IMPLEMENTATION

| Field | Value |
|-------|--------|
| Class | `LocalExecutionCapacityAdmission` |
| Module | `intergrax/runtime/execution/local_execution_capacity_admission.py` |
| Underflow | `release_one` raises if `active <= 0` (fail-closed; no silent decrement) |
| Default when omitted | `execution_capacity_admission=None` on `ExecutionRuntime` — legacy / test backward compatible |

## OVERLOAD POLICY

| Mode | Behavior |
|------|----------|
| `REJECT` | Fail fast with `ExecutionCapacityExceededError` when no immediate slot |
| `WAIT_WITH_TIMEOUT` | Requires `wait_timeout_seconds > 0`; timeout → `ExecutionCapacityAdmissionTimeoutError` |

No `WAIT_FOREVER`. No platform-owned unlimited admission backlog.

## RELEASE / CANCELLATION / EXCEPTION

Single outer `finally` on `ExecutionRuntime.execute`: `await permit.release()` after success, exception, or cancellation. Local permit: idempotent second `release()`; cancel during `release()` completes cleanup once; stale permit cannot free a newer holder's slot (regression tests).

## CONCURRENT WORK (W1-B)

| Field | Value |
|-------|--------|
| Owner | `intergrax/runtime/execution/concurrent_execution_work.py` |
| Bound | `ConcurrentExecutionWorkPolicy.max_concurrency` (≥1, caller-supplied; **no platform default**) |
| Primitives | `execute_concurrent_execution_work`, `execute_concurrent_execution_work_resilient` — **policy required** |
| Separation | Independent from `ExecutionCapacityPolicy`, `max_parallel_nodes` / `max_inflight_nodes`, and `MAX_FAN_OUT_CONCURRENCY` (64) |
| Council | `execute_council_deliberation` / parallel proposal hosts require injected `concurrent_work_policy` — no `participant_count` / `min(...)` capacity derivation |

### Production / qualification callers

```text
CALLERS FOUND:
  intergrax/runtime/execution/council_deliberation.py (internal fan-in to concurrent primitives)
  testing_support/decision_e2e/council.py
CALLERS EXPLICIT:
  council_deliberation.py — required parameter on all public council parallel entrypoints
  testing_support/decision_e2e/council.py — composition.participant_concurrent_work_policy
CALLERS MISSING POLICY:
  (none)
```

## GLOBAL DEADLINE (W1-C)

| Field | Value |
|-------|--------|
| Canonical owner | Root `RunBudget.max_wall_time_seconds` → absolute `global_deadline_monotonic` at `bind_root_execution_budget` (set once; **not** recalculated on retry) |
| Public peek | `peek_active_execution_global_deadline_monotonic()` |
| Child | Inherits parent absolute deadline via `ChildExecutionRunner` bind — no extension past parent |
| Propagation | Active execution budget → `NexusGraphRunner._transition_attempt_for_retry` → `ExecutionRetryEligibilityRequest.global_deadline_monotonic` |
| R1 | Retry eligibility policy unchanged — GraphRunner passes deadline only |

## CROSS-LAYER INTERACTIONS

| Interaction | Proof |
|-------------|--------|
| W1-A + W1-B | Orthogonal resources: root port only on `ExecutionRuntime`; concurrent work uses separate policy/state. No shared semaphore. Documented + per-layer behavioral tests (no single combined harness required). |
| W1-B + W1-C | Orthogonal: concurrent work does not touch retry deadline; deadline flows via active execution budget and GraphRunner only. |
| W1-A + W1-C | Root permit wraps full `ExecutionRuntime.execute` (acquire before `boundary.execute`, release in `finally`). In-process graph retry reads active budget while root execution scope remains active — no second root permit for retry within same root lifecycle. |
| No double root admission | `ChildExecutionRunner` does not use `ExecutionCapacityAdmissionPort`. |

## QUALIFICATION TEST MATRIX

| Invariant | Test / gate | Result |
|-----------|-------------|--------|
| Root bound (process-local) | `test_enterprise_scale_resilience_w1_a_root_capacity_admission.py` | PASS |
| Atomic REJECT | `test_reject_concurrent_acquire_capacity_one_atomic`, `test_reject_concurrent_acquire_capacity_three` | PASS |
| Permit release (success/exception/cancel) | W1-A lifecycle tests + `test_execution_runtime.py` | PASS |
| Double release / old permit vs new holder | `test_cancelled_release_terminal_state_blocks_old_permit_double_release` | PASS |
| Underflow on release | `local_execution_capacity_admission` + W1-A release tests | PASS |
| Strict concurrent failure (max=1, first fails) | `test_concurrent_execution_work.py`, W1 strict tests | PASS |
| Strict multi-worker terminal failure | W1-B strict / shutdown tests | PASS |
| Resilient per-item failure + ordering | `test_concurrent_execution_work.py`, W1 resilient tests | PASS |
| Policy explicit / no default | P0 `test_concurrent_execution_work_policy_contract_explicit_required` | PASS |
| Policy > 64 (fan-out decoupling) | W1 `test_concurrent_execution_work_policy_independent_from_fan_out_ceiling` | PASS |
| 100 requests / policy 3 | W1 bounded concurrency test | PASS |
| 4 requests / policy 100 | W1 worker count tests | PASS |
| Council policy injection | `test_council_deliberation.py` | PASS |
| Absolute deadline at bind | W1 `test_root_execution_global_deadline_fixed_at_bind` | PASS |
| Retry after deadline rejected | W1 `test_graph_runner_retry_rejected_when_global_deadline_exceeded` | PASS |
| GraphRunner propagates same deadline | W1 `test_graph_runner_propagates_active_global_deadline` | PASS |
| P0 inventory guards | `test_enterprise_scale_resilience_p0_inventory.py` | PASS |
| W0 host guardrails | `test_host_execution_capacity_guardrails.py` | PASS |
| DG_001 (execution lineage) | `test_execution_lineage_contracts.py` + `tests/unit/runtime/execution/lineage/` | PASS |
| NPSC-5B fan-out bounds | `test_npsc5b_bounded_multi_agent_fanout_gate.py` | PASS |
| NPSC-5E R1 retry | `test_npsc5e_r1_final_retry_attempt_qualification.py`, `test_npsc5e_r1_execution_retry_attempt_semantics.py` | PASS |
| GraphRunner attempt lifecycle | `test_graph_runner_attempt_lifecycle.py` | PASS |

### Regression batch (Final sign-off)

```text
uv run pytest \
  tests/unit/runtime/execution/test_enterprise_scale_resilience_w1_a_root_capacity_admission.py \
  tests/unit/runtime/execution/test_enterprise_scale_resilience_w1.py \
  tests/unit/runtime/execution/test_concurrent_execution_work.py \
  tests/unit/runtime/execution/test_council_deliberation.py \
  tests/unit/runtime/execution/test_execution_runtime.py \
  tests/unit/runtime/architecture/test_enterprise_scale_resilience_p0_inventory.py \
  tests/unit/applications/test_host_execution_capacity_guardrails.py \
  tests/unit/runtime/nexus/orchestration/test_graph_runner_attempt_lifecycle.py \
  tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py \
  tests/unit/runtime/architecture/test_npsc5e_r1_execution_retry_attempt_semantics.py \
  tests/unit/runtime/architecture/test_npsc5b_bounded_multi_agent_fanout_gate.py \
  tests/unit/contracts/test_execution_lineage_contracts.py \
  tests/unit/runtime/execution/lineage/ \
  -q
```

**Result:** 267 passed (2026-09-11, log: `.tmp/session/w1-final-qual/pytest-w1-batch.log`). No new orphan-task / pending-task warnings observed.

## REUSED COMPONENTS

- `ExecutionRetryEligibilityRequest` / `evaluate_execution_retry_eligibility`
- `ExecutionRuntime` + `bind_root_execution_budget`
- `ExecutionAttemptRetryService` / `GraphRunner` retry seam
- Fan-out platform max (64) / items (256) — **independent** from `ConcurrentExecutionWorkPolicy`

## DEFERRED (not W1 defects)

| Wave | Remaining |
|------|-----------|
| **W2** | Per-tenant fairness, provider/tool bulkheads, distributed rate limits, retry-storm isolation |
| **W3** | Checkpoint persistence scaling, SQLite contention, durable recovery scale |
| **W4** | Full cancellation hardening across adapters/providers |
| **W5** | Event/telemetry backpressure |
| **S/R4** | Distributed multi-process / multi-host qualification |

Distributed admission remains a future `ExecutionCapacityAdmissionPort` implementation — not W1.
