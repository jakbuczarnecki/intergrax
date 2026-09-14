# EE-B1.3 — Worker Isolation & Failure Containment Enterprise Certification

**Task:** EE-B1.3  
**Branch:** `development`

## Provenance

| Field | Value |
|-------|-------|
| **START_HEAD** | `b664029fb30e59ca0587569be3290a9d506301ec` |
| **START_ORIGIN** | `b664029fb30e59ca0587569be3290a9d506301ec` |
| **Remote** | `origin/development` |

## Owner decision

**REUSE EXISTING CONCURRENCY / EXECUTION OWNER** — `concurrent_execution_work.py` remains the canonical in-process worker isolation plane (`execute_concurrent_execution_work` strict, `execute_concurrent_execution_work_resilient` resilient). No new worker runtime, scheduler, queue, retry loop, or recovery engine.

**Who owns worker failure containment?** Execution reliability primitive **`concurrent_execution_work`** (bounded worker pool + typed outcomes). Root capacity release remains **`ExecutionRuntime` + EE-B1.2 admission**. Retry remains **`ExecutionAttemptRetryService`**. Recovery remains **NPSC-5E**.

## Inventory (ETAP 0)

See architecture model §2 table — primary rows: concurrent strict/resilient work, GraphExecutor gather (Nexus), fan-out bounds (NPSC-5B), child runner, capacity admission (EE-B1.2), EE-B1.1 classifier.

## Strict vs resilient

| Mode | Sibling on peer failure | Certified by |
|------|-------------------------|--------------|
| Strict | Cancelled (fail-fast) | `test_ee_b1_3_worker_exception_containment.py` |
| Resilient | Continue with typed `FAILED` | sibling + storm tests |

## Timeout / cancel / capacity

- Worker operation timeout: typed `TimeoutError` per unit (resilient).
- Host cancellation: propagates (`test_ee_b1_3_worker_cancellation_containment.py`).
- Root capacity: released once after delegate completes (`test_ee_b1_3_worker_capacity_release.py`); idempotent permit release under cancel race.

## Duplicate execution

Invocation counter test — each index executed at most once per host call; no hidden pool retry.

## Child / fan-out

Child runner does not own concurrent pool; NPSC-5E R3 regression module referenced; no EE-B1.3 fan-out limiter.

## Shutdown

Documented in architecture §17 — reuses EE-B1.1 phase order; drain + shielded capacity release (EE-B1.2).

## Tests

| Module | Focus |
|--------|-------|
| `test_ee_b1_3_worker_isolation_contract.py` | Doc + outcome typing |
| `test_ee_b1_3_worker_exception_containment.py` | Strict/resilient, duplicate guard |
| `test_ee_b1_3_worker_timeout_containment.py` | Timeout isolation |
| `test_ee_b1_3_worker_cancellation_containment.py` | Cancel propagation |
| `test_ee_b1_3_worker_capacity_release.py` | Root leak + double release race |
| `test_ee_b1_3_worker_failure_storm.py` | Mixed failure storm |
| `test_ee_b1_3_child_execution_failure_containment.py` | Child/fan-out compatibility |
| `test_ee_b1_3_worker_architecture_gate.py` | Forbidden symbols, docs |

## Regression matrix

| Slice | Result |
|-------|--------|
| EE-B1.3 + concurrent work | **32 passed** |
| Frozen slice (EE-A1, EE-A2 H1–H3, NPSC-4.2, NPSC-5B, NPSC-5F Final, W5-H1, EE-B1.1, EE-B1.2) | **157 passed** |
| NPSC-5E R3 fan-out | **22 passed** |
| **Total** | **179 passed** |

## Static quality

| Gate | Result |
|------|--------|
| `ruff check` (EE-B1.3 tests) | **PASS** |
| `ruff format --check` (EE-B1.3 tests) | **PASS** |
| `pyright` changed production scope | **N/A** (docs + tests only) |

## Final verdict

**PASS** — reuse `concurrent_execution_work` as single in-process worker failure owner; EE-B1.3 gates green; frozen matrix green; no production code drift.
