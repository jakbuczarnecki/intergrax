# EE-B1.2 — Execution Capacity & Backpressure Enterprise Certification

**Task:** EE-B1.2  
**Branch:** `development`

## Provenance

| Field | Value |
|-------|-------|
| **START_HEAD** | `59fbf6f305b70d2b74adac7cd61dd21d352dba78` |
| **START_ORIGIN** | `59fbf6f305b70d2b74adac7cd61dd21d352dba78` |
| **Remote** | `origin/development` |

## Owner decision

**REUSE EXISTING OWNER** — W1-A `ExecutionCapacityAdmissionPort` + `LocalExecutionCapacityAdmission` remain the canonical root capacity admission plane. EE-B1.2 adds typed decision contracts (`ExecutionCapacityAdmissionDecision`, `ExecutionCapacityEvaluator`) and certification gates; no second runtime or scheduler.

**Who owns execution capacity admission?** `ExecutionRuntime` wiring of `ExecutionCapacityAdmissionPort` (implementation: `LocalExecutionCapacityAdmission` for process-local slots).

## Inventory (ETAP 0)

| Mechanizm | Lokalizacja | Owner | Semantyka | Reuse / Retire / Unrelated |
|-----------|-------------|-------|-----------|----------------------------|
| Root slot admission | `local_execution_capacity_admission.py` | Execution admission | REJECT / bounded WAIT | **Reuse (canonical)** |
| Admission contracts | `execution_capacity_admission.py` | Contracts | Policy + port + permit | **Reuse** |
| Typed decision preview | `contracts/execution_capacity/` | Contracts (EE-B1.2) | ALLOW / DEFER / REJECT | **Reuse (new adjunct)** |
| Runtime wiring | `runtime.py` `execute` | ExecutionRuntime | acquire / finally release | **Reuse** |
| Recovery handoff | `task_resume_recovery_handoff.py` | Resilience handoff | optional acquire → held permit | **Reuse** |
| Graph backpressure | `GraphExecutor` | Nexus | semaphore + event | **Unrelated layer** |
| Fan-out bounds | `bounded_multi_agent_fanout.py` | Nexus platform | hard caps | **Unrelated (no duplicate)** |
| Recovery admission | `local_recovery_admission.py` | Recovery plane | start-only permit | **Unrelated** |
| Dependency bulkhead | W2-B1 | Tool/LLM plane | dependency slots | **Unrelated** |
| Concurrent work pool | `concurrent_execution_work.py` | Execution work | `max_concurrency` policy | **Unrelated** |
| Child execution | `child.py` | ChildExecutionRunner | ledger budget, no root port | **Unrelated** |

## Contract model

- `ExecutionCapacityAdmissionDecision`: `ALLOW`, `DEFER`, `REJECT`
- `ExecutionCapacityAssessmentContext`: platform counters + policy mode
- `assess_root_execution_capacity` / `RootExecutionCapacityEvaluator`: deterministic preview
- Acquire/release unchanged on `ExecutionCapacityAdmissionPort`

## Implementation

- `intergrax/contracts/execution_capacity/`
- `intergrax/runtime/execution/capacity/` (re-export `LocalExecutionCapacityAdmission`)
- Architecture: `EXECUTION_ENGINE_CAPACITY_AND_BACKPRESSURE_MODEL.md`

## Tests

| Module | Focus |
|--------|-------|
| `test_ee_b1_2_capacity_contract.py` | Typed contract, enum, policy forbid |
| `test_ee_b1_2_capacity_admission.py` | REJECT/DEFER overload mapping |
| `test_ee_b1_2_capacity_concurrency.py` | `max_active <= capacity` |
| `test_ee_b1_2_capacity_release_semantics.py` | success / exception / cancel / idempotent release |
| `test_ee_b1_2_capacity_child_execution_interaction.py` | no root port on child |
| `test_ee_b1_2_capacity_architecture_gate.py` | docs, forbidden symbols, vendor-free contract |

Legacy W1-A: `test_enterprise_scale_resilience_w1_a_root_capacity_admission.py` remains qualified regression.

## Saturation & leak proof

Covered by EE-B1.2 admission/concurrency/release tests + W1-A lifecycle tests.

## Nested execution

Child does not acquire root capacity port; documented in architecture §13 — **no root-slot nested deadlock** for qualified model.

## Fan-out compatibility

No new fan-out limiter; NPSC-5B bounds unchanged. Regression via `test_npsc5e_r3_child_fanout_partial_recovery.py` in frozen matrix.

## Regression matrix

See § Test execution evidence (filled at certification run).

## Static quality

Scope: `intergrax/contracts/execution_capacity`, `intergrax/runtime/execution/capacity`, EE-B1.2 tests.

## Final verdict

**PENDING RUN** — updated after pytest + ruff + pyright.
