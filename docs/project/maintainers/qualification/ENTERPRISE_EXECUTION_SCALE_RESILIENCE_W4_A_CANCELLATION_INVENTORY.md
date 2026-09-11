# Enterprise Execution Scale & Resilience — W4-A Cancellation Inventory

**Task:** W4-A — Cancellation Hardening (canonical lifecycle & ownership)  
**Status:** INVENTORY COMPLETE · hardening applied to cooperative backoff + tool/provider retry paths  
**Production runtime changed:** YES (cooperative delay, invoker retry, LLM retry/resilience, policy enforcer explicit cancel)

Companion: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md`](../architecture/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md).

## Invariants (W4)

| ID | Invariant | Enforcement |
|----|-----------|-------------|
| I1 | Cancellation propagates caller → execution → dependency | `CancellationCoordinator` metadata; `GraphExecutor` batch checks; `ExecutionRetryEligibilityRequest.cancelled`; dependency boundary detached completion |
| I2 | Cleanup always completes (permit, lease, worker) | `ExecutionRuntime.execute` `finally` release; admission `shield` release; `DependencyAttemptExecutionBoundary` worker-terminal release; recovery handoff `finally` |
| I3 | Cancel does not corrupt CAS / double-release | Checkpoint CAS unchanged on cancel; admission release generation tokens (W1/W2/W3 qualified) |

**No new** `CancellationManager`, coordinator scheduler, or global cancel bus — reuse `CancellationCoordinator` + asyncio + `finally`.

## Inventory by domain

### `runtime/execution`

| Location | Mechanism | Owner cleanup |
|----------|-----------|---------------|
| `runtime.py` `ExecutionRuntime.execute` | `acquire` → `try`/`finally` `capacity_permit.release()` | ExecutionRuntime |
| `local_execution_capacity_admission.py` | `shield` on release task; re-raise `CancelledError` after release | Admission port |
| `concurrent_execution_work.py` | Parent `CancelledError` → `shutdown` + cancel workers + `gather` | Concurrent work primitive |
| `fan_out_partial_recovery.py` | `recovery_permit` `try`/`finally` release (per slot + outer) | Partial recovery orchestrator |
| `decision_durable_recovery_handoff.py` | Recovery admission acquire + `finally` release | Handoff module |
| `retry/policy.py` | `request.cancelled` → `ExecutionRetryAction.CANCEL` | Policy (no sleep) |

### `runtime/resilience`

| Location | Mechanism | Notes |
|----------|-----------|-------|
| `dependency_attempt_execution_boundary.py` | `detach_if_still_running` vs `complete_attached`; worker callback release | Caller timeout ≠ permit release until worker terminal |
| `local_dependency_concurrency_admission.py` | Same shielded release as execution admission | W2 qualified |
| `local_recovery_admission.py` | Shielded release on cancel during lifecycle | W3-C qualified |

### `runtime/nexus`

| Location | Mechanism | Gap / status |
|----------|-----------|--------------|
| `graph_executor.py` | `CancellationCoordinator.is_requested` between batches; mark pending skipped | Fan-out batches cooperative |
| `graph_runner.py` | Retry eligibility passes `cancelled=` from task metadata | No backoff sleep in runner |
| `tools/invoker.py` | Tool timeout → detach; retry backoff → **W4** `cooperative_delay_seconds` + metadata check | |
| `policies/policy_enforcer.py` | `wait_for` + retry loop; **W4** explicit `CancelledError` propagate | `asyncio.sleep` cancellable |

### `llm_adapters`

| Location | Mechanism | W4 change |
|----------|-----------|-----------|
| `_shared/resilience.py` | Retry loop with budget | `should_abort` + cooperative delay |
| `_shared/retry.py` | `call_with_retry` | cooperative delay between attempts |
| Provider adapters | Streaming via `_execute_streaming` | Close on generator abandon (adapter-owned); no central stream manager |

### `long_running`

| Location | Mechanism |
|----------|-----------|
| `scheduler.py` | Loop task `cancel()` on shutdown; `CancelledError` swallowed on join |
| `scheduled_resume.py` | Cooperative checks at resume boundaries (call-site dependent) |

### `decision_recovery` / `fan_out_partial_recovery`

| Path | Cancel behavior |
|------|-----------------|
| `resume_decision_from_durable_state_with_recovery_admission` | Permit released in `finally` even if `to_thread` cancelled |
| Partial fan-out recovery | Admission per recovery wave; `finally` release |

### Legacy naming note

`CancellationCoordinator` is a **metadata helper** (Phase G.8), not a W4-forbidden central manager — no scheduling, no worker registry.

## Risk register (pre-W4 → post-W4)

| Risk | Pre | Post-W4 |
|------|-----|---------|
| Tool retry backoff after cancel | `time.sleep` ignored cancel | Cooperative delay + metadata |
| Provider retry backoff after cancel | `time.sleep` | `should_abort` hook |
| Policy retry misclassified cancel | Propagated implicitly | Explicit `CancelledError` re-raise |
| Detached tool worker permit | Qualified W2-B2 | Unchanged (by design) |
| Graph orphan asyncio tasks | Worker cancel on parent cancel in concurrent work | Qualified + 100-child test |

## Test matrix (W4-A)

| Case | Test anchor |
|------|-------------|
| Parent execution cancel → capacity | `test_enterprise_scale_resilience_w4_a_cancellation_qualification.py` |
| 100-child fan-out cancel | same |
| Provider retry cancel during backoff | same |
| Policy enforcer cancel during backoff | same |
| Recovery admission cancel | `test_decision_durable_recovery_w3_c4.py` (W3-C4, reused) |
| Dependency detach permit | `test_dependency_attempt_execution_boundary.py` (W2-B2) |
| Execution admission cancel | `test_enterprise_scale_resilience_w1_a_root_capacity_admission.py` (W1) |

## Contract decision (ETAP 7)

**No new** `intergrax/contracts/` cancellation module — existing `ExecutionRetryEligibilityRequest.cancelled`, `CancellationCoordinator` keys, and asyncio semantics suffice. Cooperative blocking wait lives in `runtime/cancellation/coordinator.py` (`cooperative_delay_seconds`, `CooperativeCancellationAbort`). Package `runtime/cancellation/__init__.py` exports coordinator symbols only (no eager `resume_admission` import) so `llm_adapters` retry paths do not create an import cycle through `execution` → `llm_adapters`.
