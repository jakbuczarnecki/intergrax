# Execution Engine — Graceful Shutdown, Drain & Termination Model

**Task:** EE-B4-B  
**Contract:** `intergrax/contracts/execution_reliability/shutdown_contract.py` (EE-B1.1)

## 1. Canonical shutdown phases (frozen order)

```text
STOP_ACCEPTING_NEW_WORK
        ↓
DRAIN_ACTIVE_EXECUTIONS
        ↓
FLUSH_REQUIRED_EVIDENCE
        ↓
PERSIST_FINAL_STATE
        ↓
TERMINATE_WORKERS
```

No second lifecycle owner. Host orchestration (`intergrax/hosting/shutdown.py`) maps application ports to bounded phases; execution semantics remain contract-owned.

## 2. Admission semantics

| Class | After STOP_ACCEPTING |
| ----- | -------------------- |
| New root execution | **Rejected** — not queued, not silently started |
| Child / continuation of already-admitted parent | **Allowed during drain** per EE-B1.2 (no second root capacity slot) |

Capacity port (`LocalExecutionCapacityAdmission`) does not bypass host stop-intake; composition must not schedule new roots once stop boundary is reached.

## 3. Active work & drain

Drain allows in-flight admitted work to reach a safe terminal or hand-off state. Bounded drain reuses host `ShutdownPolicy` drain/cancel timeouts (`run_bounded_phase`). Execution-runtime-only paths without host policy document bounded drain via dependency attempt boundary (`begin_shutdown` / `drain_and_close`) and W5-A bounded event sink sentinel drain.

Drain is **not** unbounded wait forever.

## 4. Stuck worker

On drain timeout, hosting strategies may cancel (`DRAIN_THEN_CANCEL`) or mark timed out (`WAIT_UNTIL_COMPLETE`). Dependency attempt boundary cancels pending acquires on `begin_shutdown` and escalates to close with release invariants.

## 5. Cancellation during drain

Host/worker cancellation propagates `CancelledError`; capacity permits release on all terminal paths (EE-B1.2). No hidden retry requiring new root admission after stop boundary.

## 6. Mandatory evidence vs observability

| Plane | Shutdown behavior |
| ----- | ----------------- |
| Mandatory evidence / runtime event persistence | Must flush or **fail closed** — no clean success |
| OTLP / best-effort export | Failure is secondary; must not block canonical final state persistence |

## 7. Final state persistence

`PERSIST_FINAL_STATE` runs after required evidence flush semantics and before `TERMINATE_WORKERS`. Failure → no full shutdown success report.

## 8. Worker termination

After `TERMINATE_WORKERS`, managed workers for the execution scope must be zero. Nexus tool invoker drains dependency attempt boundary and shuts down execution pool on close — no separate Nexus shutdown runtime.

## 9. Resource ownership

| Resource | Acquired by | Released by | Failure-safe | Idempotent |
| -------- | ----------- | ----------- | ------------ | ---------- |
| Root capacity permit | `ExecutionCapacityAdmissionPort.acquire` | `ExecutionCapacityPermit.release` | finally / shutdown cancel | yes (EE-B1.2) |
| Dependency attempt permit | `DependencyAttemptExecutionBoundary` | `complete` / drain close | shutdown race logged | release claimed once |
| Managed worker task | execution host / pool | drain complete or cancel | cancel on timeout | dispose once |
| Mandatory evidence buffer | runtime event bus | flush phase | fail closed | N/A |
| Final state store | shutdown orchestration | persist phase | typed failure | idempotent shutdown only |
| Best-effort exporter | observability wiring | close after persist path | secondary | close idempotent |

## 10. Deadlock model (analysis)

| Cycle | Mitigation |
| ----- | ---------- |
| shutdown waits for worker; worker waits for new admission | Stop intake before drain; reject new roots |
| worker waits on queue; queue waits on producer blocked by admission | Bounded queue sentinel (W5-A); stop intake |
| persistence flush waits on worker; worker waits on persistence | Mandatory path fail-closed; worker terminal independent of OTLP |
| nested root capacity | Child runner does not acquire root slot (EE-B1.2) |

Lock order: hosting shutdown phases use single budget; dependency boundary uses `_registry_lock` before `_close_lock` on drain.

## 11. Compound failure

Primary failure (worker / mandatory evidence) must not be masked by secondary (exporter close). Terminal authority: typed shutdown outcome, not optimistic success.

## 12. Readiness / liveness (EE-B4-A)

| Phase | Readiness | Liveness |
| ----- | --------- | -------- |
| STOP_ACCEPTING | not ready | live |
| DRAIN | not ready | live |
| FLUSH / PERSIST | not ready | live |
| TERMINATE_WORKERS | not ready | not live |

## 13. Cross-session exclusions (NPSC-5F)

Do not modify `causal_evidence.py`, `causal_evidence_export.py`, `export_boundary.py`, or `background_execution/**` in shutdown certification. Findings → handoff only.

## 14. Certification reference (non-authority)

`testing_support/shutdown/` composes capacity, drain, evidence flush, and termination for deterministic EE-B4-B gates — **not** a production runtime.
