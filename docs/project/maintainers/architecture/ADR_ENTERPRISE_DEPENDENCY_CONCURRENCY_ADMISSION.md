# ADR-ENTERPRISE-DEPENDENCY-CONCURRENCY-ADMISSION: Dependency Concurrency Admission

| Field | Value |
|-------|-------|
| **Status** | **Accepted** (contract + architecture; production wiring deferred to W2-B) |
| **Date** | 2026-09-11 |
| **Baseline** | W2-A inventory `f19e55dc9b156779442b69c35a47f43a58d46046` (ancestor); contract on current `development` |
| **Related** | [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md`](ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md) · [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W2_DEPENDENCY_ISOLATION_INVENTORY.md`](../qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W2_DEPENDENCY_ISOLATION_INVENTORY.md) · W1-A `ExecutionCapacityAdmissionPort` |
| **Contract module** | `intergrax/contracts/dependency_concurrency_admission.py` |

---

## Context

W2-A qualified that external dependency calls (tools, LLM providers, integrations, retrievers) share process-local resources without per-dependency **in-flight concurrency** caps. Root execution admission (W1-A), graph concurrent work width (W1-B), circuit breakers, and rate limiters address orthogonal concerns. Slow or storming traffic against one dependency can still saturate a shared `ThreadPoolExecutor` queue or unbounded provider fan-out.

Enterprise requirement: **acquire dependency slot → external call → release dependency slot**, keyed by typed failure domain, without embedding retry, rate limits, breakers, or tenant fairness in the same mechanism.

---

## Decision

Introduce a single shared contract surface:

| Symbol | Role |
|--------|------|
| `DependencyConcurrencyAdmissionPort` | Async pluginable admission |
| `DependencyConcurrencyAdmissionRequest` | Capacity metadata only |
| `DependencyConcurrencyIdentity` + `DependencyConcurrencyKind` | Typed failure-domain key |
| `DependencyConcurrencyPolicy` | Explicit limits (no platform magic defaults) |
| `DependencyConcurrencyPermit` | One slot; `release()` exactly once (idempotent impl allowed) |

**Lifecycle invariant:** one successful `acquire` corresponds to exactly one in-flight external dependency attempt; permit must not span retry backoff, multi-attempt retry loops, or provider failover chains.

**Configuration ownership:** composition/application maps `DependencyConcurrencyIdentity → DependencyConcurrencyPolicy`. Callers receive an injected port; runtime slot state lives in the port implementation (not in `RuntimeToolInvoker` / `LLMAdapter` as hidden defaults).

**Unknown policy (when W2 admission is enabled):**

| State | Behavior |
|-------|----------|
| No admission port injected | Legacy / feature off — existing runtime unchanged |
| Port injected, policy missing for identity | **Fail-closed** — `DependencyConcurrencyPolicyMissingError` (no silent unlimited fallback) |

Policy lookup is logically separate from slot accounting; implementations may compose both internally but must not expose a God registry in W2-ADR.

**Process-local first:** W2-B `LocalDependencyConcurrencyAdmission` (preferred location: `intergrax/runtime/resilience/`) enforces caps per process. Cluster capacity remains `N × local limit` until a future distributed coordinator implements the same port. Multi-host qualification stays in S/R4.

**Pluginability:** callers depend on `DependencyConcurrencyAdmissionPort`, not a concrete local class, so Redis/DB-backed coordinators can replace local enforcement without changing invocation seams.

---

## Contract

Public API: `intergrax/contracts/dependency_concurrency_admission.py`.

Overload modes: **`REJECT`**, **`WAIT_WITH_TIMEOUT`** only (no `WAIT_FOREVER`).

Policy validation mirrors W1-A: `WAIT_WITH_TIMEOUT` requires `wait_timeout_seconds > 0`; `REJECT` forbids a timeout field.

Typed errors:

- `DependencyConcurrencyExceededError` — saturated under `REJECT`
- `DependencyConcurrencyAdmissionTimeoutError` — wait deadline exceeded
- `DependencyConcurrencyPolicyMissingError` — enabled admission without configured policy

---

## Identity

```python
DependencyConcurrencyIdentity(kind: DependencyConcurrencyKind, value: str)
```

| Kind | Canonical `value` source (reuse existing IDs) |
|------|-----------------------------------------------|
| `TOOL` | `ToolContract.tool_id` |
| `LLM_PROVIDER` | Canonical provider slug (`LLMAdapter._provider_slug()` / adapter registry slug) |
| `INTEGRATION` | Integration catalog slug |
| `RETRIEVER` | Canonical retriever/backend identity |

**Not** a single arbitrary string key (`"tool:foo"`). Kind is part of equality and hashing.

**LLM model:** default failure domain is **provider slug only**. Model identifiers may appear later as observability metadata; they must not partition concurrency slots unless a future ADR documents infra separated per model.

**Tenant:** `tenant_id` on the request is metadata only — it must **not** partition dependency slots. Tenant fairness remains a separate axis.

No new `ToolId` / `ProviderId` types for W2.

---

## Lifecycle

```text
permit = await port.acquire(request)
try:
    # exactly one external dependency attempt
    ...
finally:
    await permit.release()
```

Future implementation requirements (W2-B qualification):

- Cancel during wait → no slot consumed; cancellation propagates
- Cancel during in-flight attempt → caller `finally` releases permit
- Cancel during `release()` → slot returned; no double-decrement (W1-A lesson)
- Second `release()` → safe no-op
- `REJECT` with capacity 1 and 32 concurrent acquirers → exactly one permit, 31 rejects
- `WAIT_WITH_TIMEOUT` → acquire when slot frees before deadline else timeout error

**Fairness:** port does not guarantee tenant fairness or strict FIFO unless a future implementation documents otherwise.

---

## Overload semantics

Same family as W1-A root admission: explicit `max_concurrent_calls`, overload mode, optional wait timeout. No embedded rate tokens or breaker state.

---

## Ownership

| Concern | Owner |
|---------|--------|
| Contract | `intergrax/contracts/` |
| Policy values | Application/composition configuration |
| Slot state | Port implementation (W2-B local, later distributed) |
| Tool wiring seam | `RuntimeToolInvoker` external attempt (W2-B) |
| Provider wiring seam | `LLMAdapter._execute` / provider attempt (W2-B) |
| Integration wiring | Deferred until canonical runtime integration invocation seam exists |
| Retriever wiring | Contract ready; RAG path deferred if W2-B prioritizes tool/provider |

Port does **not** own: retry, HTTP/SDK, circuit breaker, rate limiting, logging business payload, tenant scheduling, root admission.

---

## Tool boundary (W2-B)

Canonical seam from W2-A:

```text
_execute_external_effect → _execute_once → executor.submit(ToolExecutor.execute)
```

**Admission must occur before enqueueing** the external tool attempt into the shared `ThreadPoolExecutor`. Acquiring inside the pool worker after queueing cannot protect against unbounded queue growth.

Identity: `TOOL` + canonical `tool_id`.

Do not extend `ToolContract` with `max_concurrent_calls`; capacity is operational policy external to tool semantics.

---

## LLM provider boundary (W2-B)

Canonical path: `LLMAdapter._execute` → `execute_with_resilience` → provider attempt.

**Acquire before provider SDK/network work starts.**

Failover: separate permit per provider — release A before acquire B.

Identity: `LLM_PROVIDER` + provider slug (not model by default).

---

## Retry interaction

Permit lifetime == one in-flight dependency attempt, not the retry loop.

```text
attempt: acquire → call → release
backoff (no permit held)
attempt: acquire → call → release
```

Port must not receive attempt number, backoff, or `Retry-After`.

---

## Circuit breaker interaction

`IntegrationCircuitBreaker` (and LLM parallel breaker state) expresses **health / fail-fast**, not concurrent in-flight caps. Concerns stay separate.

Future W2-B ordering at each seam must follow the existing resilience path for that domain; this ADR does not freeze global ordering without seam analysis. Both mechanisms remain composable and independent.

---

## Rate-limit interaction

Dependency concurrency limits **how many calls are in flight now**. Rate limiting limits **throughput over time** (W2-C / existing LLM RPM). No `calls_per_minute` on `DependencyConcurrencyPolicy`.

---

## Process-local semantics

First implementation is process-local. `N` worker processes × `M` local slots ≠ global `M`.

---

## Future distributed implementation

Async port enables Redis/DB/other coordinators without caller changes. Not required for W2-B local correctness. S/R4 remains the multi-host qualification track.

---

## Rejected alternatives

| Alt | Reason |
|-----|--------|
| A — Reuse `ExecutionCapacityAdmissionPort` | Root executions ≠ dependency attempts |
| B — Reuse `ConcurrentExecutionWorkPolicy` | Per-operation worker width ≠ shared dependency isolation |
| C — Reuse `IntegrationCircuitBreaker` | Health state ≠ concurrency capacity |
| D — One semaphore in `RuntimeToolInvoker` | Global tool pool limit; does not isolate tool A from tool B |
| E — `dependency_key: str` only | No typed failure-domain semantics |
| F — `GlobalBulkheadManager` | God object; mixed ownership |

Also rejected: `BulkheadManager`, `DependencyManager`, `ConcurrencyManager`, `ResourceScheduler`, `IsolationEngine`, `WAIT_FOREVER`, magic default capacities (8/16/32/…), implicit unlimited policy when W2 is enabled.

---

## Consequences

- **Positive:** One modular, hard-typed bulkhead seam across tools, providers, integrations, retrievers; clear separation from W1/R1/resilience stacks.
- **Negative:** Composition must supply explicit policies per dependency; missing policy fails closed when enabled.
- **Neutral:** No runtime behavior change until W2-B wiring and local implementation land.

---

## Rollout / W2-B

| Phase | Status after this ADR |
|-------|------------------------|
| W2-A | CLOSED |
| W2-ADR | CLOSED (Accepted) |
| W2-B1 | **DONE** — `LocalDependencyConcurrencyAdmission` in `intergrax/runtime/resilience/local_dependency_concurrency_admission.py` (process-local, event-loop-local asyncio state; explicit policy snapshot; no runtime wiring) |
| W2-B2 | **BLOCKED — ARCHITECTURAL DECISION REQUIRED** — qualification: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W2_B2_TOOL_ADMISSION_BOUNDARY_QUALIFICATION.md`](../qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W2_B2_TOOL_ADMISSION_BOUNDARY_QUALIFICATION.md) (sync `_execute_once` + async port + caller timeout vs worker lifetime; no canonical bridge) |
| W2-B3 | OPEN — provider runtime wiring after W2-B2 seam decision |
| W2 complete | Not claimed |

Do not implement production wiring to `RuntimeToolInvoker`, `LLMAdapter`, or integration invoke paths in W2-ADR.

Regression gate: `tests/unit/contracts/test_dependency_concurrency_admission_contract.py`.
