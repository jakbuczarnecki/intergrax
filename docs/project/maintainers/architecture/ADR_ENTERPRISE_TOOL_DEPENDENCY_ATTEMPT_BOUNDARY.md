# ADR-ENTERPRISE-TOOL-DEPENDENCY-ATTEMPT-BOUNDARY: Tool dependency attempt execution boundary (W2-B2)

| Field | Value |
|-------|-------|
| **Status** | **Accepted** |
| **Date** | 2026-09-11 |
| **Baseline HEAD** | `d87f7e4c6802ef9901bfecf1e7360e684d7f0b91` (= `origin/development` at decision) |
| **Qualification** | [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W2_B2_TOOL_ADMISSION_BOUNDARY_QUALIFICATION.md`](../qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W2_B2_TOOL_ADMISSION_BOUNDARY_QUALIFICATION.md) (`505eb6af468994d3d42a7addb2d2c60afe189aeb` evidence lineage) |
| **Related** | [`ADR_ENTERPRISE_DEPENDENCY_CONCURRENCY_ADMISSION.md`](ADR_ENTERPRISE_DEPENDENCY_CONCURRENCY_ADMISSION.md) · W2-B1 `LocalDependencyConcurrencyAdmission` |

---

## Context

W2-ADR introduced `DependencyConcurrencyAdmissionPort` (async) and the invariant **one permit == one in-flight physical dependency attempt**. W2-B1 delivered process-local `LocalDependencyConcurrencyAdmission` (event-loop-local asyncio state).

W2-B2 must wire admission into the canonical Nexus tool physical seam:

```text
_execute_once → ThreadPoolExecutor.submit(ToolExecutor.execute) → future.result(timeout=...)
```

without violating:

- acquire **before** submit (wait for capacity must not occupy a pool worker),
- permit held for **physical worker lifetime** (not caller wait lifetime),
- one permit per **physical attempt** (not per `invoke` retry loop),
- replay / claim-conflict paths must not consume capacity.

Qualification closed the **boundary** analysis; runtime wiring remained **BLOCKED** on sync/async lifecycle ownership.

**Production runtime behavior in this ADR:** unchanged (decision + docs only).

---

## Problem

Three seams collide:

| Seam | Nature |
|------|--------|
| `DependencyConcurrencyAdmissionPort` | async `acquire` / `release` |
| `RuntimeToolInvoker._execute_once` | sync; blocks on `future.result(timeout=...)` |
| Pool worker | may outlive caller after `FuturesTimeoutError` |

Forbidden pattern:

```text
acquire → submit → result(timeout) → finally: release   # releases while worker still runs
```

`future.add_done_callback` alone is insufficient: callback is sync; `release()` is async and requires a loop owner.

Callers are **not** uniformly async: parallel `tool_loop` invokes `invoker.invoke` from `ThreadPoolExecutor` worker threads. There is **no** existing production bridge for per-attempt async admission at this seam.

---

## Constraints

- Keep **one** public admission contract: `DependencyConcurrencyAdmissionPort` (unchanged).
- Do **not** wrap whole `RuntimeToolInvoker.invoke` or the full retry loop in one permit.
- No per-attempt `asyncio.run` / per-attempt event loop.
- No fire-and-forget `asyncio.create_task(release())` without an owner.
- `RuntimeToolInvoker` must **not** embed loop/thread knowledge (`run_coroutine_threadsafe`, which loop, etc.).
- No new `*Manager` / `*Scheduler` / public registries.
- Hard types only in proposed surfaces (no `Any`, no `type: ignore` in new API).
- Rate limiting (W2-C), hard cancellation (W4), admission telemetry backpressure (W5) are out of scope.

---

## Caller topology

Production paths that reach `RuntimeToolInvoker.invoke` (minimum audit):

| Call path | Caller thread | Async owner at seam? | Can `await` admission at seam? |
|-----------|---------------|--------------------|--------------------------------|
| `RuntimeToolGateway.invoke` → `invoke_catalog_tool_request` → `invoker.invoke` | **asyncio event-loop thread** (sync `invoke` blocks the loop) | Gateway is async, but catalog dispatch is **sync** | **No** (must not require gateway refactor for W2-B2) |
| `tool_loop` sequential `execute_planned_tool_calls` (`max_parallel_read_only <= 1`) | **Caller thread** (often event-loop thread during async planner paths) | Optional upstream async; **none** at `invoke` | **No** |
| `tool_loop` parallel read-only batch | **`ThreadPoolExecutor` worker threads** (`copy_context` + `_invoke_planned_call`) | **No** | **No** |
| `catalog_dispatch.invoke_catalog_tool_request` / `invoke_catalog_tool_ids` | **Sync** on whoever invoked Nexus step | **No** | **No** |
| `deterministic_chain.DeterministicChainPattern.execute` | **Sync** pattern executor | **No** | **No** |
| `catalog_context.invoke_catalog_context_tool` | **Sync** on caller | **No** | **No** |
| `catalog_declarative_invoker` (agents persistence) | **Sync** | **No** | **No** |

**Conclusion:** W2-B2 must support **thread-safe sync-facing** admission coordination for **all** production callers. Option A (async-only upstream owner) is insufficient without rewriting parallel tool batch and every sync catalog entry point.

---

## Physical attempt lifecycle (canonical)

Scope: **one** call to `_execute_once` (one pool submission), not `invoke` and not the contract retry loop.

```text
prepare / governance
→ idempotency resolution (when required)
→ dependency admission acquire (when port enabled)
→ executor.submit(ToolExecutor.execute)
→ boundary.may_have_started = True (only after successful submit)
→ caller waits future.result(timeout) OR times out
→ worker runs until ToolExecutor.execute returns (independent of caller timeout)
→ attached completion: worker terminal → permit.release() on admission loop → release completes → caller receives physical result/exception
→ detached completion (caller timed out earlier): worker terminal → boundary tracked release → outstanding entry cleared
```

**Attached success path (frozen):**

```text
acquire → submit → worker returns success → permit.release() → release completes → return worker result
```

**Attached exception path (frozen):**

```text
acquire → submit → worker raises → permit.release() → release completes → propagate/map worker exception
```

**Not allowed on attached path:**

```text
worker result → schedule release → return immediately
```

**Retry loop** (`_execute_with_policy`):

```text
attempt N: acquire → submit → worker terminal → permit fully released
backoff: no permit held
attempt N+1: acquire → submit → ...
```

**Forbidden retry ordering:**

```text
attempt N worker terminal → release still pending → backoff / attempt N+1 acquire
```

---

## Options

### Option A — Async-owned physical attempt boundary upstream

Move acquire/submit/release orchestration into an async coordinator; only async Nexus paths call it.

| Criterion | Assessment |
|-----------|------------|
| Correctness | Good **if** all callers use async coordinator |
| Sync caller support | **Poor** — parallel `tool_loop`, `catalog_dispatch`, `deterministic_chain`, `catalog_context` bypass |
| Worker-thread callers | **Poor** |
| Timeout correctness | Achievable with async release on worker completion |
| API churn | **High** — broad async refactor of catalog surface |
| Complexity | Medium per path, **high** aggregate |

**Rejected** for W2-B2: does not satisfy all production callers without prohibitive seam churn.

### Option B — Runtime-owned execution-boundary adapter (selected)

One narrow runtime component:

- sync-facing API for `_execute_once`,
- owns async admission lifecycle on a **dedicated admission event loop** (one loop/thread per adapter instance),
- coordinates permit with **existing** `RuntimeToolInvoker` `ThreadPoolExecutor` (executor ownership unchanged),
- tracks worker `Future` completion and performs **tracked** async `release` on the admission loop.

| Criterion | Assessment |
|-----------|------------|
| Correctness | **Good** — matches permit == worker lifetime |
| Sync caller support | **Good** — thread-safe entry |
| Worker-thread callers | **Good** |
| Timeout correctness | **Good** — caller timeout ≠ release |
| Shutdown | Explicit `close()` on boundary owner |
| API churn | **Low** — localized to invoker + composition |
| Pluginability | Port unchanged; adapter swappable |
| Performance | Two cross-thread admissions per attempt (acquire + release scheduling); acceptable vs tool I/O |
| Testability | Public behavioral tests without asyncio leaks |

### Option C — Sync-compatible admission facade / duplicate contract

Second port or sync methods on local implementation only.

| Criterion | Assessment |
|-----------|------------|
| Contract duplication | **High** risk for distributed future |
| Evolution | Forks local vs distributed semantics |
| Thread safety | Must reimplement wait/signal in sync |

**Rejected:** conflicts with single public async port; duplicates W2-ADR pluginability.

---

## Decision

**Select Option B.**

Introduce a runtime-owned, narrowly scoped component:

| Property | Value |
|----------|-------|
| **Name** | `DependencyAttemptExecutionBoundary` |
| **Module (planned)** | `intergrax/runtime/resilience/dependency_attempt_execution_boundary.py` |
| **Owner package** | `intergrax/runtime/resilience` — generic **sync physical attempt + async admission** seam (reusable for W2-B3 provider paths; not tool-specific orchestration) |

`RuntimeToolInvoker` retains:

- `ThreadPoolExecutor` creation, `submit`, and `close()` **orchestration order** (see Shutdown),
- retry policy, idempotency, tracing, error mapping.

`DependencyAttemptExecutionBoundary` owns:

- dependency permit acquire/release lifecycle for one physical attempt,
- knowing when the worker `Future` actually finishes,
- admission event loop/thread lifecycle.

**Public W2 port:** `DependencyConcurrencyAdmissionPort` — **UNCHANGED**.

**No** `SyncDependencyConcurrencyAdmissionPort`.

---

## Ownership

| Asset | Owner |
|-------|--------|
| `DependencyConcurrencyAdmissionPort` contract | `intergrax/contracts/` |
| `LocalDependencyConcurrencyAdmission` | `intergrax/runtime/resilience/` (loop-local) |
| `DependencyAttemptExecutionBoundary` | `intergrax/runtime/resilience/` |
| `ThreadPoolExecutor` (tool physical execution) | `RuntimeToolInvoker` (unchanged) |
| Physical `submit` + `future.result(timeout)` call site | `RuntimeToolInvoker._execute_once` (delegates permit lifecycle to boundary) |
| Canonical `close()` for tool path | **`RuntimeToolInvoker.close()`** closes boundary first, then pool (single operator-facing lifecycle) |

---

## Thread / loop model

No suitable existing runtime loop owns **per-attempt** tool admission across sync + worker-thread callers. W1 `ExecutionRuntime` holds root admission for a different lifetime; vendor `run_coroutine_threadsafe` usages are application-scoped, not Nexus tool seams.

**Model (Option B):**

| Question | Answer |
|----------|--------|
| Who creates the loop? | `DependencyAttemptExecutionBoundary` at construction |
| Who owns the thread? | Same boundary instance (one dedicated background thread) |
| When does it start? | Boundary construction (or explicit `start()` called from composition before first attempt — **one** startup per boundary) |
| When does it stop? | `DependencyAttemptExecutionBoundary.close()` |
| `run_coroutine_threadsafe` | **Only inside** `DependencyAttemptExecutionBoundary` (not in callers, not in `RuntimeToolInvoker`) |
| One loop per | **One boundary instance** (typically one per configured tool invoker), **not** per request/tool/dependency |
| Per-call loop | **Forbidden** |

**Local admission loop affinity:** The `DependencyConcurrencyAdmissionPort` instance used for tool bulkhead (typically `LocalDependencyConcurrencyAdmission`) must be used **exclusively** on the boundary’s dedicated loop. Composition must **not** share that instance with the main graph/gateway asyncio loop.

**Serialization:** The admission loop may serialize short asyncio state transitions; it must **not** serialize physical tool work (pool workers remain concurrent).

---

## Public / internal API shape (planned, W2-B2 implementation)

Not implemented in this ADR. Intended shape:

- **Injected** optional `DependencyAttemptExecutionBoundary | None` on `RuntimeToolInvoker` (or equivalent composition wiring).
- **Sync** entry used only from `_execute_once`, semantically:

  1. block until acquire completes or admission error (cross-thread schedule onto admission loop),
  2. `submit` worker to invoker pool; on submit failure → release permit, `may_have_started = False`,
  3. on successful submit → `may_have_started = True`,
  4. `future.result(timeout)` for caller; on `FuturesTimeoutError` while worker not terminal → return timeout to invoker **without** releasing permit and **without** waiting for worker or release (detached lifecycle),
  5. when `future.result` returns with worker terminal (**attached** path): run tracked `permit.release()` on admission loop and **block until that release operation completes**, then return physical result or propagate/map worker exception,
  6. when worker becomes terminal after step 4 returned timeout (**detached** path): exactly one boundary-owned tracked `permit.release()`; no caller waiting.

- **Handle** (internal): ties **one acquired permit** to **one worker Future** and owns **one release ownership transition** per physical attempt — not a generic task handle; not exposed as `asyncio.Task`/`Future` to Nexus callers.

`add_done_callback` on the worker `Future` (if used) must not unconditionally release: it cooperates with the attempt handle so **attached** completion performs release in the sync return path and **detached** completion performs release once via the boundary owner — never two independent release attempts for the same handle.

---

## Acquire lifecycle

- Runs on admission loop via boundary scheduling.
- Occurs **after** idempotency resolution for the invoke (see Idempotency ordering).
- **Not** run for `REPLAY_COMPLETED` or claim paths that end before external effect.
- On `DependencyConcurrencyExceededError`, `DependencyConcurrencyAdmissionTimeoutError`, or `DependencyConcurrencyPolicyMissingError`: **no** `submit` (physical attempt not started).

Admission wait occurs **before** `submit` → **does not** occupy tool `ThreadPoolExecutor` workers.

---

## Submit lifecycle

- Only after successful acquire.
- `boundary.may_have_started = True` **only** after successful `executor.submit(...)`.
- If `submit` raises: release permit immediately (tracked), `may_have_started = False`.

---

## Timeout lifecycle

```text
caller timeout ≠ worker completion
```

If `future.result(timeout)` raises `FuturesTimeoutError` while `future.done() is False`:

- permit **remains held**,
- invoker returns `ToolExecutionResult` timeout to caller (existing behavior),
- **post-timeout release owner:** `DependencyAttemptExecutionBoundary` when worker Future completes.

---

## Completion lifecycle

**Canonical physical completion owner:** `DependencyAttemptExecutionBoundary` (same owner for attached and detached paths).

**Core invariant (attached / normal `_execute_once` return):** physical attempt completion exposed to the synchronous caller requires **both** worker `Future` terminal state **and** permit `release` terminal state. The invoker/retry loop must not observe success, mapped tool failure, or propagated exception until release for that attempt has fully completed.

### Attached completion

Caller still blocked on `future.result(timeout)`; worker reaches terminal state before or when result is delivered:

```text
worker terminal → permit.release() on admission loop → await release completion → return to RuntimeToolInvoker / caller
```

If `permit.release()` fails on this path: boundary **fail-closed** — surface invariant failure in the current lifecycle (do **not** return worker success while masking release failure; do **not** defer solely to application shutdown).

### Detached post-timeout completion

Caller already returned `TIMEOUT`; worker may still be running:

```text
worker eventually terminal → boundary performs tracked permit.release() → release completion tracked → outstanding entry removed
```

Caller does **not** wait for worker or release. Permit remains held from acquire until this detached release completes.

If `permit.release()` fails on this path: failure remains **tracked**, observable by the lifecycle owner, and must surface during `close()` / drain at minimum — not swallowed.

### Two modes, one release owner

| Mode | Caller | Release trigger | Caller waits for release? |
|------|--------|-----------------|---------------------------|
| Attached | still at `_execute_once` | sync return path after `future.result` | **Yes** |
| Detached | left after timeout | boundary when worker Future completes | **No** |

Exactly **one** release ownership transition per attempt handle. Normal attached completion and a worker `Future` done callback must **not** both independently invoke release; the handle (or equivalent terminal-completion object) records whether release was claimed by the attached path or remains for detached cleanup.

Do **not** rely on permit idempotency alone to mask double-release ownership bugs.

### Outstanding tracking

Track only attempts that still require boundary lifecycle ownership (e.g. caller timed out while `future.done() is False`). Remove entry only after **worker terminal + release terminal**. Do not track attempts that already completed attached release.

### Capacity visibility

After a normal (non-timeout) `_execute_once` return, the previous attempt’s capacity slot must already be free. Example: `capacity=1` — attempt A completes and `_execute_once` returns → immediate next `acquire` succeeds without sleep, yield, or retry polling.

### Shutdown drain

Release completion for **detached** or in-flight attempts is **awaited** during shutdown drain (see Shutdown). Attached-path release failures and drain-time failures surface as errors (not swallowed).

---

## Shutdown lifecycle

**Who calls `close()`?** Operators/composition call `RuntimeToolInvoker.close()` (existing API). That method becomes the **single** external shutdown entry for the tool execution + admission boundary stack.

**Order (frozen):**

1. Boundary: stop accepting new physical attempts (fail fast or raise closed error).
2. Boundary: cancel or fail pending **admission waits** per port semantics (no new permits).
3. **Do not** stop admission loop before permits for still-running pool workers are released.
4. Invoker pool: `ThreadPoolExecutor.shutdown(wait=True)` (existing — waits for workers).
5. Boundary: drain tracked `release()` operations on admission loop.
6. Boundary: stop admission thread/loop.
7. Surface release invariant failures (e.g. double-release bugs) — do not swallow during shutdown.

**`close()` idempotent:** second `close()` is a no-op.

**Caller timed out, worker still running, application closes:**

- Pool shutdown waits for worker (existing semantics).
- Boundary holds permit until worker Future completes, then releases during drain.
- No orphan permit, worker, or untracked release coroutine.

**Worker never returns (hung native tool):** shutdown may block on pool `wait=True` under current `ToolExecutor` semantics — **deferred to W4** (hard cancellation). ADR does not claim full cancellation.

---

## Idempotency ordering

**Frozen order:**

```text
_prepare → governance/authorization
→ idempotency coordination (when required)
→ dependency admission acquire (when port enabled)
→ submit physical attempt
```

| Case | Acquire? |
|------|----------|
| `REPLAY_COMPLETED` | **No** (no external attempt) |
| Claim conflict / terminal before effect | **No** |
| Successful claim, external effect proceeds | **Yes**, after claim |

**Claim vs admission:** **Claim before admission** (preferred). Rationale: only true idempotency owners compete for dependency slots; `WAIT_WITH_TIMEOUT` may extend claim hold, but overload modes exclude `WAIT_FOREVER`, so claim hold remains bounded by policy timeout.

Documented trade-off: capacity slot may be held briefly before a conflicting claim would have aborted — acceptable vs holding claim during unbounded wait.

---

## Error mapping

| Error | Physical attempt started? | Internal tool retry (`_execute_with_policy`)? | `RuntimeErrorCode` | Delivery |
|-------|---------------------------|-----------------------------------------------|--------------------|----------|
| `DependencyConcurrencyExceededError` | **No** | **No** | `DEPENDENCY_ERROR` | `ToolExecutionResult.fail` (controlled overload) |
| `DependencyConcurrencyAdmissionTimeoutError` | **No** | **No** | `DEPENDENCY_ERROR` | `ToolExecutionResult.fail` |
| `DependencyConcurrencyPolicyMissingError` | **No** | **No** | N/A (configuration defect) | **Propagate** typed exception |
| Tool execution timeout (`FuturesTimeoutError`) | **Yes** (submit succeeded) | **No** (current invoker returns immediately) | `TIMEOUT` | `ToolExecutionResult.fail` |
| Tool execution exception | **Yes** | Per contract retry policy | mapped / `TOOL_ERROR` | existing behavior |

Admission failures are **distinct** from tool execution failures — not one retryable `TOOL_ERROR` bucket.

**Retry classification:** Physical retry counter in `_execute_with_policy` increments only when a physical attempt **started** (`submit` succeeded). Admission failure before submit is **not** a physical attempt and must **not** trigger immediate internal tight-loop retry (no retry amplification / bulkhead storm). Saturation rejection propagates to caller / higher retry owner.

**Admission wait timeout ≠ tool execution timeout** — separate diagnostics and policy knobs.

---

## Retry interaction

Frozen in §Physical attempt lifecycle. Port does not receive attempt numbers or backoff.

Physical retry in `_execute_with_policy` may start attempt N+1 only after attempt N’s permit is **fully released**. Retryable tool errors must not cause immediate re-acquire while attempt N’s release is still pending (avoids false saturation under `capacity=1`).

---

## Failure semantics

- **Saturation:** controlled overload; external effect `NOT_STARTED`.
- **Admission wait timeout:** capacity wait expired; `NOT_STARTED`.
- **Missing policy:** configuration defect; fail closed; typed propagation.

`may_have_started` after acquire alone: **NO**. After successful submit: **YES**.

---

## Performance implications

- One cross-thread schedule for acquire; one for release completion per physical attempt.
- Admission loop does not run tool code.
- High-frequency tool calls pay bounded asyncio overhead vs network/tool latency.

---

## Future provider compatibility (W2-B3)

`DependencyAttemptExecutionBoundary` is intentionally **not** tool-named: same pattern applies to **sync physical provider attempt + async admission** if the provider seam matches pool/future completion shape. No extra abstraction in W2-B2; reuse the boundary module from provider wiring ADR.

---

## Rejected alternatives

- Option A (async-only upstream owner) — caller topology.
- Option C (sync port duplicate) — contract fork.
- Permit spanning full `invoke` / retry loop.
- `finally: release` around `future.result(timeout)`.
- Moving `ThreadPoolExecutor` into the boundary component.
- Per-call event loops; `asyncio.run` per attempt.
- Public `AttemptManager` / registries.
- Names: `AsyncBridgeManager`, `ToolBulkheadManager`, etc.

---

## W2-B2 implementation plan

| Step | Action |
|------|--------|
| 1 | Implement `DependencyAttemptExecutionBoundary` per this ADR |
| 2 | Wire optional boundary + dedicated `LocalDependencyConcurrencyAdmission` in composition |
| 3 | Delegate `_execute_once` permit lifecycle; map admission errors |
| 4 | Extend `RuntimeToolInvoker.close()` ordering |
| 5 | Behavioral qualification suite (below) |

**W2-B2 architecture:** **CLOSED** (this ADR).

**W2-B2 implementation:** **OPEN**.

---

## Qualification matrix (decision)

| Requirement | Option B |
|-------------|----------|
| Acquire before submit | Yes |
| Permit == worker lifetime | Yes |
| Caller timeout retains permit | Yes |
| Sync + worker-thread callers | Yes |
| Single async port | Yes |
| No invoker loop knowledge | Yes |
| Explicit shutdown | Yes |
| One permit per physical attempt | Yes |
| Replay / conflict skip acquire | Yes |
| Attached return after worker **and** release complete | Yes |
| Normal return before release complete | **No** |
| Retry re-acquire before prior release complete | **No** |
| Exactly-once release ownership per attempt | Yes |
| Attached release invariant failure surfaced synchronously | Yes |
| Detached release failure tracked; surfaced on drain/close | Yes |

---

## Qualification test plan (future implementation; public behavior only)

- admission before submit
- Tool A isolation / shared capacity bound
- retry reacquires; no permit during backoff; after retryable failure on attempt #1, attempt #1 release fully completes before backoff and attempt #2 `acquire` succeeds (`capacity=1`, no false saturation)
- `capacity=1`: attempt completes, `_execute_once` returns, immediate next `acquire` succeeds (capacity visible)
- caller timeout retains permit; worker completion releases (detached)
- attached completion + release invariant failure → surfaced synchronously to caller (no private-state assertions)
- detached post-timeout completion + release failure → retained and surfaced by `close()` / drain
- submit failure releases permit; `may_have_started` false
- admission rejection never reaches executor
- policy missing fails closed (propagate)
- replay path does not acquire
- claim conflict does not acquire
- shutdown drains tracked releases; `close` idempotent
- parallel `tool_loop` thread-safe
- no orphan async tasks after sustained load

---

## W4 / W5 / W2-C boundaries

| Topic | Deferral |
|-------|----------|
| Hard kill of stuck physical tools | **W4 — Cancellation hardening** |
| Admission telemetry / outstanding timed-out worker metrics | **W5** |
| Rate tokens / retry storm containment | **W2-C** |

---

## Consequences

- **Positive:** Unblocks W2-B2 implementation with correct bulkhead semantics under caller timeout and parallel tool batch.
- **Negative:** Dedicated admission thread + composition discipline for loop-local admission instance.
- **Neutral:** No production behavior change until implementation lands.
