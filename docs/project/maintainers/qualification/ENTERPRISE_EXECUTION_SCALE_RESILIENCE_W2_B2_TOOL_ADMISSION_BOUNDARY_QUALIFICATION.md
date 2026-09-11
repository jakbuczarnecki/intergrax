# W2-B2 — Tool dependency bulkhead wiring & sync-async boundary qualification

| Field | Value |
|-------|-------|
| **Status** | **BLOCKED — ARCHITECTURAL DECISION REQUIRED** |
| **Date** | 2026-09-11 |
| **Baseline HEAD** | `6f79f870808d5e44ce1dc9e11775c6e0fec4b80b` (= `origin/development` at qualification start) |
| **Related** | [`ADR_ENTERPRISE_DEPENDENCY_CONCURRENCY_ADMISSION.md`](../architecture/ADR_ENTERPRISE_DEPENDENCY_CONCURRENCY_ADMISSION.md) · W2-B1 `LocalDependencyConcurrencyAdmission` |

---

## Outcome

Production wiring of `DependencyConcurrencyAdmissionPort` into `RuntimeToolInvoker` **was not implemented**. No ad-hoc sync↔async bridge exists in the Nexus tool path; implementing permit lifecycle (especially caller timeout while pool worker continues) inside sync `_execute_once` would require a new architectural component or contract seam decision.

**Production code changed:** NO.

---

## Runtime tool invoker execution model (evidence)

| Unit | Sync / async | Module |
|------|----------------|--------|
| `RuntimeToolInvoker.invoke` | **sync** | `intergrax/runtime/nexus/tools/invoker.py` |
| `_execute_external_effect` | **sync** | same |
| `_execute_with_policy` (retry loop) | **sync** | same |
| `_execute_once` | **sync** | same |
| `ToolExecutor.execute` | **sync** | `intergrax/tools/tool_executor.py` |

Physical attempt boundary:

```text
_execute_once → self._execution_pool.submit(self._executor.execute, request)
→ boundary.may_have_started = True (after submit)
→ return future.result(timeout=timeout_s)
```

On `FuturesTimeoutError`, the invoker returns `ToolExecutionResult.fail(TIMEOUT, …)` while the pool worker **may still be running** (`future.result` timeout does not cancel the worker).

---

## Caller thread context (not single-threaded)

| Call path | Thread |
|-----------|--------|
| `RuntimeToolGateway.invoke` (async) → `invoke_catalog_tool_request` → `invoker.invoke` | **asyncio event-loop thread** (sync invoke blocks loop) |
| `tool_loop._invoke_planned_call` → `invoker.invoke` | **caller thread** (often event-loop thread for sequential batch) |
| `tool_loop.execute_planned_tool_calls` parallel read-only | **dedicated `ThreadPoolExecutor` worker threads** (`copy_context` + `_invoke_planned_call`) |
| `catalog_dispatch`, `deterministic_chain`, `catalog_context` | **sync** on whoever invoked the Nexus step |

There is **no** canonical requirement that all catalog tool calls run on one asyncio loop thread.

---

## NEAREST ASYNC OWNER

For gateway/catalog `ToolRequest` path:

```text
RuntimeToolGateway.invoke / _invoke_inner
  (intergrax/runtime/nexus/tools/tool_gateway.py)
```

Immediately followed by **sync** `invoke_catalog_tool_request` → `RuntimeToolInvoker.invoke` (no `asyncio.to_thread` wrapper).

For bounded tool planner / parallel batch paths, **`RuntimeToolInvoker.invoke` is reached without an async owner** (worker threads or sync step code).

Upstream graph execution: `GraphRunner` / `GraphExecutor.execute` are async, but catalog tool dispatch does not `await` admission before the sync invoker seam.

---

## ASYNC ADMISSION CONTRACT

- `DependencyConcurrencyAdmissionPort.acquire` / `DependencyConcurrencyPermit.release` are **async** (`intergrax/contracts/dependency_concurrency_admission.py`).
- W2-B1 `LocalDependencyConcurrencyAdmission` documents: **event-loop-local asyncio state; not safe across threads or different event loops** (`local_dependency_concurrency_admission.py` module docstring).

Parallel tool invocation already uses extra worker threads → local async admission cannot be shared safely without a thread-safe coordinator or a approved cross-thread bridge.

---

## EXISTING SYNC↔ASYNC BRIDGE (Nexus tool seam)

**NONE** suitable for high-frequency per-attempt admission inside sync `_execute_once`.

Repo-wide patterns (not production-owned for this seam):

- `asyncio.to_thread` — used to offload **sync** work from the loop (inverse of async `acquire` from sync caller).
- `asyncio.run` / `run_coroutine_threadsafe` — tests, scripts, vendor sync workers; not Nexus tool invocation lifecycle.
- W1-A `ExecutionRuntime.execute` holds **`await acquire` on the async runtime owner** for the **whole root execution** (different lifetime than W2 per physical attempt).

No `BlockingPortal`, `anyio.from_thread`, or runtime-owned “sync waiter for async port” component exists for tools.

---

## BLOCKER — permit lifecycle vs caller timeout

ADR invariant: **permit lifetime == physical external dependency attempt lifetime** (worker runs until `ToolExecutor.execute` completes).

Current invoker timeout semantics:

```text
submit → may_have_started True → future.result(timeout) → on timeout, return TIMEOUT to caller
worker may still run
```

Incorrect pattern (forbidden by W2-B2):

```text
acquire → submit → result(timeout) → finally: release   # releases while worker still holds real concurrency
```

Correct pattern requires **release only after pool future completes**, even when the sync caller stopped waiting. That implies tracked completion ownership (e.g. done-handler or background waiter) where **`release()` is async** → another sync↔async boundary unless a new approved adapter exists.

`future.add_done_callback` alone is insufficient: callback is sync; port `release` is async.

---

## IDEMPOTENCY ORDER (unchanged; no wiring)

Observed order in `RuntimeToolInvoker.invoke`:

```text
_prepare → optional pre-effect claim → _execute_external_effect → retry loop → _execute_once
```

Admission rejection before submit must keep `boundary.may_have_started = False` (submit sets the flag today). No change applied.

Waiting on async admission **after** claim could extend claim hold time — not reordered in this task.

---

## ERROR MAPPING GAP

`RuntimeErrorCode` includes `DEPENDENCY_ERROR`, but `RuntimeToolInvoker._map_error` uses contract `error_mapping` then defaults to `TOOL_ERROR`.

Typed admission errors (`DependencyConcurrencyExceededError`, `DependencyConcurrencyAdmissionTimeoutError`, `DependencyConcurrencyPolicyMissingError`) have **no** documented Nexus mapping to `ToolExecutionResult` / propagation vs terminal infrastructure error. Mapping them to generic `TOOL_ERROR` without decision would violate W2-B2 audit rules.

---

## MINIMAL OPTIONS (no implementation without approval)

### Option A — Async-aware physical attempt boundary upstream

Introduce an async-owned “single physical attempt” coordinator used from async Nexus/gateway paths only, with `await acquire` → sync worker execution via existing pool model → `await release` when worker future completes.

**Gap:** parallel `tool_loop` worker threads and direct sync `invoker.invoke` callers bypass async owner unless entire catalog invoke surface becomes async-aware (large seam change).

### Option B — One runtime execution-boundary adapter (recommended direction)

Single production-owned component that:

- accepts injected `DependencyConcurrencyAdmissionPort | None`;
- coordinates **async acquire/release** with **sync** `ThreadPoolExecutor` future lifecycle (including post-caller-timeout completion);
- exposes a sync API callable from `_execute_once` **without** per-call `asyncio.run` / thread-local loops;
- internally uses one documented event-loop / thread model (e.g. dedicated admission loop thread) satisfying cancellation and no fire-and-forget leak.

Preserves one public async port for future distributed implementations.

### Option C — Synchronous local-only admission facade

Second contract or sync methods on local implementation only.

**Requires `ARCHITECTURAL CONTRACT DECISION`** — conflicts with “do not add SyncDependencyConcurrencyAdmissionPort automatically” unless explicitly accepted.

---

## RECOMMENDED OPTION

**Option B** — one reusable runtime adapter with explicit lifecycle ownership for “permit held until worker future completes,” wired from `_execute_once` when port is injected.

**WHY:** Keeps async port; satisfies before-submit and timeout-capacity invariants; avoids duplicating policy across async gateway vs sync worker-thread call paths.

---

## W2-B2 behavioral tests

**NOT RUN / NOT IMPLEMENTED** (blocked on architecture). W2-B1 contract tests remain the regression gate for admission semantics.

---

## NEXT TASK

Architectural decision on Option B (or approved variant): adapter ownership, thread/loop model, Nexus error mapping for admission failures, then re-open W2-B2 implementation + behavioral qualification suite.
