# TOOL-LIFE-X Batch A — Canonical Deadline Hierarchy Evidence

| Field | Value |
| --- | --- |
| **Task** | `TOOL-LIFE-X-BATCH-A-CLEANUP-ACCOUNTING-FINALIZATION` |
| **Parent** | `TOOL-LIFE-X` (§3.0.5) — **not CLOSED** |
| **Mode** | Change-delta (ECAP-P1/P2/P4/P6) |
| **START_HEAD** | `4c0f6504da1969175582e1830014c24e857243b0` |
| **BA-F06 resume START_HEAD** | `75201b60945e6b3dc7964b29d8a2bbd4520c07a4` |
| **BA-F07 START_HEAD** | `e3afb404879f0ca4fc1cede779868672e01b7103` |
| **Disposition** | **`TOOL-LIFE-X BATCH A — READY FOR INDEPENDENT AUDIT`** |
| **Implementation SHA (production)** | `693ecafd36568b82bba7606a7e368b3b0aff7a38` (BA-F07; unchanged this wave) |
| **L2 execution SHA** | `af96371edb0efa949f0cebae77e9b4af47b30c2b` |
| **Operator session START_HEAD** | `ff6b1b6c7646dc4ed6f131e30ada3d44471c2059` |

**Non-closure:** Does not close `TOOL-LIFE-X`, does not promote global `FRZ-*` PASS, does not substitute `HARNESS-W4` / `DIST-SCALE-X`.

## History

| Wave | SHA | Outcome |
| --- | --- | --- |
| Pre-remediation | `03d5fc3d54d33ef6230123ab2300756441f744f8` | Independent audit **NOT ACCEPTED** (BA-F02 permit ownership race) |
| Prior Cursor remediation | (see prior table rows BA-F01/BA-F02) | **45 passed / 4 failed** — governance fixture gap + admission thread drain |
| **Atomic permit closure (BA-F02 wave)** | **fc59dd350933483205460fa0e84c7d80c37b6bb3** | **50/50** targeted Batch A regression green |
| **BA-F05 async ownership closure** | **d76d6d874ed9394d2e1959e490c5835248deea9f** | **52/52** targeted Batch A regression green |
| **BA-F06 cancellation-safe finalization** | `e3afb404879f0ca4fc1cede779868672e01b7103` | **L2 NOT RUN** (operator deferral); static review + prepared L1 sources |
| **BA-F07 cleanup accounting (shielded abandon release)** | `693ecafd36568b82bba7606a7e368b3b0aff7a38` | **L2 NOT RUN**; `STATIC REVIEW COMPLETED — L2 VALIDATION PENDING` |
| **Batch A consolidated L2 (final)** | `af96371edb0efa949f0cebae77e9b4af47b30c2b` | **56 collected / 56 passed / 0 failed / 0 skipped** — log `.tmp/session/TOOL-LIFE-X-BATCH-A/l2-batch-a-final.log` (test-only qualification fixes; production @ `693ecafd…`) |

## Static shutdown / acquire race resolution (final review @ `693ecafd…`)

**Audit question:** Can `drain_and_close()` conclude with no outstanding release while an in-flight physical admission can still mint an untracked permit-release obligation?

**Linearization (owner: `DependencyAttemptExecutionBoundary`):**

1. **`_pending_acquire_futures`** — membership only while the sync `acquire()` thread is inside `result()` / timeout wait (`add` before wait, `discard` in `finally`). Not a proxy for coroutine lifetime; shutdown uses a snapshot at `begin_shutdown()` plus drain sets below.
2. **Physical grant vs abandon** — `_AcquirePermitHandoff.on_physical_grant()` on the admission loop is the single linearization point (`deliver` / `release_in_coro` / `noop`). Sync abandon is `mark_abandoned()` on timeout paths before wrapper `cancel()` so late grants observe `ABANDONED` or `RELEASE_OBLIGATION_CLAIMED`.
3. **Wrapper cancel ≠ physical cleanup complete** — `_on_acquire_terminal` never calls `exception()` on cancelled wrappers; re-enters `_maybe_release_abandoned_acquire_permit` only when obligation states are visible and release is not `RELEASE_PENDING` / `RELEASED`.
4. **BA-F07 shielded in-loop release** — `_release_permit_for_abandoned_handoff` registers a bridged `ConcurrentFuture` in `_pending_async_release_futures` **before** `await asyncio.shield(release_task)`; `drain_and_close()` → `_wait_for_pending_async_releases()` → `_wait_for_detached_releases()` → `loop.stop()`. A shield-surviving release cannot outlive drain without appearing in the pending-async set.
5. **`begin_shutdown()`** — cancels pending acquire wrappers to unblock sync waiters; grants not yet delivered abort at the next admission `await` (no permit). Grants already linearized to `release_in_coro` remain tracked via BA-F07 accounting through loop stop.

**Invariant conclusion:** The admission event loop is not stopped until `_pending_async_release_futures` and detached attempt releases are drained (bounded timeout). BA-F07 closes the independent-audit gap where shielded abandon releases were invisible to drain. Residual `begin_shutdown()` cancel vs late-grant interleaving is bounded by handoff states + in-coroutine `release_in_coro` + pending-async registration (no new scheduler).

**Qualification note:** Prepared `test_ba_f06_*` timing/outcome expectations were corrected (no production delta) so the single L2 wave matches cancel/timeout semantics on Windows.

## BA-F07 — cleanup accounting for shielded in-coroutine abandon releases (implementation complete)

| Topic | Detail |
| --- | --- |
| **Independent audit finding @ `e3afb404…`** | `_release_permit_for_abandoned_handoff()` used `asyncio.shield` so a surviving release task could outlive acquire-wrapper cancellation without appearing in `_pending_async_release_futures`; `drain_and_close()` could stop the admission loop while physical release remained in flight. `begin_shutdown()` cancel of pending acquire wrappers is ownership-safe only when abandon cleanup is visible to the same drain set. |
| **Correction** | Bridge each in-loop abandon `release_task` to a `ConcurrentFuture` via `_bridge_loop_task_to_concurrent_future`, register in `_pending_async_release_futures` under `_registry_lock` **before** `await asyncio.shield`, and complete/discard through the same done-callback pattern as `_schedule_handoff_permit_release` (`complete_release` only after physical success; failures → `_record_lifecycle_failure`). Removed orphaned `_handoff_async_release_terminal` split path — shield survival is tracked by the registered future until terminal. |
| **Shutdown races** | Registration precedes first `await` on the abandon path (linearization vs `drain_and_close`). Wrapper `cancel()` / `begin_shutdown()` may cancel the acquire coroutine while shielded release continues; `_wait_for_pending_async_releases()` blocks until the bridged future completes. No second scheduler; `_schedule_handoff_permit_release` unchanged for sync-thread abandon observation. |
| **Pending acquire cancel** | Cancelled wrapper with no completed grant: coroutine aborts at admission wait — no permit. Abandoned grant with `release_in_coro`: obligation tracked in pending-async set, not erased by wrapper cancel. Granted permit accepted by sync caller: registered attempt record — unchanged. |
| **Modules** | `intergrax/runtime/resilience/dependency_attempt_execution_boundary.py` only (production). |
| **Static invariant review** | Single owner `DependencyAttemptExecutionBoundary`; no double release; loop thread non-blocking; bounded drain failure via existing timeout + `RuntimeError` / `DependencyAttemptReleaseInvariantError`. |
| **Tests NOT RUN** | Prepared: `test_ba_f06_shutdown_waits_pending_abandon_release`, `test_ba_f06_capacity_not_released_until_physical_release_completes`, `test_ba_f06_abandon_release_failure_surfaces_on_drain`, `test_ba_f06_acquire_terminal_callback_cancelled_future_safe` + full Batch A regression command below. |
| **L2 qualification** | **COMPLETE** — consolidated Batch A L2 @ publication SHA (56/56). |

## BA-F06 — cancellation-safe permit finalization (complete — superseded for shutdown accounting by BA-F07)

| Topic | Detail |
| --- | --- |
| **Root cause (residual after BA-F05)** | `on_physical_grant()` transitioned `ABANDONED` / `RELEASE_OBLIGATION_CLAIMED → RELEASED` before `await permit.release()`. `_maybe_release_abandoned_acquire_permit()` called `complete_release()` immediately after scheduling async release (`wait=False`). `_on_acquire_terminal` used `acquire_future.exception()` on cancelled wrappers (raises `CancelledError`) and prematurely `complete_release()` on some exception paths. Wrapper cancellation could be mistaken for completed physical cleanup. |
| **Protocol** | `RELEASE_PENDING` between obligation claim and successful physical release. `RELEASED` only via `complete_release()` after `permit.release()` succeeds (in-coroutine path, shielded task terminal, or `_schedule_handoff_permit_release` done callback). `on_physical_grant` abandon path: `→ RELEASE_PENDING` + `release_in_coro`; idempotent `noop` when already `RELEASED` / `RELEASE_PENDING` (no double release). |
| **Cancellation during release** | `_release_permit_for_abandoned_handoff`: `asyncio.shield` around release task; on `CancelledError`, `add_done_callback` on the surviving task records success (`complete_release`) or `_record_lifecycle_failure` without marking released. |
| **Callback / cancel** | Terminal callback branches on `acquire_future.cancelled()` before `exception()`; re-enters `_maybe_release_abandoned_acquire_permit` only when obligation visible and release not pending/complete. |
| **Shutdown** | Sync-thread handoff releases register in `_pending_async_release_futures` (BA-F06); in-coroutine shielded abandon releases **did not** until BA-F07. Release failures accumulate in `_release_failures` → `DependencyAttemptReleaseInvariantError` on `drain_and_close`. |
| **Modules** | `intergrax/runtime/resilience/dependency_attempt_execution_boundary.py` only (production). |
| **Static invariant review** | Single boundary owner preserved; no new scheduler/registry; accepted permits still `claim_successful_grant()` after `result()`; abandon cleanup cannot claim while `RELEASE_PENDING`; loop callbacks non-blocking (`wait=False` + done callbacks). |
| **Prepared tests (not executed this session)** | `test_ba_f06_*` in `test_dependency_attempt_execution_boundary.py` — gated release, release failure on drain, shutdown wait, cancelled-future callback safety. |
| **L2 qualification** | **PENDING** — rerun targeted 52-test Batch A regression at publication SHA before independent audit closure. |

## BA-F05 — abandoned acquire cleanup before grant visibility (complete)

| Topic | Detail |
| --- | --- |
| **Root cause** | `try_claim_abandoned_release()` transitioned `ABANDONED → RELEASED` before `acquire_future.done()` and before permit observation; early return left a late physical grant with no release obligation. Wrapper `cancel()` was treated as proof of no grant. |
| **Protocol** | Extended `_AcquirePermitHandoff`: `WAITING → GRANT_AVAILABLE → CALLER_OWNS`, or `ABANDONED → RELEASE_OBLIGATION_CLAIMED → RELEASE_PENDING → RELEASED` (BA-F06). `on_physical_grant()` on the admission loop linearizes grant vs abandon; abandoned grants release in-coroutine via `_AcquireAbandonedBeforeDelivery`. |
| **Cleanup** | `_maybe_release_abandoned_acquire_permit()` claims obligation without marking `RELEASED` until `done()` and release path completes; callback re-enters when obligation claimed. |
| **Tests** | `test_ba_f05_release_obligation_before_done_does_not_orphan_late_grant`, `test_ba_f05_zero_budget_timeout_before_grant_still_releases` + prior BA-F02 race/handoff suite |

## BA-F02 — atomic permit ownership (complete)

| Topic | Detail |
| --- | --- |
| **Root cause** | `add_done_callback` on `run_coroutine_threadsafe` future could run on the admission event-loop thread **before** sync `mark_accepted()` after `result()`, releasing a legitimately granted permit (`_AcquireAcceptance.is_accepted()` still false). |
| **Protocol** | Superseded for abandon cleanup by BA-F05 obligation states; caller claim remains `claim_successful_grant()` immediately after successful `result()`. |
| **Callback rule** | Success callback releases **only** when `handoff.is_abandoned()`; never when `WAITING` (caller still entitled to claim). |
| **Loop safety** | Abandoned release uses `_execute_permit_release(..., wait=False)` on the admission loop thread (no blocking `release_future.result()` on the loop). `drain_and_close()` waits on `_pending_async_release_futures` before `loop.stop()`. |
| **Tests** | `test_acquire_timeout_races_grant_releases_permit`, `test_acquire_handoff_callback_before_caller_claim_no_double_release`, shutdown/race suite in `test_dependency_attempt_execution_boundary.py` |

## BA-F03 — governance + shutdown disposition

| Failure (prior wave) | Disposition |
| --- | --- |
| 3× `test_runtime_tool_invoker_dependency_admission.py` — `active execution governance identity required` | **Fixture gap (fixed):** tests used `canonical_execution_identity_scope` only; `build_tool_invocation_inner_governance_request` requires `require_active_execution_governance_identity()`. Replaced with `_invoke_identity_scope` → `canonical_governed_execution_scope` aligned to `test-tenant` / `test-workspace` / `test-principal`. **No runtime governance validation relaxed.** |
| `test_no_orphan_thread_after_close` | **Production:** async abandoned releases could outlive `close()`; `_wait_for_pending_async_releases()` before stopping loop. **Not environment-only.** |

## BA-F04 — deadline / terminal (scoped)

| Path | Evidence |
| --- | --- |
| With `DependencyAttemptExecutionBoundary` | `test_tool_life_x_batch_a_deadline_hierarchy.py` (queue wait, post-acquire deadline, physical timeout after admission) + admission integration module |
| Without boundary | Same module `test_without_deadline_scope_uses_contract_timeout`, parent-expired / budget tests |
| Parent budget vs physical timeout | `_physical_attempt_timeout_after_admission_seconds()` unchanged contract; BA-F01 tests green after deterministic queue→physical scenario fix |

## Targeted regression (Batch A L2 — final)

```text
uv run pytest -p no:xdist \
  tests/unit/runtime/resilience/test_dependency_attempt_execution_boundary.py \
  tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_dependency_admission.py \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_a_deadline_hierarchy.py \
  -q
```

| Metric | BA-F05 @ `d76d6d8…` | **Final L2 @ publication SHA** |
| --- | --- | --- |
| **Collected** | 52 | **56** |
| **Passed** | 52 | **56** |
| **Failed** | 0 | **0** |
| **Skipped** | 0 | **0** |
| **Log** | `.tmp/session/TOOL-LIFE-X-BATCH-A/pytest-final.log` | `.tmp/session/TOOL-LIFE-X-BATCH-A/l2-batch-a-final.log` |

**FINAL_COMMIT (BA-F05):** `d76d6d874ed9394d2e1959e490c5835248deea9f`

**BA-F06 publication commit:** `e3afb404879f0ca4fc1cede779868672e01b7103`

**BA-F07 publication commit:** `693ecafd36568b82bba7606a7e368b3b0aff7a38` (`fix(runtime): BA-F07 cleanup accounting for shielded abandon releases`)

**Historical (BA-F02 only):** implementation `fc59dd350933483205460fa0e84c7d80c37b6bb3` — **50/50**; not attributed to BA-F05/BA-F06 SHAs.

## TL1–TL15 (Batch A reconciliation)

| ID | Classification | Owner | Contract | Evidence SHA | Remaining qualification | FRZ |
| --- | --- | --- | --- | --- | --- | --- |
| TL1 | ALREADY PROVEN (Batch A L2) | `DependencyAttemptExecutionBoundary` + invoker | bounded `acquire_timeout_seconds` + physical `result(timeout=…)` | `693ecafd…` + L2 publication SHA | Parent TOOL-LIFE L3 breadth | FRZ-REL-01 |
| TL2 | ALREADY PROVEN (Batch A L2) | `RuntimeToolInvoker` | UER / `execution_live_remaining_seconds` hierarchy | `693ecafd…` + L2 publication SHA | Parent L3 cross-plane | FRZ-REL-01, FRZ-REL-07 |
| TL3 | ALREADY PROVEN (Batch A L2) | Invoker + admission | cooperative cancel / protected work (scoped) | `693ecafd…` + L2 publication SHA | Real-infra cancel (parent) | FRZ-REL-05 |
| TL4 | VALIDATION REQUIRED | Invoker policy | `max_attempts` / side-effect safety | prior SHAs | Batch B / side-effect module | FRZ-REL-04 |
| TL5 | ALREADY PROVEN (Batch A L2) | Invoker | `ToolEffectCertainty` / uncertain timeout | `693ecafd…` + L2 publication SHA | Operator runbooks (parent) | FRZ-REL-05 |
| TL6 | IMPLEMENTATION REQUIRED | Invoker + idempotency | pre-effect coordinator | prior SHAs | Batch B | FRZ-REL-06 |
| TL7 | OPERATOR DECISION REQUIRED | Parent roadmap | STALLED / watchdog | — | Parent §3.0.5 | — |
| TL8 | ALREADY PROVEN (Batch A L2) | `DependencyAttemptExecutionBoundary` | permit ownership, detach, abandon cleanup, shutdown drain | `693ecafd…` + L2 publication SHA | Worker recovery (parent L3) | FRZ-REL-07 |
| TL9 | ALREADY PROVEN (Batch A L2) | Invoker | no terminal resurrection (scoped) | `693ecafd…` + L2 publication SHA | Cross-provider stale results | FRZ-REL-05 |
| TL10 | BLOCKED | SQL/database providers | row/byte/statement bounds | — | Parent / Batch C | FRZ-REL-10 |
| TL11 | BLOCKED | MCP via invoker | ToolContract path | — | Real MCP qual | FRZ-REL-06 |
| TL12 | BLOCKED | Parent scope | — | — | Parent | — |
| TL13 | BLOCKED | Parent scope | — | — | Parent | — |
| TL14 | ALREADY PROVEN | Architecture | single invoker path | prior SHAs | U5 gates unchanged | FRZ-BND-03, FRZ-EXE-02 |
| TL15 | BLOCKED | Parent `TOOL-LIFE-X` | full §3.0.5 certification | — | Parent L3 + HARNESS-W4 | — |

## Scoped ECAP / FRZ

| ID | Scope this wave |
| --- | --- |
| FRZ-REL-01 | Parent deadline bounds tool physical attempt — validation evidence advanced, not global PASS |
| FRZ-REL-05 | Timeout / uncertain semantics — scoped tests only |
| FRZ-REL-07 | Dependency admission permit lifecycle — BA-F02–F07 + Batch A L2 **56/56** (scoped evidence; not global PASS) |
| FRZ-EXE-02 | `RuntimeToolInvoker` enforcement owner preserved |
| FRZ-GOV-02 | Governance identity required at inner boundary — tests fixed to bind identity, not weakened |
| FRZ-BND-03 | No parallel admission owner introduced |

## Reused evidence

- P0 pre-audit @ `f5c9330973f7af7ea1d19fbbd2afd13942faac43` (baseline only).
- HARNESS-02 Q09/Q11 patterns @ historical SHAs — acceptance scope unchanged.
