# TOOL-LIFE-X Batch A — Canonical Deadline Hierarchy Evidence

| Field | Value |
| --- | --- |
| **Task** | `TOOL-LIFE-X-BATCH-A-ATOMIC-PERMIT-CLOSURE` |
| **Parent** | `TOOL-LIFE-X` (§3.0.5) — **not CLOSED** |
| **Mode** | Change-delta (ECAP-P1/P2/P4/P6) |
| **START_HEAD** | `4c0f6504da1969175582e1830014c24e857243b0` |
| **Disposition** | **`TOOL-LIFE-X BATCH A — READY FOR INDEPENDENT AUDIT`** (Cursor wave; GitHub SHA is acceptance authority) |

**Non-closure:** Does not close `TOOL-LIFE-X`, does not promote global `FRZ-*` PASS, does not substitute `HARNESS-W4` / `DIST-SCALE-X`.

## History

| Wave | SHA | Outcome |
| --- | --- | --- |
| Pre-remediation | `03d5fc3d54d33ef6230123ab2300756441f744f8` | Independent audit **NOT ACCEPTED** (BA-F02 permit ownership race) |
| Prior Cursor remediation | (see prior table rows BA-F01/BA-F02) | **45 passed / 4 failed** — governance fixture gap + admission thread drain |
| **Atomic permit closure (BA-F02 wave)** | **fc59dd350933483205460fa0e84c7d80c37b6bb3** | **50/50** targeted Batch A regression green |
| **BA-F05 async ownership closure (this wave)** | **d76d6d874ed9394d2e1959e490c5835248deea9f** | **52/52** targeted Batch A regression green |

## BA-F05 — abandoned acquire cleanup before grant visibility (complete)

| Topic | Detail |
| --- | --- |
| **Root cause** | `try_claim_abandoned_release()` transitioned `ABANDONED → RELEASED` before `acquire_future.done()` and before permit observation; early return left a late physical grant with no release obligation. Wrapper `cancel()` was treated as proof of no grant. |
| **Protocol** | Extended `_AcquirePermitHandoff`: `WAITING → GRANT_AVAILABLE → CALLER_OWNS`, or `ABANDONED → RELEASE_OBLIGATION_CLAIMED → RELEASED` only after release completes. `on_physical_grant()` on the admission loop linearizes grant vs abandon; abandoned grants release in-coroutine via `_AcquireAbandonedBeforeDelivery`. |
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

## Targeted regression (this wave)

```text
uv run pytest -p no:xdist \
  tests/unit/runtime/resilience/test_dependency_attempt_execution_boundary.py \
  tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_dependency_admission.py \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_a_deadline_hierarchy.py \
  -q
```

| Metric | Value |
| --- | --- |
| **Collected** | 52 |
| **Passed** | 52 |
| **Failed** | 0 |
| **Skipped** | 0 |
| **Log** | `.tmp/session/TOOL-LIFE-X-BATCH-A/pytest-final.log` |

**FINAL_COMMIT:** `d76d6d874ed9394d2e1959e490c5835248deea9f` (BA-F05 implementation + evidence)

**Historical (BA-F02 only):** implementation `fc59dd350933483205460fa0e84c7d80c37b6bb3` — **50/50**; not attributed to BA-F05 SHA.

## TL1–TL15 (Batch A reconciliation)

| Criterion | Owner | Contract | Evidence SHA | Gap | Action | Test | FRZ | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TL1 bounded waits | `DependencyAttemptExecutionBoundary` + invoker | `acquire_timeout_seconds`, physical `future.result(timeout=…)` | fc59dd350933483205460fa0e84c7d80c37b6bb3 | — | Implemented | Batch A + boundary | FRZ-REL-01 | VALIDATION REQUIRED |
| TL2 deadline hierarchy | `RuntimeToolInvoker` | UER / `execution_live_remaining_seconds` | fc59dd350933483205460fa0e84c7d80c37b6bb3 | — | BA-F01 preserved | `test_tool_life_x_batch_a_deadline_hierarchy.py` | FRZ-REL-01, FRZ-REL-07 | VALIDATION REQUIRED |
| TL3 cancellation | Invoker + admission | cooperative cancel / protected work | fc59dd350933483205460fa0e84c7d80c37b6bb3 | — | Scoped tests unchanged | Batch A cancel test | FRZ-REL-05 | VALIDATION REQUIRED |
| TL4 retry bound | Invoker policy | `max_attempts` / side-effect safety | prior SHAs | — | Not expanded Batch B | existing side-effect module (out of this wave) | FRZ-REL-04 | VALIDATION REQUIRED |
| TL5 uncertain outcome | Invoker | `ToolEffectCertainty` on timeout | fc59dd350933483205460fa0e84c7d80c37b6bb3 | — | Preserved | Batch A timeout tests | FRZ-REL-05 | VALIDATION REQUIRED |
| TL6 side-effect retry | Invoker + idempotency | pre-effect coordinator | prior SHAs | — | Not in Batch A delta | — | FRZ-REL-06 | IMPLEMENTATION REQUIRED |
| TL7 | — | — | — | — | Open (parent) | — | — | OPERATOR DECISION REQUIRED |
| TL8 ownership/recovery | `DependencyAttemptExecutionBoundary` | permit + worker detach | BA-F05 FINAL_COMMIT | — | BA-F02 + BA-F05 closure | boundary matrix + BA-F05 L1 | FRZ-REL-07 | VALIDATION REQUIRED |
| TL9 terminal late results | Invoker | no resurrection | fc59dd350933483205460fa0e84c7d80c37b6bb3 | — | Preserved | Batch A | FRZ-REL-05 | VALIDATION REQUIRED |
| TL10–TL13 | — | — | — | — | Open per parent | — | — | BLOCKED / parent scope |
| TL14 no lifecycle bypass | Architecture | single invoker path | prior SHAs | — | No new bypass | U5 gates (unchanged) | FRZ-BND-03, FRZ-EXE-02 | ALREADY PROVEN |
| TL15 parent certification | Parent `TOOL-LIFE-X` | full §3.0.5 close | — | Parent not CLOSED | — | — | — | **BLOCKED** |

## Scoped ECAP / FRZ

| ID | Scope this wave |
| --- | --- |
| FRZ-REL-01 | Parent deadline bounds tool physical attempt — validation evidence advanced, not global PASS |
| FRZ-REL-05 | Timeout / uncertain semantics — scoped tests only |
| FRZ-REL-07 | Dependency admission permit lifecycle — BA-F02 + BA-F05 atomic ownership |
| FRZ-EXE-02 | `RuntimeToolInvoker` enforcement owner preserved |
| FRZ-GOV-02 | Governance identity required at inner boundary — tests fixed to bind identity, not weakened |
| FRZ-BND-03 | No parallel admission owner introduced |

## Reused evidence

- P0 pre-audit @ `f5c9330973f7af7ea1d19fbbd2afd13942faac43` (baseline only).
- HARNESS-02 Q09/Q11 patterns @ historical SHAs — acceptance scope unchanged.
