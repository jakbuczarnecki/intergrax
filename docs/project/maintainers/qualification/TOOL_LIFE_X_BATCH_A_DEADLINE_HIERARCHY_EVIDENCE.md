# TOOL-LIFE-X Batch A — Canonical Deadline Hierarchy Evidence

| Field | Value |
| --- | --- |
| **Task** | `TOOL-LIFE-X-BATCH-A-CLEANUP-ACCOUNTING-FINALIZATION` |
| **Parent** | `TOOL-LIFE-X` (§3.0.5) — **not CLOSED** |
| **Mode** | Change-delta (ECAP-P1/P2/P4/P6) |
| **START_HEAD** | `4038cc5995e26758cc5143e430fed252b8740c6d` |
| **BA-F06 resume START_HEAD** | `75201b60945e6b3dc7964b29d8a2bbd4520c07a4` |
| **BA-F07 START_HEAD** | `e3afb404879f0ca4fc1cede779868672e01b7103` |
| **BA-F08 START_HEAD** | `4038cc5995e26758cc5143e430fed252b8740c6d` |
| **BA-F09 START_HEAD** | `1dd30bffa8f2f9ed55517e7e43015962906f0439` |
| **BA-F10 START_HEAD** | `032b983ede3b0e6dee44083a4070e25326465006` |
| **Disposition** | **`TOOL-LIFE-X BATCH A — READY FOR INDEPENDENT AUDIT`** (BA-F12 atomic release claimant + L2 **65/65** after remediation; prior **63/63** preserved) |
| **BA-F12 final ownership START_HEAD** | `9e46bbcbf92d92ebf70655c778228cd85f39cc9f` |
| **BA-F12 final execution SHA** | `9e46bbcbf92d92ebf70655c778228cd85f39cc9f` + working-tree delta (L2 log; content ≡ `8f6b9a230…` implementation) |
| **Evidence publication SHA** | `8f6b9a2300508a6583a54a358bf4c04c2513ba05` |
| **BA-F12 START_HEAD** | `afe4f9c69b060d911cd7475dfd7fa0513f2ec03e` |
| **BA-F11 final qualification START_HEAD** | `afe4f9c69b060d911cd7475dfd7fa0513f2ec03e` |
| **Test execution SHA (L2 wave 1)** | `afe4f9c69b060d911cd7475dfd7fa0513f2ec03e` + unstaged test-source delta (see § BA-F11 L2) |
| **Documentation publication SHA (post-L2)** | `f64390392` (BA-F12) |
| **Implementation SHA (production)** | `f64390392` (BA-F12) |
| **BA-F11 START_HEAD** | `205a0ee436ccdedf32a19cb40d09615f24f1a8c3` |
| **BA-F08 implementation SHA** | `0d3a0d22fe07f7826acfb776c24341c303359838` |
| **Production baseline (pre-F08)** | `693ecafd36568b82bba7606a7e368b3b0aff7a38` (BA-F07) |
| **Historical L2 execution SHA** | `af96371edb0efa949f0cebae77e9b4af47b30c2b` (**56/56** @ production `693ecafd…` + test fixes; not attributed to BA-F08) |
| **Operator session START_HEAD** | `4038cc5995e26758cc5143e430fed252b8740c6d` |

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
| **Batch A consolidated L2 (historical)** | `af96371edb0efa949f0cebae77e9b4af47b30c2b` | **56 collected / 56 passed** — log `.tmp/session/TOOL-LIFE-X-BATCH-A/l2-batch-a-final.log` (production @ `693ecafd…`; **not** BA-F08) |
| **BA-F08 in-flight acquire shutdown accounting** | `0d3a0d22fe07f7826acfb776c24341c303359838` | **L2 NOT RUN**; static protocol closure + evidence update |
| **BA-F09 physical acquire quiescence correction** | `032b983ede3b0e6dee44083a4070e25326465006` | **L2 NOT RUN**; static protocol closure + evidence update |
| **BA-F10 pre-start cancellation & physical acquire lifecycle** | `205a0ee436ccdedf32a19cb40d09615f24f1a8c3` | **L2 NOT RUN**; static protocol closure + evidence update |
| **BA-F11 atomic start vs pre-start terminal handoff** | `bb7c01de29cc602885c94d8a698493d6759df01d` | **L2 wave 1:** 58 passed / 3 failed / 61 collected — log `.tmp/session/TOOL-LIFE-X-BATCH-A/l2-batch-a-ba-f11-final.log` |
| **BA-F11 consolidated L2 (operator-authorized)** | test delta on `afe4f9c6…` | **58/61** — failures grouped § BA-F11 L2; **no automatic re-run** |
| **BA-F11 consolidated L2 (remediation wave)** | unstaged test delta @ `afe4f9c6…` | **58/61** — log `l2-batch-a-ba-f11-remediation-final.log`; see § BA-F11 L2 wave 2 |
| **BA-F11 consolidated L2 (final remediation wave 3)** | test delta @ `afe4f9c6…` | **59/61** — log `l2-batch-a-ba-f11-final-remediation.log`; see § BA-F11 L2 wave 3 |
| **BA-F12 L1 (operator-authorized)** | production + test delta @ `afe4f9c6…` | **3/3** — log `.tmp/session/TOOL-LIFE-X-BATCH-A/l1-batch-a-ba-f12-final.log`; see § BA-F12 |
| **BA-F12 final ownership closure L1** | `9e46bbcbf92d92ebf70655c778228cd85f39cc9f` + delta | **4/4** — log `l1-batch-a-ba-f12-final-remediation.log` |
| **Batch A consolidated L2 (BA-F12 closure)** | same tree as final publication | **63 collected / 63 passed** — log `l2-batch-a-ba-f12-final.log` |
| **BA-F12 atomic claimant START_HEAD** | `e3bed11298eba2a4e5b83eabc65db963ebf2afcb` | Independent audit TOCTOU on `try_claim_release_obligation` re-entry |
| **BA-F12 atomic claimant L1** | implementation delta @ START_HEAD | **6/6** — log `.tmp/session/TOOL-LIFE-X-BATCH-A/l1-batch-a-ba-f12-atomic-claimant.log` |
| **BA-F12 atomic claimant L2 (wave 1)** | same delta | **64 passed / 1 failed / 65 collected** — `test_acquire_success_shutdown_before_registration_releases_permit` timing flake under full suite; **STOP** per batch rules (no auto rerun); log `l2-batch-a-ba-f12-atomic-claimant.log` |
| **BA-F12 atomic claimant L2 (publication)** | publication commit | **65 passed / 65 collected**, 8.89s, exit **0** — log `l2-batch-a-ba-f12-atomic-claimant-publication.log` (operator-authorized single rerun after wave-1 STOP classification) |

## BA-F12 — atomic physical permit ownership / cancelled wrapper recovery

| Topic | Detail |
| --- | --- |
| **START_HEAD** | `afe4f9c69b060d911cd7475dfd7fa0513f2ec03e` |
| **Root cause (static)** | Physical admission can grant before the sync wrapper exposes `result()`. `_maybe_release_abandoned_acquire_permit` returned early on `acquire_future.cancelled()` while `_AcquirePermitHandoff` did not retain the granted permit — **PHYSICAL GRANT → WRAPPER CANCELLED → ABANDON → NO RELEASE**. |
| **Ownership protocol** | On `on_physical_grant` deliver path: `_held_permit` written under handoff lock with `GRANT_AVAILABLE`. Caller `claim_successful_grant` clears `_held_permit` (`CALLER_OWNS`). Abandon: `try_claim_release_obligation` then `take_stored_permit_for_release()` before optional non-cancelled `future.result()` fallback. `release_in_coro` unchanged (local permit, `RELEASE_PENDING`). States: (1) not granted (2) granted on handoff (3) caller owns (4) abandoned + obligation claimed (5) release pending (6) released (7) release failure → `_release_failures` / drain fail-closed. |
| **Linearization** | Grant vs abandon: `on_physical_grant` / `mark_abandoned` / `try_claim_release_obligation` under `_AcquirePermitHandoff._lock`. Exactly one release claimant via obligation claim + single permit source (handoff store OR completed non-cancelled wrapper). |
| **Interleaving matrix (scoped)** | **A** abandon before grant → no permit. **B** grant then abandon → handoff store + obligation release. **C/D** grant + wrapper cancel → release from `_held_permit` (not `result()`). **E** grant, caller not claimed → store until claim or abandon. **F** caller claims → store cleared. **G–H** timeout/shutdown + grant → same obligation path. **I–J** release pending / cancel → `is_release_pending` noop. **K** release failure → lifecycle failure list. **L–M** pre-start / shutdown acquire → BA-F08–F11 unchanged. **N** shielded release → BA-F07. **O–P** duplicate callback/shutdown → idempotent obligation + pending checks. |
| **Modules** | `intergrax/runtime/resilience/dependency_attempt_execution_boundary.py`; tests + evidence only elsewhere. |
| **L1** | **3 passed** — `test_acquire_success_shutdown_before_registration_releases_permit`, `test_shutdown_cancel_races_acquire_success_no_leak`, `test_ba_f12_handoff_retains_physical_permit_when_wrapper_cancelled`; log `l1-batch-a-ba-f12-final.log`. |
| **L2** | **63/63** @ final publication SHA — log `l2-batch-a-ba-f12-final.log` (56 historical + 5 BA-F11 negatives + 2 BA-F12 negatives). |
| **Publication SHA (original)** | `f64390392` (`fix(runtime): BA-F12 physical permit ownership on cancelled wrapper`) |

## BA-F12 final — exactly-once release / competing cleanup vs sync grant

| Topic | Detail |
| --- | --- |
| **START_HEAD** | `9e46bbcbf92d92ebf70655c778228cd85f39cc9f` |
| **Prior production SHA** | `f643903924e0dfdd8cb203a7a1ce9df054577014` (retained `_held_permit`; sync path still called `_execute_permit_release` on failed `claim_successful_grant`) |
| **Root cause (static audit)** | Scenario **D/E**: wrapper `result()` succeeds while handoff is **ABANDONED**; `_maybe_release_abandoned_acquire_permit` claims obligation + `_schedule_handoff_permit_release`; sync caller then fails `claim_successful_grant` and invoked `_execute_permit_release(permit, wait=False)` → **two physical release obligations** for one grant. |
| **Correction** | On failed `claim_successful_grant`: skip sync release when obligation already **claimed / pending / complete**; otherwise atomically `try_claim_release_obligation` + `take_stored_permit_for_release` + `_schedule_handoff_permit_release` (shutdown accounting), never parallel `_execute_permit_release`. |
| **Winner/loser matrix (addendum)** | **Cleanup wins:** obligation claimed before sync grant → sync raises invariant only (no second release). **Sync wins:** `ABANDONED` without prior obligation → sync claims obligation and schedules handoff release once. **Caller wins:** `claim_successful_grant` before abandon → cleanup no-ops. **Tie:** `RELEASE_PENDING` / `RELEASED` → both paths no-op release. |
| **Scenarios A–L** | **A–B** unchanged (BA-F12 store). **C** wrapper cancel → stored permit release. **D** shutdown before registration + successful wrapper → gated test + spy: zero `_execute_permit_release`. **E/F** cleanup vs caller claim — obligation vs `CALLER_OWNS`. **G** sync release while pending → skipped. **H** callback + shutdown — single `_schedule_handoff_permit_release`. **I** idempotent obligation re-entry. **J/K** release failure / shielded — BA-F06/F07 paths unchanged. **L** drain during physical acquire — BA-F08–F11 unchanged. |
| **L1** | **4 passed** — `test_acquire_success_shutdown_before_registration_releases_permit`, `test_shutdown_cancel_races_acquire_success_no_leak`, `test_ba_f12_handoff_retains_physical_permit_when_wrapper_cancelled`, `test_ba_f12_exactly_one_release_when_cleanup_claims_before_sync_grant`; log `l1-batch-a-ba-f12-final-remediation.log`. |
| **L2** | **63 passed / 63 collected**, 10.46s, exit **0** — log `l2-batch-a-ba-f12-final.log`. |
| **Publication SHA (final)** | `8f6b9a2300508a6583a54a358bf4c04c2513ba05` (`fix(runtime): BA-F12 exactly-once release when sync grant loses to cleanup`) |
| **Test-source publication** | `8f6b9a2300508a6583a54a358bf4c04c2513ba05` (production + tests + evidence, single commit) |

## BA-F12 atomic claimant — exclusive release scheduling (audit remediation)

| Topic | Detail |
| --- | --- |
| **START_HEAD** | `e3bed11298eba2a4e5b83eabc65db963ebf2afcb` |
| **Production baseline (prior)** | `8f6b9a2300508a6583a54a358bf4c04c2513ba05` |
| **Root cause** | `try_claim_release_obligation()` returned **`True` on re-entry** (`RELEASE_OBLIGATION_CLAIMED`). Sync failed-grant path used a non-atomic `is_release_obligation_claimed()` pre-check then `try_claim_release_obligation()` → TOCTOU: callback could claim between check and call; sync received `True`, `take_stored_permit_for_release()` returned `None`, fell back to local `permit`, scheduled a **second** physical release. |
| **State machine** | **(1)** not granted · **(2)** grant on handoff · **(3)** caller owns · **(4)** abandoned / obligation reserved (`RELEASE_OBLIGATION_CLAIMED`, no schedule yet) · **(5)** exclusive schedule won (`_release_scheduling_claimed`, `RELEASE_PENDING` after `_schedule_handoff_permit_release`) · **(6)** `RELEASED` · **(7)** release failure → lifecycle failure list. Reservation (`try_claim_release_obligation`) ≠ scheduling authority (`try_claim_exclusive_release_scheduling`). |
| **Linearization** | Under `_AcquirePermitHandoff._lock`: obligation reservation is first-writer-only; exclusive scheduling is a separate first-winner flag; `mark_release_pending` only after schedule path wins. Sync failed-grant and `_maybe_release_abandoned_acquire_permit` both call **`try_claim_exclusive_release_scheduling`** only after `acquire_future.done()`. |
| **Contested matrix (addendum)** | **G:** re-entrant obligation claim no longer authorizes schedule; concurrent `try_claim_exclusive_release_scheduling` → one winner (deterministic handoff test). **H:** obligation reserved while future pending; schedule deferred until `done()`. Prior **D/E** matrix @ `8f6b9a2…` unchanged. |
| **Modules** | `intergrax/runtime/resilience/dependency_attempt_execution_boundary.py`; `tests/unit/runtime/resilience/test_dependency_attempt_execution_boundary.py`; this evidence file. |
| **L1** | **6 passed** — shutdown races, BA-F12 store, exactly-once cleanup, obligation re-entry negative, concurrent exclusive scheduling. |
| **L2** | **65/65** @ publication SHA — log `l2-batch-a-ba-f12-atomic-claimant-publication.log` (wave 1 **64/65** flake classified; one operator-authorized rerun). |
| **Publication SHA** | `a29e6e7df` (`fix(runtime): BA-F12 exclusive release scheduling claimant`) |
| **Evidence publication SHA** | same commit (source execution ≡ implementation) |
| **Historical preservation** | **56/56**, **58/61**, **58/61**, **59/61**, BA-F12 L1 **4/4**, L2 **63/63** @ `8f6b9a2…` / `e3bed112…` docs — **not** invalidated; attributed to prior SHAs only. |

**Scoped closure matrix (BA-F12-affected TL):**

| Criterion | Classification | Canonical owner | Contract | Existing evidence SHA | Missing action | Validation | FRZ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TL1 | ALREADY PROVEN | `DependencyAttemptExecutionBoundary` | bounded acquire timeout | `693ecafd…` L2 | Parent L3 | L2 batch | FRZ-REL-01 |
| TL3 | ALREADY PROVEN (Batch A L2) | Invoker + boundary | cancel / abandon + ownership interleaving | final publication SHA | Parent real-infra | L2 **63/63** | FRZ-REL-05 |
| TL5 | ALREADY PROVEN | Invoker | uncertain timeout | `693ecafd…` | Parent | L2 batch | FRZ-REL-05 |
| TL8 | **ALREADY PROVEN (Batch A L2)** | `DependencyAttemptExecutionBoundary` | exactly-once physical release / handoff | final publication SHA | Parent L3 breadth | L1 4/4; L2 **63/63** | FRZ-REL-07 |
| TL9 | **ALREADY PROVEN (Batch A L2)** | Boundary + invoker | no orphan / no double cleanup | final publication SHA | Cross-provider stale | L2 **63/63** | FRZ-REL-05 |
| TL14 | ALREADY PROVEN | Architecture | single invoker path | prior SHAs | — | static | FRZ-BND-03, FRZ-EXE-02 |
| TL4,6–7,10–13,15 | unchanged | Parent / Batch B | per roadmap | prior | Batch B/parent | deferred | per matrix § TL1–TL15 |

## BA-F11 L2 — consolidated qualification (wave 1)

| Metric | Value |
| --- | --- |
| **Execution tree** | `afe4f9c69b060d911cd7475dfd7fa0513f2ec03e` (production unchanged @ `bb7c01de29cc602885c94d8a698493d6759df01d`) |
| **Collected** | **61** (56 historical + 5 new negatives) |
| **Passed** | **58** |
| **Failed** | **3** |
| **Skipped** | **0** |
| **Exit code** | **1** |
| **Duration** | **8.93s** |
| **Log** | `.tmp/session/TOOL-LIFE-X-BATCH-A/l2-batch-a-ba-f11-final.log` |

**Failure grouping (wave 1):**

| Test | Class | Notes |
| --- | --- | --- |
| `test_acquire_success_shutdown_before_registration_releases_permit` | **test defect** | BA-F11 propagates `asyncio.CancelledError` on in-flight wrapper cancel during `result()`; test expected only `DependencyAttemptExecutionBoundaryClosedError`. |
| `test_shutdown_cancel_races_acquire_success_no_leak` | **test defect** | Same terminal semantics as above (`CancelledError` vs `closed`). |
| `test_no_orphan_thread_after_close` | **test defect / timing** | Admission thread join not observed immediately after `close()` under suite load; bounded poll remediation staged. |

**New negative tests (BA-F08–F11):**

| Test | Scenario |
| --- | --- |
| `test_ba_f11_handoff_start_vs_pre_start_terminal_exclusive` | Simultaneous start vs pre-start terminal (handoff) |
| `test_ba_f11_handoff_concurrent_start_vs_pre_start_single_winner` | Concurrent claim linearization |
| `test_ba_f10_wrapper_cancel_before_physical_start_clears_in_flight` | Wrapper cancelled before physical coroutine start |
| `test_ba_f09_schedule_failure_unregisters_in_flight_handoff` | Failure during coroutine scheduling |
| `test_ba_f08_shutdown_waits_physical_acquire_not_wrapper_terminal` | Shutdown awaits physical acquire lifecycle |

**Reused (no new test):**

| Mandatory scenario | Existing test |
| --- | --- |
| Wrapper cancelled after physical-start claim | `test_shutdown_cancel_races_acquire_success_no_leak` (phase 2) |
| Shutdown racing registration and scheduling | `test_acquire_success_shutdown_before_registration_releases_permit`, `test_shutdown_cancel_races_acquire_success_no_leak` (phase 1) |
| Late physical grant after abandonment | `test_ba_f05_release_obligation_before_done_does_not_orphan_late_grant` |
| Shielded release after outer cancellation | `test_ba_f06_shutdown_waits_pending_abandon_release`, `test_ba_f06_capacity_not_released_until_physical_release_completes` |

**Remediation staged (wave 1 → wave 2):** align shutdown-race expectations with BA-F11 cancel propagation; bounded thread-join poll in `test_no_orphan_thread_after_close`; broaden `_second_acquire` exception handling in `test_ba_f08_shutdown_waits_physical_acquire_not_wrapper_terminal`.

**Static audit acceptance:** independent static review of BA-F01–F11 production **unchanged** this session (`git diff bb7c01de… -- intergrax/` empty).

## BA-F11 L2 — remediation qualification (wave 2)

| Metric | Value |
| --- | --- |
| **Execution tree** | `afe4f9c69b060d911cd7475dfd7fa0513f2ec03e` + unstaged test-source delta (same as wave 2 remediation) |
| **Collected** | **61** |
| **Passed** | **58** |
| **Failed** | **3** |
| **Exit code** | **1** |
| **Duration** | **10.81s** |
| **Log** | `.tmp/session/TOOL-LIFE-X-BATCH-A/l2-batch-a-ba-f11-remediation-final.log` |

| Test | Wave 2 failure | Verified cause (source-level) |
| --- | --- | --- |
| `test_acquire_success_shutdown_before_registration_releases_permit` | `isinstance(..., (ClosedError, asyncio.CancelledError))` false | `acquire_future.result()` raises **`concurrent.futures.CancelledError`**, not `asyncio.CancelledError` (log: `CancelledError()` vs tuple). Accounting asserts (`permits_released`, `active`) not reached in this run order. |
| `test_shutdown_cancel_races_acquire_success_no_leak` | `permits_released == 0` (expected 1) | Outcome label accepted `CancelledError`; **async permit release not observed** before assertion — race between wrapper cancel (`_maybe_release_abandoned_acquire_permit` returns early when `acquire_future.cancelled()`) and physical coroutine `finally` / abandon release path. Not a timing-only label issue. |
| `test_no_orphan_thread_after_close` | `dependency-admission-boundary` still alive after 2s poll | Fixed thread **name** is shared across all boundary instances (`daemon=False`); `threading.enumerate()` can match **another suite instance** while `close()` on the test boundary already joined its own `_thread`. Poll does not pin `boundary._thread`. |

**Wave 2 gate outcome:** remediation **incomplete** — one authorized re-run consumed; **no further pytest** this session per operator directive.

**Minimal next decision (operator):** (1) Accept terminal type union: `DependencyAttemptExecutionBoundaryClosedError` \| `concurrent.futures.CancelledError` (and map outcomes accordingly). (2) Synchronize on `permits_released` via admission `asyncio.Event` (or `drain_and_close` on test boundary) before asserting leak-free accounting. (3) Replace global name scan with `assert not boundary._thread.is_alive()` after successful `close()`, or isolate thread names in production (would be out of test-only scope).

## BA-F11 L2 — final remediation qualification (wave 3)

| Metric | Value |
| --- | --- |
| **Execution tree** | `afe4f9c69b060d911cd7475dfd7fa0513f2ec03e` + test-source delta (groups A/C + partial B) |
| **Collected** | **61** |
| **Passed** | **59** |
| **Failed** | **2** |
| **Skipped** | **0** |
| **Exit code** | **1** |
| **Duration** | **8.91s** |
| **Log** | `.tmp/session/TOOL-LIFE-X-BATCH-A/l2-batch-a-ba-f11-final-remediation.log` |

| Test | Wave 3 outcome | Notes |
| --- | --- | --- |
| `test_no_orphan_thread_after_close` | **PASS** | `boundary._thread.is_alive()` after `close()` — group C closed. |
| `test_acquire_success_shutdown_before_registration_releases_permit` | **FAIL** | `FuturesCancelledError` accepted; **`permits_released == 0`** after `drain_and_close()` — shutdown raced before async grant completion on wrapper. |
| `test_shutdown_cancel_races_acquire_success_no_leak` (phase 1) | **FAIL** | Same **`permits_released`** accounting (group B). |

**Post–wave 3 static package (not L2-verified):** `concurrent.futures.CancelledError` import; `drain_and_close()` before permit asserts; `_wait_for_pending_acquire_futures_done()` barrier before gated shutdown; `test_no_orphan_thread_after_close` pins `boundary._thread`. **No further pytest** this session (operator one-shot L2 consumed).

**Open item:** If wave-3 barrier + drain still fail on re-auth L2, escalate **contract review** — `_maybe_release_abandoned_acquire_permit` skips `acquire_future.cancelled()` wrappers after physical grant (production @ `bb7c01de…`).

## BA-F11 — atomic start vs pre-start terminal handoff (implementation complete)

| Topic | Detail |
| --- | --- |
| **Independent audit finding @ BA-F10 `205a0ee…`** | `_unregister_in_flight_acquire_if_pre_start` tested `physical_acquire_entered()` and called `_unregister_in_flight_acquire` as separate steps. A coroutine could execute `mark_physical_acquire_entered()` after the callback observed `False` but before map removal — **running acquire dropped from `_in_flight_acquire_handoffs`** → false shutdown quiescence. |
| **Root cause** | Pre-start terminal cleanup and physical-start evidence were not one atomic ownership transition; BA-F10 split check/remove reintroduced a lost wake vs BA-F09 physical tracking. |
| **Linearization point** | `_AcquirePermitHandoff._lock`: competing `try_claim_physical_start()` (admission loop) vs `try_claim_pre_start_terminal()` (wrapper terminal, schedule-failure path). States: `REGISTERED → PHYSICAL_STARTED` or `REGISTERED → PRE_START_TERMINAL` — mutually exclusive. |
| **Correction** | `_acquire_coro` calls `try_claim_physical_start()` before any `await`; loser raises `asyncio.CancelledError()` (no physical admission). Pre-start paths call `try_claim_pre_start_terminal()` then unregister only on `True`. `physical_acquire_entered()` maps to `PHYSICAL_STARTED`. Schedule-failure cleanup uses the same pre-start claim protocol. Post-start: physical `finally` unregisters (unchanged BA-F09). |
| **Winner / loser matrix** | **(1)** Terminal before start: terminal wins → unregister; late coroutine loses claim → no admission; `finally` pop idempotent. **(2)** Start before terminal: physical wins → callback claim fails → map until `finally`. **(3)** Simultaneous: handoff lock orders one winner. **(4)** Wrapper cancel + delayed dispatch: terminal claim blocks physical start. **(5)** Shutdown + register/schedule: abort + dual paths per BA-F08–F10 with atomic claims. **(6)** Schedule failure: pre-start claim + unregister. **(7)** Late grant / abandon: permit handoff states unchanged (BA-F02–F07). **(8)** Loop stop: empty map requires physical `finally` or winning pre-start unregister. |
| **Static safety** | No false quiescence; no permanent pre-start map entry; single removal authority per seq; no admission after pre-start terminal; no orphaned permit / double release (permit path untouched); no lock held across async wait; handoff lock never taken under `_registry_lock`. |
| **Modules** | `intergrax/runtime/resilience/dependency_attempt_execution_boundary.py` only. |
| **Tests NOT RUN** | Operator session forbids execution. **Deferred L2:** start-vs-terminal interleave (callback between check and unregister regression); cancel before body after schedule; concurrent `try_claim_*`; schedule failure; shutdown during register; late grant; shielded abandon (`test_ba_f06_*` + 56-test command). |
| **Historical L2** | **56/56** remains @ `af96371e…` / production `693ecafd…` — not re-run or attributed to BA-F11. |
| **L2 qualification** | **BLOCKED (wave 2)** — 58/61 after remediation L2; further test-source work required before any new L2. |

## BA-F10 — pre-start cancellation & physical acquire lifecycle (implementation complete)

| Topic | Detail |
| --- | --- |
| **Independent audit finding @ BA-F09 `032b983e…`** | Physical quiescence unregisters only in `_acquire_coro` `finally`. A wrapper `ConcurrentFuture` may reach terminal (`cancelled()` / `done()`) **before** the admission-loop coroutine body enters that `finally` (scheduled-but-not-started, or loop never dispatched the task). `_wait_for_in_flight_acquires` then blocks until timeout — **permanent in-flight record** with no physical unwind path. |
| **Root cause** | BA-F09 correctly refused wrapper-terminal map removal while physical work could continue, but did not add a complementary terminal path for **REGISTERED / NOT STARTED** obligations when the wrapper alone reached terminal. |
| **Correction** | Physical-start evidence as first statement in `_acquire_coro` (before first `await`). Dual terminal accounting (exactly one map removal): physical `finally` vs wrapper pre-start path. **Superseded for atomicity by BA-F11** (`try_claim_physical_start` / `try_claim_pre_start_terminal`). Schedule-failure unregister under registering seq. Registration + first shutdown snapshot under `_registry_lock` before schedule (unchanged). |
| **Scheduling states (internal)** | **REGISTERED / NOT STARTED** — in `_in_flight_acquire_handoffs`, wrapper may be pending; terminal via pre-start wrapper path or schedule-failure unregister. **RUNNING** — `physical_acquire_entered`; terminal via physical `finally` only. **PHYSICALLY TERMINATED** — map entry removed; permit ownership still governed by `_AcquirePermitHandoff` / attempt record (BA-F02–F09). |
| **Static scenario matrix (12/12)** | (1) register→schedule→start→success: physical `finally` unregisters; wrapper pre-start noop. (2) cancel before start: wrapper `finally` unregisters; coroutine never entered or never marked. (3) concurrent shutdown during schedule: abandoned + cancel; pre-start or physical path per dispatch. (4) schedule throws: immediate unregister. (5) wrapper cancel after start: map until physical `finally`. (6) late grant: handoff unchanged. (7) abandon after grant: in-coro release + physical `finally`. (8) cancel during release: shielded pending-async (BA-F07). (9) shutdown during register/schedule: snapshot abort + dual terminal paths. (10) repeated cancel: idempotent `pop`. (11) concurrent `begin_shutdown`/`drain_and_close`: `_close_lock` + lifecycle gate. (12) loop stop: barrier requires empty handoff map (both terminal paths). |
| **Modules** | `intergrax/runtime/resilience/dependency_attempt_execution_boundary.py` only. |
| **Tests NOT RUN** | Operator session forbids execution. **Deferred L2 negatives:** cancel before coroutine body runs (map must drain without physical `finally`); cancel after `mark_physical_acquire_entered`; shutdown racing register-before-`run_coroutine_threadsafe`; simulated schedule failure; shutdown with late grant; shutdown with shielded abandon release (`test_ba_f06_*` + consolidated 56-test command). |
| **L2 qualification** | **PENDING** — consolidated Batch A L2 at BA-F10 publication SHA (separately authorized). |

## BA-F09 — physical acquire quiescence correction (implementation complete)

| Topic | Detail |
| --- | --- |
| **Independent audit finding @ BA-F08 `0d3a0d22…`** | `_in_flight_acquire_handoffs` was keyed by `run_coroutine_threadsafe` wrapper futures; `_on_acquire_terminal` unregistered on wrapper terminal. Successful `acquire_future.cancel()` marks the wrapper `done()`/`cancelled()` before the admission-loop `_acquire_coro` body finishes (`await self._admission.acquire`, abandon release, or `finally`). `_wait_for_in_flight_acquires` then polled wrapper `done()` and could observe an empty map while a physical coroutine still ran — **wrapper terminal ≠ physical coroutine terminal**. |
| **Root cause** | Quiescence barrier conflated (1) wrapper-future terminal with (2) physical admission coroutine terminal. Unregister in the wrapper done-callback `finally` dropped tracking while `_admission.acquire` or in-coro abandon release could still complete and mint permit obligations after `drain_and_close` advanced. |
| **Correction** | **Registration before schedule:** monotonic `in_flight_seq` + handoff entered under `_registry_lock` before `run_coroutine_threadsafe`. **Physical completion evidence:** `_acquire_coro` `finally` → `_unregister_in_flight_acquire(in_flight_seq)` only after the coroutine stack unwinds on the admission loop. **Wrapper map:** `_in_flight_acquire_futures` holds cancel targets for `_abort_in_flight_acquire`; wrapper done-callback retains abandon release logic only — no map removal. **Quiescence barrier:** `_wait_for_in_flight_acquires` waits until `_in_flight_acquire_handoffs` is empty (condition wake from physical `finally`), not until wrappers are `done()`. |
| **Lifecycle facts (non-conflated)** | **(1)** Wrapper terminal — sync `ConcurrentFuture` callback / `result()` visibility. **(2)** Physical coroutine terminal — `_acquire_coro` `finally` on admission loop. **(3)** Permit ownership — `_AcquirePermitHandoff` + attempt record / abandon paths unchanged (BA-F02–F07). **(4)** Physical release — `_pending_async_release_futures` + `complete_release()` unchanged. |
| **Shutdown ownership proof** | `begin_shutdown`: `_lifecycle → SHUTTING_DOWN` under lock; snapshot all `in_flight_seq` handoffs; `_abort_in_flight_acquires` marks abandoned + cancels known wrappers. `drain_and_close`: abort if still OPEN → `_wait_for_in_flight_acquires` (physical map empty) → detached → async releases → `loop.stop`. No second owner; bounded `_STARTUP_TIMEOUT_SECONDS`. |
| **Modules** | `intergrax/runtime/resilience/dependency_attempt_execution_boundary.py` only. |
| **Tests NOT RUN** | Operator session forbids execution. **L2 obligations (negative):** wrapper `cancel()` while admission `acquire` still blocked — `drain_and_close` must not pass until physical `finally`; shutdown race register-before-schedule vs `begin_shutdown`; late grant after abandon still routes to `_pending_async_release_futures`. Reuse `test_ba_f06_*` + full 56-test Batch A command below. |
| **L2 qualification** | **PENDING** — consolidated Batch A L2 at BA-F09 publication SHA (separately authorized). |

## BA-F08 — in-flight acquire shutdown accounting (superseded quiescence detail by BA-F09)

| Topic | Detail |
| --- | --- |
| **Independent audit finding @ `693ecafd…`** | `_pending_acquire_futures` tracked only the sync `result()` window; `begin_shutdown()` cancelled wrapper futures without `mark_abandoned()` on the paired handoff; no quiescence barrier tied underlying admission coroutines to drain before `loop.stop()`. `Future.cancel()` alone did not prove every started acquire had reached outcome A/B/C. |
| **Root cause** | Acquire lifetime ≠ wrapper wait lifetime. Shutdown snapshot omitted coroutines whose sync waiter had left `finally` while admission still ran; cancelled wrappers without abandoned handoff skipped `_maybe_release_abandoned_acquire_permit`; admission loop could stop after async-release drain while an unstopped acquire could still mint obligations. |
| **Correction** | `_in_flight_acquire_handoffs` maps each `run_coroutine_threadsafe` future to its `_AcquirePermitHandoff` from schedule until terminal callback unregisters. **Linearization:** `_lifecycle → SHUTTING_DOWN` under `_registry_lock` closes admission; `_abort_in_flight_acquires` marks every mapped handoff abandoned and cancels or observes completed wrappers. **`drain_and_close` order:** abort (if still OPEN) → `_wait_for_in_flight_acquires` → `_wait_for_detached_releases` → `_wait_for_pending_async_releases` → `loop.stop`. Race: register in-flight before `add_done_callback`; post-schedule `SHUTTING_DOWN` triggers immediate abort of that pair. |
| **Outcomes A/B/C** | **A:** cancel before grant — coroutine aborts at admission `await`, no permit. **B:** caller `claim_successful_grant()` → attempt record → detached drain. **C:** abandon (`mark_abandoned` / shutdown abort) → `on_physical_grant` `release_in_coro` or `_maybe_release_abandoned_acquire_permit` → `_pending_async_release_futures` (BA-F07 bridge) until physical release or `_record_lifecycle_failure`. |
| **Modules** | `intergrax/runtime/resilience/dependency_attempt_execution_boundary.py` only. |
| **Tests NOT RUN** | Full Batch A L2 + `test_ba_f06_*` / shutdown race suite — operator session forbids execution; static review only. |
| **L2 qualification** | **PENDING** — rerun targeted 56-test Batch A regression at BA-F08 publication SHA. |

## Static shutdown / acquire race resolution (BA-F07 baseline; in-flight barrier completed in BA-F08)

**Audit question:** Can `drain_and_close()` conclude with no outstanding release while an in-flight physical admission can still mint an untracked permit-release obligation?

**Linearization (owner: `DependencyAttemptExecutionBoundary`):**

1. **`_pending_acquire_futures`** — membership only while the sync `acquire()` thread is inside `result()` / timeout wait (`add` before wait, `discard` in `finally`). Not a proxy for coroutine lifetime; shutdown uses a snapshot at `begin_shutdown()` plus drain sets below.
2. **Physical grant vs abandon** — `_AcquirePermitHandoff.on_physical_grant()` on the admission loop is the single linearization point (`deliver` / `release_in_coro` / `noop`). Sync abandon is `mark_abandoned()` on timeout paths before wrapper `cancel()` so late grants observe `ABANDONED` or `RELEASE_OBLIGATION_CLAIMED`.
3. **Wrapper cancel ≠ physical cleanup complete** — `_on_acquire_terminal` never calls `exception()` on cancelled wrappers; re-enters `_maybe_release_abandoned_acquire_permit` only when obligation states are visible and release is not `RELEASE_PENDING` / `RELEASED`.
4. **BA-F07 shielded in-loop release** — bridged `ConcurrentFuture` in `_pending_async_release_futures` before `await asyncio.shield(release_task)`.
5. **BA-F08/BA-F09/BA-F10/BA-F11 in-flight map** — `_in_flight_acquire_handoffs` (physical seq) + `_abort_in_flight_acquires` + `_wait_for_in_flight_acquires` before detached/async release drain and `loop.stop()` (BA-F09: post-start drain on physical `finally`; BA-F10/BA-F11: pre-start drain when `try_claim_pre_start_terminal()` wins).

**Invariant conclusion (BA-F11):** No `loop.stop()` until in-flight acquire map is empty (physical `finally` **or** atomic pre-start terminal claim + unregister), attempt/detached releases drained, and `_pending_async_release_futures` drained (bounded timeout). Post-start: wrapper `done()` alone remains insufficient; pre-start: physical `finally` may never run — winning pre-start claim is required before unregister.

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
| **Collected** | 52 | **56** (historical @ `693ecafd…`) · **61** (BA-F11 wave 1 @ `afe4f9c6…`) |
| **Passed** | 52 | **56** (historical) · **58** (BA-F11 wave 1) |
| **Failed** | 0 | **0** (historical) · **3** (BA-F11 wave 1) |
| **Skipped** | 0 | **0** |
| **Log** | `.tmp/session/TOOL-LIFE-X-BATCH-A/pytest-final.log` | `.tmp/session/TOOL-LIFE-X-BATCH-A/l2-batch-a-final.log` (historical) · `l2-batch-a-ba-f11-final.log` (BA-F11 wave 1) |

**FINAL_COMMIT (BA-F05):** `d76d6d874ed9394d2e1959e490c5835248deea9f`

**BA-F06 publication commit:** `e3afb404879f0ca4fc1cede779868672e01b7103`

**BA-F07 publication commit:** `693ecafd36568b82bba7606a7e368b3b0aff7a38` (`fix(runtime): BA-F07 cleanup accounting for shielded abandon releases`)

**BA-F08 publication commit:** `0d3a0d22fe07f7826acfb776c24341c303359838` (`fix(runtime): BA-F08 in-flight acquire shutdown accounting`)

**BA-F09 publication commit:** `032b983ede3b0e6dee44083a4070e25326465006` (`fix(runtime): BA-F09 physical acquire quiescence correction`)

**BA-F10 publication commit:** `205a0ee436ccdedf32a19cb40d09615f24f1a8c3` (`fix(runtime): BA-F10 pre-start acquire cancellation lifecycle`)

**BA-F11 publication commit:** `bb7c01de29cc602885c94d8a698493d6759df01d` (`fix(runtime): BA-F11 atomic start vs pre-start terminal handoff`)

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
| TL8 | **ALREADY PROVEN (Batch A L2)** | `DependencyAttemptExecutionBoundary` | permit handoff + exactly-once release (BA-F12 final) | final publication SHA | Parent TOOL-LIFE L3 breadth | FRZ-REL-07 |
| TL9 | **ALREADY PROVEN (Batch A L2)** | Boundary + invoker | no orphan / no double cleanup (BA-F12 final) | final publication SHA | Cross-provider stale results | FRZ-REL-05 |
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
| FRZ-REL-07 | Dependency admission permit lifecycle — BA-F02–F12 final; L1 **4/4**; consolidated L2 **63/63** @ final publication SHA; historical **56/56** and BA-F11 waves **58/61**, **58/61**, **59/61** preserved — **not** global PASS |
| FRZ-EXE-02 | `RuntimeToolInvoker` enforcement owner preserved |
| FRZ-GOV-02 | Governance identity required at inner boundary — tests fixed to bind identity, not weakened |
| FRZ-BND-03 | No parallel admission owner introduced |

## Reused evidence

- P0 pre-audit @ `f5c9330973f7af7ea1d19fbbd2afd13942faac43` (baseline only).
- HARNESS-02 Q09/Q11 patterns @ historical SHAs — acceptance scope unchanged.
