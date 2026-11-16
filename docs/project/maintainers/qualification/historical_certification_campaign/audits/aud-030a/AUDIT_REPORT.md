# AUDIT_REPORT - aud-030a - T-IGX-030-A (tests-only)

Auditor: aud-030a (independent; not implementer, planner, approver, MASTER, HIST worker or reviewer). Date: 2026-11-08.
Audited object: implementation SHA `683f955727fb4d90fe402b53de4c6dbf9a8748e1` (C1 `ec105a789`, C2 `683f95572`), base `492e361c0706e1aedeff44900a8a228345bf1d75`, checkout `D:\Projekty\_wt\intergrax\aud-030a` (detached, clean before and after).

## Verdict: PASS_SCOPED

Scope the verdict covers: the two tests-only commits (C1, C2) at the pinned SHA, as a faithful, non-weakening integration of source commits `0adb73380` and `10089121e`; their green exact-SHA replay; the real red state of the pre-change C1 file; ACT-N11 staying green; ticket/approval process hygiene.
Why scoped and not PASS: the authority for the ABANDONED semantics (OD-TLX-A, OD-TLX-C Option A, delegated, signed outside the repo) cannot be verified from the repo (see "Could not verify"); the broad N6 replay (F8, L3/TL11, TL14) was not in my brief.

This verdict does NOT close TOOL-LIFE-X. It covers conditions C-1 / C-2 only (stale-test reconciliation and the F3 cancel-test stabilisation). The EBH-4 R1 gate (T-IGX-030-B) and the product batch (T-IGX-030-C) are separate and HELD. No FRZ row is promoted and no stage status changes.

## Evidence table
Logs under `D:\Projekty\intergrax\.tmp\session\hist-cert-campaign\audits\aud-030a\`. All tests ran under `heavy.lock` (mkdir/rmdir per step; lock dir empty at the end), one process at a time, pytest's own exit code captured to the log (`EXIT_CODE=` line), `uv run --extra dev-ci` (+ `--extra mcp` for F3).

| # | Check | Command / method | Result | Exit | Log |
|---|---|---|---|---|---|
| 1 | Scope integrity | `git diff --name-only 492e361c0 683f95572` | 2 test files + `governance/tickets/T-IGX-030-{A,B,C}.md` + `plans/PLAN_T-IGX-030-{A,B,C}.md`; 0 paths under intergrax/, agents/, applications/, scripts/ | n/a | (inline) |
| 2 | Blob equality | `git rev-parse <sha>:<path>` | C1 file `2325c416...` = `0adb73380`; C2 file `23483824...` = `10089121e`; base file blobs `05b5fccc...` / `c18e28ca...` = parents `0adb73380^` / `10089121e^` | n/a | (inline) |
| 3 | Side-effect module | `pytest tests/unit/runtime/tools/test_tools_side_effect_safety.py -q` | 37 passed in 30.5 s | 0 | a_side_effect.log |
| 4 | F3 module x3 | `pytest ...f3_streamable_http_physical_l3.py -q` (3 separate runs) | 5 passed each: 79.2 s (cold first run, incl. 10.2 s fixture setup), 40.4 s, 40.3 s | 0,0,0 | c_f3_run1..3.log |
| 5 | Combined set | side-effect + Batch B retry/idempotency + Batch B governance authorization + p0_safety_8 + Batch C stalled/abandoned recovery | 76 passed in 5.4 s | 0 | d_combined.log |
| 6 | ruff | `ruff check` and `ruff format --check` on the 2 files | All checks passed; 2 files already formatted | 0, 0 | e_ruff_check.log, e_ruff_format.log |
| 7 | ACT-N11 verifier | `python -m scripts.governance.verify_iiac_policy_activation --require-in-force` | `IN_FORCE @ IIAC-2026-11-05.1 from f8c29fb8...` | 0 | f_iiac_verify.log |
| 8 | ACT-N11 governance tests | test_iiac_activation_gate.py, test_iiac_negative_guards.py, test_evol_x_real_repo_fresh_clone.py | 56 passed, 1 skipped (symlink creation unavailable, environment) in 452.8 s | 0 | f_gov_tests.log |
| 9 | NEGATIVE PROBE | same side-effect module in read-only `pin-ba07eba` (HEAD `ba07ebaa3`, file blob `05b5fccc...` = pre-change) | 4 failed, 32 passed | 1 | b_negprobe_pin.log |
| 10 | Plan approval | `verify_plan_approval --plan PLAN_T-IGX-030-A.md T-IGX-030-A.md` | OK | 0 | (inline) |

Negative probe detail (item 9). Failing tests at `ba07eba`: `test_side_effect_timeout_marks_uncertain_and_blocks_replay`, `test_mapped_validation_error_after_executor_marks_uncertain`, `test_close_shuts_down_execution_pool`, `test_replay_before_close_preserves_exactly_once` - exactly the four named in "C-1". The close-test failure reads `assert 'cannot schedule new futures after shutdown' in 'abandoned'`, i.e. the code already returns ABANDONED and only the stale assertion is red. Probe validity: `git diff --name-only ba07ebaa3 492e361c0 -- intergrax agents applications` is empty (ba07eba is an ancestor of the base), and between `ba07eba` and the audited SHA only 6 `scripts/` files differ, none in the tools/invoker path. So the same product code was exercised: 4 red before, 37 green after (the 5th new test N2 is an addition).

## Gate-weakening review (my own, from the real diffs)
C1 (`git diff 0adb73380^ 0adb73380`):
- Timeout test: `TIMEOUT` -> `ABANDONED`, exact enum swap (no set/`in`), plus new `error_message == "abandoned"`. Ledger `UNCERTAIN` and replay `InvocationUncertaintyError` kept. Not weakened.
- Mapped-validation test: `VALIDATION_ERROR` -> `ABANDONED`, exact swap. `external_effects["n"] == 1`, ledger `UNCERTAIN`, and replay-block assertions kept. Not weakened (the contract `error_mapping` code is no longer surfaced for admitted side-effecting failures; this is the OD-TLX-A decision).
- Two close tests: a CPython `ThreadPoolExecutor` substring replaced by exact code + exact message + a NEW ledger-`UNCERTAIN` assertion on the new key. `executor.calls == 1` kept. Stronger on the contract; no longer coupled to a stdlib string.
- New test N2 (`test_post_close_uncertain_key_replay_is_blocked_without_executor_call`): pure addition (no blind replay of a poisoned key, executor never called).
- Import reordering and signature re-wrapping (ruff/isort): behavior-neutral.
- No skip, xfail, widened set, dropped effect-count/ledger/replay check, or swallowed failure found.

C2 (`git diff 10089121e^ 10089121e`):
- `started` changed from `asyncio.Event` to `threading.Event` (the invoker runs on a worker thread; the event is set from `invoke`). The cancel is now event-driven (`invoker.started.wait(8.0)` with `TimeoutError`) instead of `asyncio.sleep(0.06)`: removes a timing race, adds a failure path.
- Added `step_id` to the catalog call, `assert invoker.results` (cancel really was mid-child), `_is_error_text(catalog) is None`, and failure messages. The accepted set `{"execution_cancelled","execution_admission_denied"}` is unchanged (not widened). Strengthened.
- Two new tests: cancel-before-admission (child must never run; accepts either `missing_execution_binding` error text or `execution_cancelled` payload - both fail-closed, backed by `assert not invoker.results`) and unknown-task-id negative control (`missing_execution_binding`, child never runs). Additions only.
- No skip/xfail/swallowed failure found.

My result matches the plan ledger (section 5 of the plan); I derived it from the diffs, not from the ledger.

## Semantics check against the code at the pinned SHA
Read: `intergrax/runtime/nexus/tools/invoker.py` (lines ~367-410, 1330-1560, 1840-1990), `intergrax/runtime/tools/tool_operation_recovery.py` (60-110). Batch C `a5f813a2c` is an ancestor of the base.
- Durable effect admission (`_record_durable_effect_admission_before_submit`, sets `boundary.may_have_started = True`) happens before `self._execution_pool.submit(...)`.
- Timeout path: `except FuturesTimeoutError` -> `must_abandon_uncertain_outcome(...)` (UNCERTAIN certainty and NON_IDEMPOTENT_WRITE class) -> `_tool_result_terminal_disposition(ABANDONED)`; `runtime_error_for_disposition` -> `RuntimeErrorCode.ABANDONED`; message `disposition_message` = `"abandoned"`. Matches the C1 timeout assertions.
- Executor failure / post-close submit failure: generic `except Exception` after retries -> `must_abandon_after_retry_exhaustion(effect_may_have_started, ...)` (side-effecting, not READ_ONLY) -> ABANDONED/UNCERTAIN. This matches the mapped-validation and both close tests, including why a never-started executor after `close()` is still reported ABANDONED/UNCERTAIN (admission is recorded before submit).
- Replay block of the UNCERTAIN key is exercised and green (37/37 and Batch B/C 76/76).
Conclusion: the code implements the semantics the C1 assertions encode. Whether those semantics are the AUTHORISED ones is a decision question I could not verify (below).

## Flakiness (C2)
3/3 green; durations stable after the cold first run (79 s -> 40 s -> 40 s; first-run delta = fixture setup 10.2 s vs ~2 s). Slowest tests identical each run: disconnect-during-child ~13.2 s, canonical-root ~9.4-10.5 s; the changed cancel tests take 1.9-2.9 s. No hangs, no process left running after the summary, no retries. Windows only, unloaded machine; no stress run.

## Process hygiene
- Ticket `T-IGX-030-A` at 683f95572: lifecycle `plan_approved`; at tip `90fae38aa` `ready_for_audit` (the tip diff touches only the ticket file: lifecycle + checkpoint). The checkpoint claims no PASS.
- `Approved:` line has the exact required form (`operator-delegate cert-030a @ 2026-11-08T10:42:59+01:00 | Delegation: .tmp/.../DELEG-HIST-030A-2026-11-08.md`); timestamp is within the delegation window (expires 2026-11-08T23:59:59+01:00); `verify_plan_approval` OK.
- Plan digest: `git show 683f95572:governance/tickets/plans/PLAN_T-IGX-030-A.md | sha256sum` = `c6d1dca6...e342`, equal to the ticket's `Plan digest:` line and unchanged since plan commit `78657d445`. (Raw `sha256sum` of the CRLF working-tree file gives `84a03429...`; that is a line-ending artefact of the checkout, not a plan change - do not use it.)
- Delegation record sha256 file = `e17d1cb1...0364` = sha256 of the `.md` (match).
- Commits C1/C2 carry `Source-Commit:` and `Ref: governance/tickets/T-IGX-030-A.md` trailers plus the Co-Authored-By line.
- English: all added lines are English; non-ASCII chars are only em-dashes in ticket/plan headings. C1 blob keeps 5 historical `T-IGX-023-A` labels by design (blob equality; map in ticket).
- Audited checkout stayed clean (`git status` empty before/after); `.venv` was created by my `uv run` in `aud-030a` (gitignored). `pin-ba07eba` untouched (clean).

## Findings (consolidated)
| ID | Severity | Finding | Required action |
|---|---|---|---|
| F-1 | Low (assurance) | The delegation record is a MASTER transcription, not signed by the Operator (`process_enforced`, no cryptographic provenance). The signed OD-TLX-A/C records are outside the repo. | None for this ticket. MASTER/Operator keep the signed OD-TLX-A/C and the chat transcript retrievable for the certifier; the certifier must not treat this audit as having verified them. |
| F-2 | Low (design residual) | C1 pins the CURRENT conservative result: a NEW call after `close()` whose executor never ran is reported `ABANDONED` with the key `UNCERTAIN` (admission recorded before `submit`). C1 does not prove the durable external-operation record is terminalised. | Track in HELD `T-IGX-030-C` (`29c387879` + `98b4bf514`); 030-C hard-depends on C1. No action on 030-A. |
| F-3 | Info | Generic `error_message == "abandoned"` replaces the specific root cause text in tests; diagnosis relies on trace events. | None. |
| F-4 | Info (coverage) | No test in these two files guards that read-only tool timeouts stay `TIMEOUT` (N1 declined by Operator Q-A3 = no). Code limits timeout-abandon to NON_IDEMPOTENT_WRITE (`must_abandon_uncertain_outcome`); I did not verify other tests cover the read-only case. | Certifier/MASTER decide whether to accept; no action on 030-A. |
| F-5 | Info | Ticket test contract lists `-p no:xdist`; my replay used the default invocation. Both green. | None. |
| F-6 | Info | `T-IGX-030-B` (HELD) still says "any approval line is written only by the Operator", which conflicts with the delegated-signing practice recorded for the campaign. | Align wording when 030-B is un-held; not a 030-A defect. |

No blocking finding. No finding requires a change to the audited commits.

## Could NOT verify
1. The signed OD-TLX-A / OD-TLX-C Option A records (outside the repo): whether ABANDONED is the authorised semantics. I verified only that the code at the pinned SHA implements it.
2. Operator's original chat instruction behind the delegation (only the MASTER transcription exists).
3. The broad N6 replay named in the handoff (F8 12-module inventory expecting 67, L3/TL11 17 modules, TL14 gates) and the review of the `5df074fcb` invoker delta: not in my brief; not run.
4. `igx_gov ticket` / `verify_ticket_lifecycle` (not run; broad tier, budget).
5. Behavior under CI/Linux, xdist, or a loaded machine for the real-uvicorn C2 test (3 Windows runs only).
6. The one skipped governance test (symlink creation unavailable on this machine).

## Residual risks
- C2 is a real-network L3 test; low-probability timing flakiness on a loaded runner cannot be excluded from 3 runs.
- C1 encodes a conservative, arguably over-broad ABANDONED for never-started post-close calls (F-2); a later product fix (030-C) may change the observable result and require re-touching these assertions (030-C is designed to build on this C1).
- ACT-N11 stays green only while no `intergrax/`, `agents/`, `applications/` path changes; HELD 030-B/030-C turn it red by design until the IIAC re-audit/re-activation batch.

## Statement
This verdict does not close TOOL-LIFE-X: it covers conditions C-1 and C-2 only; the EBH-4 R1 gate and the product batch are separate. It promotes no FRZ row, activates nothing, and changes no stage status. I made no edit to any checkout, no commit, no push.
