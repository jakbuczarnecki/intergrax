# AUDIT REPORT — TOOL-LIFE-X re-audit at the post-fix SHA (AUD-TOOLLIFE2, OD-TLX-D)

AUDITOR RECORD — NOT AN OPERATOR CLOSURE SIGNATURE. Independent exact-SHA audit. The auditor is not IMPL-TLX-A/B, not the `aud_toollife` session, not the certifier, not a signing session. Nothing is signed.

## 1. Verdict
**CONDITIONAL** — `chain_gate` = **STOP**.

Both prior audit conditions are technically closed at the post-fix SHA (C-1 and C-2 below). All three OD-TLX-D replay sets are green. The verdict cannot be PASS or PASS_SCOPED because of the C0 finding, which I reproduced independently at the post-fix SHA (see section 4), and because the OD-TLX-D disclosure addendum is not yet in the tree.

## 2. Object and SHAs (resolved, existence and ancestry verified)
| Item | SHA | Note |
|---|---|---|
| Acceptance baseline | `1548640708c850b3fbfca3f2db0e5ecfee4274dd` | 2026-10-26. Ancestor of the post-fix SHA. |
| Publication commit | `35344557d3ce67a631f82df96c86b79b92b4db93` | Docs only. The reconciliation record has had **no commit since** it. |
| EBH-7 (changed `invoker.py`) | `5df074fcb` | Ancestor of the post-fix SHA. |
| Campaign base | `2f6664d53e3fcd1cb6010e33506e2d22201dc9f3` | Ancestor of the post-fix SHA. |
| Fix-wave base | `50ddc04fbabd24ac062267d5d33f14ccef89840c` | Parent 1 of the merge. |
| T-IGX-023-A | `ae5d0e6958d9389ac432007f67dc94a803e99580` | Test-only. |
| T-IGX-023-B | `94fa985741a07babfc7fd8250274c06457bfe49a` | Test-only. |
| **Audited post-fix SHA** | `2d14bd76f29366a6eda562c22cf857b55d515700` | Local merge, audit-only. The scratch worktree was detached here. |

Scratch worktree: `D:/Projekty/_wt/_scratch/aud-toollife2`. It was used read-only; its only addition is a throw-away copy of the C0 repro under its gitignored `.tmp/`. It is removed at finish.

The merge against `50ddc04fb` changes only 4 governance ticket and plan docs and 2 test files. It has **no `intergrax/` change** (git diff, section 3.1).

## 3. Questions asked by the mission

### 3.1 F-10: does `invoker.py` drift since the baseline affect any prior conclusion? — NO
`invoker.py` changed once after the baseline, in `5df074fcb` (+182/−211). I compared baseline and post-fix with an AST diff, normalising imports and annotations:
- Five methods differ in raw AST. Four differ only in function-local import order (`_require_agent_governance_approval_consumption_pair`, `_require_agent_runtime_governance`, `_require_canonical_inner_execution_guard`, `_require_canonical_meaningful_side_effect_authorization`). They have **zero semantic diff**.
- **The only semantic change** is in `_require_current_attempt_authorization`. It adds a fail-closed **tenant-mismatch deny guard**: a projected `invocation_context.tenant_id` that differs from `state.tenant_id` raises `ToolScopeViolationError`. It sits before any effect admission.
- Module-level and class-level non-function statements: zero diff, for `invoker.py`, `nexus_loop.py` and `resilience/dependency_attempt_execution_boundary.py`.
- `nexus_loop.py` and `dependency_attempt_execution_boundary.py` have **zero function-level semantic diff** (formatting and typing only).
- All three replay sets ran against the **current** `invoker.py`, so the drift is covered by execution and not only by inspection.

Conclusion: the drift is additive and fail-closed and does not undermine any prior conclusion. F-10 is RESOLVED.

### 3.2 C-1 (4 red Batch B L2 tests) — CLOSED technically, disclosure part pending
- `tests/unit/runtime/tools/test_tools_side_effect_safety.py` is now green: 37 passed, including the 4 formerly red tests and the new N2 test `test_post_close_uncertain_key_replay_is_blocked_without_executor_call`.
- I read the test diff. Only the error-code and message assertions changed, to the signed OD-TLX-A and OD-TLX-C semantics (`ABANDONED` / `"abandoned"`).
- The safety assertions are kept: ledger `UNCERTAIN`, replay raising `InvocationUncertaintyError`, effect count 1, `executor.calls == 1`. Two new `UNCERTAIN` ledger assertions were added for the post-close keys.
- Nothing was weakened, skipped or marked xfail.
- The part of C-1 that asked for the history to be **disclosed in the reconciliation record** is **NOT yet done** at the audited SHA. See condition C-4.

### 3.3 C-2 (F3 order dependence) — CLOSED
- The module is now 5 tests and was green in the 18-module run. That run is in-order, in-process, and includes the earlier modules that previously triggered the failure.
- Three further whole-module runs, `--extra mcp`, `-p no:xdist`: **5 passed, 5 passed, 5 passed**.
- Root cause was a timing defect in the test: a hard-coded `sleep(0.06)` raced the MCP handshake. The fix is event-driven, and `started` is now a `threading.Event` (it was an `asyncio.Event` set from a worker thread and never awaited).
- The fix **strengthens** the test: it asserts that the child actually ran (`invoker.results` non-empty) and that the result is not a `ToolError`. It also adds two negative controls (cancel-before-admission, and an unknown task id).
- I found no `intergrax/` change. This matches the signed "timing defect, not a product defect" classification, to the extent a test-only fix passing 4 of 4 times can show it.

### 3.4 C0 finding: scope and ruling — IN SCOPE; verdict cannot exceed CONDITIONAL
**Claim under test.** After `RuntimeToolInvoker.close()`, a new call whose `pool.submit` raises leaves a durable external-operation record in physical state `RUNNING`. It pre-dates Batch C and is not fixed.

**Independent verification at `2d14bd76f`:**
1. Static: `close()` sets `_dependency_attempt_boundary = None` (`invoker.py:257`). A post-close call therefore takes the `dep_boundary is None` branch (`:1905`). That branch runs `ext_op.mark_running()` (`:1914`) and then `self._execution_pool.submit(...)` (`:1915`) with **no** `except BaseException: ext_op.mark_failed()`. The boundary branch has that guard (`:1976-1977`).
2. Runtime: I ran the C0 repro once under the heavy-test mutex, with `intergrax/` identical to base `50ddc04fb`. **Both parametrisations reproduce.** `executor.calls == 1`, so the executor never ran for call 2.

   | Case | Call-2 user-visible result | Durable record for call 2 |
   |---|---|---|
   | side-effecting tool with idempotency coordinator | `ABANDONED` / `abandoned` | **`RUNNING`** |
   | non-side-effecting tool, no coordinator | `TOOL_ERROR` / `RuntimeError: cannot schedule new futures after shutdown` | **`RUNNING`** |

   Log: `logs/c0_repro_at_postfix.log`. The test "fails" on its own observation assertion, and that failure is the finding. A first attempt (`logs/c0_repro_attempt1_usage_error.log`) failed on my own pytest flag (`-p no:cacheprovider` conflicts with the repo `cache_dir` config); it contains no result.

**Is it in scope of the TOOL-LIFE-X acceptance? Yes.**
- Roadmap §3.0.5 states, as a certification requirement for this stage: "**No Tool invocation may remain RUNNING without end**" (`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md:866`). Its "Worker shutdown" row adds "no permanent RUNNING", and "No eternal RUNNING without a live owner".
- The acceptance record claims TL15, "zero unresolved mandatory Tool lifecycle blockers". The reconciliation names no exclusion that covers a post-close or submit-failure edge.
- OD-TLX-C expressly did not sanction this: it "does NOT sanction any durable external-operation record left RUNNING after a failed pool.submit", and it names it as an open condition.
- It pre-dates Batch C and the baseline, so it was present at the original self-published acceptance and was not caught.

**Does it make the verdict at best CONDITIONAL? Yes.**
- It contradicts a verbatim roadmap invariant that the stage claims to have certified, and no declared, Operator-signed exclusion covers it. PASS_SCOPED requires bounded exclusions that match the stage's own declared scope, so it is not available.
- The record stays `RUNNING` for the lifetime of the store. Retry of that operation id is blocked and recovery gates report "still RUNNING". There is no sweeper: `ensure_terminal_recorded` has zero callers, and the orphan reconcile is a documented no-op after Batch C. (These last points come from the C0 analyst's static analysis, which I treat as a claim; I checked the guard asymmetry myself.)

**Bounds, stated fairly:**
- The caller-visible result is correct, because the executor did not run.
- The trigger needs an external-operation store wired **and** a call on an already-closed invoker (a shutdown or lifecycle edge, or reuse of a closed invoker).
- The in-memory store dies with the process.
- The durable SQLite store variant was **not** run, by C0 or by me, so its behaviour is unknown.
- Other exceptions between `mark_running()` and `future.result()` in the same branch (for example `bind_future`) share the gap by inspection. Only the post-close case was run.
- No production wiring check was performed (see not_reproduced).

These bounds argue for severity MEDIUM, not for dismissal. The scope ruling above does not depend on severity.

**Resolution paths (not decided by this audit):**
- (a) A product fix under a new ticket, which is out of scope here: guard the `None` branch like the boundary branch (terminalise the record if `submit` or `bind_future` fails), plus a regression test, then a replay at a new exact SHA.
- (b) An Operator-signed, scoped exclusion or known-limitation record. It would need the durable-store variant stated explicitly, and it must be a real Operator decision, not delegated inference.

### 3.5 Independence of the original acceptance (carried forward, re-checked)
The only record of the "external independent review (ChatGPT) @ 1548640708" is text inside `TOOL_LIFE_X_PARENT_FORMAL_CLOSURE_RECONCILIATION.md` (lines 9, 26, 133). The other TOOL-LIFE-X hit, the roadmap, only repeats it. I found no transcript, findings list or reviewer identity in the tree. Nothing was reconstructed. The fix wave does not touch this, and OD-TLX-D leaves it out of scope. **F-03 remains OPEN.**

### 3.6 FRZ and IIAC
- The merge range adds no `docs/` change except the 4 governance ticket and plan files.
- The FRZ checklist differs from the baseline only by changes from other stages: OD-IC-01, IIAC Session A, BND-03 and W7-03 (FRZ-EVOL-01..10 promotion). No TOOL-LIFE-X-owned FRZ row is promoted by this stage or by the fix wave.
- The IIAC record and matrix additions come from other stages. I did not touch them.
- This audit promotes nothing.

## 4. Test replay (one pytest at a time, heavy-test mutex held, `-p no:xdist`, no `-n`)
| # | Set | Command (abridged) | Result | Log |
|---|---|---|---|---|
| 1 | Batch B L2 (4 modules, including `test_tools_side_effect_safety.py`) | `uv run pytest -p no:xdist -W default -v <4 mods>` | **64 passed**, 0 failed, rc 0 | `logs/batchB.log` |
| 2 | F8 inventory (12 modules named in R4 FINAL_ROOT_CAUSE §5) | `uv run pytest ... <12 mods>` | **67 passed** (67/67), 0 failed, 0 skipped, rc 0 | `logs/f8.log` |
| 3 | L3/TL11 set, `--extra mcp`: **18 candidate modules, a superset of the 17** | `uv run --extra mcp pytest ... <18 mods>` | **230 passed**, 0 failed, 0 skipped, rc 0, 356 s | `logs/l3_tl11_mcp.log` |
| 4 | F3 streamable-HTTP module, whole, 3 consecutive runs | `uv run --extra mcp pytest -q <module>` x3 | **5 passed x3** | `logs/f3_module_run{1,2,3}.log` |
| 5 | C0 repro (throw-away copy) | `uv run pytest -s --rootdir=. .tmp/c0/...` | **2 failed on the observation assertion = C0 reproduced** | `logs/c0_repro_at_postfix.log` |

- **Batch B arithmetic.** 63 historical + 1 new N2 test = 64.
- **17-module set.** The mission and the prior audit give the "17-module set" only by description ("L3-R1/R2/R6–R10, host bootstrap, TL11 MCP, F3 physical, detachment"), without enumerating it. I ran 18 modules. Arithmetic check: 230 total minus 11 from `test_mcp_trust_a_principal_binding.py` is 219, which equals the prior 217 plus the 2 new F3 tests. So the prior 17 is, by inference, my 18 minus `trust_a`. I did not execute the prior 17 as a separate set, and that set was not reproduced as a named list. Every module in the superset passed, so any reading of "17" is covered. See F-12.
- **F-07 (pytest not exiting).** All runs exited rc 0 with no hang, including the combined runs. I do not claim it is fixed, only that I did not observe it.
- Not run, by freeze: Docker PostgreSQL L3 (C10/C15 freeze, no Docker).

## 5. Status of every earlier finding at the post-fix SHA
| ID | Sev | Status |
|---|---|---|
| F-01 four red Batch B L2 tests | HIGH | **CLOSED** (tests green, safety assertions kept). The disclosure part is open: condition C-4. |
| F-02 F3 module-order failure | MEDIUM | **CLOSED** (in-order set run, plus 3 of 3 whole-module runs) |
| F-03 no independent review artifact | MEDIUM | OPEN (out of scope per OD-TLX-D) |
| F-04 exit criteria 7 and 8 narrowed via qualification records | MEDIUM | OPEN (OD-EC7; out of scope per OD-TLX-D) |
| F-05 P0 SHA citation slip | LOW | OPEN (documentation slip, not an evidence defect) |
| F-06 large production delta; unverifiable wave approvals | LOW | OPEN |
| F-07 pytest hang at exit | LOW | NOT OBSERVED in this audit's runs |
| F-08 stale in-doc status wording | LOW | OPEN |
| F-09 optional `mcp` extra needed | INFO | OPEN (documented here: `--extra mcp` was needed and used) |
| F-10 `invoker.py` drift | INFO | **RESOLVED** (section 3.1) |
| **F-11 C0: `RUNNING` record after post-close `submit` failure** | MEDIUM, blocking for PASS | **NEW, OPEN, reproduced** (section 3.4) |
| F-12 "17-module" set not enumerated in the stage docs | LOW | NEW, OPEN |
| F-13 OD-TLX-D disclosure addendum not yet in the tree | LOW (administrative) | NEW, OPEN |

## 6. Conditions to reach PASS or PASS_SCOPED
1. **C-3 (F-11):** resolve C0 with one of the two paths in section 3.4. Either (a) a product fix with a regression test and a replay at a new exact SHA, or (b) an Operator-signed, scoped exclusion or known-limitation record that states the durable-store variant explicitly.
2. **C-4 (F-13):** publish the OD-TLX-D item 1 disclosure addendum to the reconciliation record. It must state that the 4 tests were red from `a5f813a2c` to the baseline, the root cause (stale tests vs Batch C ABANDONED), that "63 passed" is valid only at `60a3ad53b`, and the fix at the post-fix SHA. The original verdict is not rewritten. Record `test_tools_side_effect_safety.py` in the standing mandatory L2 inventory (OD-TLX-D item 2).

Items that stay open for any final closure signature, and are not conditions on code: F-03 and F-04 (Operator-level).

## 7. Verified clean
Batch B, F8 and L3/TL11 replays green at the exact post-fix SHA. The fix wave is test-only and weakens no assertion. The `invoker.py` drift is additive and fail-closed. No TOOL-LIFE-X FRZ promotion. The IIAC anchor and impact matrix were untouched by this audit.
