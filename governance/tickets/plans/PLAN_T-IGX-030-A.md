## PLAN `T-IGX-030-A` — mechanical integration of the three unpublished TOOL-LIFE-X corrective commits (Operator decision DR-7 option 1)

Campaign: Integrax Historical Certification Closure, workstream IMPL-TLX. Implementer != auditor != certifier: this plan claims no PASS, no closure, no FRZ promotion.

### 1. Scope

Re-create, under the new ticket ID `T-IGX-030-A`, exactly the content of three local, unpublished commits (branch `cert-wave/publish-v1-with-tlx`, not on origin) on top of the current `origin/development` (`492e361c0`):

| New commit (this branch) | Source commit | Old ticket ID | Content |
|---|---|---|---|
| C1 | `0adb73380` | `T-IGX-023-A` (collides with the published EVOL-X W7-06 ticket, PR #398) | test-only: reconcile 4 stale side-effect-safety tests with ABANDONED semantics (+1 new test N2) |
| C2 | `10089121e` | `T-IGX-023-B` | test-only: streamable-HTTP parent-cancel test made event-driven (+2 named negative tests) |
| C3 | `29c387879` | `T-IGX-024-A` (collides with the ENT-AUDIT-X stream) | PRODUCT: `intergrax/runtime/nexus/tools/invoker.py` +5/-1 |

The old ticket/plan files that these commits carry (`governance/tickets/T-IGX-023-A.md`, `T-IGX-023-B.md`, `plans/PLAN_T-IGX-023-A.md`, `plans/PLAN_T-IGX-023-B.md`) are NOT brought over: `T-IGX-023-A.md` already exists on origin with different content. Their normative text is cited by reference (source SHAs above) and summarised in section 5.

### 2. Exact files (edit list for phase 2; nothing else may change)

| Commit | File | Change |
|---|---|---|
| C1 | `tests/unit/runtime/tools/test_tools_side_effect_safety.py` | blob identical to that at `0adb73380` (+48/-12 vs. HEAD, see section 5) |
| C2 | `tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py` | blob identical to that at `10089121e` (+~121/-12 vs. HEAD) |
| C3 | `intergrax/runtime/nexus/tools/invoker.py` | blob identical to that at `29c387879` (+5/-1) |

Pre-verified in phase 1 (read-only): `git diff HEAD <parent> -- <file>` is empty for all three files, i.e. HEAD holds the exact pre-change blobs, so the post-change blobs apply with no merge. Phase 2 MUST re-check this and STOP if HEAD moved such that it is no longer true.

Mechanics (no cherry-pick, no checkout, no rebase): per commit, `git restore --source=<sha> --staged --worktree -- <file>`, then assert `git rev-parse :<file>` equals `git rev-parse <sha>:<file>`, then commit with the message below. Commit trailer on all three: `Ref: governance/tickets/T-IGX-030-A.md` (C3 touches `intergrax/`; C1/C2 carry it for traceability).

Commit subjects (original subjects with the ID replaced; original authorship/date are recorded in the body as `Source-Commit: <sha>`):
- C1 `test(tool-life-x): reconcile four stale side-effect-safety tests with ABANDONED semantics (T-IGX-030-A)`
- C2 `test(f3): make streamable-HTTP parent-cancel test event-driven (T-IGX-030-A)`
- C3 `fix(nexus-tools): terminalise external-operation record when pool.submit fails without dependency boundary (T-IGX-030-A)`

### 3. Design choice and alternatives rejected

Chosen: path-scoped restore of the three exact post-change blobs, three commits in the original order (C1, C2, C3), one ticket.

Rejected:
- `git cherry-pick`: would carry the old ticket/plan files (add/add conflict with origin `T-IGX-023-A.md`) and old IDs in messages; needs manual surgery anyway.
- Rebuild the changes by hand: risks drift from the already-audited content; the point of DR-7 is mechanical integration.
- Squash into one commit: loses the test-only vs product-path separation that the auditor (and the `Ref:` trailer gate) needs.
- Keep old IDs: collides with published tickets (DR-7 forbids).
- Rewrite the tests more strictly than the source: out of scope for a mechanical integration; any strengthening goes to a separate ticket.

### 4. Open scope question (needs Operator decision BEFORE approval; default = narrow)

DR-7 names three commits. HIST-01 handoff N5 lists a fourth artefact: `98b4bf514` "red-first tests for no-boundary pool.submit failure terminal state" (new file `tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py`, +330, test-only), which is the ONLY regression proof for C3. Without it, C3 lands as a product change with no test that exercises its `except BaseException: ext_op.mark_failed(); raise` path (C1's N2 test covers the post-`close()` path, not this branch). This plan does NOT include `98b4bf514` (DR-7 does not authorise it). Recommended: Operator extends DR-7 to include it as C4 (test-only, no further product risk); then this plan is re-digested. Until then C3 ships un-proven by a dedicated test and the audit handoff must say so.

### 5. Assertion-change ledger for C1 (`0adb73380`) — for the auditor to re-check

File `tests/unit/runtime/tools/test_tools_side_effect_safety.py`. Authority cited by the source commit: OD-TLX-A, OD-TLX-C Option A (delegated, process-enforced; signed record outside the repo; I cannot re-verify it and record it as UNVERIFIED here). Basis in code: Batch C `a5f813a2c` introduced ABANDONED/STALLED (`intergrax/runtime/tools/tool_operation_recovery.py`).

| # | Test | Old assertion | New assertion | Kept unchanged (same test) | Why this does not weaken the gate |
|---|---|---|---|---|---|
| 1 | `test_side_effect_timeout_marks_uncertain_and_blocks_replay` | `error_code == RuntimeErrorCode.TIMEOUT` | `error_code == RuntimeErrorCode.ABANDONED` and (added) `error_message == "abandoned"` | `result.success is False`, `error is not None`, ledger status `UNCERTAIN`, replay raises `InvocationUncertaintyError` | single exact enum value replaced by another single exact value (not a set or `in`), plus one new exact-message assertion; the safety properties (no blind replay, ledger UNCERTAIN) are untouched |
| 2 | `test_mapped_validation_error_after_executor_marks_uncertain` | `error_code == RuntimeErrorCode.VALIDATION_ERROR` | `error_code == RuntimeErrorCode.ABANDONED` | `external_effects["n"] == 1` (effect ran exactly once), ledger `UNCERTAIN`, replay raises `InvocationUncertaintyError`, `n == 1` after replay | same exact-value swap; effect-count and replay-block assertions kept. Residual: contract `error_mapping` no longer surfaces for an admitted side-effecting failure — this is the semantic decision (OD-TLX-A), not a test artefact |
| 3 | `test_close_shuts_down_execution_pool` | `"cannot schedule new futures after shutdown" in error_message` | `error_code == ABANDONED`, `error_message == "abandoned"`, (added) ledger of key `r6a-close-pool-new` is `UNCERTAIN` (needs `_` renamed to `store` in the `_idempotent_invoker` unpacking) | `not after_close.success`, `error is not None`, `executor.calls == 1` | substring of a CPython `ThreadPoolExecutor` message replaced by exact code + exact message + a NEW ledger assertion: strictly stronger on the contract, no longer coupled to a stdlib string. Residual: the code asserts the CURRENT conservative fail-closed result; see section 7 |
| 4 | `test_replay_before_close_preserves_exactly_once` | same substring | `error_code == ABANDONED`, `error_message == "abandoned"`, (added) ledger of `r6a-replay-before-close-new` is `UNCERTAIN` | `not after_close.success`, `error is not None`, `executor.calls == 1` | as #3 |
| 5 | NEW `test_post_close_uncertain_key_replay_is_blocked_without_executor_call` (N2) | n/a | asserts ABANDONED, ledger `UNCERTAIN`, `executor.calls == 0`, replay raises `InvocationUncertaintyError`, `executor.calls == 0`, ledger still `UNCERTAIN` | n/a | pure addition (guards against blind replay of a poisoned key after `close()`) |
| 6 | non-semantic | import block order (testing_support/pydantic moved), one signature wrap | — | — | formatting only (the source commit states the pre-commit ruff gate forced it). Phase 2 re-runs ruff on the final file; if the base config sorts differently, apply ruff's output and say so |

N1 of the old plan was not added by the source commit (equivalent `test_runtime_tool_invoker_timeout` exists) — the auditor should confirm that claim.

C2 (`10089121e`) change summary, same standard: `started` becomes `threading.Event`; cancel is triggered from `invoker.started` instead of `asyncio.sleep(0.06)`; step_id added to the arguments; new asserts `invoker.results` non-empty and `_is_error_text(catalog) is None`; the accepted outcome set stays `{"execution_cancelled","execution_admission_denied"}` (not widened); two new named negative tests (cancel-before-admission, unknown task_id -> `missing_execution_binding`, child never runs). Import/format hunks are ruff-forced. No assertion is removed.

### 6. Blast radius

- C1/C2: single test files; no importers. No `intergrax/` change.
- C3: `RuntimeToolInvoker._execute_once`, branch `dep_boundary is None` only. Importers of `invoker.py`: found by `git grep` in phase 2 (list recorded in the audit handoff); behaviour change is confined to the case where `ThreadPoolExecutor.submit` raises (after-`close()` or resource failure): the external-operation record is moved to FAILED before re-raising, instead of staying RUNNING. No new error code/state/API/ledger change.
- **ACT-N11 (critical, see ticket and handoff):** C3 changes a path under `intergrax/`. `verify_iiac_policy_activation --require-in-force` (check ACT-N11, `scripts/governance/iiac_activation_gate.py`) fails when any `intergrax/`, `agents/`, `applications/` path differs from the activation record's audit SHA `ae467c8c4`. At HEAD the product-path diff is currently 0 files. After C3 it is 1 file. Expected consequence: ACT-N11 red and the governance-fast tests that run the gate (`tests/unit/governance/test_iiac_activation_gate.py`, `test_iiac_negative_guards.py`, `test_evol_x_real_repo_fresh_clone.py`) go red. This plan does not and must not relax the gate; it needs an Operator decision (section 9, Q2).

### 7. Compatibility decision

No alias/re-export concerns (no symbol moved). C1 locks in ABANDONED behaviour for post-`close()` submit failure as the CURRENT fail-closed result; it does not sanction a durable external-operation record left RUNNING (C3 addresses exactly the no-boundary branch; the boundary branch already had the guard).

### 8. Test contract (phase 2; ONE pytest/ruff process at a time, each under `<campaign>\locks\heavy.lock`; capture pytest's OWN exit code with `; echo EXIT=$?` into the log, never through a pipe; logs under `.tmp/session/t-igx-030-a/`; no `--basetemp` outside the repo)

1. After C1: `uv run --extra dev-ci pytest -p no:xdist tests/unit/runtime/tools/test_tools_side_effect_safety.py -q`
2. After C2: `uv run --extra dev-ci --extra mcp pytest -p no:xdist tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py -q` — whole module alone, run 3 consecutive separate invocations (the original acceptance rule: the flake needs an earlier test in the same process).
3. After C3 (final tree): repeat step 1 once (module now also exercises C3 interaction), then one combined run: `uv run --extra dev-ci --extra mcp pytest -p no:xdist tests/unit/runtime/tools/test_tools_side_effect_safety.py tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_retry_idempotency_unknown_outcome.py tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_real_governance_authorization.py tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py tests/unit/runtime/nexus/tools/test_tool_life_x_batch_c_stalled_abandoned_recovery.py -q`
4. Lint: `uv run --extra dev-ci ruff check <3 files>` then `uv run --extra dev-ci ruff format --check <3 files>` (two sequential invocations).
5. Diff guard: `git diff --name-only 492e361c0..HEAD` must list only the 3 files above plus `governance/tickets/T-IGX-030-A.md`, its plan and (if approved) nothing else. Blob guard: for each file, `git rev-parse HEAD:<f>` equals `git rev-parse <source-sha>:<f>`.
6. Gate check (informational, expected red, DO NOT FIX): `uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force`, record the ACT-N11 line verbatim.
7. `igx_gov ticket` if feasible.
The broader replay (Batch B L2 inventory, F8 67-test inventory, L3/TL11 17 modules with `--extra mcp`, TL14 gates, review of the invoker delta against TL1-TL14) belongs to the independent auditor at the post-fix SHA (HIST-01 N6), not to the implementer. W5 b2/c/d, if they appear in any inventory, run alone; they are not part of this contract.

### 9. Open questions for the Operator

- Q1 include `98b4bf514` as C4 (recommended) or ship C3 without its dedicated proof test?
- Q2 ACT-N11: accept that C3 invalidates the IIAC activation (needs IIAC re-audit + non-delegable operator re-activation), or hold C3 until that path exists? The other two commits (tests only) do not touch product paths and are unaffected.
- Q3 does the Operator also want N1 (read-only tool timeout stays TIMEOUT) added, since the claim "equivalent exists" is unverified? (default: no, mechanical integration only).

### 10. Rollback

`git revert` of C3 (and/or C2, C1) individually; no data migration. The branch is unpublished; MASTER owns publication.

### 11. Out of scope

`test_sec_ent.py` (F-006) and the W5 b2/c/d isolation batch (F-016); any `intergrax/` change other than the `invoker.py` hunk of C3; ENT-AUDIT-X artefacts; IIAC records/activation; roadmap/FRZ edits; the F-013 nexus-encapsulation fix (ticket `T-IGX-030-B`); FRZ promotion; PASS/closure claims; Docker/vendor/production config (C10/C15); pushing, merging, opening PRs.
