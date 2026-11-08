## PLAN `T-IGX-030-A` — TESTS-ONLY integration of two unpublished TOOL-LIFE-X corrective test commits (Operator DR-7 option 1, re-scoped by DR-12 = option B)

Campaign: Integrax Historical Certification Closure, workstream IMPL-TLX. Implementer != auditor != certifier: this plan claims no PASS, no closure, no FRZ promotion.

Re-plan note (DR-12 = B): the product-path work was split out. `T-IGX-030-A` now touches NOTHING under `intergrax/`, `agents/`, `applications/`, so IIAC ACT-N11 must stay green. The `invoker.py` fix (`29c387879`) and its red-first test (`98b4bf514`) moved to HELD ticket `T-IGX-030-C`; the F-013 enum move is HELD ticket `T-IGX-030-B`.

### 1. Scope

Re-create, under the new ID `T-IGX-030-A`, exactly the content of two local unpublished commits (branch `cert-wave/publish-v1-with-tlx`, not on origin) on top of `origin/development` (`492e361c0`):

| New commit | Source commit | Old ticket ID | Content |
|---|---|---|---|
| C1 | `0adb73380` | `T-IGX-023-A` (collides with the published EVOL-X W7-06 ticket, PR #398) | test-only: reconcile 4 stale side-effect-safety tests with ABANDONED semantics (+1 new test N2) |
| C2 | `10089121e` | `T-IGX-023-B` | test-only: streamable-HTTP parent-cancel test made event-driven (+2 named negative tests) |

The old ticket/plan files carried by the source commits (`governance/tickets/T-IGX-023-A.md`, `T-IGX-023-B.md`, `plans/PLAN_T-IGX-023-A.md`, `plans/PLAN_T-IGX-023-B.md`) are NOT brought over (`T-IGX-023-A.md` already exists on origin with different content). Their content is summarised in section 5.

### 2. Exact files (edit list for phase 2; nothing else may change)

| Commit | File | Change |
|---|---|---|
| C1 | `tests/unit/runtime/tools/test_tools_side_effect_safety.py` | blob identical to that at `0adb73380` (+48/-12 vs. HEAD) |
| C2 | `tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py` | blob identical to that at `10089121e` (about +121/-12 vs. HEAD) |

Pre-verified in phase 1 (read-only): `git diff HEAD <parent> -- <file>` is empty for both files, i.e. HEAD holds the exact pre-change blobs, so the post-change blobs apply without a merge. Phase 2 re-checks this and STOPs if HEAD moved.

Mechanics (no cherry-pick, no checkout, no rebase): per commit, `git restore --source=<sha> --staged --worktree -- <file>`, assert `git rev-parse :<file>` equals `git rev-parse <sha>:<file>`, then commit. Trailer on both commits: `Ref: governance/tickets/T-IGX-030-A.md`; body line `Source-Commit: <sha>`.

Commit subjects (original subjects, ID replaced):
- C1 `test(tool-life-x): reconcile four stale side-effect-safety tests with ABANDONED semantics (T-IGX-030-A)`
- C2 `test(f3): make streamable-HTTP parent-cancel test event-driven (T-IGX-030-A)`

Decision on stale IDs inside the blobs: the C1 blob contains 5 comments citing `T-IGX-023-A` (pre-rename name). They are kept byte-identical so the auditor can prove mechanical equality by blob hash; they are historical labels. The mapping `T-IGX-023-A -> T-IGX-030-A (C1)`, `T-IGX-023-B -> T-IGX-030-A (C2)`, `T-IGX-024-A -> T-IGX-030-C (C3)` is recorded in this plan and the ticket. A comment-only rename is a separate follow-up if the Operator wants it.

### 3. Design choice and alternatives rejected

Chosen: path-scoped restore of the two exact post-change blobs, two commits in original order, one ticket, test-only.

Rejected: `git cherry-pick` (carries old ticket/plan files, add/add conflict with origin `T-IGX-023-A.md`, old IDs in messages); rebuilding by hand (drift from the already-reviewed content); squashing (loses per-commit traceability); keeping old IDs (collision, forbidden by DR-7); strengthening the tests beyond the source (not mechanical; a separate ticket); including any `intergrax/` hunk (violates DR-12 = B).

### 4. Blast radius and compatibility

Two test files; no importers except `98b4bf514`'s test (HELD `T-IGX-030-C`), which imports helpers (`CountingExecutor`, `_register_tool`, `_request`, `_side_effect_safety_governed_scope`, `_state_with_enforce_allow`) from the C1 file: those names are not changed by C1, but it means `T-IGX-030-C` can only land on a branch that already contains C1. No `intergrax/` change, no gate, no inventory, no IIAC record changes. C1 locks in ABANDONED as the CURRENT conservative fail-closed result for a post-`close()` submit failure; it does not sanction a durable external-operation record left RUNNING (that is what HELD `T-IGX-030-C` addresses).

### 5. Assertion-change ledger for C1 (`0adb73380`) — for the auditor to re-check

File `tests/unit/runtime/tools/test_tools_side_effect_safety.py`. Authority cited by the source commit: OD-TLX-A, OD-TLX-C Option A (delegated, process-enforced; the signed record is outside the repo and is UNVERIFIED by this implementer). Code basis: Batch C `a5f813a2c` (`intergrax/runtime/tools/tool_operation_recovery.py`).

| # | Test | Old assertion | New assertion | Kept unchanged (same test) | Why this does not weaken the gate |
|---|---|---|---|---|---|
| 1 | `test_side_effect_timeout_marks_uncertain_and_blocks_replay` | `error_code == RuntimeErrorCode.TIMEOUT` | `error_code == RuntimeErrorCode.ABANDONED` and (added) `error_message == "abandoned"` | `result.success is False`, `error is not None`, ledger `UNCERTAIN`, replay raises `InvocationUncertaintyError` | one exact enum value replaced by another exact value (not a set, not `in`), plus a new exact-message assertion; no-blind-replay and ledger-UNCERTAIN properties untouched |
| 2 | `test_mapped_validation_error_after_executor_marks_uncertain` | `error_code == RuntimeErrorCode.VALIDATION_ERROR` | `error_code == RuntimeErrorCode.ABANDONED` | `external_effects["n"] == 1`, ledger `UNCERTAIN`, replay raises `InvocationUncertaintyError`, `n == 1` after replay | same exact-value swap; effect-count and replay-block assertions kept. Residual: the contract `error_mapping` code is no longer surfaced for an admitted side-effecting failure — that is the semantic decision OD-TLX-A, not a test artefact |
| 3 | `test_close_shuts_down_execution_pool` | `"cannot schedule new futures after shutdown" in error_message` | `error_code == ABANDONED`, `error_message == "abandoned"`, (added) ledger of key `r6a-close-pool-new` is `UNCERTAIN` (unpacking `_` renamed `store`) | `not after_close.success`, `error is not None`, `executor.calls == 1` | substring of a CPython `ThreadPoolExecutor` message replaced by exact code + exact message + a NEW ledger assertion: stronger on the contract, no longer coupled to a stdlib string. Residual: asserts the CURRENT conservative fail-closed result |
| 4 | `test_replay_before_close_preserves_exactly_once` | same substring | `error_code == ABANDONED`, `error_message == "abandoned"`, (added) ledger of `r6a-replay-before-close-new` is `UNCERTAIN` | `not after_close.success`, `error is not None`, `executor.calls == 1` | as #3 |
| 5 | NEW `test_post_close_uncertain_key_replay_is_blocked_without_executor_call` (N2) | n/a | asserts ABANDONED, ledger `UNCERTAIN`, `executor.calls == 0`; replay raises `InvocationUncertaintyError`, `executor.calls == 0`, ledger still `UNCERTAIN` | n/a | pure addition (no blind replay of a poisoned key after `close()`) |
| 6 | non-semantic | import block order (`testing_support`, `pydantic` moved), one signature wrap | — | — | formatting only (the source commit says the pre-commit ruff gate forced it). Phase 2 re-runs ruff on the final file; if base config disagrees, apply ruff's output and report |

N1 of the old plan was not added (Operator Q-A3: no). C2 (`10089121e`) change summary, same standard: `started` becomes `threading.Event`; cancel is triggered from `invoker.started` instead of `asyncio.sleep(0.06)`; `step_id` added to the arguments; new asserts `invoker.results` non-empty and `_is_error_text(catalog) is None`; the accepted outcome set stays `{"execution_cancelled","execution_admission_denied"}` (not widened); two new named negative tests (cancel-before-admission, unknown task_id -> `missing_execution_binding`, child never runs); import/format hunks ruff-forced; no assertion removed.

### 6. Test contract (phase 2; ONE pytest/ruff/uv process per heavy-lock hold, sequential; pytest's OWN exit code captured via `; echo EXIT=$?` into the log, never through a pipe; logs under `.tmp/session/t-igx-030-a/`; no `--basetemp` outside the repo; reuse the existing `.venv`, build no other)

1. After C1: `uv run --extra dev-ci pytest -p no:xdist tests/unit/runtime/tools/test_tools_side_effect_safety.py -q`
2. After C2: `uv run --extra dev-ci --extra mcp pytest -p no:xdist tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py -q` — whole module alone, 3 separate consecutive invocations (the flake needs an earlier test in the same process).
3. Final tree, one combined run: `uv run --extra dev-ci --extra mcp pytest -p no:xdist tests/unit/runtime/tools/test_tools_side_effect_safety.py tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_retry_idempotency_unknown_outcome.py tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_real_governance_authorization.py tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py tests/unit/runtime/nexus/tools/test_tool_life_x_batch_c_stalled_abandoned_recovery.py -q`
4. Lint: `uv run --extra dev-ci ruff check <2 files>` then `uv run --extra dev-ci ruff format --check <2 files>`.
5. Diff guard: `git diff --name-only 492e361c0..HEAD` lists only the 2 test files plus `T-IGX-030-*` tickets/plans (and nothing under `intergrax/`, `agents/`, `applications/`); blob guard: `git rev-parse HEAD:<f>` equals `git rev-parse <source-sha>:<f>` for both files.
6. **ACT-N11 pre-check (requested by MASTER), after C1 and C2 are both committed, executed ONCE each, sequentially, each under its own heavy-lock hold, own exit code captured:**
   a. `uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force` — expected exit 0 (IN_FORCE; product-path diff vs `ae467c8c4` is 0 files).
   b. `uv run --extra dev-ci pytest -p no:xdist tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_evol_x_real_repo_fresh_clone.py -q` — expected all green.
   If either is red: STOP, do not fix, do not touch the gate or IIAC records, report to MASTER (it would mean the tests-only premise is wrong). Not executed in phase 1.
7. `igx_gov ticket` if feasible.
The broad replay (Batch B L2 inventory, F8 67-test inventory, L3/TL11 17 modules with `--extra mcp`, TL14 gates) belongs to the independent auditor at the post-fix SHA (HIST-01 N6). W5 b2/c/d are not part of this contract; if any inventory includes them they run alone.

### 7. Rollback

`git revert` of C2 and/or C1; no data migration. The branch is unpublished; MASTER owns publication.

### 8. Open questions

None blocking approval. (Resolved by the Operator: Q-A1 `98b4bf514` goes to `T-IGX-030-C`; Q-A2 ACT-N11 handled by DR-12 = B split; Q-A3 no N1.) Residual for the auditor: the delegated OD-TLX-A/C signatures cannot be re-verified from the repo.

### 9. Out of scope

Everything under `intergrax/`, `agents/`, `applications/` (incl. `invoker.py` fix `29c387879` -> HELD `T-IGX-030-C`; F-013 enum move -> HELD `T-IGX-030-B`); `98b4bf514`; `test_sec_ent.py` (F-006) and the W5 b2/c/d isolation batch (F-016); N1; ENT-AUDIT-X artefacts; IIAC records/activation and the gate; roadmap/FRZ edits; FRZ promotion; PASS/closure claims; Docker/vendor/production config (C10/C15); pushing, merging, rebasing, PRs.
