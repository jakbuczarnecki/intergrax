## PLAN `T-IGX-030-C` — HELD — product fix `29c387879` (no-boundary `pool.submit` failure terminalises the external-operation record) together with its red-first test `98b4bf514`

**HELD - ACT-N11 sequencing (DR-12=B): do not approve or implement until MASTER confirms the IIAC re-audit/re-activation batch.**

Campaign: Integrax Historical Certification Closure, workstream IMPL-TLX. Implementer != auditor != certifier: this plan claims no PASS, no closure, no FRZ promotion. Authority: Operator DR-7 option 1 (C3) extended by Operator Q-A1 = YES (C4 joins, same batch because C4 is red without C3), sequencing by DR-12 = B.

### 1. Scope

Re-create under the new ID `T-IGX-030-C` (formerly `T-IGX-024-A`, which collides with the ENT-AUDIT-X stream) on top of the branch state that already contains `T-IGX-030-A`:

| New commit | Source commit | Content |
|---|---|---|
| C3 | `29c387879` | PRODUCT `intergrax/runtime/nexus/tools/invoker.py` +5/-1: in the `dep_boundary is None` branch of `RuntimeToolInvoker._execute_once`, wrap only `self._execution_pool.submit(...)` in `try/except BaseException: ext_op.mark_failed(); raise`, mirroring the boundary branch. `mark_running()` and `bind_future()` stay outside the guard. No new error code, state, API, result mapping or ledger change (OD-TLX-C Option A intact). |
| C4 | `98b4bf514` | TEST-ONLY new file `tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py` (+330): T1-T4 pin FAILED-terminalisation on post-close / generic / retry>1 submit failure; N1 pins that a `bind_future` failure after a real submit is NOT marked FAILED (residual R-1). |

The old ticket/plan files (`governance/tickets/T-IGX-024-A.md`, `plans/PLAN_T-IGX-024-A.md`, commit `8a1f71de1`) are NOT brought over (ID in use on origin by a different ticket).

### 2. Exact files (phase 2 edit list)

| File | Change |
|---|---|
| `intergrax/runtime/nexus/tools/invoker.py` | blob identical to that at `29c387879` |
| `tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py` | blob identical to that at `98b4bf514` (new file) |

Phase-1 verification: HEAD's `invoker.py` is byte-identical to its pre-fix parent (`git diff HEAD 98b4bf514^ -- intergrax/runtime/nexus/tools/invoker.py` empty); the C4 file does not exist at HEAD. The C4 file imports helpers (`CountingExecutor`, `_register_tool`, `_request`, `_side_effect_safety_governed_scope`, `_state_with_enforce_allow`) from `tests/unit/runtime/tools/test_tools_side_effect_safety.py`: this ticket therefore requires `T-IGX-030-A` C1 to be present on the branch first (hard dependency; STOP if absent). The C4 blob keeps its in-file ID `T-IGX-024-A (C0)` (historical label, byte-identical for blob-equality proof).

Mechanics as in `T-IGX-030-A` (`git restore --source=<sha> --staged --worktree -- <file>`, blob-equality assertion, no cherry-pick/checkout/rebase). Commit order: C3 then C4 (never a red commit on the branch). Trailer `Ref: governance/tickets/T-IGX-030-C.md` on both; body `Source-Commit: <sha>`. Subjects:
- C3 `fix(nexus-tools): terminalise external-operation record when pool.submit fails without dependency boundary (T-IGX-030-C)`
- C4 `test(tool-life-x): red-first tests for no-boundary pool.submit failure terminal state (T-IGX-030-C)`

### 3. Design choice and alternatives rejected

Chosen: mechanical restore, one cohesive batch (C3 + C4) because C4 is red without C3 and C3 has no other regression proof. Rejected: C3 alone (no proof of the new branch; Operator rejected by Q-A1); C4 before C3 as a committed red commit (breaks bisect); `git cherry-pick` (old ticket/plan files, old IDs); hand-rewriting the guard (drift from the reviewed content); guarding `bind_future` too (changes OD-TLX-C Option A semantics; C4's N1 pins the opposite).

### 4. Blast radius and compatibility

Behaviour change is confined to the case where `ThreadPoolExecutor.submit` raises in the no-boundary branch (after `close()` or resource failure): the durable external-operation record goes to FAILED before the exception propagates, instead of staying RUNNING. Importers of `invoker.py`: recompute in phase 2 with `git grep -n "nexus.tools.invoker" -- intergrax agents applications tests scripts` and record. No signature/result/ledger change; OD-TLX-C Option A results asserted by `T-IGX-030-A` C1 tests (post-close -> `ABANDONED`, ledger `UNCERTAIN`) must stay green with C3 present — that interaction is why the side-effect module is re-run at the final tree.

**ACT-N11 (known, by design):** C3 changes a path under `intergrax/`; after it, `verify_iiac_policy_activation --require-in-force` fails (check ACT-N11, product path changed after `ae467c8c4`) and the governance-fast tests that run it go red. Hence this ticket is HELD until MASTER confirms the IIAC re-audit/re-activation batch. The gate is not relaxed.

### 5. Test contract (phase 2, only after the HOLD is lifted; ONE pytest/ruff/uv process per heavy-lock hold; own exit code captured; logs in `.tmp/session/t-igx-030-c/`; reuse the existing `.venv`)

1. Red proof: restore the C4 file only (uncommitted), `uv run --extra dev-ci pytest -p no:xdist tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py -q` — expected T1-T4 failed, N1 passed (record).
2. Restore `invoker.py` (C3), same command — expected all green. Then commit C3, then C4.
3. Final tree, one combined run: the C4 file + `tests/unit/runtime/tools/test_tools_side_effect_safety.py` + the Batch B (2 files) and Batch C recovery modules listed in `PLAN_T-IGX-030-A` section 6.3 + `tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_policy.py`, `test_runtime_tool_invoker_dependency_admission.py`, `test_tool_life_x_batch_a_deadline_hierarchy.py`, `test_harness_w4_r1_production_tool_admission_behavior.py` (`--extra mcp` if required).
4. Lint on the 2 files (`ruff check`, then `ruff format --check`, sequential).
5. Diff guard: only the 2 files plus ticket/plan files; blob guard against `29c387879` and `98b4bf514`.
6. Informational, expected RED, do not fix: `verify_iiac_policy_activation --require-in-force` (ACT-N11), recorded verbatim.
The invoker-delta review against TL1-TL14 and the broad replay belong to the independent auditor (HIST-01 N6).

### 6. Rollback

`git revert` of C4 then C3; no data migration.

### 7. Open questions

None for the plan content. Sequencing question owned by MASTER/Operator: when the IIAC re-audit/re-activation batch happens relative to `T-IGX-030-B` and this ticket (both product-path; file-disjoint: `invoker.py` here vs `error_codes.py`/`tool_operation_recovery.py`/new contract module in B; `invoker.py` imports `RuntimeErrorCode` through the alias, so either order works and one IIAC cycle can cover both).

### 8. Out of scope

`T-IGX-030-A` content; `T-IGX-030-B`; any other `intergrax/` hunk; N1 of old plan 023-A; a fix for `bind_future` failures (R-1 stays an accepted residual); `test_sec_ent.py` (F-006), W5 b2/c/d (F-016); IIAC records/gate; roadmap/FRZ edits; PASS/closure claims; Docker/vendor/production config (C10/C15); push, merge, rebase, PR.
