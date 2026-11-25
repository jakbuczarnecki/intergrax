## PLAN — `T-IGX-024-A` (proposed id): terminalise the durable external-operation record when `pool.submit` fails in the no-boundary branch of `RuntimeToolInvoker._execute_once` (C0-TLX correction)

### 1. What this is (and is not)
A **defect correction**, not new functionality. Roadmap §3.0.5 already requires "No Tool invocation may remain RUNNING without end" (lines ~857-866, 886). The code violates that rule in one branch; the sibling branch of the same function already implements the required behaviour. The change makes the first branch do what the second branch already does. No new `RuntimeErrorCode`, no new typed rejection, no new state, no new API, no change to result mapping, retry logic or ledger semantics.

**This changes product code (`intergrax/`) inside a certification campaign.** Why that is acceptable as a correction and not scope widening: (a) the defect is confirmed by a runtime reproduction (C0 finding, run5: 2/2 parametrisations leave a RUNNING record, executor never invoked); (b) it pre-dates Batch C (introduced with `mark_running()` in `e8264ff2e`) and is not a regression of any certified behaviour; (c) OD-TLX-C expressly left it unsanctioned and open ("does NOT sanction any durable external-operation record left RUNNING after a failed pool.submit"); (d) it needs its own ticket, approval and independent audit, because it is not covered by T-IGX-023-A/B (test-only). It does **not** adopt OD-TLX-C Option B/C and does not alter the signed Option A behaviour.

### 2. Verified facts (base `50ddc04fb`; `invoker.py` byte-identical to `2f6664d53`; `intergrax/` and `tests/` identical to `origin/development` `65ccf681a`)
- `intergrax/runtime/nexus/tools/invoker.py`, `_execute_once`, branch `if dep_boundary is None:` (L1905-1942):
  - L1914 `ext_op.mark_running()` (durable RUNNING when an external-operation store is wired; sets `_physical_execution_admitted`).
  - L1915 `future = self._execution_pool.submit(self._executor.execute, request)` — **no handler**.
  - L1916-1917 `bind_future(...)`.
  - The only exception handling starts at L1919 (`try: future.result(...)`), i.e. after submit. A `submit` exception leaves the function with the record RUNNING; `ext_op` is local, nothing else terminalises it (`ensure_terminal_recorded()` has no caller in `intergrax/runtime`; `reconcile_orphaned_running_operations` is a documented no-op).
- Boundary branch (L1972-1979) already has the guard: `try: mark_running(); submit(); bind_future()` / `except BaseException: ext_op.mark_failed(); dep_boundary.release_after_submit_failure(...); raise`.
- `close()` (L248-258) sets `_dependency_attempt_boundary = None`, so every post-close call takes the unguarded branch; a post-close `submit` raises `RuntimeError("cannot schedule new futures after shutdown")`. Any other `submit` failure in a no-boundary invoker shares the gap.
- `mark_failed()` → physical `FAILED`, `finalize_intent=False`. `assert_may_begin_physical_attempt` (external_operation_ownership.py:107-124) accepts a retry (attempt>1) when physical state is FAILED/CANCELLED/UNKNOWN and blocks it when RUNNING — so FAILED removes the permanent retry block, and is the same classification the boundary branch already uses for the same event.
- Observable result of the call is unchanged by this fix: ABANDONED/"abandoned" (side-effecting + coordinator) or TOOL_ERROR (non-side-effecting), executor not invoked, idempotency ledger UNCERTAIN (OD-TLX-C Option A).

### 3. Exact minimal change (recommended: C0 Option 1, split try)
File `intergrax/runtime/nexus/tools/invoker.py`, only the no-boundary branch, +4 / -0 lines (re-indent of the `submit` line only):

```diff
             ext_op.mark_running()
-            future = self._execution_pool.submit(self._executor.execute, request)
+            try:
+                future = self._execution_pool.submit(self._executor.execute, request)
+            except BaseException:
+                ext_op.mark_failed()
+                raise
             if self._tool_termination_port is not None and ext_op.operation_id is not None:
                 self._tool_termination_port.bind_future(ext_op.operation_id, future)
```

Design decisions (all binding for the implementer):
1. **Guard `submit` only.** Rationale for the split (differs from the boundary branch, which wraps `mark_running`+`submit`+`bind_future`): a `submit` failure proves the executor never ran → FAILED is truthful. A `bind_future` failure happens **after** a successful `submit`; the worker is already running, so FAILED would assert "never ran", which may be false. `mark_running()` stays outside the guard: if it fails the record never reached RUNNING, so the defect does not exist and no terminal write is needed (avoids a terminal write from a non-RUNNING state masking the original error).
2. **`bind_future` path and everything after it: unchanged.** Residual (open, explicitly NOT fixed here, register as R-1): an exception from `bind_future` after a successful `submit` still leaves RUNNING with a live worker and no awaiter. Correct handling would be a classification decision (UNKNOWN via `mark_unknown()` plus worker/future handling), outside OD-TLX-C Option A and outside a pure correction of the confirmed defect. Probability is low (in-memory dict operation in `ToolExecutorTerminationPort`). It is reported to the operator as a separate finding/optional follow-up ticket (see §9, OQ-2).
3. **Terminal state FAILED**, identical to the existing boundary-branch guard. Not NOT_STARTED, not UNKNOWN, not ABANDONED. If the operator wants a different classification that is a semantic ruling (OD-TLX-C Option B/C territory) and invalidates this plan.
4. No other file under `intergrax/` changes. `ToolExternalOperationAttempt`, stores, recovery gates, `close()`, result mapping, `_ToolPhysicalAttemptParentDeadlineExceeded` path are untouched.
5. If `mark_failed()` itself raises inside the handler it replaces the original exception (same behaviour as the existing boundary guard); accepted for parity, not changed here.

### 4. Tests (one new test file; no existing test file edited)
New file: `tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py`. Harness: follow `test_tool_life_x_r4_terminal_lifecycle_invoker.py` (direct `_execute_once`, `InMemoryExternalOperationStateStore`, `ProcessLocalExternalOperationOwner.mint()`, `dependency_attempt_boundary=None`, `canonical_governed_execution_scope`). Do not import from `test_tools_side_effect_safety.py` except where stated for T2 (import order matters; C0 repro run5 shows the working order: import that module first).

Positive (defect closed):
- **T1 post-close, store-level.** Call 1 (step "1") succeeds → its record SUCCEEDED. `invoker.close()`. Call 2 (step "2", new operation id — operation id is deterministic per tenant/tool/step/attempt, reusing a step is suppressed before submit and proves nothing) via `_execute_once` raises `RuntimeError`. Assert: call-2 record `physical_state is FAILED` and **not RUNNING**; call-1 record still SUCCEEDED; `executor.calls == 1` (counting handler; executor not invoked by call 2).
- **T2 post-close, public path, both C0 parametrisations** (side-effecting tool + `IdempotencyPreEffectCoordinator`; non-side-effecting tool without coordinator), via `invoke()` using the helpers proven in the C0 repro. Assert: call 2 not successful; `executor.calls == 1`; new record FAILED (terminal, not RUNNING). **OD-TLX-C Option A unchanged:** for the side-effecting case `error_code == ABANDONED`, `error_message == "abandoned"`, ledger entry for the new key `UNCERTAIN`, replay of that key still raises `InvocationUncertaintyError` with `executor.calls` unchanged. The non-side-effecting case asserts only `not success` and that the result code equals the value recorded from the **unfixed** code in step A below (do not hard-code a guess).
- **T3 generic submit failure, not close-specific.** Fresh (open) invoker; monkeypatch `invoker._execution_pool.submit` to raise `RuntimeError("c0-boom")`; `_execute_once` raises it unchanged (same exception object/type, not replaced); record FAILED; executor not invoked.

Negative / boundary (what must NOT change):
- **N1 bind failure is not marked FAILED.** Monkeypatch the `ToolExecutorTerminationPort.bind_future` of the invoker to raise after a real `submit`; assert the exception propagates and the record `physical_state is not FAILED` (do **not** assert RUNNING or any other value — that would pin residual R-1). Call `invoker.close()` at the end to join the worker.
- **N2 success path unchanged:** cite existing `test_no_dependency_boundary_success_records_succeeded_before_return` (must stay green; run in the test contract); do not duplicate.
- **N3 boundary branch unchanged:** existing `test_runtime_tool_invoker_dependency_admission.py` (must stay green; run in the test contract).
- **N4 ledger/replay semantics of OD-TLX-C Option A:** the four 023-A tests in `test_tools_side_effect_safety.py` (incl. `test_close_shuts_down_execution_pool`, `test_replay_before_close_preserves_exactly_once`) must stay green with **zero edits**; a change in their outcome is a STOP-and-report.

Procedure (red-first, mandatory evidence):
- **A.** Add the tests first, run the new file on the **unfixed** tree: T1, T2 (record assertion), T3 must fail on the record-state assertion only; T2's result/ledger assertions, N1 must pass; record the observed non-side-effecting result code. Log it.
- **B.** Apply the 4-line change; the new file is fully green.
- **C.** Keep two separate local commits: (1) tests only, (2) the `invoker.py` fix. Step A is then reproducible at commit (1) and step B at commit (2); an auditor can re-run A without reverting anything. No stash, no history rewrite.

### 5. Test contract (implementer; one `uv run pytest` at a time under the heavy-test mutex, `-p no:xdist`)
1. New file + direct neighbours, one command:
   `uv run pytest -p no:xdist tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py tests/unit/runtime/nexus/tools/test_tool_life_x_r4_terminal_lifecycle_invoker.py tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_dependency_admission.py tests/unit/runtime/tools/test_tools_side_effect_safety.py tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_retry_idempotency_unknown_outcome.py tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_real_governance_authorization.py tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py tests/unit/runtime/nexus/tools/test_tool_life_x_batch_c_stalled_abandoned_recovery.py -q`
2. EBH-7 C4 change-class set for `invoker.py` (RUNTIME_BEHAVIOR + SECURITY_GOVERNANCE_TENANT, `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py:222-226,242-251`), one command: `uv run pytest -p no:xdist tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_policy.py tests/unit/runtime/nexus/tools/test_tool_runtime_scope_enforcement.py tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_cross_tenant.py tests/unit/runtime/security/test_tenant_scope.py tests/qualification/ebh_7/tenant_platform_e2e/test_ebh7_integrated_tenant_boundary_proofs.py -q` (the dependency-admission module already ran in step 1).
3. `uv run ruff check` on the two changed files (one invocation); the repo type check on the changed files if `igx_gov ticket` requires it; `uv run python -m scripts.governance.igx_gov ticket` before handoff.
4. Diff guard (mandatory evidence): `git diff --name-only <base>..HEAD` = exactly `intergrax/runtime/nexus/tools/invoker.py`, the new test file and the ticket file (if the operator permits the ticket in the branch); `git diff <base>..HEAD -- intergrax` shows only the hunk in §3.
Logs → `.tmp/session/t-igx-024-a/`. Record exact HEAD SHA and test counts; counts are recorded, not assumed.

### 6. Base and branching
Implement on a branch/worktree from the campaign integration tip that already contains T-IGX-023-A/B (local post-fix tree `2d14bd76f`, a descendant of `50ddc04fb`; not on `origin/development` at plan time) or, if the operator prefers an independent change, from `origin/development`. `invoker.py` is identical in both for the touched region. The new test file does not touch any file edited by 023-A/B, so the order of merging is irrelevant. No push, no PR, no commit to `development` by the implementer unless the operator says so. Commit trailer `Ref: governance/tickets/T-IGX-024-A.md` on the invoker.py commit.

### 7. Re-audit impact (invoker.py is in the TOOL-LIFE-X replay sets)
Any change to `invoker.py` invalidates replay evidence taken at earlier SHAs. After this change an **independent AUDITOR** (neither implementer nor certifier of T-IGX-024-A; also not a rubber-stamp of AUD-TOOLLIFE2 — a successor or fresh session) must replay at the exact post-fix SHA:
1. **Batch B L2 set** (4 modules + side-effect-safety, as in 023-A contract) — mandatory (the changed branch is the one these exercise).
2. **F8 consolidated L2 inventory** (12 modules, `TOOL_LIFE_X_R4_FINAL_ROOT_CAUSE…§5`; 67/67 at baseline) — mandatory (R4 terminal-lifecycle invoker module is part of the durable-terminal family).
3. **Batch A–E L2 union** (Batch E L2 8 modules + Batch A boundary/dependency-admission + Batch B p0_safety_8 + side_effect_safety) — mandatory; includes Batch C stalled/abandoned/recovery which asserts ABANDONED semantics.
4. **W4-C distributed-cancellation qualification** (`tests/unit/runtime/architecture/test_enterprise_scale_resilience_w4_c_*` and `w4_d_provider_cancellation`, which use `RuntimeToolInvoker`) — recommended (touches RUNNING/cancel terminal logic).
5. **L3 / TL11 17-module set with `--extra mcp`** (L3-R1/R2/R6-R10, host bootstrap wiring, TL11 MCP modules, F3 physical L3, detachment supervision) — **recommended, not strictly required**: the change is confined to the no-boundary submit-failure path which the L3 physical tests do not enter on success; but the set exercises `RuntimeToolInvoker` end-to-end and its F3 module is the OD-TLX-B test; the auditor decides and records the reason if skipped.
6. **EBH-7 C4 set** of §5 step 2 and **harness_01 / trace_x / gr13 / ebh_7 F10 catalogs** that name `invoker.py` (`tests/qualification/harness_01/test_harness_01_gates.py`, `tests/qualification/trace_x/_trace_x_p0_support.py`/`_p3_support.py`, `tests/qualification/governance/gr13/catalog.py`, `tests/qualification/ebh_7/test_ebh_7_b6_f10_certification_adversarial.py`): the implementer/auditor must check whether any pins invoker.py content (hash/line) and replay those modules; static inspection at plan time found path references only, **no hash pin verified** (not analysed in depth). `tests/qualification/control_planes/ctrl_x/wide_pyright_*` records per-file pyright diagnostic counts for `invoker.py` (3 `reportArgumentType` at plan time): the change must not add diagnostics — run the configured pyright on the file and compare.
7. **IIAC global authority impact matrix** (protected file; do not touch): the implementer only reports whether `invoker.py` is a covered path; any matrix regeneration is the integration owner's, not this ticket's.
Not replayed / not required: DIST-SCALE-X, CT-ENT, SKILL-X, TRACE-X modules unrelated to `invoker.py` content.
Disclosure: the TOOL-LIFE-X reconciliation/acceptance record is **not** rewritten. A corrective note ("a pre-Batch-C defect in the no-boundary submit path was found by C0 and corrected by T-IGX-024-A") is a proposed patch for the integration owner only.

### 8. Risk and rollback
- Risk **low**: +4 lines on an error-only path that previously terminated with an unhandled stuck record; no effect on success/timeout/cancel paths; result mapping and ledger unchanged (verified by T2/N4). Residual risk: classification FAILED vs NOT_STARTED (matches the existing guard; operator may rule otherwise); R-1 bind-failure gap remains; `mark_failed()` raising masks the original error (parity with existing guard).
- Change class: `RUNTIME_BEHAVIOR` + `SECURITY_GOVERNANCE_TENANT` per EBH-7 C4 map (any `invoker.py` edit) → §5 step 2 and §7 item 6.
- Rollback: `git revert` of the fix commit (the tests-only commit's T1/T2/T3 then correctly fail again, documenting the defect); no data migration; existing stuck RUNNING records created by the unfixed code (if any exist in a durable store) are **not** repaired by this change (no sweep exists since Batch C) — the C0 finding assumed process-local InMemory stores; a durable-store inventory is out of scope.

### 9. Roles, scope limits, open questions
- **Roles:** IMPLEMENTER != AUDITOR != CERTIFIER — three distinct sessions. The planner (this document's author) is none of them and signs nothing. The implementer never records PASS; the auditor never implements; the certifier is independent of both. The signing/approval of the plan is by the operator or a designated delegate under IIAC Amendment 01 — **not** by the planner, implementer, or auditor of this ticket.
- **Out of scope:** OD-TLX-C Options B/C; new `RuntimeErrorCode`; typed NOT_STARTED rejection; pre-admission "closed" check; central `ensure_terminal_recorded()` net (C0 Option 3); the `bind_future` residual R-1; any edit of `test_tools_side_effect_safety.py` or other existing tests; durable SQLite store variant; sweep/repair of existing RUNNING records; FRZ-* promotion; closing TOOL-LIFE-X or any stage parent; IIAC paths and the impact matrix; roadmap/acceptance text rewrite; Docker/vendor/production configuration (C10/C15); push/PR.
- **OQ-1 (operator):** confirm FAILED as the terminal classification for a provable never-started submit failure (parity with the boundary branch). Default in this plan: yes.
- **OQ-2 (operator):** accept R-1 (bind_future failure after submit) as a separately tracked residual, or request a follow-up ticket (candidate fix: `mark_unknown()` + worker handling). Default: residual, not fixed here.
- **OQ-3 (operator):** which base — campaign post-fix tree (contains 023-A/B) or `origin/development`. Default: campaign post-fix tree.
- **OQ-4 (operator):** confirm that a product change under a certification campaign is authorised for this correction (it is not covered by any signed OD).
