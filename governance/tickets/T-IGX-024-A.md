# T-IGX-024-A — C0 correction: terminalise the durable external-operation record when `pool.submit` fails in the no-boundary branch of `RuntimeToolInvoker._execute_once`

## Scope

**Product-code correction in a certification campaign.** Defect (C0-TLX, CONFIRMED, runtime reproduction): in `intergrax/runtime/nexus/tools/invoker.py` `_execute_once`, branch `dep_boundary is None`, `ext_op.mark_running()` is followed by an unguarded `self._execution_pool.submit(...)`. If `submit` raises (always after `RuntimeToolInvoker.close()`, which also nulls the boundary), the durable physical record stays `RUNNING` although the executor never ran. Violates roadmap section 3.0.5 ("No Tool invocation may remain RUNNING without end"); blocks retry of that operation id; nothing sweeps it (Batch C removed orphan sweeps). The boundary branch of the same function already has the guard (`except BaseException: ext_op.mark_failed(); ...; raise`). Pre-dates Batch C (since `e8264ff2e`).

Why this is a correction, not new functionality: it applies to one branch the exact terminalisation the sibling branch already performs for the same event; no new state, error code, API or typed rejection; result mapping and idempotency-ledger semantics are unchanged; it stays inside the signed OD-TLX-C Option A (Options B/C unsigned).

Change: wrap only the `submit` call in `try/except BaseException: ext_op.mark_failed(); raise`. `mark_running()` and `bind_future()` stay OUTSIDE the guard (a `bind_future` failure after a successful `submit` must not be marked FAILED because the worker already runs; it remains an open residual R-1, not fixed here). The diff text of plan section 3 is binding, not the +/- counts.

Files (exact): `intergrax/runtime/nexus/tools/invoker.py` (modify, one hunk); `tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py` (new); this ticket and `plans/PLAN_T-IGX-024-A.md`. No other file.

Approved plan (byte-unedited copy of the signed plan): [`plans/PLAN_T-IGX-024-A.md`](plans/PLAN_T-IGX-024-A.md).

Signed decisions: `.tmp/session/cert-campaign/SIGNING/SIGN-7_C0/SIGNED_DECISIONS.md` (OD-C0-PRODUCT-CORRECTION-2026-11-07, OD-C0-PLAN-APPROVAL-2026-11-07; both SIGNED_WITH_CONDITIONS).

Signed conditions (binding): (1) the product diff under `intergrax/` is exactly the one hunk of plan section 3 and nothing else; (2) the independent AUDITOR additionally checks, at the post-fix SHA, a retry case (effective max attempts > 1, `pool.submit` failing) and confirms final result mapping and ledger semantics are unchanged; (3) the TOOL-LIFE-X replay of plan section 7 is mandatory and the TOOL-LIFE-X acceptance remains open until it is done. No PASS, closure, FRZ promotion or stage-parent change is implied.

Roles: IMPLEMENTER != AUDITOR != CERTIFIER (three distinct sessions). The implementer records no PASS.

## Plan digest

Plan digest: 40a9672ef0c9a2f153b42113eb29159ad699a614bbffb7784ea1ee48d91083be

## Approved

Approved: operator-delegate @ 2026-11-07T21:59:15+01:00 (Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab)

Signature lines (verbatim from the signed record):

```text
Approved OD-C0-PRODUCT-CORRECTION-2026-11-07: operator-delegate @ 2026-11-07T21:59:15+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
Approved OD-C0-PLAN-APPROVAL-2026-11-07: operator-delegate @ 2026-11-07T21:59:15+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```

## Test contract

Heavy-test mutex; one `uv run pytest` at a time; `-p no:xdist`; logs under the IMPL-C0 dossier `logs/`.

1. Red-first: tests-only commit; run the new file on the unfixed tree — T1/T2(record)/T3 fail only on the record-state assertion; T2(result/ledger), N1 pass.
2. After the fix commit:
   ```powershell
   uv run pytest -p no:xdist tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py tests/unit/runtime/nexus/tools/test_tool_life_x_r4_terminal_lifecycle_invoker.py tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_dependency_admission.py tests/unit/runtime/tools/test_tools_side_effect_safety.py tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_retry_idempotency_unknown_outcome.py tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_real_governance_authorization.py tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py tests/unit/runtime/nexus/tools/test_tool_life_x_batch_c_stalled_abandoned_recovery.py -q
   ```
3. EBH-7 C4 change-class set for `invoker.py`:
   ```powershell
   uv run pytest -p no:xdist tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_policy.py tests/unit/runtime/nexus/tools/test_tool_runtime_scope_enforcement.py tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_cross_tenant.py tests/unit/runtime/security/test_tenant_scope.py tests/qualification/ebh_7/tenant_platform_e2e/test_ebh7_integrated_tenant_boundary_proofs.py -q
   ```
4. `uv run ruff check` (two files, one invocation); configured type check on the changed files; `uv run python -m scripts.governance.igx_gov ticket`.
5. Diff guard: `git diff --name-only <base>..HEAD` = the product/test files above + this ticket and the plan copy; `git diff <base>..HEAD -- intergrax` = the single hunk of the plan.

Required tests (new file): T1 post-close store-level (record FAILED, not RUNNING; call-1 record SUCCEEDED; `executor.calls == 1`); T2 post-close via `invoke()`, both C0 parametrisations (OD-TLX-C Option A unchanged: ABANDONED/"abandoned", ledger UNCERTAIN, replay blocked, `executor.calls` unchanged); T3 generic `submit` failure on an open invoker (exception propagated unchanged, record FAILED); N1 `bind_future` failure after a real `submit` is NOT marked FAILED (asserts `!= FAILED` only); N2/N3/N4 existing success-path, boundary-branch and 023-A tests unchanged and green. Signed condition (2) retry case (effective max attempts > 1, `submit` failing): added by the implementer as T4 inside the same planned new test file (no other file touched); the auditor performs the condition (2) check.

## Out of scope

OD-TLX-C Options B/C; new `RuntimeErrorCode`; NOT_STARTED typed rejection; pre-admission "closed" check; `ensure_terminal_recorded()` safety net; `bind_future` residual R-1; editing any existing test (incl. `test_tools_side_effect_safety.py`); durable SQLite store variant; repair/sweep of existing RUNNING records; FRZ-* promotion; closing TOOL-LIFE-X or any stage parent; IIAC paths and the IIAC global authority impact matrix; roadmap/acceptance text rewrite; Docker, vendor, production configuration (C10/C15); push, PR, commit to `development`.

## Risk and rollback

Risk low (error-only path, parity with existing guard). Residuals: FAILED vs NOT_STARTED classification (decided: FAILED, OQ-1); R-1 `bind_future` (accepted, unfixed, OQ-2); `mark_failed()` raising would mask the original error (parity); with the fix a failed submit leaves FAILED so attempt>1 is admitted (condition 2). Change class RUNTIME_BEHAVIOR + SECURITY_GOVERNANCE_TENANT (EBH-7 C4). Rollback: `git revert` of the fix commit; no migration; previously stuck records are not repaired.

## Re-audit impact

`invoker.py` is in the TOOL-LIFE-X replay sets. An independent AUDITOR replays at the exact post-fix SHA (plan section 7): Batch B L2 set; F8 12-module inventory (67/67 at baseline); Batch A-E L2 union (incl. Batch C); W4-C/W4-D cancellation modules (recommended); L3/TL11 17-module set with `--extra mcp` (recommended; reason recorded if skipped); EBH-7 C4 set; catalogs naming `invoker.py` and the wide-pyright per-file counts; report whether `invoker.py` is a covered path of the IIAC impact matrix. Original acceptance/reconciliation verdicts are not rewritten.

## Lifecycle state

quality_gate

## Lease holder

IMPL-C0 (worktree `D:/Projekty/_wt/intergrax/cx-ic3`, branch `cert-wave/impl-c0`)

## Checkpoint

IMPL-C0 implementer checkpoint (not an audit; no PASS claimed; handed over READY_FOR_INDEPENDENT_AUDIT). Base `2d14bd76f29366a6eda562c22cf857b55d515700`.

- `verify_plan_approval.py` ticket vs plan: OK.
- Red-first (tests-only commit `98b4bf514`, unfixed tree): T1, T2 (x2), T3, T4 fail only on the record-state assertion (RUNNING != FAILED); N1 passes (5 failed, 1 passed).
- After fix: test contract command 1 = 102 passed; command 2 (EBH-7 C4) = 20 passed. Mutation check (hunk reverted): 5 new tests fail, N1 passes.
- `ruff check` on both changed Python files: OK.
- `igx_gov ticket` stages run individually: `verify_commit_ticket_refs.py --mode range --rev-range <base>..HEAD` OK (branch has no upstream, so `--mode push` degrades to a whole-history walk); `validate_policy_registry.py` OK; `run_regression_gate_ci.py --profile fast` PASS.
- Diff guard: `intergrax/` diff = the single hunk of plan section 3 in `invoker.py`.
- Full logs and report: IMPL-C0 dossier (`.tmp/session/cert-campaign/IMPL-C0/`).
