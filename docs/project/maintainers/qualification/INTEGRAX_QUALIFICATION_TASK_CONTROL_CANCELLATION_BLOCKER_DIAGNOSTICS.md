# INTEGRAx-QUALIFICATION-TASK-CONTROL-CANCELLATION-BLOCKER-DIAGNOSTICS

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-QUALIFICATION-TASK-CONTROL-CANCELLATION-BLOCKER-DIAGNOSTICS` |
| Session scope | Qualification optimization · final certification closure (task-control / cancellation blocker) |
| Diagnostic HEAD (start context) | `4d64d0874513443f7f3ecb53a9338be698c37d38` |
| Remediation HEAD | see git at commit time |

## Session Scope

Diagnose and remediate the known `npsc5f-final` failure on `test_taskcpm_r21_product_host_uses_canonical_bundle_authority` without opening Task Control / Cancellation as a platform project and without `intergrax/` production edits in this task.

## Business Scenario

| Element | Detail |
| --- | --- |
| Actor | Operator (HTTP / product host principal) |
| Operation | Governed checkpoint resume (`resume_task_execution`) |
| Authority | Canonical harness host policy bundle (`harness.control_plane`) via `build_harness_control_plane_governance` |
| Governance | `ControlPlaneMutationAuthorizationBoundary` → bundle-backed evaluator → explicit `harness.task_control.resume_task_execution` allow rule |
| Expected | Resume mutation is ALLOW when bundle rule matches and request identity is coherent |
| Failure | Boundary returned DENY with `cpm-deny:evaluator_failure` (evaluator raised; fail-closed wrapper) |

## Technical Scenario

Product-profile harness host composes `ApprovalConsumingControlPlaneMutationEvaluator` over `BundleBackedControlPlaneMutationEvaluator` + immutable bundle. Test calls `boundary.authorize(resume_request)` synchronously (no HTTP), expecting ALLOW and rule id `harness.task_control.resume_task_execution`.

## Initial Failure

| Item | Detail |
| --- | --- |
| Exact node | `tests/unit/applications/test_task_control_governed_resume.py::test_taskcpm_r21_product_host_uses_canonical_bundle_authority` |
| Assertion | `result.permitted is True` |
| Expected | `permitted=True`, policy rule `harness.task_control.resume_task_execution` |
| Actual | `permitted=False`, `reason=evaluator_failure`, `decision_id=cpm-deny:evaluator_failure` |
| Underlying exception | `RuntimeError: active execution identity required` in `resolve_meaningful_side_effect_execution_identity` ← `control_plane_mutation_to_meaningful_side_effect_request` (GR-1) |
| Minimal command | `uv run pytest tests/unit/applications/test_task_control_governed_resume.py::test_taskcpm_r21_product_host_uses_canonical_bundle_authority -q --tb=short` |

## Exact Reproduction

Reproduced on `development` before fix: single test FAIL in ~2.7s with assertion on `permitted`.

## Application Flow

```text
Product host profile (PRODUCT)
  → build_harness_control_plane_governance(env)
  → resolve_harness_task_control_mutation_boundary
  → boundary.authorize(build_resume_task_execution_mutation_request(...))
  → ApprovalConsumingControlPlaneMutationEvaluator (no approval ref → inner)
  → BundleBackedControlPlaneMutationEvaluator
  → control_plane_mutation_to_meaningful_side_effect_request
  → resolve_meaningful_side_effect_execution_identity (requires active run/attempt/execution)
  → exception → boundary fail-closed → cpm-deny:evaluator_failure
```

## Task-Control Boundary

Governed resume mutation type and resource scope are correct; boundary was invoked (not bypassed). Failure occurred inside policy adapter identity resolution, not in resume admission or runner.

## Canonical Bundle Authority

`build_harness_host_control_plane_policy_bundle()` includes explicit allow rule `harness.task_control.resume_task_execution`. Bundle id/version/digest wiring unchanged; evaluator never reached bundle match logic.

## Evaluator Resolution

| Item | Value |
| --- | --- |
| Outer | `ApprovalConsumingControlPlaneMutationEvaluator` |
| Inner | `BundleBackedControlPlaneMutationEvaluator` |
| Bundle | `harness.control_plane` v1.0.0 |
| Policy input | `MeaningfulSideEffectRequest` derived from mutation (GR-1 path) |

## Identity Scope

Request carried `tenant_id`, `task_id`, `run_id`, `mutation_id`, checkpoint-derived revisions. GR-1 requires bound `attempt_id` + `execution_id` (from active context or explicit pair). Test invoked authorize **without** binding execution identity despite checkpoint runtime carrying `_ATTEMPT_ID` / `_ROOT_EXECUTION_ID`.

## Root Cause Classification

**INVALID TEST FIXTURE** (GR-1 follow-up gap in synchronous bundle-authority proof — same class as `test_control_plane_mutation_policy.py` using `_with_active_execution_for_request`).

## Platform Gap Assessment

**Separate follow-up (not fixed here):** `intergrax/applications/_shared/task_control.py` calls `mutation_boundary.authorize` without binding execution identity from active task binding or checkpoint runtime. Operator HTTP cancel/resume with canonical bundle may fail with `evaluator_failure` until application composition binds GR-1 identity at the authorization callsite. Requires dedicated application design; out of scope for test-only qualification closure.

## Architecture Boundary Assessment

No contract change. No `intergrax/` edits in remediation. Evaluator fail-closed on exception preserved.

## Remediation

1. Added `tests/unit/applications/task_control_policy_evaluation_support.py` with `authorize_bundle_backed_control_plane_mutation`, binding `run_id` / `attempt_id` / `execution_id` from checkpoint runtime before authorize.
2. Updated `test_taskcpm_r21_product_host_uses_canonical_bundle_authority` to use helper with `_checkpoint()` identity (realistic paused-run resume scope).

## Targeted Verification

```bash
uv run pytest tests/unit/applications/test_task_control_governed_resume.py::test_taskcpm_r21_product_host_uses_canonical_bundle_authority -q
```

Result: PASS.

## Bounded Application Regression

```bash
uv run pytest tests/unit/applications/test_task_control_governed_resume.py -q
```

Result: 28 passed.

## Cancellation Leaf Verification

```bash
uv run pytest tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py -k "not survives_process_restart" tests/unit/runtime/cancellation/test_p0c5a_explicit_terminal_wiring.py tests/unit/applications/test_task_control_governed_resume.py -q
```

Result: 43 passed.

Nested gate:

```bash
uv run pytest "tests/unit/runtime/architecture/test_npsc5e_r2_h2_q1_frozen_regression_closure.py::test_mandatory_frozen_suite_passes[Cancellation]" -q
```

Result: PASS.

## Qualification Regression

```bash
uv run pytest tests/unit/testing_support/execution_qualification/ -q
```

Result: 181 passed, 7 skipped.

## Embedded Harness Regression

```bash
uv run pytest tests/unit/testing_support/execution_qualification/test_embedded_harness_elimination.py -q
```

Result: included in qualification run — PASS.

## Full Canonical Verification

```bash
uv run python -m testing_support.execution_qualification.performance --profile npsc5f-final --repetitions 1 --max-parallel 2
```

Result: **FAIL** (`run_status_pass: false`). Slowest leaf: `npsc5e-r2.mandatory.r2-h2-q1` (wall ~189s). Nested mandatory suites (e.g. Checkpoint store, Long-running, NPSC-5A–5D) fail inside `test_npsc5e_r2_h2_q1_frozen_regression_closure.py` — **not** the task-control cancellation leaf.

## Performance Observation

| Metric | Value |
| --- | --- |
| Wall | ~189.4 s |
| Slowest leaf | `npsc5e-r2.mandatory.r2-h2-q1` (~108.7 s) |
| Recovery leaf | ~12.7 s (`npsc5f-final.recovery`) |
| Total leaf work | ~378.2 s |
| Effective concurrency | ~2.0 @ max_parallel=2 |

## Production Changes

**NONE** (`intergrax/` untouched).

## Findings

- `cpm-deny:evaluator_failure` was a fail-closed wrapper over missing GR-1 execution identity at policy evaluation time, not a broken bundle rule.
- Cancellation qualification path for `test_task_control_governed_resume.py` is green after test fixture fix.
- Full `npsc5f-final` remains blocked by unrelated nested mandatory failures under R2-H2-Q1.

## Decision

**TASK-CONTROL CANCELLATION BLOCKER = REMEDIATED — PASS** (scoped blocker).

Full canonical closure: **FULL CANONICAL CLOSURE BLOCKED BY NEW LEAF** (R2-H2-Q1 orchestrated nested suites).

## Final Verdict

Task-control / cancellation qualification blocker remediated with test-only GR-1 identity binding. Platform application wiring gap documented for follow-up. Full `npsc5f-final` single-run PASS not achieved in this task due to downstream R2-H2-Q1 leaf failures.
