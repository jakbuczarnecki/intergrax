# Execution Certification Acceleration — R1 Qualification Record

**Status:** `QUALIFIED` (mechanism only — no final gate integration)

**Task:** Execution Certification Acceleration/R1

**Branch:** `development`

---

## Scope

Reusable, typed, bounded-parallel qualification coordinator for isolated `uv run pytest` child processes. **Does not** replace NPSC finals, `_MANDATORY_SUITES`, or existing `_run_pytest` gates (R2).

---

## Implemented architecture

```text
QualificationRunManifest (typed tuple[QualificationSuite, ...])
        ↓
QualificationCoordinator (scheduling, exclusive resources, collect-all)
        ↓
QualificationSuiteExecutor Protocol
        ↓
PytestSubprocessSuiteExecutor (default)
```

**Location:** `testing_support/execution_qualification/`

---

## Contracts

| Type | Role |
| --- | --- |
| `QualificationSuite` | `suite_id`, `pytest_arguments`, optional `exclusive_resource_id`, `environment_overrides` |
| `QualificationRunManifest` | Ordered suites; duplicate IDs rejected at construction |
| `QualificationRunConfig` | `max_parallel` (required ≥ 1), `run_artifact_root`, `suite_timeout_seconds`, `run_id` |
| `ExecutionQualificationSuiteResult` | PASS/FAIL/SKIP, `outcome_kind` (pytest vs launch vs timeout), log path |
| `ExecutionQualificationRunResult` | Manifest-order `suite_results`, aggregate PASS/FAIL |

---

## Proofs (unit / integration tests)

| Property | Evidence |
| --- | --- |
| Bounded concurrency (`max_parallel=2`) | `test_bounded_concurrency_never_exceeds_max_parallel` (measured `max_active`) |
| Actual parallelism | `test_actual_parallel_overlap_with_synchronization` (`threading.Barrier`) |
| Exclusive resource | `test_exclusive_resource_suites_do_not_overlap` |
| Deterministic aggregation | `test_result_order_follows_manifest_not_completion_order` |
| Collect-all | `test_collect_all_runs_remaining_after_failure` |
| Infrastructure vs pytest failure | `test_infrastructure_failure_distinct_from_pytest_failure` |
| Timeout classification | `test_suite_timeout_terminates_slow_pytest` |
| Timeout direct-child termination | `test_subprocess_run_timeout_terminates_direct_child_process` (`subprocess.run` timeout kills its child) |
| Separate logs | `test_separate_logs_created`, subprocess smoke log file |
| Environment per suite | `test_environment_override_recorded_per_suite` |
| Duplicate manifest | `test_duplicate_suite_ids_rejected` |
| Real gate-shaped manifest | `test_manifest_real_targets.py` (declaration only) |
| Subprocess smoke | `test_subprocess_smoke_passes_single_file` |
| Contract hardening | `test_contract_hardening.py` (executor identity, infrastructure failures, result invariants) |

**Tests:** `tests/unit/testing_support/execution_qualification/`

---

## Failure semantics

- Pytest `exit_code != 0` → suite FAIL; remaining suites still run.
- Typed `LAUNCH_FAILURE` / `TIMEOUT` from the executor remain normal suite FAIL rows; collect-all continues.
- Unexpected executor exceptions or executor result `suite_id` mismatch → `QualificationCoordinatorError` (not `LAUNCH_FAILURE`).
- Invalid manifest / config → no child processes (`QualificationRunManifest` / `QualificationRunConfig` validation).
- Coordinator artifact prep failure → `QualificationCoordinatorError` before execution.
- `ExecutionQualificationSuiteResult` rejects impossible `status` / `outcome_kind` / `exit_code` combinations at construction.

---

## Known R1 limitations

- No wiring into NPSC-5E Final or `_MANDATORY_SUITES` (R2).
- No `os.cpu_count()` auto parallelism; `max_parallel` is explicit.
- Exclusive resources are process-local mutex keys (sufficient for P0 `cross.db` pattern).
- Subprocess timeout (`subprocess.run(..., timeout=...)`) terminates the **direct child** (`uv run pytest …`); descendant process-tree cleanup is **not** guaranteed by R1 (no process-group/session isolation).
- Parent/child composition parity (AD-R1-2) deferred to R2.

---

## R2 boundary

Integrate coordinator into maintainer certification entrypoints; parity-qualify against serial composed finals without changing frozen test semantics.
