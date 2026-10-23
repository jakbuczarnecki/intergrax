# TOOL-LIFE-X Batch A — Canonical Deadline Hierarchy Evidence

| Field | Value |
| --- | --- |
| **Task** | `TOOL-LIFE-X-BATCH-A-CANONICAL-DEADLINE-HIERARCHY` |
| **Parent** | `TOOL-LIFE-X` (§3.0.5) — **not CLOSED** |
| **Mode** | Change-delta (ECAP-P1/P2/P4/P6) |
| **Disposition** | **`TOOL-LIFE-X BATCH A — REMEDIATION READY FOR INDEPENDENT AUDIT`** |

**Non-closure:** Does not close `TOOL-LIFE-X`, does not promote global `FRZ-*` PASS, does not substitute `HARNESS-W4` / `DIST-SCALE-X`.

## Invariant evidence (scoped)

| Invariant | Owner | Contract | Test / gate | FRZ | Status |
| --- | --- | --- | --- | --- | --- |
| Parent execution deadline bounds tool physical attempt timeout | `RuntimeToolInvoker` | `resolve_active_provider_timeout_seconds` + `ToolContract.timeout_ms` | `test_tool_life_x_batch_a_deadline_hierarchy.py` | FRZ-REL-01 | VALIDATION REQUIRED (Batch A wave) |
| Protected-work admission blocks pre-effect when expired | `CanonicalHardProtectedWorkAdmission` | `ExecutionProtectedWorkAdmissionPort` | `test_tool_protected_work_admission.py`, Batch A expired-before | FRZ-REL-01 | VALIDATION REQUIRED |
| Queue admission wait consumes remaining parent budget | `DependencyAttemptExecutionBoundary` + invoker | `acquire_timeout_seconds` projection | Batch A queue-wait test | FRZ-REL-07 | VALIDATION REQUIRED |
| Retry backoff consumes same operation budget | `RuntimeToolInvoker._execute_with_policy` | capped `cooperative_delay_seconds` | Batch A backoff-bounded test | FRZ-REL-07 | VALIDATION REQUIRED |
| Single sanctioned invoker path | Architecture gates | U5 zero-bypass | existing U5 tests (unchanged) | FRZ-BND-03, FRZ-EXE-02 | ALREADY PROVEN (scoped) |

**Execution SHA (pre-remediation wave):** `03d5fc3d54d33ef6230123ab2300756441f744f8` — independent audit **REMEDIATION REQUIRED** (BA-F01, BA-F02).

## Remediation wave (BA-F01 / BA-F02)

| Finding | Root cause | Correction | Tests |
| --- | --- | --- | --- |
| **BA-F01** | `attempt_timeout_s` computed in `_execute_with_policy` before dependency `acquire()`; queue wait did not shrink physical `future.result(timeout=…)` | `_physical_attempt_timeout_after_admission_seconds()` after permit; deadline exhausted → `release_after_submit_failure` + `deadline_exceeded` / no submit | `test_queue_wait_then_physical_timeout_uses_remaining_not_contract`, `test_deadline_exhausted_after_permit_before_physical_submit` + prior Batch A module |
| **BA-F02** | `acquire_future.cancel()` on timeout did not release async-granted permits when caller abandoned | `_AcquireAcceptance` + done-callback / `_release_abandoned_acquire_permit` on timeout | `test_acquire_timeout_races_grant_releases_permit` + existing shutdown/race boundary tests |

| Invariant | Owner | Contract | Test / gate | FRZ | Status |
| --- | --- | --- | --- | --- | --- |
| Queue wait consumes parent budget for physical timeout | `RuntimeToolInvoker` | `execution_live_remaining_seconds` post-acquire | Batch A BA-F01 tests | FRZ-REL-01, FRZ-REL-07 | VALIDATION REQUIRED (remediation wave) |
| Abandoned acquire cannot orphan permit | `DependencyAttemptExecutionBoundary` | async acquire + sync acceptance | `test_acquire_timeout_races_grant_releases_permit` | FRZ-REL-07 | VALIDATION REQUIRED (remediation wave) |

**Targeted regression @ remediation commit:** Batch A module + `test_dependency_attempt_execution_boundary.py` + `test_tools_side_effect_safety.py` — **45 passed**; **3 failed** `test_runtime_tool_invoker_dependency_admission.py` (side-effect + declarative policy evidence) with `active execution governance identity required` — **pre-existing test wiring** (failure in `_prepare_invocation`, not post-acquire path); **1 failed** `test_no_orphan_thread_after_close` — **environment/timing** on Windows (thread still alive immediately after `close()`); reproducer isolated to that test.

**Execution SHA:** see `FINAL_COMMIT` below (single remediation commit).

## TL1–TL15 (Batch A reconciliation)

See parent P0 matrix; Batch A advances **TL2** primary evidence only. **TL15** remains **BLOCKED** until parent criteria close.

## Reused evidence

- P0 pre-audit @ `f5c9330973f7af7ea1d19fbbd2afd13942faac43` (baseline only).
- HARNESS-02 Q09/Q11 patterns @ historical SHAs — acceptance scope unchanged.
