# INTEGRAx Qualification — Final SHA Canonical Verification

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-QUALIFICATION-FINAL-SHA-CANONICAL-VERIFICATION` |
| Mode | Verification-only (no production / qualification-architecture edits by this task) |
| Profile | `npsc5f-final` |
| Run artifacts | `.tmp/session/FINAL-SHA-CANONICAL-VERIFICATION/npsc5f-final/` |
| Expected remediation reference (historical) | `a2625a39f6216f5b5e74eca7f3e4ed069854f9ef` |

## Session Scope

Formal single canonical run plus pre-run smokes and targeted pytest slices. No feature, refactor, or optimization work in this task.

## Repository State

| Item | Value |
| --- | --- |
| Branch | `development` |
| Session-start `HEAD` | `9979e3b3af64b436708d0e95fac669dc9ef6e3d1` |
| Session-start `origin/development` | `9979e3b3af64b436708d0e95fac669dc9ef6e3d1` (in sync) |
| Ahead/behind | 0 / 0 at session start |
| Runner-reported `git_head` (JSON) | `71a522b551e8a914fda5e614509ff36a846c2f87` |
| `HEAD` after run (external commits) | `ac488ce7ffaf4124c7c813cd80b85831d0e8b295` |

## Worktree Integrity

**Not acceptable for formal certification evidence.**

| Phase | Tracked WIP | Untracked WIP |
| --- | --- | --- |
| Session start | 5 modified test files under `tests/unit/` (debug, runtime execution, task) | `build/` pytest caches (ignored session noise) |
| Pre canonical run | Above plus **modified `intergrax/memory/*` production files**, modified observability/memory tests | Multiple `intergrax/memory/` and contract files |
| During run | Commits landed on `development` while the benchmark was executing (`71a522b…`, later `ac488ce7…`) | — |

Tracked production and test WIP on the qualification path (including `runtime-observability` leaf inputs) invalidates a clean SHA-bound certification.

## Verification SHA

```text
FULL CANONICAL RUN SHA (pre-run checkpoint) = 9979e3b3af64b436708d0e95fac669dc9ef6e3d1
VERIFIED PLATFORM SHA (intended)             = 9979e3b3af64b436708d0e95fac669dc9ef6e3d1
Effective code under test during run         = contaminated (see Worktree Integrity)
```

## Memory Cold Import Smoke

| Check | Result |
| --- | --- |
| `import intergrax.memory.contracts` | PASS |
| `MemoryLifecycleOutcome` import | PASS |
| `UserProfile`, `UserProfileMemoryEntry` public path | PASS |
| `PublicUserProfile is CanonicalUserProfile` | PASS |

## Cancellation Reproducer

`test_pre_existing_cancellation_fixture_unrelated_to_h2_revision` — **PASS** (6.58 s).

## R2-H2-Q1 Semantic Revalidation

`test_npsc5e_r2_h2_q1_frozen_regression_closure.py` with `-k "not test_mandatory_frozen_suite_passes"` — **12 passed**, 17 deselected (11.33 s).

## R3 Implementation Semantic Revalidation

`test_npsc5e_r3_child_fanout_partial_recovery.py` with `-k "not test_mandatory_frozen_suite_passes"` — **13 passed**, 9 deselected (5.63 s).

## Qualification Regression

`tests/unit/testing_support/execution_qualification/` — **216 passed**, 7 skipped (95.05 s).

## Canonical Run Command

```bash
uv run python -m testing_support.execution_qualification.performance \
  --profile npsc5f-final \
  --repetitions 1 \
  --max-parallel 2 \
  --artifact-dir .tmp/session/FINAL-SHA-CANONICAL-VERIFICATION/npsc5f-final
```

## Canonical Run Evidence

| Field | Value |
| --- | --- |
| Wall time | **166.45 s** (measured); shell **166.88 s** |
| `run_status_pass` | **false** |
| `certification_decision` | **blocked** |
| Total leaf work | **332.12 s** |
| Effective concurrency | **1.995** |
| Slowest leaf | `runtime-observability` — **58.42 s** |
| Failed leaves | `runtime-observability` (1 pytest failure) |
| Skipped leaves | none reported |
| Gate failures | observability suite gate (pytest FAIL) |
| Drift / protected drift | **PASS** (`npsc5f-final.drift-sentinel` 13 passed; R2/R3/R4 drift classifiers passed in artifact logs) |

## Leaf Timing Summary

Top 10 slowest suites (wall-share from JSON):

| suite_id | duration (s) | wall % |
| --- | ---: | ---: |
| `runtime-observability` | 58.42 | 35.1 |
| `npsc5e-r3.mandatory.r2-original` | 27.58 | 16.6 |
| `runtime-events` | 22.97 | 13.8 |
| `npsc5f-final.recovery` | 15.22 | 9.1 |
| `npsc5e-r2.mandatory.r2-h2-q1` | 13.14 | 7.9 |
| `npsc5e-r2.final-semantic` | 12.44 | 7.5 |
| `npsc5e-r3.mandatory.long-running` | 11.05 | 6.6 |
| `npsc5e-r3.mandatory.p0a` | 9.09 | 5.5 |
| `dg001-lineage` | 8.63 | 5.2 |
| `npsc5e-r3.mandatory.r1-final` | 8.42 | 5.1 |

Key leaves (pytest outcome in artifact logs):

| suite_id | Leaf timing (s) | Outcome |
| --- | ---: | --- |
| `npsc5e-r2.final-semantic` | 12.44 | PASS (19 passed, 22 deselected) |
| `npsc5e-r2.mandatory.r2-h2-q1` | 13.14 | PASS (12 passed, 17 deselected) |
| `npsc5e-r3.implementation-semantic` | (in profile) | PASS (13 passed, 9 deselected) |
| `npsc5e-r3.final-semantic` | n/a | Not a separate physical leaf in `npsc5f-final` (44 leaves); R3 final semantics covered via mandatory / implementation receipt paths |
| `runtime-observability` | 58.42 | **FAIL** |
| `runtime-events` | 22.97 | PASS (313 passed) |
| `npsc5f-final.recovery` | 15.22 | PASS (93 passed, 26 deselected) |

## Nested Harness Regression Check

No `test_mandatory_frozen_suite_passes` invocation in any leaf log under `perf-npsc5f-final-0/`.

| Path | Expected |
| --- | --- |
| R2 Final canonical | semantic only (`-k` deselect frozen suite in `npsc5e-r2.final-semantic.log`) |
| R2-H2-Q1 canonical | semantic only (deselected counts in `npsc5e-r2.mandatory.r2-h2-q1.log`) |
| R3 implementation canonical | semantic only (`npsc5e-r3.implementation-semantic.log`) |
| R3 Final | receipt/mandatory decomposition (no nested frozen harness in logs) |

## Drift Assessment

Protected drift sentinels and classifiers **passed** in this run. Certification blocked by **functional** observability failure, not drift.

## Failure Classification

| Dimension | Classification |
| --- | --- |
| Formal certification eligibility | **ENVIRONMENT/CONTAMINATION** (dirty worktree + moving `HEAD` during run) |
| Observed leaf failure | **FLAKY/UNKNOWN** (thread timing) pending isolated repro on clean `HEAD` |

### Failing leaf detail

| Field | Value |
| --- | --- |
| `suite_id` | `runtime-observability` |
| Gate | pytest suite gate for runtime observability qualification bundle |
| Node/test | `tests/unit/runtime/observability/test_enterprise_scale_resilience_w5_a_observability_backpressure.py::test_critical_overflow_fail_closed_no_silent_loss` |
| Exception | `AssertionError`: expected at least one `EventDeliveryDisposition.REJECTED`; none observed |
| Duration | ~55 s pytest wall inside leaf (~58.4 s suite attribution) |
| Category | Platform vs test undetermined without clean-tree isolation; threading/barrier race suspected under `max_parallel=2` host load |

## Performance Comparison

| Historical wall (s) | This run |
| ---: | ---: |
| ~938, ~510, ~189, **~129**, ~208, **~167** | **~166** |

Wall time is in line with recent ~167 s runs; not treated as performance regression.

## Production Changes

```text
NONE (by this verification task)
```

## Findings

1. Session-start `HEAD` (`9979e3b3`) is ahead of referenced remediation `a2625a39` (`OBS-COVERAGE-1-R1` and follow-on commits exist).
2. Pre-run and in-run worktree was **not clean**; additional commits appeared during the benchmark.
3. Pre-run smokes and semantic slices **passed** on session-start tree.
4. Optional `tests/unit/memory/` at session start: **1 failed** (`test_memory_contracts_do_not_import_implementation_modules`) — not re-run after mid-session commits.
5. Full canonical run: **`certification_decision=blocked`**, single failure in `runtime-observability`.
6. Nested frozen harness regression: **no evidence** of `test_mandatory_frozen_suite_passes` in canonical leaves.

## Decision

Do **not** treat this run as final SHA certification. **STOP** per task policy (no fix in this task).

**Recommended follow-up tasks:**

1. Restore clean `development` at intended certification SHA; prohibit concurrent commits during run.
2. Re-run formal `npsc5f-final` once on clean tree.
3. If observability test fails again on clean tree, isolate `test_critical_overflow_fail_closed_no_silent_loss` (direct pytest, mp=1/mp=2) and classify platform vs test.

## Final Verdict

```text
FINAL SHA CANONICAL VERIFICATION = BLOCKED — CONTAMINATED WORKTREE
```

Additional note: observed functional blocker on `runtime-observability` leaf (see Failure Classification). Not a protected-drift block.

```text
VERIFIED PLATFORM SHA (intended)     = 9979e3b3af64b436708d0e95fac669dc9ef6e3d1
DOCUMENTATION COMMIT SHA             = fc81cc5fd606fddcf2a7896b37a58a9f4649a0fe
```
