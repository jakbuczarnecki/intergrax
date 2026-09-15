# INTEGRAx Qualification — Clean Final-SHA Canonical Verification

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-QUALIFICATION-CLEAN-FINAL-SHA-CANONICAL-VERIFICATION` |
| Mode | Verification-only (no production / qualification-architecture / test fixes in this task) |
| Profile | `npsc5f-final` |
| Session date | 2026-09-15 |
| OBS reconstruction alignment closure | `c7a33f915121af4006f47fc80a79c77e27247fdd` |

## Session Scope

Single formal `npsc5f-final` canonical certification on a **clean, stable** `development` HEAD. Mandatory pre-canonical smokes, then exactly one full orchestrator run (`repetitions=1`, `max_parallel=2`). No implementation work.

## Repository State

| Item | Value |
| --- | --- |
| Branch | `development` |
| `HEAD` (at verification start) | `6758e538d7349402ae251bad9f8adefab30c04ac` |
| `origin/development` (at verification start) | `6758e538d7349402ae251bad9f8adefab30c04ac` |
| Ahead / behind | 0 / 0 |
| Tracked modified | none |
| Tracked staged | none |
| Untracked | none affecting import/pytest/catalog |

Recent tip at verification: `6758e538d` — prior doc stub commit; platform code unchanged for this run.

## Worktree Integrity

| Check | Result |
| --- | --- |
| `git diff --name-only` | empty |
| `git diff --cached --name-only` | empty |
| Untracked under `intergrax/`, `tests/`, `testing_support/` | none |
| Formal clean-SHA precondition | **SATISFIED** |

## Verification SHA

```text
VERIFICATION_SHA = 6758e538d7349402ae251bad9f8adefab30c04ac
```

## HEAD Stability Evidence

| Checkpoint | `git rev-parse HEAD` | Match `VERIFICATION_SHA` |
| --- | --- | --- |
| Session start | `6758e538d7349402ae251bad9f8adefab30c04ac` | yes |
| After memory smoke | `6758e538d7349402ae251bad9f8adefab30c04ac` | yes |
| After mandatory pytest slices | `6758e538d7349402ae251bad9f8adefab30c04ac` | yes |
| Pre-canonical freeze | `6758e538d7349402ae251bad9f8adefab30c04ac` | yes |
| Post-canonical run | `6758e538d7349402ae251bad9f8adefab30c04ac` | yes |

## Memory Smoke

| Check | Result |
| --- | --- |
| `import intergrax.memory.contracts` | PASS |
| `MemoryLifecycleOutcome` import | PASS |
| `UserProfile`, `UserProfileMemoryEntry` public path | PASS |
| `PublicUserProfile is CanonicalUserProfile` | PASS |

## Cancellation Reproducer

| Node | Result |
| --- | --- |
| `test_npsc5e_r2_h2_q1_frozen_regression_closure.py::test_pre_existing_cancellation_fixture_unrelated_to_h2_revision` | PASS (~6.6 s) |
| `tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py::test_terminal_cancellation_survives_process_restart` | PASS (~2.6 s) |

## R2-H2-Q1 Semantic Revalidation

`test_npsc5e_r2_h2_q1_frozen_regression_closure.py` with `-k "not test_mandatory_frozen_suite_passes"` — **12 passed**, 17 deselected (~8.3 s).

## R3 Implementation Semantic Revalidation

`test_npsc5e_r3_child_fanout_partial_recovery.py` with `-k "not test_mandatory_frozen_suite_passes"` — **13 passed**, 9 deselected (~3.4 s).

## W5-A Deterministic Revalidation

`test_critical_overflow_fail_closed_no_silent_loss` — **PASS** (~0.7 s).

## OBS Reconstruction Alignment Revalidation

`tests/unit/runtime/observability/test_obs_trace_1_qualification.py` — **18 passed** (~0.5 s).

## Runtime Observability Regression

`tests/unit/runtime/observability/` — **505 passed**, 5 warnings (~40.0 s).

## Qualification Regression

`tests/unit/testing_support/execution_qualification/` — **216 passed**, 7 skipped (~64.3 s).

Mandatory reference drift (`test_orchestrator_mandatory_reference_drift.py`) — **3 passed** (R1/R2/R3 alignment).

## Pre-Canonical Integrity Check

| Item | Value |
| --- | --- |
| `FULL_CANONICAL_SHA` | `6758e538d7349402ae251bad9f8adefab30c04ac` (= `VERIFICATION_SHA`) |
| Tracked worktree | clean |
| Mandatory smokes §13–§25 | **PASS** |

## Canonical Run Command

```bash
uv run python -m testing_support.execution_qualification.performance \
  --profile npsc5f-final \
  --repetitions 1 \
  --max-parallel 2 \
  --artifact-dir .tmp/session/CLEAN-FINAL-SHA-CANONICAL-VERIFICATION/npsc5f-final
```

## Canonical Run SHA

```text
FULL_CANONICAL_SHA = 6758e538d7349402ae251bad9f8adefab30c04ac
```

Orchestrator-reported `git_head`: `6758e538d7349402ae251bad9f8adefab30c04ac`.

## Canonical Result

| Field | Value |
| --- | --- |
| `run_status_pass` | `true` |
| `certification_decision` | `pass` |
| `run_id` | `perf-npsc5f-final-0` |
| Failed leaf count | 0 |
| Skipped leaf count | 0 (orchestrator receipt) |
| Gate failures | none |

## Certification Decision

```text
pass
```

## Wall Time

```text
191.49 s (measured wall_seconds; session stopwatch ~192 s)
```

## Total Leaf Work

```text
328.03 s (total_leaf_work_seconds)
```

## Effective Concurrency

```text
1.713 (effective_concurrency)
```

Scheduler parallel efficiency estimate: `0.857`.

## Physical Leaf Count

```text
44 (canonical_physical_leaf_count)
```

Legacy logical subprocess count (reference model): `267`. Duplicate execution eliminated: `223` (~83.5%).

## Top Slowest Leaves

| Suite ID | Duration (s) | Result |
| --- | ---: | --- |
| `npsc5f-final.recovery` | 79.20 | PASS |
| `runtime-observability` | 45.58 | PASS |
| `runtime-events` | 17.13 | PASS |
| `npsc5e-r3.mandatory.r2-original` | 15.56 | PASS |
| `npsc5f-final.evidence` | 10.61 | PASS |
| `npsc5e-r2.final-semantic` | 9.73 | PASS |
| `npsc5e-r2.mandatory.r2-h2-q1` | 9.56 | PASS |
| `npsc5e-r3.mandatory.long-running` | 9.23 | PASS |
| `npsc5f-r3.export-boundary` | 7.48 | PASS |
| `npsc5e-r3.mandatory.p0a` | 7.06 | PASS |

## Key Leaf Timings

| Suite ID | Duration (s) | Canonical notes |
| --- | ---: | --- |
| `npsc5e-r2.final-semantic` | 9.73 | 19 selected tests; `test_mandatory_frozen_suite_passes` deselected |
| `npsc5e-r2.mandatory.r2-h2-q1` | 9.56 | 12 semantic tests; frozen nested matrix deselected |
| `npsc5e-r3.implementation-semantic` | ~5.15 (pytest wall in leaf log) | 13 selected; mandatory harness deselected |
| `runtime-observability` | 45.58 | includes W5-A overflow gate; no failure on `test_critical_overflow_fail_closed_no_silent_loss` |
| `runtime-events` | 17.13 | PASS |
| `npsc5f-final.recovery` | 79.20 | slowest leaf |

Leaves **> 10 s** beyond key set: `npsc5f-final.recovery`, `runtime-observability`, `runtime-events`, `npsc5e-r3.mandatory.r2-original`, `npsc5f-final.evidence`.

## Nested Harness Regression

Artifact logs under `.tmp/session/CLEAN-FINAL-SHA-CANONICAL-VERIFICATION/npsc5f-final/perf-npsc5f-final-0/`: **no** execution of `test_mandatory_frozen_suite_passes` on migrated R2/R3 semantic paths. R2/R3 semantic leaves show explicit deselection counts in pytest output.

## R2/R3 Canonical Path Assessment

| Path | Assessment |
| --- | --- |
| R2 Final | Canonical semantic leaf (`npsc5e-r2.final-semantic`); receipt/gate model; no nested subprocess mandatory matrix |
| R2-H2-Q1 | Semantic-only selection (12 tests); frozen suite node deselected |
| R3 implementation | `npsc5e-r3.implementation-semantic`; semantic-only (13 selected, 9 deselected) |
| R3 Final | Mandatory leaves use catalog receipt model; no embedded harness regression observed |
| Execution reconstruction | `npsc5f-r2.execution-reconstruction` → `tests/unit/runtime/observability/reconstruction/test_execution_reconstruction.py` (13 passed) |

## Drift Assessment

`npsc5f-final.drift-sentinel` / `test_npsc5f_final_protected_drift.py` — **13 passed**. Protected drift did **not** block certification.

## Failure Classification

```text
NONE — functional canonical PASS
```

## Performance Comparison

| Reference wall (historical) | Delta vs current (~191.5 s) |
| --- | --- |
| ~167 s | +24.5 s (~+14.7%) |
| ~129 s | +62.5 s (~+48.4%) |

Performance is informational only; correctness gates satisfied.

## Production Changes

```text
NONE (this task)
```

## Qualification Architecture Changes

```text
NONE (this task)
```

## Test Changes

```text
NONE (this task)
```

## Findings

1. Clean tracked worktree and stable `VERIFICATION_SHA` held for entire session including canonical run.
2. All mandatory pre-canonical smokes passed, including OBS reconstruction qualification and R1/R2/R3 reference drift guards.
3. Exactly one `npsc5f-final` run: `certification_decision=pass`, `run_status_pass=true`.
4. W5-A critical overflow test passed in isolation and within `runtime-observability` canonical leaf.
5. Reconstruction SSOT uses `observability/reconstruction`; no stale `runtime/diagnostics` reconstruction path in canonical leaves.
6. No nested `test_mandatory_frozen_suite_passes` harness execution in canonical R2/R3 semantic paths.

## Decision

Formal canonical qualification on `VERIFICATION_SHA` is **complete**. Proceed to **Performance Re-Certification** as separate task.

## Final Verdict

```text
CLEAN FINAL-SHA CANONICAL VERIFICATION = PASS
```

```text
VERIFIED_PLATFORM_SHA = 6758e538d7349402ae251bad9f8adefab30c04ac
DOCUMENTATION_COMMIT_SHA = (recorded after doc-only commit for this task)
```
