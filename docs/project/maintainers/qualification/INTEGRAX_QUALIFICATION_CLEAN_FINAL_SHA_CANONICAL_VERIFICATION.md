# INTEGRAx Qualification — Clean Final-SHA Canonical Verification

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-QUALIFICATION-CLEAN-FINAL-SHA-CANONICAL-VERIFICATION` |
| Mode | Verification-only (no production / qualification-architecture / test fixes in this task) |
| Profile (intended) | `npsc5f-final` |
| Session date | 2026-09-15 |
| Historical W5-A remediation SHA | `38d9d43944a537470a1013b4d6fd407e7b41389b` |

## Session Scope

Single formal `npsc5f-final` canonical certification on a **clean, stable** `development` HEAD. Mandatory pre-canonical smokes, then exactly one full orchestrator run. No implementation work.

## Repository State

| Item | Value |
| --- | --- |
| Branch | `development` |
| `HEAD` | `9af3c82b198af73c19c36e0bca2fb8a05f860565` |
| `origin/development` | `9af3c82b198af73c19c36e0bca2fb8a05f860565` |
| Ahead / behind | 0 / 0 |
| Tracked modified | none |
| Tracked staged | none |
| Untracked (visible to git) | none — `working tree clean` |

Recent tip: `9af3c82b1` — `OBS-RECONSTRUCTION-1: advance R4 quality drift baseline to migration commit`.

## Worktree Integrity

| Check | Result |
| --- | --- |
| `git diff --name-only` | empty |
| `git diff --cached --name-only` | empty |
| Untracked contamination (`intergrax/`, `tests/`, `testing_support/`) | none reported by git |
| Formal clean-SHA precondition | **SATISFIED** |

## Verification SHA

```text
VERIFICATION_SHA = 9af3c82b198af73c19c36e0bca2fb8a05f860565
```

## HEAD Stability Evidence

| Checkpoint | `git rev-parse HEAD` | Match `VERIFICATION_SHA` |
| --- | --- | --- |
| Session start | `9af3c82b198af73c19c36e0bca2fb8a05f860565` | yes |
| After memory smoke | (implicit, same session) | yes |
| After targeted pytest slices | (implicit) | yes |
| After observability + qualification regression | `9af3c82b198af73c19c36e0bca2fb8a05f860565` | yes |
| Post-canonical (not run) | — | — |

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
| `test_npsc5e_r2_h2_q1_frozen_regression_closure.py::test_pre_existing_cancellation_fixture_unrelated_to_h2_revision` | PASS |
| `tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py::test_terminal_cancellation_survives_process_restart` | PASS |

Wall for pair: ~8.2 s.

## R2-H2-Q1 Semantic Revalidation

`test_npsc5e_r2_h2_q1_frozen_regression_closure.py` with `-k "not test_mandatory_frozen_suite_passes"` — **12 passed**, 17 deselected (~8.7 s).

## R3 Implementation Semantic Revalidation

`test_npsc5e_r3_child_fanout_partial_recovery.py` with `-k "not test_mandatory_frozen_suite_passes"` — **13 passed**, 9 deselected (~4.1 s).

## W5-A Deterministic Revalidation

`test_critical_overflow_fail_closed_no_silent_loss` — **PASS** (~0.8 s).

## Runtime Observability Regression

`tests/unit/runtime/observability/` — **504 passed**, **1 failed** (~40.1 s).

Failure:

| Test | Exception |
| --- | --- |
| `test_obs_trace_1_qualification.py::test_gate_factual_reconstruction_does_not_import_trace_event` | `FileNotFoundError`: `intergrax/runtime/diagnostics/execution_reconstruction.py` (module now at `intergrax/runtime/observability/reconstruction/execution_reconstruction.py`) |

**Mandatory smoke: FAIL.**

## Qualification Regression

`tests/unit/testing_support/execution_qualification/` — **214 passed**, 7 skipped, **2 failed** (~59.8 s).

Failures:

| Test | Drift |
| --- | --- |
| `test_orchestrator_mandatory_reference_drift.py::test_npsc5f_r2_final_orchestrator_reference_matches_canonical` | Index 7: orchestrator `_MANDATORY_SUITES` still lists `tests/unit/runtime/diagnostics/test_execution_reconstruction.py`; canonical catalog expects `tests/unit/runtime/observability/reconstruction/test_execution_reconstruction.py` |
| `test_orchestrator_mandatory_reference_drift.py::test_npsc5f_r3_final_orchestrator_reference_matches_canonical` | Same reconstruction path mismatch at index 10 |

**Mandatory smoke: FAIL.**

## Pre-Canonical Integrity Check

| Item | Value |
| --- | --- |
| `FULL_CANONICAL_SHA` (would-be) | `9af3c82b198af73c19c36e0bca2fb8a05f860565` (= `VERIFICATION_SHA`) |
| Tracked worktree | clean |
| Mandatory smokes §22–§23 | **FAIL** |

Full canonical run **not started** (STOP per mandatory pre-canonical gates).

## Canonical Run Command

```bash
uv run python -m testing_support.execution_qualification.performance \
  --profile npsc5f-final \
  --repetitions 1 \
  --max-parallel 2 \
  --artifact-dir .tmp/session/CLEAN-FINAL-SHA-CANONICAL-VERIFICATION/npsc5f-final
```

## Canonical Run SHA

Not executed.

## Canonical Result

| Field | Value |
| --- | --- |
| `run_status_pass` | n/a (not run) |
| `certification_decision` | n/a (not run) |

## Certification Decision

n/a — canonical orchestrator not invoked.

## Wall Time

n/a (no full run).

## Total Leaf Work

n/a.

## Effective Concurrency

n/a.

## Slowest Leaves

n/a (no orchestrator receipt).

## Key Leaf Timings

n/a for canonical context. Pre-canonical targeted slices passed where executed (R2-H2-Q1, R3 implementation semantic, W5-A exact).

## Nested Harness Regression

Not evaluated in canonical path (run skipped). Semantic slices explicitly excluded `test_mandatory_frozen_suite_passes` and passed.

## Drift Assessment

Protected drift baseline not exercised (no orchestrator). Pre-run catalog/orchestrator reference drift tests **failed** on reconstruction path SSOT — indicates incomplete alignment after OBS-RECONSTRUCTION-1 migration on `9af3c82b`.

## Failure Classification

```text
QUALIFICATION DEFECT (SSOT / gate reference lag after reconstruction migration)
```

With secondary **TEST/FIXTURE DEFECT** surface: OBS-TRACE-1 gate still enumerates removed diagnostics module path.

## Performance Comparison

| Reference | Delta |
| --- | --- |
| Current wall | n/a |
| vs ~167 s | n/a |
| vs ~129 s | n/a |

## Production Changes

```text
NONE (this task)
```

## Qualification Architecture Changes

```text
NONE (this task)
```

## Findings

1. Clean-SHA and HEAD-stability preconditions are met on `9af3c82b`.
2. Memory contract smokes, cancellation pair, R2-H2-Q1 / R3 semantic slices, and W5-A overflow test pass.
3. Runtime observability directory regression fails because `test_obs_trace_1_qualification.py` references a deleted diagnostics module file.
4. Qualification catalog drift tests fail because R2/R3 Final orchestrator mandatory lists still point at diagnostics reconstruction tests while canonical catalog SSOT lists observability/reconstruction tests.
5. These failures block mandatory §22–§23 gates; full `npsc5f-final` was correctly **not** executed (no second attempt).

## Decision

Do not run canonical certification until reconstruction path alignment is fixed in a **separate** implementation task. This verification task performs no fixes.

## Final Verdict

```text
CLEAN FINAL-SHA CANONICAL VERIFICATION = BLOCKED — MANDATORY PRE-CANONICAL SMOKE FAILED
```

```text
PRIMARY BLOCKER: OBS-RECONSTRUCTION-1 incomplete SSOT — gate module list and R2/R3 Final `_MANDATORY_SUITES` still reference `runtime/diagnostics` reconstruction paths removed on VERIFICATION_SHA.
SECONDARY CONTRIBUTORS: none equivalent; W5-A and targeted semantic slices passed.
```

```text
VERIFIED_PLATFORM_SHA = 9af3c82b198af73c19c36e0bca2fb8a05f860565
DOCUMENTATION_COMMIT_SHA = (not committed — canonical PASS prerequisite not met; §50 commit applies only on PASS)
```
