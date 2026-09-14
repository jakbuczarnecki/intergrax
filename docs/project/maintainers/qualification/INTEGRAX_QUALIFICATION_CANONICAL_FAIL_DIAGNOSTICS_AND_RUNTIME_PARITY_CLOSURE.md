# INTEGRAx Qualification — Canonical FAIL Diagnostics and Runtime Parity Closure

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-QUALIFICATION-CANONICAL-FAIL-DIAGNOSTICS-AND-RUNTIME-PARITY-CLOSURE` |
| Diagnostic HEAD | `5f2ec87000af7ff258a191e5fe107901f5d2b649` |
| Semantic parity reference | `dffe2ae52a6938e0620ae4a3cc5e4be760e7f2f7` |
| Profile | `npsc5f-final` |
| Closure re-run | `repetitions=1`, `max_parallel=2`, wall **938.35 s**, `run_status_pass=true` |

## Scope

Diagnose two live FAIL leaves from performance certification, classify root cause, apply minimal remediation, confirm canonical runtime parity. No production `intergrax/` changes, no semantic weakening, no scheduler tuning.

## Initial Failures

| suite_id | Symptom (performance run) |
| --- | --- |
| `npsc5f-final.cancellation` | `test_terminal_cancellation_survives_process_restart` — `CheckpointNotResumableError` (`TaskState.CREATED`) |
| `npsc5f-r4.final-drift-sentinel` | Protected drift: `testing_support/npsc5f_r4_regression_matrix.py` since `b0a465fc…` |

## Canonical Suite Definitions

### `npsc5f-final.cancellation`

| Field | Value |
| --- | --- |
| `pytest_arguments` | `tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py`, `tests/unit/runtime/cancellation/test_p0c5a_explicit_terminal_wiring.py` |
| Canonical command | `uv run pytest tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py tests/unit/runtime/cancellation/test_p0c5a_explicit_terminal_wiring.py` |
| `exclusive_resource_id` | `None` |
| `environment_overrides` | `()` |
| Suite timeout (live runner) | `21600` s |

### `npsc5f-r4.final-drift-sentinel`

| Field | Value |
| --- | --- |
| `pytest_arguments` | `tests/unit/testing_support/test_npsc5f_r4_final_protected_drift.py` |
| Canonical command | `uv run pytest tests/unit/testing_support/test_npsc5f_r4_final_protected_drift.py` |
| `exclusive_resource_id` | `None` |
| `environment_overrides` | `()` |
| Suite timeout (live runner) | `21600` s |

## Reproduction Matrix

| Suite | Direct pytest | Executor | Coordinator single | Profile mp=1 | Profile mp=2 | Classification |
| --- | --- | --- | --- | --- | --- | --- |
| `npsc5f-final.cancellation` | FAIL → **PASS** after fix | PASS | PASS | not isolated (full profile) | **PASS** (closure run) | **STALE TEST EXPECTATION** (+ dependent meta-tests) |
| `npsc5f-r4.final-drift-sentinel` | FAIL → **PASS** after baseline | PASS | PASS | not isolated | **PASS** (closure run) | **DRIFT BASELINE MISMATCH** (expected SSOT migration) |

Pre-remediation: both leaves **FAIL** under direct canonical pytest — not coordinator/concurrency.

## Cancellation Diagnostics

- **Failing test:** `test_terminal_cancellation_survives_process_restart`
- **Assertion:** `LongRunningCoordinator.persist_checkpoint` raised `CheckpointNotResumableError` for default `TaskState.CREATED`
- **Evidence:** Same failure in `.tmp/session/qualification-performance/perf-npsc5f-final-1/npsc5f-final.cancellation.log`
- **Fix:** Set `state=TaskState.WAITING_FOR_HUMAN` on the fixture `Task` (aligned with `test_checkpoint_store.py` and R2 persist-gate semantics). Terminal cancellation / restart behavior under test unchanged.
- **Follow-on:** Frozen regression meta-tests expected the old invalid fixture to fail (`proc.returncode != 0`); updated to expect PASS. Removed obsolete `-k not test_terminal_cancellation_survives_process_restart` exclusion from `npsc5f_final_regression_matrix.py`.

## Drift Sentinel Diagnostics

| Item | Value |
| --- | --- |
| Previous baseline | `b0a465fc0b8e9f1c9b9e2879f94510fc88b77516` |
| New baseline | `dffe2ae52a6938e0620ae4a3cc5e4be760e7f2f7` |
| Previous fingerprint (drift) | `testing_support/npsc5f_r4_regression_matrix.py` |
| Actual at diagnostic HEAD | Same path changed in `1dc93afab` (SSOT import from `mandatory_sources`) |
| Protected scope | R4 reconstruction surfaces + `npsc5f_r4_regression_matrix.py` — **no `intergrax/` drift** |
| Why allowed | Expected consequence of audited canonical qualification architecture (`INTEGRAx-QUALIFICATION-FULL-ORCHESTRATOR-LEAF-ELIMINATION-AND-SSOT-CLOSURE`); qualification-only matrix wiring |
| Architecture reopen | **Not required** — production protected paths unchanged since new baseline |

## Isolation Analysis

- Per-run artifact roots: `perf-npsc5f-final-{n}/` under `.tmp/session/qualification-performance/`
- Unique `run_id` per repetition (`perf-npsc5f-final-0`, …)
- Suite logs per leaf under run artifact root
- No basetemp collision identified for the two original FAIL leaves

## Concurrency Analysis

- Original FAILs reproduced standalone — **not** `max_parallel=2` overlap
- Closure run `max_parallel=2`: **PASS** (all 43 leaves)

## Resource Exclusivity Analysis

Neither failing leaf declares `exclusive_resource_id`. No catalog metadata change required.

## Root Cause Classification

| suite_id | Primary classification |
| --- | --- |
| `npsc5f-final.cancellation` | **STALE TEST EXPECTATION** |
| `npsc5f-r4.final-drift-sentinel` | **DRIFT BASELINE MISMATCH** |

## Remediation

1. Cancellation fixture resumable task state + meta-test expectation updates (qualification tests only).
2. `R4_POST_QUALIFIED_BASELINE_SHA` → `dffe2ae52a6938e0620ae4a3cc5e4be760e7f2f7` with recorded test assertion update.
3. Regression matrix kexpr: allow `test_terminal_cancellation_survives_process_restart` in final matrix.

## Regression

- `uv run pytest tests/unit/testing_support/execution_qualification/ -q` — **169 passed**, 7 skipped
- Failing leaf targets — **PASS**
- `uv run python -m testing_support.execution_qualification.performance --profile npsc5f-final --repetitions 1 --max_parallel 2` — **PASS**

## Canonical Full Re-run

| Field | Value |
| --- | --- |
| Wall | 938.35 s |
| `run_status_pass` | `true` |
| Slowest leaf | `npsc5f-final.recovery` (~682.6 s) |

## Runtime Parity Decision

**CANONICAL RUNTIME PARITY CLOSURE = PASS**

## Performance Certification Status

Runtime parity restored. Full multi-repetition performance re-certification remains **NEXT** (out of scope for this task).

## Production Changes

**NONE**

## Findings

1. Performance FAIL was not scheduler/isolation — direct pytest reproduced both leaves.
2. First closure attempt fixed original two leaves but exposed dependent meta-tests encoding the old invalid cancellation fixture.
3. Drift sentinel correctly flagged SSOT migration until baseline advanced to semantic parity commit.

## Final Verdict

**CANONICAL RUNTIME PARITY CLOSURE = PASS**
