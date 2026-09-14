# INTEGRAx-QUALIFICATION-SLOW-LEAF-ANALYSIS-AND-RECOVERY-DECOMPOSITION

## Metadata

| Field | Value |
| --- | --- |
| Task | `INTEGRAx-QUALIFICATION-SLOW-LEAF-ANALYSIS-AND-RECOVERY-DECOMPOSITION` |
| Branch | `development` |
| Analysis HEAD | `a189282b35e7a4ac549489f0d06b7acd271e9a5b` |
| Prior closure | `INTEGRAx-QUALIFICATION-CANONICAL-FAIL-DIAGNOSTICS-AND-RUNTIME-PARITY-CLOSURE` |
| Full profile baseline (observed) | `npsc5f-final`, `max_parallel=2`, wall ≈ 938.35 s, PASS |
| Slow leaf baseline (observed) | `npsc5f-final.recovery` ≈ 682.6 s |

## Scope

Analyze compiled definition and runtime cost of `npsc5f-final.recovery`, attribute wall time to subcomponents, detect overlap with other canonical leaves, and apply catalog decomposition only when semantically safe. No production code changes. No full `npsc5f-final` re-run for this task (analysis-only outcome).

## Baseline

From runtime parity closure on `a189282b35e7a4ac549489f0d06b7acd271e9a5b`:

- Dominant leaf: `npsc5f-final.recovery` (~73% of full-profile wall @ mp=2).
- Remaining leaves run largely in parallel; shrinking recovery wall is the primary qualification-runtime lever.

## Recovery Suite Definition

Compiled from `build_default_qualification_catalog().compile_profile("npsc5f-final")`:

| Field | Literal value |
| --- | --- |
| `suite_id` | `npsc5f-final.recovery` |
| `pytest_arguments` | `('tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py', 'tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py', 'tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py')` |
| `exclusive_resource_id` | `None` |
| `environment_overrides` | `()` |
| Per-suite subprocess timeout (runner) | `LIVE_SUITE_TIMEOUT_SECONDS` = `21600.0` |

SSOT declaration: `testing_support/execution_qualification/catalog/mandatory_sources.py` → `_NPSC5F_FINAL_EXTRA_SUITES` label `"Recovery"`.

## Structural Breakdown

| Kind | Content |
| --- | --- |
| Files (3) | R1 final retry qual; R2 final checkpoint durable resume qual; R3 final child fanout partial recovery qual |
| Directories | — |
| `-k` / marks / extra flags | — |
| Orchestrator-as-leaf | **No** — multi-file vector bypasses single-file orchestrator expansion (`is_nested_orchestrator_leaf` → false) |

Note: each path alone is listed in `CANONICAL_ORCHESTRATOR_PATHS`; a **single-file** mandatory entry would expand to full R2/R3 mandatory subgraphs. The three-file bundle is intentional: one pytest collection, no orchestrator expansion.

## Measurement Methodology

- **NO FULL PROFILE RUNS** during analysis.
- Subprocess command matches qualification executor: `uv run pytest <pytest_arguments>`.
- Artifacts under `.tmp/session/slow-leaf-recovery/`.
- Each timing labeled **MEASURED** below; baseline 682.6 s from prior certified run is **observation**, not re-measured here.

## Subgroup Timing

| Target | Wall (s) | Tests / outcome | PASS/FAIL |
| --- | ---: | --- | --- |
| Whole recovery (3 files) | **626.47** MEASURED | 119 passed | PASS |
| R1 file only | **8.85** MEASURED | (subset) | not used for decomposition |
| R2 file only | **307.71** MEASURED | 36 passed, 5 failed | FAIL |
| R3 file only | **472.81** MEASURED | (subset) | isolated run unreliable vs bundle |
| R1+R2 | **308.39** MEASURED | 90 passed, 5 failed | FAIL |
| R2+R3 | **711.68** MEASURED | 59 passed, 6 failed | FAIL |
| R1+R3 | **377.41** MEASURED | 77 passed, 1 failed | FAIL |

Pairwise/single-file runs fail mandatory frozen-suite harness tests that pass when all three modules are collected in **one** pytest process.

## Slowest Components

**MEASURED** `--durations=15` on the three-file vector (representative run):

| Duration (s) | Test |
| ---: | --- |
| 399.34 | `test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py::test_mandatory_frozen_suites_pass_via_parallel_qualification` |
| 121.83 | `test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py::test_mandatory_frozen_suite_passes[R2-H2-Q1]` |
| 18.30 | `...::test_mandatory_frozen_suite_passes[R2 Original]` |
| (remaining top entries) | Mostly `test_mandatory_frozen_suite_passes[...]` in R2 module |

Root cost is **in-test re-execution of mandatory qualification slices** (nested qualification / parallel harness), not pytest collection overhead.

## Existing Suite Overlap

**Same pytest path on another physical leaf (duplicate execution in full plan):**

| Recovery path | Also on leaf |
| --- | --- |
| `test_npsc5e_r1_final_retry_attempt_qualification.py` | `npsc5e-r3.mandatory.r1-final` |

**Logical overlap (not same leaf pytest vector):** R2/R3 final modules embed `test_mandatory_frozen_suite_passes[...]` targeting suites that also exist as separate canonical leaves elsewhere in `npsc5f-final` — work is re-run **inside** the recovery subprocess by design of the final qualification tests.

Removing the standalone `npsc5e-r3.mandatory.r1-final` leaf would save ~9 s wall but requires catalog gate/composition rules; savings &lt;10% of recovery → out of scope for structural decomposition.

## Decomposition Feasibility

| Criterion | Result |
| --- | --- |
| Independent pytest vectors per semantic slice | **No** — R2/R3 alone fail; only the 3-file vector matches closure semantics |
| Single-file leaves without orchestrator expansion | **No** — triggers expansion or fails harness |
| Order / shared collection state | **Yes** — shared pytest session required |
| Coverage-preserving split | **No** without changing test or compiler semantics |
| Parallelism via split | **No** — subprocesses cannot be split; dominant tests are sequential inside one leaf |

## Proposed / Applied Decomposition

**Applied:** none (analysis-only).

**Rejected:** splitting into R1/R2/R3 leaves, R2+R3 sub-leaf, or recovery aggregate gate over child suites — would break mandatory frozen-suite qualification or change coverage via orchestrator expansion.

## Coverage Parity

No catalog change. Parity unchanged vs `a189282b`.

## Suite Identity

`npsc5f-final.recovery` remains the single physical leaf for the three-file vector per global registry (`NPSC5F_FINAL_EXTRA_LABEL_TO_SUITE_ID`).

## Resource Classification

| Leaf | `exclusive_resource_id` | Parallelism |
| --- | --- | --- |
| `npsc5f-final.recovery` | `None` | May run concurrently with other non-exclusive leaves @ mp=2; internally heavy CPU/time |

## Parallel Safety

No new leaves introduced. Existing catalog parallel-safe metadata unchanged.

## Recovery-Only Benchmark

| Variant | Physical suites | Wall (s) | Total work | Effective concurrency |
| --- | ---: | ---: | --- | --- |
| Before (observed leaf) | 1 | ~682.6 (observation) | 1× 3-file pytest | 1 subprocess |
| Before (MEASURED direct pytest) | 1 | 626.47 | 119 tests | 1 subprocess |
| After (proposed split) | — | — | — | **Not applied** |

Splitting would not reduce total work; isolated subprocesses fail or inflate wall (e.g. R2+R3 **711.68 s** MEASURED with failures).

## Full Verification

Not executed — no composition change (per task §67).

## Production Changes

```text
NONE
```

## Findings

1. Recovery is a **fixed three-file pytest vector**, not a gate orchestrator leaf.
2. ~64% of measured recovery pytest time sits in one R3 test (`test_mandatory_frozen_suites_pass_via_parallel_qualification`).
3. ~19% in R2 `test_mandatory_frozen_suite_passes[R2-H2-Q1]`.
4. Subprocess **cannot** be decomposed without semantic regression (pairwise/single-file runs fail).
5. Minor duplicate: R1 module also runs as `npsc5e-r3.mandatory.r1-final` (~9 s), &lt;10% recovery improvement if removed safely.
6. Next optimization lever: **test-runtime** of embedded mandatory harness tests, not qualification DAG decomposition.

## Decision

```text
DECOMPOSITION NOT JUSTIFIED
```

## Final Verdict

```text
RECOVERY DECOMPOSITION = NOT JUSTIFIED
```
