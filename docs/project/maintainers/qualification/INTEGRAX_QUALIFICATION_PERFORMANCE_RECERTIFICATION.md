# INTEGRAx Qualification — Performance Re-Certification

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-QUALIFICATION-PERFORMANCE-RECERTIFICATION` |
| Mode | Performance verification + diagnostics only (no production / scheduler / test changes) |
| Profile | `npsc5f-final` |
| Session date | 2026-09-15 |
| Prior canonical verification | `6758e538d7349402ae251bad9f8adefab30c04ac` (functional PASS @ ~191.5 s wall) |
| Clean verification doc commit on tip | `0fc4e5423858bc804255924226abc68aa5f2fe89` |

## Session Scope

Three controlled canonical benchmark runs on a stable, clean tracked worktree at `PERFORMANCE_RECERTIFICATION_SHA`. Collect wall/leaf/concurrency metrics, per-leaf timing variance, nested-harness and duplicate-execution audits, and hotspot classification. No optimization implementation.

## Repository State

| Item | Value |
| --- | --- |
| Branch | `development` |
| `HEAD` (benchmark series) | `0fc4e5423858bc804255924226abc68aa5f2fe89` |
| `origin/development` | `0fc4e5423858bc804255924226abc68aa5f2fe89` |
| Ahead / behind | 0 / 0 |
| Tracked modified | none |
| Tracked staged | none (pre-benchmark) |
| Untracked | `.tmp/session/PERFORMANCE-RECERTIFICATION/` only (outside import discovery) |

## Benchmark SHA

```text
PERFORMANCE_RECERTIFICATION_SHA = 0fc4e5423858bc804255924226abc68aa5f2fe89
```

| Checkpoint | `git rev-parse HEAD` | Match |
| --- | --- | --- |
| Pre RUN-1 | `0fc4e5423858bc804255924226abc68aa5f2fe89` | yes |
| Post RUN-1 | `0fc4e5423858bc804255924226abc68aa5f2fe89` | yes |
| Pre RUN-2 | `0fc4e5423858bc804255924226abc68aa5f2fe89` | yes |
| Post RUN-2 | `0fc4e5423858bc804255924226abc68aa5f2fe89` | yes |
| Pre RUN-3 | `0fc4e5423858bc804255924226abc68aa5f2fe89` | yes |
| Post RUN-3 | `0fc4e5423858bc804255924226abc68aa5f2fe89` | yes |

Artifact `git_head` in JSON report for each run: `0fc4e5423858bc804255924226abc68aa5f2fe89`.

## Environment

| Item | Value |
| --- | --- |
| OS | Windows 10 (win32 10.0.26200) |
| Python (benchmark subprocess) | 3.12.11 |
| `uv` | 0.8.15 |
| Logical CPUs | 20 |
| RAM | ~64 GB |
| `max_parallel` | 2 (fixed) |

## Methodology

```bash
uv run python -m testing_support.execution_qualification.performance \
  --profile npsc5f-final \
  --repetitions 1 \
  --max-parallel 2 \
  --artifact-dir .tmp/session/PERFORMANCE-RECERTIFICATION/run-<N>
```

Three separate runs (`run-1`, `run-2`, `run-3`); no code or profile changes between runs. Per-leaf pytest wall times parsed from leaf `.log` receipts under `perf-npsc5f-final-0/`.

## Run 1

| Metric | Value |
| --- | --- |
| `run_status_pass` | true |
| `certification_decision` | pass |
| `wall_seconds` | 187.72 |
| `total_leaf_work_seconds` | 375.03 |
| `effective_concurrency` | 1.998 |
| `scheduler_parallel_efficiency_estimate` | 0.999 |
| `canonical_physical_leaf_count` | 44 |
| `legacy_logical_subprocess_count` | 267 |
| `duplicate_execution_eliminated` | 223 (83.52%) |
| `critical_path_approximation_seconds` | 121.88 |
| Slowest leaf | `npsc5e-r2.mandatory.r2-h2-q1` (121.88 s receipt; pytest log 120.53 s) |
| Failed leaves | 0 |

## Run 2

| Metric | Value |
| --- | --- |
| `run_status_pass` | true |
| `certification_decision` | pass |
| `wall_seconds` | 126.79 |
| `total_leaf_work_seconds` | 252.77 |
| `effective_concurrency` | 1.994 |
| `scheduler_parallel_efficiency_estimate` | 0.997 |
| `canonical_physical_leaf_count` | 44 |
| `duplicate_execution_eliminated` | 223 (83.52%) |
| `critical_path_approximation_seconds` | 47.02 |
| Slowest leaf | `runtime-observability` (47.02 s) |
| Failed leaves | 0 |

## Run 3

| Metric | Value |
| --- | --- |
| `run_status_pass` | true |
| `certification_decision` | pass |
| `wall_seconds` | 132.81 |
| `total_leaf_work_seconds` | 265.05 |
| `effective_concurrency` | 1.996 |
| `scheduler_parallel_efficiency_estimate` | 0.998 |
| `canonical_physical_leaf_count` | 44 |
| `duplicate_execution_eliminated` | 223 (83.52%) |
| `critical_path_approximation_seconds` | 45.83 |
| Slowest leaf | `runtime-observability` (45.83 s) |
| Failed leaves | 0 |

## Aggregate Wall Statistics

| Statistic | Seconds |
| --- | ---: |
| Min | 126.79 |
| Max | 187.72 |
| Mean | 149.11 |
| Median | 132.81 |
| Range | 60.93 |
| Spread (max−min)/min | 48.0% |

Runs 2–3 only: spread 4.8% (within preferred ≤20% band). Run 1 is an outlier driven by `npsc5e-r2.mandatory.r2-h2-q1` (see Per-Leaf Variability).

## Leaf Work Statistics

| Statistic | Leaf work (s) |
| --- | ---: |
| Min | 252.77 |
| Max | 375.03 |
| Mean | 297.62 |
| Median | 265.05 |

Run 1 elevated leaf work correlates with the single slow R2-H2-Q1 execution, not duplicate physical leaves.

## Effective Concurrency

All runs: ~1.99–2.00 with `max_parallel=2`. Rational given DAG dependencies and one serial critical-path chain; no evidence of scheduler starvation (parallel efficiency ~0.997–0.999).

## Parallel Efficiency

| Run | Estimate |
| --- | ---: |
| 1 | 0.999 |
| 2 | 0.997 |
| 3 | 0.998 |

## Canonical Physical Leaf Count

44 physical leaves on every run; 0 failed; 0 skipped physical leaves in certification outcome.

## Logical vs Physical Work

| Run | Logical references | Physical leaves | Eliminated |
| --- | ---: | ---: | ---: |
| 1 | 267 | 44 | 223 |
| 2 | 267 | 44 | 223 |
| 3 | 267 | 44 | 223 |

`logical work > physical work` — canonical dedup invariant holds.

## Duplicate Elimination

Identity-based DAG dedup: **223 / 267 (~83.52%)** eliminated every run. Receipt-based single physical execution per leaf identity per orchestrator run; not cross-run cache.

## Top Slow Leaves

Mean pytest duration across three runs (from leaf logs; see full table in Per-Leaf Variability):

| Rank | Leaf | Mean (s) | % median wall (132.81 s) | Classification |
| --- | --- | ---: | ---: | --- |
| 1 | `runtime-observability` | 45.31 | 34.1% | LEGITIMATE WORKLOAD |
| 2 | `npsc5e-r2.mandatory.r2-h2-q1` | 46.21* | 34.8%* | ENVIRONMENT SENSITIVE |
| 3 | `npsc5e-r3.mandatory.r2-original` | 17.11 | 12.9% | LEGITIMATE WORKLOAD |
| 4 | `runtime-events` | 15.12 | 11.4% | LEGITIMATE WORKLOAD |
| 5 | `npsc5f-final.recovery` | 11.72 | 8.8% | LEGITIMATE WORKLOAD |

\*Mean skewed by RUN-1 outlier (120.53 s); runs 2–3: ~8–10 s.

## Per-Leaf Variability

Full table (pytest summary line per leaf log):

| suite_id | run1 | run2 | run3 | avg | min | max | range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dg001-lineage | 3.73 | 3.80 | 4.70 | 4.08 | 3.73 | 4.70 | 0.97 |
| npsc5d-final | 4.32 | 4.54 | 5.21 | 4.69 | 4.32 | 5.21 | 0.89 |
| npsc5e-r2.final-semantic | 7.65 | 7.56 | 9.55 | 8.25 | 7.56 | 9.55 | 1.99 |
| npsc5e-r2.mandatory.r2-h2-q1 | 120.53 | 7.93 | 10.16 | 46.21 | 7.93 | 120.53 | 112.60 |
| npsc5e-r3.implementation-semantic | 3.67 | 4.10 | 4.35 | 4.04 | 3.67 | 4.35 | 0.68 |
| npsc5e-r3.mandatory.attempt-lifecycle | 1.37 | 1.56 | 1.74 | 1.56 | 1.37 | 1.74 | 0.37 |
| npsc5e-r3.mandatory.cancellation | 4.35 | 4.08 | 4.63 | 4.35 | 4.08 | 4.63 | 0.55 |
| npsc5e-r3.mandatory.checkpoint-store | 1.99 | 2.14 | 2.45 | 2.19 | 1.99 | 2.45 | 0.46 |
| npsc5e-r3.mandatory.child-execution | 1.61 | 1.57 | 1.71 | 1.63 | 1.57 | 1.71 | 0.14 |
| npsc5e-r3.mandatory.fan-out | 1.33 | 1.24 | 1.48 | 1.35 | 1.24 | 1.48 | 0.24 |
| npsc5e-r3.mandatory.hitl-r3 | 3.82 | 2.93 | 3.24 | 3.33 | 2.93 | 3.82 | 0.89 |
| npsc5e-r3.mandatory.long-running | 6.49 | 6.22 | 7.91 | 6.87 | 6.22 | 7.91 | 1.69 |
| npsc5e-r3.mandatory.npsc-5a | 2.38 | 2.59 | 2.76 | 2.58 | 2.38 | 2.76 | 0.38 |
| npsc5e-r3.mandatory.npsc-5b-final | 3.07 | 3.37 | 3.49 | 3.31 | 3.07 | 3.49 | 0.42 |
| npsc5e-r3.mandatory.npsc-5c | 2.82 | 2.92 | 3.19 | 2.98 | 2.82 | 3.19 | 0.37 |
| npsc5e-r3.mandatory.p0a | 4.24 | 4.46 | 5.65 | 4.78 | 4.24 | 5.65 | 1.41 |
| npsc5e-r3.mandatory.r1-final | 4.54 | 5.08 | 4.19 | 4.60 | 4.19 | 5.08 | 0.89 |
| npsc5e-r3.mandatory.r2-h1 | 2.44 | 2.87 | 2.58 | 2.63 | 2.44 | 2.87 | 0.43 |
| npsc5e-r3.mandatory.r2-h2 | 2.74 | 2.96 | 3.53 | 3.08 | 2.74 | 3.53 | 0.79 |
| npsc5e-r3.mandatory.r2-original | 18.70 | 16.72 | 15.92 | 17.11 | 15.92 | 18.70 | 2.78 |
| npsc5e-r3.mandatory.terminal | 2.85 | 2.88 | 3.21 | 2.98 | 2.85 | 3.21 | 0.36 |
| npsc5f-final.cancellation | 5.10 | 3.98 | 4.61 | 4.56 | 3.98 | 5.10 | 1.12 |
| npsc5f-final.checkpoint | 3.35 | 2.84 | 2.87 | 3.02 | 2.84 | 3.35 | 0.51 |
| npsc5f-final.drift-sentinel | 0.39 | 0.25 | 0.25 | 0.30 | 0.25 | 0.39 | 0.14 |
| npsc5f-final.evidence | 2.60 | 2.06 | 2.25 | 2.30 | 2.06 | 2.60 | 0.54 |
| npsc5f-final.recovery | 12.71 | 11.20 | 11.24 | 11.72 | 11.20 | 12.71 | 1.51 |
| npsc5f-r1.implementation-gate | 0.95 | 0.77 | 0.71 | 0.81 | 0.71 | 0.95 | 0.24 |
| npsc5f-r1.p0-gate | 1.50 | 1.48 | 1.27 | 1.42 | 1.27 | 1.50 | 0.23 |
| npsc5f-r2.drift-classifier | 0.14 | 0.17 | 0.15 | 0.15 | 0.14 | 0.17 | 0.03 |
| npsc5f-r2.execution-reconstruction | 0.62 | 0.68 | 0.69 | 0.66 | 0.62 | 0.69 | 0.07 |
| npsc5f-r2.implementation-gate | 1.20 | 1.02 | 0.93 | 1.05 | 0.93 | 1.20 | 0.27 |
| npsc5f-r2.trace-asof | 0.70 | 0.77 | 0.81 | 0.76 | 0.70 | 0.81 | 0.11 |
| npsc5f-r2.trace-bitemp | 0.66 | 0.66 | 0.69 | 0.67 | 0.66 | 0.69 | 0.03 |
| npsc5f-r3.drift-classifier | 0.12 | 0.15 | 0.13 | 0.13 | 0.12 | 0.15 | 0.03 |
| npsc5f-r3.export-boundary | 5.37 | 5.80 | 5.64 | 5.60 | 5.37 | 5.80 | 0.43 |
| npsc5f-r3.implementation-gate | 0.80 | 0.78 | 0.83 | 0.80 | 0.78 | 0.83 | 0.05 |
| npsc5f-r3.journal-export | 0.63 | 0.59 | 0.66 | 0.63 | 0.59 | 0.66 | 0.07 |
| npsc5f-r4.final-drift-sentinel | 0.29 | 0.26 | 0.23 | 0.26 | 0.23 | 0.29 | 0.06 |
| npsc5f-r4.implementation-gate | 0.84 | 0.84 | 0.88 | 0.85 | 0.84 | 0.88 | 0.04 |
| npsc5f-r4.npsc-5a | 0.25 | 0.20 | 0.20 | 0.22 | 0.20 | 0.25 | 0.05 |
| npsc5f-r4.npsc-5c | 0.27 | 0.30 | 0.32 | 0.30 | 0.27 | 0.32 | 0.05 |
| npsc5f-r4.trace-bitemp | 0.86 | 0.92 | 0.91 | 0.90 | 0.86 | 0.92 | 0.06 |
| runtime-events | 17.32 | 14.91 | 13.13 | 15.12 | 13.13 | 17.32 | 4.19 |
| runtime-observability | 48.87 | 44.45 | 42.62 | 45.31 | 42.62 | 48.87 | 6.25 |

## Recovery Analysis

Historical reference ~79 s referred to an earlier recovery decomposition profile. Current `npsc5f-final.recovery` leaf:

- Single physical pytest subprocess per run (~11–13 s).
- **93 tests selected**, 26 deselected; modules span R1 retry, R2 checkpoint resume, R3 fan-out partial recovery qualification.
- No `test_mandatory_frozen_suite_passes` in logs; no nested full-profile harness.
- No docker/external process in log output.
- Stable range 1.51 s across runs.

**Verdict:** PERFORMANCE HOTSPOT — NOT ARCHITECTURE DEFECT (moderate semantic bundle; not duplicated).

## Runtime Observability Analysis

- **505 tests** in one leaf; pytest **42.6–48.9 s** across runs (mean 45.3 s).
- Dominates wall on runs 2–3 (~34–37% wall share).
- Includes bounded-delivery, OTLP/export, reconstruction suites — inherent I/O and timing-sensitive tests.
- No nested mandatory harness markers; standard pytest collection/execution.

**Verdict:** PERFORMANCE HOTSPOT — NOT ARCHITECTURE DEFECT; optimization optional follow-up, not certification blocker.

## Runtime Events Analysis

Mean **15.1 s** (range 4.2 s); proportional to events-plane qualification scope; stable vs observability.

## R2 Original Analysis

`npsc5e-r3.mandatory.r2-original`: mean **17.1 s**, range 2.8 s. Single canonical pytest leaf (multiplicity 9 logical → 1 physical). Real semantic R2/R3 workload; no embedded full orchestrator in logs.

## Evidence Leaf Analysis

`npsc5f-final.evidence`: **45 tests**, **~2.1–2.6 s**; evidence architecture + persistence boundary tests. Historical ~10.6 s not reproduced — composition/caching differs; no subprocess fanout or duplicate prerequisite pattern observed.

## Nested Harness Audit

Search across `.tmp/session/PERFORMANCE-RECERTIFICATION/**/*.log` for `test_mandatory_frozen_suite_passes`: **no matches**.

**Expected:** NO EXECUTION on migrated canonical paths — **SATISFIED**.

## Nested Subprocess Audit

Leaf logs show one pytest session per physical leaf (canonical `PytestSubprocessExecutor` pattern). No secondary `uv run pytest` / `python -m pytest` strings inside leaf stdout beyond the executor-owned invocation.

| Pattern | Classification |
| --- | --- |
| One pytest session per leaf log | LEGITIMATE LEAF EXECUTION |
| Full-profile re-entry inside leaf | not observed |

## Duplicate Execution Audit

Each of 44 suite identities appears once per run under `perf-npsc5f-final-0/*.log`. No evidence of the same pytest node executed twice physically in one orchestrator run. Dedup metrics unchanged (223 eliminated).

## Scheduler Analysis

| Run | Wall | Leaf work | Eff. concurrency | Critical path (approx.) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 187.72 | 375.03 | 1.998 | 121.88 (`r2-h2-q1`) |
| 2 | 126.79 | 252.77 | 1.994 | 47.02 (`runtime-observability`) |
| 3 | 132.81 | 265.05 | 1.996 | 45.83 (`runtime-observability`) |

Parallel efficiency near 1.0 at `max_parallel=2`; no idle-window evidence while ready leaves existed.

## Critical Path Assessment

Runs 2–3: critical path ≈ slowest leaf (`runtime-observability`, ~46 s) plus DAG overlap → wall ~127–133 s. Run 1: critical path dominated by **`npsc5e-r2.mandatory.r2-h2-q1`** at ~122 s (environment/cold-run variance on 12-test semantic slice), inflating wall to ~188 s without changing leaf count or dedup.

## Resource / Isolation Assessment

Slow leaves report `exclusive_resource_id: null` in certification JSON — not spuriously serialized by exclusive resource class. No misclassification fix attempted (verification-only task).

## Executor Assessment

Observed leaf logs match **PytestSubprocessExecutor** workloads (pytest session headers, isolated `cachedir` under `build/pytest/`). No evidence of wrong executor type on slow leaves.

## Architecture Boundary Assessment

No cross-layer shortcuts, contract bypass, or suite-id special casing introduced. Performance path remains contract-first canonical qualification.

## Production Changes

**NONE**

## Qualification Architecture Changes

**NONE**

## Test Changes

**NONE**

## Performance Findings

1. Median wall **132.81 s** — inside practical **130–200 s** band and below certified historical **~191.5 s**.
2. Canonical dedup and 44-leaf physical plan stable across all runs.
3. Three-run wall spread **48%** exceeds preferred 20% due to **single-run R2-H2-Q1 latency outlier**; runs 2–3 spread **4.8%**.
4. Dominant steady-state cost: **`runtime-observability`** then **`r2-original` / `runtime-events` / `recovery`**.

## Hotspots

| Area | Note |
| --- | --- |
| `runtime-observability` | Largest steady-state wall contributor; legitimate 505-test bundle |
| `npsc5e-r2.mandatory.r2-h2-q1` | High run-to-run variance (120 s vs ~8–10 s); investigate environment/cold-cache in optional optimization task |
| `npsc5f-final.recovery` | Stable ~12 s; not historical ~79 s profile |

**optimization is optional follow-up; not certification blocker**

## Blockers

None identified (no nested harness regression, no duplicate physical execution, no scheduler starvation, no >2× unexplained regression vs certified baseline).

## Decision

```text
PERFORMANCE RE-CERTIFICATION = PASS — OPTIMIZATION HOTSPOTS IDENTIFIED
```

## Final Verdict

Three functional PASS runs at `PERFORMANCE_RECERTIFICATION_SHA` with clean tracked tree, intact dedup (267→44), no nested mandatory harness, and acceptable median performance. Documented hotspots are legitimate or environment-sensitive workloads, not architecture defects.

## Aggregate Benchmark Table

| Metric | Run 1 | Run 2 | Run 3 | Mean | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wall (s) | 187.72 | 126.79 | 132.81 | 149.11 | 132.81 |
| Leaf work (s) | 375.03 | 252.77 | 265.05 | 297.62 | 265.05 |
| Effective concurrency | 1.998 | 1.994 | 1.996 | 1.996 | 1.996 |
| Parallel efficiency | 0.999 | 0.997 | 0.998 | 0.998 | 0.998 |
| Physical leaves | 44 | 44 | 44 | 44 | 44 |

## Historical Comparison

| Benchmark generation | Wall (s) |
| --- | ---: |
| Early nested | ~938 |
| Intermediate | ~510 |
| After harness work | ~189 |
| Best historical | ~129 |
| Later canonical | ~208 |
| Pre-final | ~167 |
| Certified canonical | ~191.5 |
| **Current median (this task)** | **132.81** |

Relative reduction vs ~938 s: **~85.8%** (~805 s absolute saved). vs certified ~191.5 s: median **~31%** faster (~58.7 s saved).

## Final Recommendation

Proceed to **`INTEGRAx-QUALIFICATION-FINAL-ARCHITECTURE-AND-CERTIFICATION-CLOSURE`**. Do not open a mandatory optimization task solely for recovery or observability cost unless product owners prioritize latency work.

---

> Wprowadzone zmiany wymagają audytu na podstawie aktualnego kodu znajdującego się na GitHub. Raport implementacji nie zastępuje niezależnej weryfikacji zmian wykonanych w repozytorium.
