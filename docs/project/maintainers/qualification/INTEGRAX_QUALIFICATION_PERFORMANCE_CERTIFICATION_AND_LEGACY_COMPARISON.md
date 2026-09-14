# INTEGRAx Qualification — Performance Certification and Legacy Comparison

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-QUALIFICATION-PERFORMANCE-CERTIFICATION-AND-LEGACY-COMPARISON` |
| Git HEAD (benchmark) | `5f2ec87000af7ff258a191e5fe107901f5d2b649` |
| Semantic parity base | `dffe2ae52a6938e0620ae4a3cc5e4be760e7f2f7` |
| Primary profile | `npsc5f-final` |
| Tooling | `testing_support/execution_qualification/performance/` |

## Scope

Measure canonical qualification DAG wall time, duplicate execution elimination vs recursive legacy expansion, critical-path approximation, and parallel metrics. No production runtime changes, no semantic re-certification, no cross-run cache.

## Benchmark Methodology

- Path: `QualificationCatalog` → compile profile → `QualificationPlanRunner` → `QualificationCoordinator` → `PytestSubprocessSuiteExecutor`.
- Wall clock: external `time.perf_counter()` around full plan run (matches coordinator measured semantics).
- Leaf work: `sum(receipt.duration_seconds)` from suite receipts.
- `max_parallel`: qualified default **2** (`EXECUTION_QUALIFICATION_DEFAULT_MAX_PARALLEL`), overridable via CLI/`INTERGRAX_EXECUTION_QUALIFICATION_MAX_PARALLEL`.
- Pytest cache: **default** (not disabled).
- Suite timeout: **21600 s** (6 h) per live runner constant.
- Artifacts: `.tmp/session/qualification-performance/perf-{profile}-{repetition-index}/` (gitignored).
- CLI: `uv run python -m testing_support.execution_qualification.performance --profile <id> [--repetitions N] [--max-parallel N] [--structural-matrix]`.

## Environment

| Field | Value |
| --- | --- |
| Python | 3.12.11 |
| uv | 0.8.15 |
| Platform | Windows |
| CPU logical count | 20 |
| max_parallel | 2 |

## Profiles Benchmarked

| Profile | Live wall measured | Structural counts |
| --- | --- | --- |
| `npsc5f-r1-final` | No | Yes |
| `npsc5f-r2-final` | No | Yes |
| `npsc5f-r3-final` | No | Yes |
| `npsc5f-r4-final` | No | Yes |
| `npsc5f-final` | Yes (2 repetitions) | Yes |
| `npsc5e-final` | No (optional; not run — cost) | Available via tooling |

## Legacy Baseline

| Source | Provenance | Notes |
| --- | --- | --- |
| `npsc5e-r3-final` coordinator wall **892.31 s** @ mp=2 | **historical_measured** | R3 record in graph audit doc; legacy flat mandatory matrix — **not** equivalent to `npsc5f-final` scope |
| Full legacy `npsc5f-final` recursive orchestration wall | **not_available** | No verified historical full-run wall for equivalent obligation |
| Graph audit “~84 s” NPSC-5F Final matrix | **historical_measured** | Single orchestrator pytest — **not** equivalent to full recursive legacy subprocess tree |

Legacy logical subprocess counts use `expand_mandatory_subprocesses` on catalog `mandatory_sources` (same reference as semantic parity).

## Canonical Baseline

Primary live benchmark @ HEAD `5f2ec8700`, mp=2, cold/warm: repetition 0 cold, repetition 1 warm (same session).

| Repetition | Wall (s) | Total leaf work (s) | Run PASS |
| --- | ---: | ---: | --- |
| 0 | 878.99 | 1397.31 | **FAIL** |
| 1 | 904.34 | 1405.92 | **FAIL** |

Primary aggregate wall (mean of 2 runs): **891.67 s** (**measured**).

Failed leaf suites (both runs): `npsc5f-final.cancellation`, `npsc5f-r4.final-drift-sentinel`.

## Execution Count Comparison

| Profile | Legacy logical exec | Legacy semantic unique | Canonical physical | Eliminated | Eliminated % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `npsc5f-r1-final` | 40 | 24 | 24 | 16 | 40.0 |
| `npsc5f-r2-final` | 85 | 29 | 29 | 56 | 65.9 |
| `npsc5f-r3-final` | 171 | 33 | 33 | 138 | 80.7 |
| `npsc5f-r4-final` | 341 | 38 | 38 | 303 | 88.9 |
| **`npsc5f-final`** | **384** | **43** | **43** | **341** | **88.8** |
| `npsc5e-r3-final` | 34 | 20 | 20 | 14 | 41.2 |
| `npsc5e-final` | 34 | 20 | 20 | 14 | 41.2 |

## Duplicate Work Elimination

Mandatory `npsc5f-final` proof: **384 > 43** physical executions; **341** duplicate logical invocations removed (**88.80%** of legacy logical count).

Top legacy invocation multiplicity (pytest argument vectors, legacy expansion):

| Legacy multiplicity | Representative suite_id (canonical) |
| ---: | --- |
| 27 | `npsc5d-final` |
| 18 | `npsc5e-r3.mandatory.long-running`, `npsc5e-r3.mandatory.p0a` |
| 9 | `npsc5e-r3.mandatory.r3-implementation-gate`, `npsc5e-r2.mandatory.r2-original`, others |
| 7 | `runtime-events`, `runtime-observability` |

## Wall-Time Comparison

| Profile | Canonical wall | Legacy wall | Reduction | Speedup |
| --- | --- | --- | --- | --- |
| `npsc5f-final` | **891.67 s measured** | not_available | n/a | n/a |
| `npsc5e-r3-final` | not measured | **892.31 s historical** | n/a | n/a |

No speedup/reduction percent is reported without equivalent-scope legacy wall (**measured** or **historical_measured**).

## Speedup

Not certified for `npsc5f-final` (legacy full wall unavailable). Structural duplicate elimination is certified separately.

## Per-Leaf Timing

Primary run slowest leaves (repetition 1 wall **904.34 s**):

| Rank | suite_id | Duration (s) | ~% wall | Exclusive resource | Legacy multiplicity |
| ---: | --- | ---: | ---: | --- | ---: |
| 1 | `npsc5f-final.recovery` | 665.92 | 74.7 | — | 1 |
| 2 | `npsc5e-r3.mandatory.r3-implementation-gate` | 332.44 | 37.3 | `npsc5e-r3-cross-db` | 9 |
| 3 | `npsc5e-r2.mandatory.r2-h2-q1` | 137.47 | 15.4 | — | 9 |
| 4 | `runtime-observability` | 55.80 | 6.3 | — | 7 |
| 5 | `runtime-events` | 17.69 | 2.0 | — | 7 |

## Critical Path

**Critical path approximation:** max leaf duration **665.92 s** (`npsc5f-final.recovery`) — leaves are mostly parallel under mp=2; exclusive `npsc5e-r3-cross-db` serializes R3 implementation gate (**332 s** additional serialized work class).

Measured wall **~892 s** >> max leaf → parallelism effective but dominated by long leaves + mutex.

## Exclusive Resource Impact

| Resource | Suite | Approx. duration |
| --- | --- | ---: |
| `npsc5e-r3-cross-db` | `npsc5e-r3.mandatory.r3-implementation-gate` | 332 s |

## Parallel Efficiency

| Metric | Value (primary / rep 0) |
| --- | ---: |
| Total leaf work | 1397.31 s |
| Wall | 878.99 s |
| Effective concurrency (`work/wall`) | **1.59** |
| Scheduler parallel efficiency estimate (`work/(wall×mp)`) | **0.79** |

## Slowest Suites

See per-leaf table; **`npsc5f-final.recovery`** dominates (~74% of wall on rep 1).

## Bottleneck Classification

| Rank | Driver | Class |
| ---: | --- | --- |
| 1 | `npsc5f-final.recovery` (~11 min leaf) | **SLOW LEAF** |
| 2 | R3 implementation gate + cross-db mutex | **EXCLUSIVE RESOURCE SERIALIZATION** + **DUPLICATE EXECUTION** (legacy 9×; canonical 1×) |
| 3 | Directory suites `runtime-events` / `runtime-observability` | **DUPLICATE EXECUTION** (legacy 7× each; canonical 1×) — wall share smaller than recovery |

## Stability / Variance

| Stat | Wall (s) |
| --- | ---: |
| min | 878.99 |
| max | 904.34 |
| mean | 891.67 |
| median | n/a (2 samples) |
| spread vs mean | ~2.8% |

## Semantic Safety Reference

**GLOBAL SEMANTIC PARITY CERTIFICATION = PASS** @ `dffe2ae52a6938e0620ae4a3cc5e4be760e7f2f7` (not re-run in this task).

## Tests

```bash
uv run pytest tests/unit/testing_support/execution_qualification/ -q
```

Includes `test_qualification_performance_metrics.py` (T1–T7 calculations, provenance, serialization).

## Static Quality

```bash
uv run ruff check testing_support/execution_qualification/performance tests/unit/testing_support/execution_qualification/test_qualification_performance_metrics.py
uv run ruff format --check testing_support/execution_qualification/performance
uv run pyright testing_support/execution_qualification/performance
```

## Production Changes

**NONE**

## Findings

1. In-run dedup removes **341** redundant subprocesses for `npsc5f-final` vs legacy expansion (**88.8%** logical invocations).
2. Canonical full run still **~15 min** wall @ mp=2 — dominated by **`npsc5f-final.recovery`**, not duplicate re-execution.
3. Live canonical run **failed** pytest in `npsc5f-final.cancellation` and `npsc5f-r4.final-drift-sentinel` — blocks performance PASS despite timing evidence.
4. Equivalent-scope legacy full wall for `npsc5f-final` remains unmeasured; do not infer speedup from `npsc5e-r3` historical **892.31 s**.

## Performance Outcome

**PERFORMANCE OUTCOME = WEAK** (wall ~892 s; no certified legacy wall reduction; dedup proven structurally).

## Certification Decision

**QUALIFICATION PERFORMANCE CERTIFICATION = BLOCKED**

Reason: **PERFORMANCE CERTIFICATION BLOCKED — CANONICAL RUN FAILED** (suites above). Duplicate accounting and timing methodology are valid; verdict certification requires successful canonical PASS run.

## Final Verdict

**Performance problem solved: NO** (still ~15 min; wall reduction vs legacy not certified; run FAIL).

Duplicate execution problem **structurally solved** (384 → 43). Next work: fix failing mandatory leaves, then re-run benchmark; optional follow-up to measure one controlled legacy-equivalent wall or bound via historical evidence with matched scope.

---

Typed JSON evidence (local, gitignored): `.tmp/session/qualification-performance/npsc5f-final-report.json`.
