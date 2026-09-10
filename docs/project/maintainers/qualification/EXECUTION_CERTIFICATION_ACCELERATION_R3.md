# Execution Certification Acceleration — R3 Qualification Record

**Status:** `QUALIFIED` (performance evidence collected; production `max_parallel` unchanged)

**Task:** Execution Certification Acceleration/R3 — Performance Qualification & Hardening

**Branch:** `development`

---

## Session anchors

| Field | Value |
| --- | --- |
| `START_HEAD` | `5c7f6867b78bfe31b367c04be62b0984f49beabe` |
| `START_ORIGIN` | `d6d6fb94b11ef20553d9d4af36ef5cb36ac31e9c` |
| R2 canonical reference commit | `73d64eef34d18a9f647228d48be0a4644f763988` |

---

## Measurement methodology

- Child suite timings: `ExecutionQualificationSuiteResult.duration_seconds` from the R1 subprocess executor (not parsed pytest output).
- Coordinator wall clock: `time.monotonic()` around bounded parallel scheduling in `QualificationCoordinator.run_measured`.
- Overlap metric (not historical speedup): `observed_overlap_ratio = sum(child durations) / wall_duration_seconds`.
- Evidence artifacts: `.tmp/session/r3-performance-qualification/` (JSON + Markdown tables); run logs under `build/qualification/<run_id>/`.
- Live collection tests: `tests/unit/testing_support/execution_qualification/test_r3_live_performance_qualification.py` with `INTERGRAX_R3_LIVE_PERF=1`.

Typed projections: `testing_support/execution_qualification/performance_snapshot.py`, `performance_evidence.py`.

---

## R2 baseline (reference only)

| Field | Value |
| --- | --- |
| Observed R3 Final file wall (R2 record) | **1717 s** (pytest reported 1708.30 s) |
| `max_parallel` | 2 |
| Pre-R2 serial baseline | **NOT AVAILABLE** |

R3 does **not** claim speedup versus pre-R2 serial.

---

## R3 serial-equivalent baseline (full matrix)

| Field | Value |
| --- | --- |
| Full matrix `max_parallel=1` | **NOT RUN** — deferred to stay within full-matrix execution budget (max 3); representative subset provides serial-equivalent signal |

### Representative subset serial-equivalent (`max_parallel=1`, 6 suites)

| Field | Value |
| --- | --- |
| Wall time | **94.49 s** |
| Sum child durations | 94.49 s |
| Overlap ratio | 1.000 |
| Result | PASS |

---

## Representative subset concurrency (stage 1)

Six suites: Terminal, Fan-out, P0A, NPSC-5C, Checkpoint store, Long-running (no R3 implementation gate).

| max_parallel | Scope | Wall time (s) | Result | Overlap ratio | Timing validity |
| --- | --- | ---: | --- | ---: | --- |
| 1 | SUBSET | 94.49 | PASS | 1.000 | VALID |
| 2 | SUBSET | 48.41 | PASS | 1.907 | VALID |
| 3 | SUBSET | 37.36 | PASS | 2.623 | VALID |
| 3 | SUBSET (repeat) | 42.03 | PASS | 2.638 | VALID |
| 4 | SUBSET | 32.66 | PASS | 2.880 | VALID |

---

## Full matrix measurements (stage 2)

| max_parallel | Scope | Wall time (s) | Sum child (s) | Overlap | Result | Timing validity |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 2 | FULL (17 suites) | **892.31** | 1778.50 | 1.993 | PASS | VALID |
| 3 | FULL (candidate) | **868.02** | 1886.50 | 2.173 | PASS | VALID |

---

## Per-suite duration table (frozen manifest order, full `max_parallel=2`)

| Frozen label | Duration (s) | Status | Notes |
| --- | ---: | --- | --- |
| R1 Final | 19.84 | PASS | Parent final; local static + in-process gates |
| R2 Final | 701.86 | PASS | Composed parent; dominates child work |
| R3 implementation gate | 831.08 | PASS | Exclusive `npsc5e-r3-cross-db`; longest single suite |
| P0A | 19.34 | PASS | |
| DG_001 | 16.30 | PASS | |
| NPSC-5A | 13.50 | PASS | |
| NPSC-5B Final | 15.83 | PASS | |
| NPSC-5C | 13.91 | PASS | |
| NPSC-5D Final | 20.09 | PASS | |
| HITL R3 | 17.56 | PASS | |
| Attempt lifecycle | 14.41 | PASS | |
| Child execution | 7.98 | PASS | |
| Terminal | 14.84 | PASS | |
| Cancellation | 19.73 | PASS | |
| Checkpoint store | 12.47 | PASS | |
| Long-running | 24.19 | PASS | |
| Fan-out | 15.56 | PASS | |

---

## Critical-path analysis

1. **R3 implementation gate** (~831 s measured child time) — exclusive cross-db resource; defines a long pole.
2. **R2 Final** (~702 s) — second pole; largely orchestration subprocess composition.
3. **Long-running** (~24 s) — largest remaining leaf after finals.
4. **NPSC-5D Final / R1 Final / Cancellation / P0A** (~17–20 s each).
5. **Fan-out** (~16 s).

Observed full-matrix wall **892.31 s** ≈ parallel overlap of the two dominant poles (R3 gate + R2 Final), explaining `observed_overlap_ratio ≈ 2.0` at `max_parallel=2`.

---

## Nested R1/R2 Final duplication

| Question | Finding |
| --- | --- |
| Unique parent-local work? | **Yes** — R1 Final is primarily in-process static/canonical gates; R2 Final file parametrize-subprocesses its own `_MANDATORY_SUITES` including nested **R1 Final** target. |
| Duplicated leaf targets vs R3 matrix? | **OBSERVED** — R2 Final subprocess composition re-executes targets also scheduled directly in R3 (`P0A`, `DG_001`, `NPSC-5D Final`, `HITL R3`, `NPSC-5A`, `NPSC-5B`, `NPSC-5C`, lifecycle/child/terminal/cancellation/long-running/fan-out per R2 `_MANDATORY_SUITES`). |
| Mostly orchestration? | R2 Final child time (~702 s) is **mostly repeated leaf pytest**, not empty wrapper. |
| Flattening | **DEFERRED** — duplication is real, but flattening fails R3 §15 without mechanical parity proof for all parent-local assertions; scope/semantics freeze forbids opportunistic flattening. |

---

## Hardening / contract changes

| Item | Change |
| --- | --- |
| Wall-clock measurement | `QualificationCoordinator.run_measured` + `validate_and_run_measured` |
| Performance snapshot | `QualificationPerformanceSnapshot`, suite timing rows, overlap ratio |
| Evidence Markdown/JSON | `performance_evidence.py`; opt-in live tests |
| Scheduler | **Unchanged** (no starvation evidence) |
| `max_parallel` production constant | **Unchanged (`2`)** — full-matrix `max_parallel=3` wall improvement **2.7%** vs `max_parallel=2` (below 10% qualification threshold) |
| Production `intergrax/**` | **NONE** |

---

## Parity

| Check | Result |
| --- | --- |
| Manifest 17/17 | PASS (`test_r2_npsc5e_r3_parity.py`) |
| Label / order / pytest args | PASS (unchanged adapter) |
| Exclusive resource | PASS (`npsc5e-r3-cross-db` on R3 implementation gate) |

---

## Unexpected failures / skips

| Item | Count |
| --- | --- |
| Unexpected failures | 0 |
| Unexpected skips | 0 |

---

## Resource collisions / timing contamination

| Item | Result |
| --- | --- |
| SQLite / tmp collisions observed | **NONE** |
| Timing contamination | **NO** (full matrices run sequentially) |

---

## Known limitations

- Process-tree descendant cleanup: unchanged from R1 (no orphan observation in R3 runs).
- Full-matrix `max_parallel=1` not executed; overlap vs subset may differ from full critical path.
- R3 full-matrix wall time varies by host load (892 s vs R2-recorded 1717 s on a prior run).

---

## Next step

Final Execution Engine Documentation Consolidation

---

## Configuration comparison (full matrix)

| max_parallel | Scope | Wall time (s) | Result | Timing validity |
| --- | --- | ---: | --- | --- |
| 2 | FULL | 892.31 | PASS | VALID |
| 3 | FULL | 868.02 | PASS | VALID |

Observed wall-clock improvement `max_parallel=3` vs `2` on full matrix: **2.7%** (`1 - 868.02/892.31`).

## Measured speedup vs R3 serial baseline

| Metric | Value |
| --- | --- |
| Full-matrix serial-equivalent | **NOT AVAILABLE** (full `max_parallel=1` not run) |
| Subset-only speedup `2` vs `1` | **48.8%** wall reduction (subset scope only; not used for production policy) |

---

## Qualified policy decision

**QUALIFIED EXECUTION QUALIFICATION MAX_PARALLEL = 2**

`max_parallel=3` (and `4` on subset) showed additional overlap on small suites but **did not** deliver ≥10% full-matrix wall-clock improvement; correctness remained PASS with no resource collisions. Conservative enterprise policy keeps the R2-qualified default.
