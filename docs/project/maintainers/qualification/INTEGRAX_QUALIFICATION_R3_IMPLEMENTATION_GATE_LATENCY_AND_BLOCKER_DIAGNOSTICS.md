# INTEGRAx-QUALIFICATION-R3-IMPLEMENTATION-GATE-LATENCY-AND-BLOCKER-DIAGNOSTICS

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-QUALIFICATION-R3-IMPLEMENTATION-GATE-LATENCY-AND-BLOCKER-DIAGNOSTICS` |
| Branch | `development` |
| HEAD (diagnostics) | `125b5160ae9e6290b122922609e830fe077f88c8` |
| `origin/development` | `1bdb0798bc2f604401b291857142c1f279e92d27` (local **ahead 1**) |
| Diagnostics date | 2026-09-15 |
| Production changes | **NONE** |
| Qualification architecture changes | **NONE** (this task) |

## Session Scope

Root-cause analysis only for reported ~310 s on leaf `npsc5e-r3.mandatory.r3-implementation-gate` and separation of latency vs qualification blocker. No scheduler, resource classifier, platform, or gate-model edits.

## Canonical Suite Definition

**Registry truth (fail-closed `suite_id_for_pytest_arguments`):**

| `suite_id` | Pytest target | Harness |
| --- | --- | --- |
| `npsc5e-r3.mandatory.r3-implementation-gate` | `tests/unit/runtime/architecture/test_npsc5e_r3_child_fanout_partial_recovery.py` | **Embedded** (`test_mandatory_frozen_suite_passes` → `uv run pytest` subprocess) |
| `npsc5f-r3.implementation-gate` | `tests/unit/runtime/architecture/test_npsc5f_r3_governed_evidence_export.py` | **Direct semantic** (no nested pytest orchestration) |

Operator brief cited `test_npsc5f_r3_governed_evidence_export.py` under `npsc5e-r3.mandatory.r3-implementation-gate`; that mapping does **not** match the catalog registry. The ~310 s leaf in full canonical runs is the **npsc5e** file above, not the npsc5f OBS-03 gate file.

## Suite ID / Target / Resource Classification

| Property | Value |
| --- | --- |
| Slow leaf `suite_id` | `npsc5e-r3.mandatory.r3-implementation-gate` |
| Profile membership | `npsc5f-final` (via `NPSC-5E Final` / expanded `NPSC5E_R3_FINAL_MANDATORY` leaves) |
| Gate id | `npsc5f-final.requires.npsc5e-r3.mandatory.r3-implementation-gate` |
| Pytest arguments | `tests/unit/runtime/architecture/test_npsc5e_r3_child_fanout_partial_recovery.py` |
| `exclusive_resource_id` | `npsc5e-r3-cross-db` (only this suite in `npsc5f-final` `suite_by_id`) |
| Suite timeout (live runner) | `LIVE_SUITE_TIMEOUT_SECONDS` = 21600 s (`testing_support/execution_qualification/performance/runner.py`) |
| Gate dependencies | Flat aggregate: no extra `leaf_gate_extra_requires` on this suite; root aggregate lists all mandatory leaves in parallel under `max_parallel` |

## Semantic Test Inventory

### `test_npsc5e_r3_child_fanout_partial_recovery.py` (qualification leaf)

| Node | Invariant (short) |
| --- | --- |
| `test_mandatory_frozen_suite_passes[*]` (×8) | Nested full pytest runs of R1/R2/P0A/DG/NPSC/HITL/child/checkpoint predecessors |
| `test_no_forbidden_recovery_runtime_names_in_production` | No shadow recovery runtime symbols in production paths |
| `test_no_reflection_in_partial_recovery_production` | No reflection in partial recovery production code |
| `test_submission_port_exposes_recover_failed_slot` | Submission port API for recovery |
| `test_one_failed_slot_recovery_preserves_siblings` | Partial fan-out recovery semantics |
| `test_all_success_recovery_is_noop` | No-op recovery when all slots succeed |
| `test_result_order_and_cardinality_preserved_after_recovery` | Ordering invariant after recovery |
| `test_cross_process_partial_recovery` | Cross-process recovery (uses `.tmp/session/npsc5e-r3/cross.db`) |
| `test_wrong_revision_blocked` | Stale revision fail-closed |
| `test_policy_deny_blocked` | Policy deny on recovery |
| `test_stale_recovery_writer_blocked` | Writer safety |
| `test_duplicate_recovery_request_idempotent_via_correlation` | Idempotency |
| `test_no_direct_child_execution_runner_in_partial_recovery` | Boundary / no direct runner |
| `test_runtime_checkpoint_topology_recovery_field_compatible_v2` | Checkpoint field compatibility |

### `test_npsc5f_r3_governed_evidence_export.py` (separate leaf `npsc5f-r3.implementation-gate`)

22 direct unit tests for OBS-03 governed export (forbidden fields, envelopes, OTLP, pluginability, tenant isolation, determinism, static surface guards). Not the subprocess driver for `npsc5e-r3.mandatory.r3-implementation-gate`.

## Nested Harness Check

| Check | Result |
| --- | --- |
| `test_npsc5f_r3_governed_evidence_export.py` | No `subprocess.run` / `pytest.main` / `_run_pytest` — **direct semantic** |
| `test_npsc5e_r3_child_fanout_partial_recovery.py` | **`EMBEDDED HARNESS DISCOVERED`**: `_run_pytest` + `test_mandatory_frozen_suite_passes` runs 8 predecessor modules via nested `uv run pytest` |

## Standalone Timing

Measured on HEAD `125b5160`, Windows, `uv run pytest`, logs under `.tmp/session/R3-GATE-DIAG/`.

| Target | Wall (s) | Pytest reported | Notes |
| --- | ---: | --- | --- |
| **npsc5e leaf file** (`npsc5e-r3.mandatory…`) | **347.02** | **344.28** (22 tests) | Dominated by embedded suites |
| **npsc5f OBS-03 file** (`npsc5f-r3.implementation-gate`) | 2.40 | 0.95 | 1 local FAIL (reflection symbol; unrelated WIP on export boundary) |

Historical canonical receipts (~310–390 s) align with **npsc5e** standalone wall, not npsc5f (~2 s).

## Per-Test Timing

### Dominant nodes — npsc5e leaf (full file, `--durations=10`)

| node | wall (call, s) | pytest duration | result |
| --- | ---: | ---: | --- |
| `test_mandatory_frozen_suite_passes[R2 Final]` | **284.84** | (nested run 276–282 s) | **FAIL** |
| `test_mandatory_frozen_suite_passes[R1 Final]` | 10.16 | — | PASS |
| `test_mandatory_frozen_suite_passes[NPSC-5D Final]` | 8.68 | — | PASS |
| `test_mandatory_frozen_suite_passes[P0A]` | 8.30 | — | PASS |
| `test_mandatory_frozen_suite_passes[NPSC-5B]` | 7.22 | — | PASS |
| `test_mandatory_frozen_suite_passes[HITL R3]` | 5.96 | — | PASS |
| `test_mandatory_frozen_suite_passes[Checkpoint store]` | 4.55 | — | PASS |
| `test_mandatory_frozen_suite_passes[DG_001]` | 3.80 | — | PASS |
| `test_mandatory_frozen_suite_passes[Child execution]` | 3.64 | — | PASS |
| Remaining semantic tests | &lt;1 each | — | PASS |

**Sum of embedded `test_mandatory_frozen_suite_passes` call times ≈ 337 s** vs **full-file wall 347 s** → small gap (~10 s) = import/collection/overhead, not hidden parallelization.

### npsc5f file — all nodes &lt;0.15 s setup; slowest call ~0.01 s (`test_r3_no_second_export_framework_symbols`).

## Collection / Import Timing

| Artifact | Wall (s) |
| --- | ---: |
| Collect npsc5e leaf | 6.59 |
| Collect npsc5f OBS-03 | 2.38 |
| Import npsc5f module (`importlib`) | 1.38 |

Collection/import are **not** drivers of ~310 s.

## Fixture Analysis

| Fixture | Scope | Location |
| --- | --- | --- |
| `_diagnostic_problem_list_cursor_secret` | function, **autouse** | `tests/unit/runtime/architecture/conftest.py` |
| Session autouse fixtures | session | `tests/conftest.py` (e.g. agent fleet inventory — visible in repro setup stdout) |

No module/session fixtures in the R3 gate files themselves. Latency is not fixture-contention dominated.

## Exclusive Resource Analysis

- Resource `npsc5e-r3-cross-db` assigned only to `npsc5e-r3.mandatory.r3-implementation-gate` in `npsc5f-final`.
- On-disk cross-db path used inside semantic test: `.tmp/session/npsc5e-r3/cross.db` (child fanout cross-process test).
- **No second leaf** competes for the same `exclusive_resource_id` in the compiled profile → **queue wait for this mutex is not the ~310 s story**; duration is almost entirely subprocess work inside the leaf executor.

## Scheduler / Queue Analysis

- `PytestSubprocessSuiteExecutor` sets `duration_seconds = time.monotonic()` around the **single** top-level `uv run pytest` child (`executor.py`).
- Coordinator exclusive lock is held **during** that execution only; wait-before-start is **not** included in `duration_seconds`.
- **Equation (this leaf):** `leaf_total ≈ executor_duration` (queue_wait ≈ 0 for exclusive resource here; finalization negligible vs 300+ s).

```text
scheduled
   ↓
queue/resource wait          ~0 s (sole holder of npsc5e-r3-cross-db)
   ↓
executor start
   ↓
pytest execution             ~344 s (nested R2 Final ~285 s of that)
   ↓
executor finish
   ↓
gate/receipt finalization    ≪ 1 s
```

Reported canonical leaf ~310–332 s ≈ **executor subprocess duration**, not mis-attributed queue wait.

## Previous Run Artifact Analysis

No `.tmp/session/...` canonical run logs present in workspace at diagnostics time. Relied on prior qualification docs (`INTEGRAX_QUALIFICATION_PERFORMANCE_CERTIFICATION…`, `INTEGRAX_OBSERVABILITY_RUNTIME_EVENT_EXPORT…`) for cross-check (~332 s leaf, multiplicity 9 in historical accounting).

## Failure Inventory

**Leaf `npsc5e-r3.mandatory.r3-implementation-gate` (standalone run):**

1. `tests/unit/runtime/architecture/test_npsc5e_r3_child_fanout_partial_recovery.py::test_mandatory_frozen_suite_passes[R2 Final]`

**Nested inside R2 Final subprocess (from failure output):**

2. `tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py::test_mandatory_frozen_suite_passes[R2-H2-Q1]`
3. `tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py::test_mandatory_frozen_suite_passes[DG_001]`

**Separate leaf `npsc5f-r3.implementation-gate` (not the ~310 s id):**

4. `tests/unit/runtime/architecture/test_npsc5f_r3_governed_evidence_export.py::test_r3_no_reflection_on_safe_projection_symbols` (`IndexError` — symbol rename/absence in `export_boundary.py` vs test expectation; local WIP on observability contracts)

## Minimal Reproducer

```bash
uv run pytest "tests/unit/runtime/architecture/test_npsc5e_r3_child_fanout_partial_recovery.py::test_mandatory_frozen_suite_passes[R2 Final]" -q --tb=short
```

Observed: **FAIL** after ~284 s; nested R2-H2-Q1 and DG_001 failures inside `test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py`.

## Root Cause Classification

| Class | Applies |
| --- | --- |
| **EMBEDDED HARNESS** | **PRIMARY (latency)** |
| SUBPROCESS SEMANTIC COST | Secondary (legitimate cross-process test is small vs nested R2) |
| DIRECT SEMANTIC TEST SLOW | No (npsc5f gate is ~2 s) |
| EXCLUSIVE RESOURCE WAIT | No (sole consumer) |
| SCHEDULER/QUEUE WAIT | No |
| QUALIFICATION WRAPPER COST | No (leaf duration tracks pytest child) |
| QUALIFICATION TIMING MODEL DEFECT | No |
| PLATFORM DEFECT | No evidence for ~310 s |
| TEST DEFECT | Blocker path: nested mandatory suite failures / drift |

## Primary Root Cause

**~310 s is real executor time for `npsc5e-r3.mandatory.r3-implementation-gate` because the catalog leaf runs `test_npsc5e_r3_child_fanout_partial_recovery.py`, which embeds eight nested full `uv run pytest` predecessor suites; `test_mandatory_frozen_suite_passes[R2 Final]` alone spends ~285 s (~83% of leaf wall) launching the full R2 Final orchestrator module without the canonical embedded-harness `-k` slice used elsewhere.**

## Secondary Contributors

- Nested R2 Final still runs its own embedded mandatory matrix (R2-H2-Q1, DG_001, …) — duplicate work vs DAG-optimized leaves.
- `exclusive_resource_id` serializes only this one leaf (correctness path for cross-db); does not explain hundreds of seconds.
- Separate fast leaf `npsc5f-r3.implementation-gate` does not substitute for the slow `npsc5e-r3.mandatory…` id in receipts.

## Architecture Boundary Assessment

**No `ARCHITECTURE BOUNDARY CHANGE REQUIRED` for diagnosis.** Remediation is qualification/catalog/harness migration (same class as R2 Final embedded harness elimination), not Execution Engine / observability / scheduler contracts.

## Platform Gap Assessment

**Not `R3 IMPLEMENTATION GATE = PLATFORM DEFECT` for latency.** Failures may involve test/qualification drift in nested R2 mandatory suites; not validated as production defect in this task.

## Qualification Architecture Assessment

Timing model is consistent (leaf receipt ≈ subprocess). Structural issue: **legacy embedded predecessor orchestration inside a mandatory leaf** inflates wall and duplicates DAG work. **`QUALIFICATION ARCHITECTURE CHANGE REQUIRED`** only if operator chooses to migrate/remove harness — **out of scope for this diagnostics-only task** (STOP before implementation).

## Recommended Next Action

1. **Migration task** (parallel pattern to R2 Final / R2-H2-Q1): replace `test_npsc5e_r3_child_fanout_partial_recovery.py` mandatory subprocess matrix with canonical semantic slice + receipt-based predecessors, or repoint `NPSC5E_R3_FINAL_MANDATORY` “R3 implementation gate” to direct semantic coverage only.
2. **Unblock failures**: fix or re-slice nested `test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py` mandatory entries (R2-H2-Q1, DG_001) — likely qualification/test alignment, not platform.
3. **Clarify operator docs**: map `npsc5f-r3.implementation-gate` vs `npsc5e-r3.mandatory.r3-implementation-gate` explicitly.
4. Resolve `test_r3_no_reflection_on_safe_projection_symbols` vs current `export_boundary.py` on a separate observability WIP track.

## Changed Files

| Path | Change |
| --- | --- |
| `docs/project/maintainers/qualification/INTEGRAX_QUALIFICATION_R3_IMPLEMENTATION_GATE_LATENCY_AND_BLOCKER_DIAGNOSTICS.md` | Added (this document) |

## Verification

Diagnostics-only — no code fixes. Measurements:

- `uv run pytest tests/unit/runtime/architecture/test_npsc5e_r3_child_fanout_partial_recovery.py -q --durations=10`
- `uv run pytest tests/unit/runtime/architecture/test_npsc5f_r3_governed_evidence_export.py -q --durations=0`
- `uv run python -c "… suite_id_for_pytest_arguments …"`

## Performance Evidence

| Metric | Value |
| --- | ---: |
| Standalone npsc5e leaf wall | 347.02 s |
| Standalone npsc5f OBS-03 wall | 2.40 s |
| Sum embedded mandatory calls (npsc5e) | ~337 s |
| Slowest node | `test_mandatory_frozen_suite_passes[R2 Final]` ~285 s |
| Queue wait (exclusive) | ~0 s (single leaf) |
| Executor duration | ≈ leaf total |
| Reported historical leaf | ~310–332 s |

## Production Changes

**NONE**

## Findings

1. Operator target file `test_npsc5f_r3_governed_evidence_export.py` is **`npsc5f-r3.implementation-gate`**, not `npsc5e-r3.mandatory.r3-implementation-gate`.
2. The slow mandatory id runs **`test_npsc5e_r3_child_fanout_partial_recovery.py`** with **embedded harness**.
3. Latency and blocker decouple: **slow** = nested R2 Final subprocess; **fail** = that nested run still fails (R2-H2-Q1, DG_001).
4. Exclusive resource is assigned but not the dominant wait contributor.

## Decision

Do not optimize `max_parallel`, timeouts, or resource ownership in this task. Proceed to a **dedicated harness migration / canonical gate alignment** task after operator sign-off.

## Final Verdict

**Latency:** `R3 IMPLEMENTATION GATE = BLOCKED — EMBEDDED HARNESS DISCOVERED`

**Blocker (fail):** embedded `R2 Final` mandatory subprocess inside the npsc5e R3 implementation gate leaf (not HITL, not scheduler artifact).

**Not:** `QUALIFICATION TIMING MODEL DEFECT`, `EXCLUSIVE RESOURCE SERIALIZATION` (as primary), or `PLATFORM DEFECT` for the ~310 s measurement.
