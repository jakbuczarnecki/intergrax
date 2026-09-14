# INTEGRAx-QUALIFICATION-EXECUTION-GRAPH-AUDIT-AND-OPTIMIZATION-DESIGN

## Metadata

| Field | Value |
| ----- | ----- |
| **Task** | `INTEGRAx-QUALIFICATION-EXECUTION-GRAPH-AUDIT-AND-OPTIMIZATION-DESIGN` |
| **Date** | 2026-09-14 |
| **Branch** | `development` |
| **Audit HEAD** | `a9830f09245718736a23c97f0420c758defa6044` |
| **Scope** | Qualification orchestration audit + canonical DAG design (**no** orchestrator refactor, **no** production runtime) |
| **Parent architecture** | [`EXECUTION_QUALIFICATION_ACCELERATION_ARCHITECTURE.md`](../architecture/EXECUTION_QUALIFICATION_ACCELERATION_ARCHITECTURE.md) |
| **Evidence (no re-run of full matrix)** | P0 inventory, R3 performance record, Current HEAD revalidation log summary (~84 s nested matrix) |

## Problem Statement

Platform certification is **correct** (`CURRENT HEAD PLATFORM REVALIDATION = PASS WITH OBSERVATIONS`) but **expensive** because qualification gates compose other gates via **nested `uv run pytest` subprocesses**, often re-running the same leaf paths (directories and final gate files) multiple times within one top-level qualification run.

Optimization target:

```text
same tests · same invariants · same gates · same fail-closed semantics
→ fewer physical executions of identical leaf suites
```

Hard invariant for future implementation:

```text
Qualification outcome before optimization == Qualification outcome after optimization
(for same commit SHA and environment)
```

## Current Qualification Architecture

Two dominant patterns coexist:

| Pattern | Mechanism | Typical location |
| ------- | --------- | ---------------- |
| **Aggregate subprocess matrix** | One child `uv run pytest` with flattened targets + `-k` exclusions for known orchestrator test names | `testing_support/npsc5f_*_regression_matrix.py`, invoked from NPSC-5F R4/Final gate tests |
| **Parametrize subprocess gate** | `test_mandatory_frozen_suite_passes` → `_run_pytest(targets)` per label | NPSC-5E/5F R1–R3 Final architecture gate modules |
| **Parallel coordinator (partial)** | `QualificationRunManifest` + `QualificationCoordinator` | NPSC-5E R3 Final only (`test_mandatory_frozen_suites_pass_via_parallel_qualification`) |
| **In-process leaf** | Static AST / import / E2E in parent interpreter | Most tests inside each gate file after subprocess composition |

**Recursive hazard:** Parent runs a **directory target** (e.g. `tests/unit/runtime/events/`) that contains no orchestrator, while a **sibling mandatory label** in the same parent file subprocess-invokes the **same directory** again. Parent matrix runs **multiple** final gate files that each subprocess-compose overlapping upstream finals.

## Existing Acceleration Architecture

Implemented and **R3 QUALIFIED** for **one** canonical matrix:

```text
_MANDATORY_SUITES (NPSC-5E R3 Final qualification module)
  → adapt_frozen_pytest_suites + label→suite_id map
  → QualificationRunManifest
  → QualificationCoordinator (max_parallel=2, collect-all)
  → ExecutionQualificationSuiteResult / ExecutionQualificationRunResult
  → QualificationPerformanceSnapshot (run_measured)
```

See [`EXECUTION_CERTIFICATION_ACCELERATION_R1.md`](EXECUTION_CERTIFICATION_ACCELERATION_R1.md)–[`R3.md`](EXECUTION_CERTIFICATION_ACCELERATION_R3.md).

**Gap:** NPSC-5F evidence-plane finals, NPSC-5E Final aggregator, and R1–R2 Final parametrize gates **do not** use the coordinator; they remain nested trees.

## Orchestrator Inventory

Node classes used in this audit:

`LEAF_TEST` · `LEAF_SUITE` · `AGGREGATE_GATE` · `ORCHESTRATOR` · `STATIC_GATE` · `EXCLUSIVE_SUITE` · `PERFORMANCE_GATE`

| ID | File | Symbol | Spawn type | Targets (summary) | Parent qualification |
| -- | ---- | ------ | ---------- | ----------------- | -------------------- |
| O-01 | `testing_support/execution_qualification/executor.py` | `PytestSubprocessSuiteExecutor.execute` | `uv run pytest` | manifest `pytest_arguments` | NPSC-5E R3 Final coordinator path |
| O-02 | `testing_support/npsc5f_r4_regression_matrix.py` | `run_mandatory_regression_matrix` | single subprocess pytest | R4 `MANDATORY_REGRESSION_SUITES` flattened | NPSC-5F R4 Final, NPSC-5F Final (via extended suites) |
| O-03 | `testing_support/npsc5f_final_regression_matrix.py` | `run_mandatory_regression_matrix` | single subprocess pytest | R4 suites + 5F extras | NPSC-5F Final |
| O-04 | `tests/unit/runtime/architecture/test_npsc5f_r4_final_*.py` | `test_mandatory_regression_matrix_passes` | calls O-02 | R4 matrix | NPSC-5F R4 Final gate |
| O-05 | `tests/unit/runtime/architecture/test_npsc5f_final_evidence_plane_qualification.py` | `test_npsc5f_final_mandatory_regression_matrix_passes` | calls O-03 | Full 5F matrix | NPSC-5F Final |
| O-06 | `tests/unit/runtime/architecture/test_npsc5f_r1_final_*.py` | `_run_pytest` / `test_mandatory_frozen_suite_passes` | subprocess per label | 7 labels incl. `events/`, `observability/`, NPSC-5E Final | NPSC-5F R1 Final; nested inside 5F matrix |
| O-07 | `tests/unit/runtime/architecture/test_npsc5f_r2_final_*.py` | same | subprocess per label | R1 Final file + dirs + TRACE + NPSC-5E | NPSC-5F R2 Final; nested inside matrix |
| O-08 | `tests/unit/runtime/architecture/test_npsc5f_r3_final_*.py` | same | subprocess per label | R2→R1 chain + dirs + export suites | NPSC-5F R3 Final; nested inside matrix |
| O-09 | `tests/unit/runtime/architecture/test_npsc5e_final_*.py` | `_run_pytest` / `test_mandatory_frozen_suite_passes` | subprocess | NPSC-5E R3 Final qualification file | NPSC-5E Final; in 5F matrix |
| O-10 | `tests/unit/runtime/architecture/test_npsc5e_r3_final_*.py` | `run_npsc5e_r3_mandatory_qualification` | coordinator (O-01) | 17 suite_ids | NPSC-5E R3 Final; nested from O-09 |
| O-11 | `tests/unit/runtime/architecture/test_npsc5e_r3_final_*.py` | `test_mandatory_frozen_suites_pass_via_parallel_qualification` | coordinator | same 17 suites | excluded from 5F matrix `-k` |
| O-12 | `tests/unit/runtime/architecture/test_npsc5e_r2_final_*.py` | `test_mandatory_frozen_suite_passes` | subprocess per label | R1 Final + many shared leaves | NPSC-5E R2 Final; nested under R3 coordinator |
| O-13 | `tests/unit/runtime/architecture/test_npsc5e_r3_child_fanout_partial_recovery.py` | `test_mandatory_frozen_suite_passes` | subprocess | section-94 matrix labels | R3 implementation; exclusive DB path |
| O-14 | `tests/unit/runtime/architecture/test_npsc5e_r2_h2_q1_*.py` | `test_mandatory_frozen_suite_passes` | subprocess | frozen regression subset | R2 Final nested label |
| O-15 | `tests/unit/runtime/architecture/test_npsc5e_final_*.py` | `test_ruff_*` / `test_pyright_*` | `uv run ruff` / `uv run pyright` | recovery production surfaces | NPSC-5E Final (excluded from 5F matrix `-k`) |
| O-16 | `tests/unit/runtime/architecture/test_npsc5e_r3_final_*.py` | `test_ruff_*` / `test_pyright_*` | ruff / pyright | R3 surfaces | excluded from matrix `-k` |
| O-17 | `scripts/gates/check_audit_ideal_gates.py` | gate runner | pytest subprocess | ideal harness gate tests | CI / release scripts (out of NPSC finals) |
| O-18 | `scripts/release/phase_w_ops_evidence.py` | `_run_pytest` | many sequential pytest | ops evidence unit paths | release tooling |

## Nested Subprocess Inventory

### Depth model (qualification-critical)

| Depth | Example chain |
| ----- | ------------- |
| **D0** | Operator / CI → `uv run pytest <gate file>` |
| **D1** | `test_npsc5f_final_mandatory_regression_matrix_passes` → **one** matrix pytest (O-03) |
| **D2** | Matrix collects `test_npsc5f_r1_final_*` → `test_mandatory_frozen_suite_passes` → `_run_pytest` per label |
| **D3** | Label `NPSC-5E Final` → entire `test_npsc5e_final_*` → `test_mandatory_frozen_suite_passes` → R3 Final file |
| **D4** | R3 Final file → `run_npsc5e_r3_mandatory_qualification` → up to **17** coordinator child pytest processes (when parallel test not excluded) |

**Matrix `-k` exclusions** (`testing_support/npsc5f_final_regression_matrix.py`) suppress **orchestrator test node names** only; they do **not** suppress parametrize subprocess labels inside collected gate files. Exclusions are **partial** — they prevent re-entry to the **same** matrix/coordinator test, not duplicate leaf execution across labels.

### Recursive orchestration (directory fan-out)

| Parent | Child orchestrator inside target? | Risk |
| ------ | --------------------------------- | ---- |
| `tests/unit/runtime/events/` | No pytest orchestrator in tree | **Duplicate leaf** when R1/R2/R3 each subprocess the directory in one matrix run |
| `tests/unit/runtime/observability/` | Same | **Duplicate leaf** (3× minimum per 5F matrix) |
| `test_npsc5e_final_recovery_plane_qualification_and_freeze.py` | **Yes** (O-09) | **Duplicate gate file** when listed in flatten **and** spawned from R1/R2/R3/Recovery labels |
| `test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py` | **Yes** (coordinator + ruff/pyright) | Spawned from NPSC-5E Final mandatory label while also indirectly covered by Recovery block |

## Qualification Execution Graph

Legend: `→` subprocess pytest composition; `⇢` in-process / static gates in same invocation; `[coord]` QualificationCoordinator children.

### NPSC-5F FINAL (gate file + mandatory matrix)

```text
NPSC-5F Final (test_npsc5f_final_evidence_plane_qualification.py)
├─ test_npsc5f_final_mandatory_regression_matrix_passes [ORCHESTRATOR → O-03]
│  └─ single pytest (flattened targets, -k excludes named orchestrators)
│     ├─ R4 implementation gate file ⇢ static
│     ├─ test_npsc5f_r3_final_* ⇢ test_mandatory_frozen_suite_passes × N → R2,R1,events/,observability/,5E,…
│     ├─ test_npsc5f_r2_final_* ⇢ test_mandatory_frozen_suite_passes × M → R1,events/,observability/,5E,…
│     ├─ test_npsc5f_r1_final_* ⇢ test_mandatory_frozen_suite_passes × 7
│     ├─ TRACE / DG_001 / NPSC-5A–5D / reconstruction paths ⇢ LEAF
│     ├─ test_npsc5e_final_* ⇢ mandatory → R3 Final file; other in-process proofs
│     ├─ Recovery block (R1/R2/R3 qual files + 5E Final) ⇢ overlaps above
│     └─ Evidence / Cancellation / Child / Checkpoint extras ⇢ LEAF
├─ drift / ownership / E2E in-process gates ⇢ STATIC_GATE
└─ (orchestrator test excluded when matrix re-invoked from parent — N/A at top)
```

### NPSC-5F R4 FINAL

```text
NPSC-5F R4 Final
└─ test_mandatory_regression_matrix_passes → O-02 (R4 MANDATORY_REGRESSION_SUITES only)
   ├─ R3 Final file → nested O-08 chain
   ├─ R2 Final file → nested O-07 chain
   ├─ R1 Final file → nested O-06 chain
   └─ shared leaves (5E Final, 5D, DG_001, TRACE, …)
```

### NPSC-5F R3 / R2 / R1 FINAL (standalone operator invocation)

```text
R3 Final → parametrize mandatory labels → subprocess each (includes R2 Final file, R1 Final file, dirs, 5E Final)
R2 Final → parametrize → subprocess (includes R1 Final file, dirs, 5E Final)
R1 Final → parametrize → subprocess (events/, observability/, 5E Final, …)
```

### NPSC-5E FINAL

```text
NPSC-5E Final
├─ test_mandatory_frozen_suite_passes → subprocess R3 Final qualification file
│  └─ (inside child) test_mandatory_frozen_suites_pass_via_parallel_qualification [coord 17 suites]
│     └─ each suite may include R1/R2 Final files → further nested parametrize (when not excluded)
├─ in-process recovery E2E / static ⇢
└─ ruff/pyright subprocess gates (excluded from 5F matrix -k)
```

### NPSC-5E R3 FINAL

```text
NPSC-5E R3 Final
├─ test_mandatory_frozen_suites_pass_via_parallel_qualification [coord] — canonical accelerated path
├─ in-process qualification tests ⇢
└─ ruff/pyright ⇢ STATIC_GATE (spawn subprocess)
```

### NPSC-5E R2 / R1 FINAL

```text
R2 Final → parametrize → many subprocess labels (includes nested R1 Final target)
R1 Final → in-process frozen regression import + static gates
```

### CURRENT_HEAD_REVALIDATION (observed profile, not yet manifest-driven)

From [`INTEGRAX_CURRENT_HEAD_PLATFORM_REVALIDATION.md`](INTEGRAX_CURRENT_HEAD_PLATFORM_REVALIDATION.md):

```text
CURRENT_HEAD_REVALIDATION
├─ NPSC sentinels + baseline provenance (targeted pytest)
├─ Critical subset (EE-A1, U5, governance, identity, 5F P0)
├─ Architecture guard matrix batch (-k excludes nested orchestrators)
├─ test_npsc5f_final_mandatory_regression_matrix_passes (~84s nested matrix)
├─ R3 Final slice (-k "not test_mandatory_frozen_suite_passes")
└─ EE-B4-B/C architecture gates
```

**Design note:** Future `CURRENT_HEAD_REVALIDATION` profile should declare **gate IDs** and compile to the same DAG with dedup — not duplicate matrix + partial R3 slice overlapping matrix leaves.

## Duplicate Execution Analysis

### Physical vs ideal (per top-level run)

**Ideal:** each distinct `suite_id` (stable identity, not raw path alone) executes **once**.

### Multiplicity table (NPSC-5F Final mandatory matrix — static lower bounds)

| Leaf suite (normalized) | Requested by (labels / parents) | Physical executions (lower bound) | Ideal |
| ----------------------- | --------------------------------- | --------------------------------: | ----: |
| `runtime-events-dir` (`tests/unit/runtime/events/`) | R1/R2/R3 Final mandatory labels inside one matrix pytest | **3** | **1** |
| `runtime-observability-dir` | R1/R2/R3 Final mandatory labels | **3** | **1** |
| `npsc5f-r1-final-gate-file` | R4 flatten; R2 label; R3 label | **3+** (plus in-process collection) | **1** |
| `npsc5f-r2-final-gate-file` | R4 flatten; R3 label | **2+** | **1** |
| `npsc5e-final-gate-file` | flatten; R1/R2/R3 labels; Recovery block | **4+** | **1** |
| `npsc5e-r3-final-gate-file` | Recovery block; spawned from 5E Final mandatory | **2+** | **1** |
| `npsc5f-p0-gate` | R4; R1/R2/R3 labels; Evidence block | **4+** | **1** |
| `dg-001-lineage` | R4; R1/R2/R3; section-94 | **4+** | **1** |
| `npsc5d-final` | R4; R1/R2/R3 | **4+** | **1** |
| `trace-asof-pair` | R4; R2/R3 | **3** | **1** |
| `trace-bitemp-triple` | R4; R2/R3 (+ observability file in R4) | **3** | **1** |
| NPSC-5E R3 coordinator leaf suites (17) | Only when parallel test runs | **1× each** when coord used; **+N** when R2/R1 parametrize paths also run same targets | **1× each** |

Upper bounds depend on how many parametrize labels fire in one matrix invocation (sum of `_MANDATORY_SUITES` lengths across embedded finals). **Conservative removable duplicate count:** **≥ 30** distinct subprocess invocations eliminable per full 5F matrix run via dedup (3× dirs + repeated finals + overlapping Recovery/R4 blocks), before counting NPSC-5E R3 internal R2/R1 re-composition documented in R3 record.

### Top duplication ranking (qualification-critical)

| Rank | Duplicate pattern | dup count (order of magnitude) | Est. cost driver | Fan-out depth |
| ---- | ----------------- | ------------------------------ | ---------------- | ------------- |
| 1 | `events/` + `observability/` directory respawn per 5F R1/R2/R3 label | 3× each / matrix | large directory collection | D2 |
| 2 | NPSC-5E Final gate file re-run | 4+ / matrix | spawns R3 subtree | D3–D4 |
| 3 | NPSC-5F R1 Final file re-run | 3+ / matrix | 7 subprocess labels each | D2–D3 |
| 4 | NPSC-5F P0 gate | 4+ / matrix | medium file | D2 |
| 5 | DG_001 + lineage dir | 4+ / matrix | medium | D2 |
| 6 | NPSC-5D Final | 4+ / matrix | medium | D2 |
| 7 | TRACE-ASOF / TRACE-BITEMP | 3+ / matrix | medium | D2 |
| 8 | NPSC-5E R3 coordinator leaves duplicated via R2 Final parametrize | 2× for overlapping labels | high (892 s full coord wall at mp=2) | D4 |
| 9 | `test_npsc5e_r2_final_*` entire file under R3 coord + R2 label | 2+ | high | D3 |
| 10 | Recovery block duplicates R1/R2/R3 qual paths already in R4 flatten | 1× extra each | medium | D2 |
| 11 | Current HEAD: matrix + guard batch + R3 slice | overlapping leaf files | medium (operator time) | D1 |
| 12 | NPSC-5F R2 Final spawned from R3 and flattened | 2+ | high | D2–D3 |
| 13 | Child execution / Cancellation / Checkpoint in 5F extras vs 5E section-94 | 2× semantic overlap | medium | D2 |
| 14 | Retry file listed twice in 5F Final extras | 2× same path in flatten (deduped once in flatten — **1 pytest collect**, OK) | low | — |
| 15 | R4 matrix embedded in 5F Final matrix | full R4 subtree re-executed | high | D2–D3 |
| 16 | ruff/pyright gates | excluded in matrix but run on standalone 5E/R3 invocations | policy duplication | D1 |
| 17 | `phase_w_ops_evidence.py` sequential pytest | independent product surface | CI time | D0 |
| 18 | Ideal harness L3 gates (`scripts/gates/*`) | CI | separate profile | D0 |
| 19 | R3 implementation `cross.db` exclusive path | must stay serial | small but mandatory mutex | D2 |
| 20 | Parallel qual test excluded but serial parametrize still runs same leaves | parity gap | largest long-term | D2–D4 |

**POTENTIAL SEMANTIC DUPLICATION (no removal in this task):** Recovery extras vs NPSC-5E section-94 child/cancellation/checkpoint labels; multiple drift sentinel tests across finals.

## Cost / Critical Path Analysis

| Metric | Source | Value / note |
| ------ | ------ | ------------ |
| NPSC-5F Final matrix wall | Current HEAD revalidation log | **~84 s** (single orchestrator test) |
| NPSC-5E R3 full coordinator wall | R3 record | **892.31 s** @ `max_parallel=2` |
| R3 Final file wall (pre-coordinator era) | R3 record | **1717 s** reference |
| Overlap @ mp=2 full matrix | R3 record | ratio **~1.99** (good parallelism **within** coordinator only) |
| Duplicate time avoided (in-run dedup target) | static | **≥ sum of repeated directory + repeated final gate durations** — lower bound **>> 84 s** for full platform certification if 5E R3 coord + 5F matrix composed without dedup |

Critical path today: **longest chain D1→D4** (5F matrix → 5E Final → R3 Final → coordinator → R2/R1 nested parametrize), not CPU-bound unit tests alone.

Scheduler wait: exclusive `npsc5e-r3-cross-db` serializes one coord slot; irrelevant to directory triple-spawn.

## Existing Framework Reuse Decision

**Answer: PARTIALLY — extend existing architecture**

| Criterion | Assessment |
| --------- | ---------- |
| Typed manifest + coordinator exist | **YES** — reuse |
| Covers all qualification surfaces | **NO** — 5F finals and 5E aggregators still nested |
| Receipt-like suite results | **YES** — extend `ExecutionQualificationSuiteResult` fields for dedup telemetry |
| Performance snapshots | **YES** — reuse `QualificationPerformanceSnapshot` |
| Blocker for full reuse | **None architectural** — requires manifest profiles + graph compiler + aggregate gates consuming receipts |

Do **not** introduce `NewQualificationEngine` / parallel ad-hoc runners.

## Target Canonical DAG Architecture

```text
Qualification Request (profile: TASK_TARGETED | SUBSYSTEM_REQUALIFICATION |
                        CURRENT_HEAD_REVALIDATION | FULL_PLATFORM_CERTIFICATION)
        ↓
Qualification Manifest Resolver (explicit gate IDs → suite definitions)
        ↓
Dependency Graph Compiler (requires = suite_id list; cycle detect; missing ID fail-closed)
        ↓
Leaf Deduplicator (same suite_id → one execution node; track deduplicated_request_count)
        ↓
Resource / Isolation Classifier (exclusive_resource_id mutex groups)
        ↓
Bounded Scheduler (QualificationCoordinator — max_parallel unchanged until dedup proven)
        ↓
Leaf Executor (PytestSubprocessSuiteExecutor — only layer allowed to spawn pytest/ruff/pyright)
        ↓
Immutable Suite Receipts (ExecutionQualificationSuiteResult + provenance)
        ↓
Aggregate Gate Evaluator (in-process only: consume receipts, no subprocess test tools)
        ↓
Qualification Report (manifest order / stable topo index)
```

### Architecture invariants (design)

```text
AGGREGATE_GATE MUST NOT spawn pytest/ruff/pyright subprocesses
one leaf suite_id → at most one physical execution per qualification run
parent gates MUST consume receipts, not re-run children
```

## Contracts

Extend existing types in `testing_support.execution_qualification.contracts` (no `dict[str, Any]` public surfaces):

| Design name | Reuse / extend |
| ----------- | -------------- |
| `QualificationSuiteDefinition` | extend `QualificationSuite` (+ `requires`, `gate_kind`, `profile_tags`) |
| `QualificationGateDefinition` | new frozen dataclass: `gate_id`, `requires: tuple[suite_id, ...]`, `gate_kind: AGGREGATE_GATE` |
| `QualificationDependency` | explicit edge: `(from_gate_id, to_suite_id)` or gate→gate via suite |
| `QualificationExecutionPlan` | compiled DAG: execution nodes + aggregate evaluation order |
| `QualificationReceipt` | alias/extend `ExecutionQualificationSuiteResult` (+ `commit_sha`, `artifact_log_ref`, `deduplicated_request_count`) |

Gate declaration example (future):

```python
requires = ("npsc5f-r1-final", "runtime-events", "npsc5e-r3-coord-matrix", ...)
```

## Suite Identity

Stable `suite_id` examples (manifest-owned, not path-only):

| suite_id | Primary pytest target(s) |
| -------- | ------------------------ |
| `npsc5e-r3-coord-leaf-r1-final` | R1 Final file (single leaf invocation) |
| `runtime-events-dir` | `tests/unit/runtime/events/` |
| `runtime-observability-dir` | `tests/unit/runtime/observability/` |
| `npsc5e-final-aggregate` | receipt consumer only — **no pytest** |
| `npsc5f-final-matrix-leaves` | compiled from current `MANDATORY_REGRESSION_SUITES` flatten |

**FAIL CLOSED:** same `suite_id` with differing `pytest_arguments`.  
**Report:** same targets, different IDs → `POTENTIAL DUPLICATE SUITE DEFINITION`.

## Graph Compilation

1. Load profile manifest (gates + suite definitions).
2. Expand `requires` transitively to leaf suite nodes.
3. Detect cycles → typed `QualificationGraphCycleError`.
4. Validate all `suite_id` exist → typed plan validation failure.
5. Deduplicate execution nodes by `suite_id`.
6. Assign exclusive resource mutex edges from `exclusive_resource_id`.
7. Emit `QualificationExecutionPlan` with stable ordering index.

## Deduplication

In-run only (no cross-commit cache):

```text
same suite_id requested N times → one execution node
receipt referenced by all requesting gates
```

## Scheduling / Isolation

Preserve:

- isolated subprocess per leaf
- `apply_invocation_pytest_basetemp` per child
- env snapshot/restore in executor
- `requires_exclusive_paths` → serial mutex (e.g. `npsc5e-r3-cross-db`)

**Priority:** `DEDUPLICATION > MORE PARALLELISM` (`max_parallel` stays 2 until dedup parity certified).

## Receipt Model

Minimum fields (extend existing suite result):

```text
suite_id, commit_sha, command/targets, status, exit_code, duration_seconds,
artifact/log ref, deduplicated_request_count, parent_gate_ids (consumers)
```

Aggregate result lists **consumed receipt IDs** (immutable references).

## Aggregate Gate Model

Future NPSC-5F Final / 5E Final gate tests:

- in-process static / E2E / drift checks unchanged
- mandatory regression **orchestrator test replaced by** receipt aggregation over compiled plan
- parity: every path in current flatten still executed exactly once as some leaf `suite_id`

## Failure Semantics

- **Collect all** suite failures; aggregate FAIL if any mandatory leaf FAIL.
- **Fail fast** only: invalid manifest, graph cycle, coordinator catastrophic error, isolation provisioning failure.
- One failing leaf → dependent aggregate gates FAIL **without** re-executing leaf (receipt reuse).

## Performance Telemetry

Extend coordinator / `QualificationPerformanceSnapshot` (no second telemetry model):

```text
suite_id, collection/start, start, end, duration, test_count (optional collect-only),
exit_code, status, deduplicated_request_count, parent_gate_count, resource_group
```

Report sections:

```text
qualification wall time · sum physical suite durations · duplicate time avoided ·
scheduler wait · critical path · parallel overlap (observed_overlap_ratio)
```

## Parity Strategy

### Semantic parity matrix (excerpt)

| Current gate | Required old targets | Future suite IDs | Coverage preserved |
| ------------ | -------------------- | ---------------- | ------------------ |
| NPSC-5F Final matrix | `MANDATORY_REGRESSION_SUITES` flatten + `-k` | compiled leaf set + same `-k` policy per leaf | **YES** |
| NPSC-5F R1 Final parametrize | `_MANDATORY_SUITES` | one leaf per label | **YES** |
| NPSC-5E R3 coord | 17 coordinator suites | unchanged IDs from adapter | **YES** |
| NPSC-5E Final mandatory | R3 Final file subprocess | `requires npsc5e-r3-coord-plan` receipt | **YES** |
| CURRENT_HEAD profile | steps 1–6 in revalidation doc | profile manifest | **YES** |

Future tests (design):

- `Old qualification required target set == New DAG required target set` (normalized)
- representative runs: old PASS → new PASS; injected leaf FAIL → aggregate FAIL without rerun
- failure injection: one leaf FAIL poisons dependent aggregates

## Anti-Regression Gates

| Test name | Intent |
| --------- | ------ |
| `test_no_nested_pytest_orchestrators_outside_canonical_qualification_executor` | fail if qualification-marked aggregate/orchestrator calls `subprocess` pytest outside executor layer |
| `test_each_leaf_suite_executes_at_most_once_per_run` | instrumented plan / mock executor |
| `test_aggregate_gate_never_launches_external_test_process` | static allowlist of gate modules |

## Migration Plan

| Phase | Responsibility |
| ----- | -------------- |
| **P1** | Canonical `suite_id` registry + gate/suite contracts + graph compiler (typed, cycle/missing/duplicate definition checks) — implementation: [`INTEGRAX_CANONICAL_QUALIFICATION_DAG_CONTRACTS_AND_GRAPH_COMPILER.md`](INTEGRAX_CANONICAL_QUALIFICATION_DAG_CONTRACTS_AND_GRAPH_COMPILER.md) |
| **P2** | Replace nested `_run_pytest` / matrix orchestrators with `requires` declarations; profiles (`CURRENT_HEAD_REVALIDATION`, `FULL_PLATFORM_CERTIFICATION`) |
| **P3** | Central DAG execution via extended `QualificationCoordinator` + in-run dedup — **implementation (execution plane, no orchestrator migration):** [`INTEGRAX_QUALIFICATION_IN_RUN_DEDUP_AND_RECEIPT_REUSE.md`](INTEGRAX_QUALIFICATION_IN_RUN_DEDUP_AND_RECEIPT_REUSE.md) |
| **P4** | Parity certification (target set equality + PASS/FAIL injection) |
| **P5** | Performance certification (wall time vs baseline; duplicate time avoided) |

P1–P3 may ship in one engineering epic if parity tests pass incrementally.

### Incremental qualification (design only)

```text
git diff → impacted guard families → impacted suite_ids → subgraph profile
```

Full certification retains ability to run entire canonical DAG.

## Risks

| Severity | Risk |
| -------- | ---- |
| **CRITICAL** | Dedup without `-k`/orchestrator parity could skip mandatory subprocess-isolated paths |
| **MAJOR** | Aggregate gate still spawning pytest via helper bypass |
| **MAJOR** | Same targets, different env between duplicate runs today — dedup must use strongest isolation superset |
| **MINOR** | suite_id naming drift across NPSC labels |
| **OBSERVATION** | Further gains after dedup from incremental profiles |

## Findings

| Severity | ID | Summary |
| -------- | -- | ------- |
| **MAJOR** | F-01 | NPSC-5F Final matrix runs R1/R2/R3 parametrize chains **inside** one pytest → **≥3×** directory suite respawn |
| **MAJOR** | F-02 | NPSC-5E Final embedded in matrix + upstream labels → **multi×** full 5E gate execution |
| **MAJOR** | F-03 | R3 coordinator path proven, but **not** wired to 5F/5E aggregators — duplicate trees remain |
| **OBSERVATION** | F-04 | Matrix `-k` excludes orchestrator **test names** only — insufficient for dedup |
| **OBSERVATION** | F-05 | Current HEAD profile overlaps matrix leaves with guard batch + R3 slice |
| **CRITICAL** | — | **None** — dedup design preserves full target set if parity gates enforced |

## Final Design Verdict

**QUALIFICATION DAG OPTIMIZATION DESIGN = APPROVED**

Conditions met: full orchestration graph documented; nested subprocess paths identified; duplicate executions bounded (≥30 removable subprocesses per 5F matrix); reuse **PARTIALLY — extend existing architecture**; no second framework; target DAG preserves test targets; leaf once per run; parent receipts; isolation and fail semantics preserved; telemetry and migration defined; **production runtime impact: NONE**.

---

> Implementation changes require audit against code on GitHub; this design does not replace independent verification.
