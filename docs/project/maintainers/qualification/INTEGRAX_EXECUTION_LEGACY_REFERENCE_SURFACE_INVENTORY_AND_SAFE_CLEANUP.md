# INTEGRAx-EXECUTION-LEGACY-REFERENCE-SURFACE-INVENTORY-AND-SAFE-CLEANUP

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAX-EXECUTION-LEGACY-REFERENCE-SURFACE-INVENTORY-AND-SAFE-CLEANUP` |
| Class | A — dead-code audit / compatibility hygiene (documentation closure) |
| Execution Engine | FROZEN / ENTERPRISE-CERTIFIED — no semantic changes |
| Prior closed follow-ups | F-01 testing_support boundary; UE-10R4.1 execution local-import hygiene |

## Scope

Inventory legacy / reference / compatibility surfaces tied to Execution Engine admission, host task intake, harness scheduling, and long-running resume wiring. Remove **only** surfaces with proven zero supported consumers. No redesign of `ExecutionRuntime`, `ExecutionBoundary`, `StrategyExecutionRouter`, or frozen recovery semantics.

**Search roots:** `intergrax/runtime/execution/`, `intergrax/runtime/task/`, `intergrax/runtime/long_running/`, `intergrax/applications/_shared/`, `testing_support/` (qualification references only), historical RB-2A/RB-2B candidates.

## Repository State

Audit on branch `development` at baseline SHA below. No production runtime edits in this task — prior RB-2B1 already removed dead harness surfaces.

## Baseline SHA

`d9d3a24eb1cbc7902c8ea4bbe9e7342f6d0cb186`

## Legacy / Reference Inventory

| Candidate | Location | Type | Consumers | Production? | Qualification? | Public/API? | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `UnifiedTaskRunner` | `intergrax/runtime/task/unified_task_runner.py` | Harness/scheduling adapter → `execute_root_task` | Eval (`nexus_eval_runner`), long-running tests, integration J3/J4, architecture gates (UE-9D, NPSC-4.1), many unit tests | No Tier-3 factory entry | Yes (identity/orchestration proofs) | Lazy export via `runtime/task/__init__.py` | **KEEP** |
| `HostTaskExecutionExecutor` | `intergrax/runtime/interactions/task_executor.py` | Interaction → `HostTaskExecutionPort` | All Tier-3 hosts, harness, lab, debug, task control wiring | **Yes** | Yes (NPSC-3C-D, U1 EP-01/04) | `__all__` on interactions module | **KEEP** |
| `mount_canonical_harness_task_routes` | `intergrax/applications/_shared/harness_task_routes.py` | Harness HTTP task control | `task_control_wiring`, extensive unit tests | Harness opt-in | U1 EP-04 | Documented harness API | **KEEP** |
| `wire_long_running_scheduler_with_host_execution` | `intergrax/runtime/long_running/wiring.py` | Scheduler + host execution resume | `harness_host_auxiliary_wiring`, lab paths | When LR enabled | Convergence gates | Internal wiring | **KEEP** |
| `task_run_bridge` (`new_run_id`, `task_from_*`) | `intergrax/runtime/task/task_run_bridge.py` | Transport / harness alias | Worker bootstrap, MCP/run adapter, LKW ask, gates (NPSC-4.1, HARDENING-6) | Partial (LKW `new_run_id`) | Yes | Lazy exports in `task/__init__.py` | **KEEP** |
| `task_metadata_bridge` | `intergrax/runtime/task/task_metadata_bridge.py` | Flat metadata ↔ typed Task | Task hydration across API/JSON | **Yes** (compat path) | Indirect | Internal | **KEEP** |
| `HostTaskExecutionRunAdapter` | `intergrax/runtime/task/host_task_execution_run_adapter.py` | FastAPI run dispatch → executor | Legal/dispute queue workers, `queue_worker_wiring` | **Yes** | NPSC-3G | Exported lazy symbol | **KEEP** |
| `NexusWorkerRuntime` | `intergrax/runtime/task/nexus_worker_execution.py` | Queue worker execution surface | `worker_bootstrap`, legal/dispute factories, background tests | **Yes** | UE-9A/11E, J3 integration | Worker payload schema | **KEEP** |
| `CanonicalExecutionRuntimeAdapter` | `intergrax/runtime/execution/canonical_intake_adapter.py` | AW trusted intake bridge | Autonomous work dispatch tests, GR-2-R3 gates | AW composition | GR-2-R3 policy | Documented intake implementor | **KEEP** |
| `resume_decision_from_durable_state` | `intergrax/runtime/execution/decision_recovery.py` | Test-only durable resume wrapper | Unit tests + W3-C4 gate (forbids prod wiring) | No (handoff module for prod) | W3-C4 | Internal | **KEEP (B)** |
| Legacy pickle blob guard | `decision_durable_wire_codec.py` | Runtime rejection | Durable decision load path | **Yes** | Decision qualification | Error type exported | **KEEP** |
| In-memory reference persistence modules | `in_memory_decision_*_persistence.py` | Test/reference stores | Unit/conformance tests | No | Yes | Internal | **KEEP (B)** |
| `mount_harness_task_routes` | — | Legacy harness mount | — | — | — | — | **ALREADY CLOSED @ RB-2B1** |
| `build_task_runner_with_enricher` | — | Legacy runner factory | — | — | — | — | **ALREADY CLOSED @ RB-2B1** |
| `wire_long_running_scheduler` (UTR) | — | Legacy LR wiring | — | — | — | — | **ALREADY CLOSED @ RB-2B1** |
| `NexusLoopTaskExecutor` | — | Retired intake executor | Gate tokens only in docs/history | — | Retirement gates | — | **ALREADY CLOSED** |
| `NexusTaskExecutionAdapter` / `QueuedNexusExecutionAdapter` | — | Retired queue adapters | Absent from `intergrax/` (NPSC-3G gate) | — | — | — | **ALREADY CLOSED** |
| EP-17 `WorkStageCapabilityLoop` | `autonomous_work` + integration tests | Non-prod capability loop | Tests only | No | U5 P3 inventory | — | **KEEP (C)** |
| EP-20 experiments `UnifiedTaskRunner` | `intergrax/experiments/workflow.py` | Dev workflow | Experiments only | No | P0 inventory LEGACY | — | **KEEP (C)** |
| EP-21 `nexus_eval_runner` | `intergrax/eval/nexus_eval_runner.py` | Eval orchestration | Eval tooling | No | P0 inventory LEGACY | — | **KEEP (C)** |
| `testing_support/execution_qualification/performance/legacy_multiplicity.py` | testing_support | Qualification accounting | Performance qualification runner | No | **Authority** | — | **KEEP (C)** |

## Consumer Analysis

- **UnifiedTaskRunner:** Delegates to `resolve_root_task_identity` + `execute_root_task` (canonical EE). Production Tier-3 factories do not construct it (NPSC-4.1 / NPSC-3G gates). Remaining callers are scheduler coordination, eval, experiments, and architecture/qualification tests — all classified non-production or harness-only per P0 inventory EP-20/21 and docstring contract.
- **HostTaskExecutionExecutor:** Thin 1:1 delegate to `HostTaskExecution` / host port; required for interaction intake and harness HTTP (U1 inventory).
- **Historical RB-2B targets:** Grep confirms no `def wire_long_running_scheduler` without `_with_host_execution`; no `mount_harness_task_routes` in source (only negative gate assertion).

## Public API Analysis

- `intergrax.runtime.execution` package `__all__` exports frozen facade types only — unchanged.
- `intergrax.runtime.task` lazy-loads `UnifiedTaskRunner`, `task_from_execution_request`, `HostTaskExecutionRunAdapter` — all have active consumers; removal would break documented harness/queue paths.
- `HostTaskExecutionExecutor` in `task_executor.__all__` — production contract.

## Qualification Reference Analysis

Drift guards explicitly reference: `UnifiedTaskRunner` (UE-9D, NPSC-4.1, EE-FINAL-ARCH legacy section), `HostTaskExecutionExecutor` (NPSC-3C-D, U1), `resume_decision_from_durable_state` (W3-C4 — must remain for tests, prod uses handoff), `legacy_multiplicity` (R3 performance qualification). Historical qualification markdown retains old names by policy — not edited.

## Plugin Compatibility Analysis

No exported execution symbol was proven dead with zero external/plugin risk. Default rule: **KEEP** unless formal deprecation record exists.

## Candidate Classification

| Category | Surfaces |
| --- | --- |
| **A. DEAD** | None at this revision (RB-2B1 already removed harness dead APIs) |
| **B. TEST-ONLY BUT REQUIRED** | `resume_decision_from_durable_state`, in-memory reference persistence, parts of `task_run_bridge` |
| **C. QUALIFICATION REFERENCE** | `legacy_multiplicity`, mandatory suite labels, EP-17 integration bindings |
| **D. COMPATIBILITY API** | `HostTaskExecutionExecutor`, `task_metadata_bridge`, harness canonical routes, `task/__init__` lazy exports |
| **E. PLUGIN / EXTENSION** | `CanonicalExecutionRuntimeAdapter`, fan-out/coordination adapters (internal contracts) |
| **F. INTERNAL LEGACY ADAPTER — SAFE TO REMOVE** | **None proven** |
| **G. DOCUMENTATION-ONLY** | Historical RB-2A/RB-2B records (evidence, not code) |
| **H. UNKNOWN** | **None acted upon** |

## Safe Removals

**None.** No candidate met the safe-delete rule (zero direct/indirect/exported/qualification/plugin references) without violating frozen semantics or compatibility.

### Removed surfaces table

| Surface | Proof of no consumers | Replacement/canonical path | Verdict |
| --- | --- | --- | --- |
| — | — | — | **No new removals this task** |

Prior session (RB-2B1) evidence retained in [`EXECUTION_ADOPTION_RESIDUAL_AUDIT_RB2A.md`](../audits/EXECUTION_ADOPTION_RESIDUAL_AUDIT_RB2A.md).

## Retained Compatibility Surfaces

| Surface | Why retained | Owner | Future action |
| --- | --- | --- | --- |
| `UnifiedTaskRunner` | Scheduler/eval/experiments + qualification gates; thin canonical delegate | `runtime/task` | Document-only unless EP-20/21 formally retired |
| `HostTaskExecutionExecutor` | Canonical interaction/harness intake adapter | `runtime/interactions` | None — production contract |
| `task_run_bridge.new_run_id` | Harness/eval alias; gate-enforced not for HTTP intake | `runtime/task` | Track under intake convergence docs |
| `task_metadata_bridge` | API flat-metadata compatibility | `runtime/task` | Migrate callers to typed `Task.options` over time |
| `mount_canonical_harness_task_routes` | Harness HTTP task control | `applications/_shared` | None |
| `wire_long_running_scheduler_with_host_execution` | LR resume via host execution | `runtime/long_running` | None |
| EP-17/20/21 non-prod paths | P0 inventory P3 legacy rows | AW / experiments / eval | Optional future gate or removal with separate task |

## Already Closed Historical Debt

| Item | Status |
| --- | --- |
| `mount_harness_task_routes` | **DELETED @ RB-2B1** |
| `build_task_runner_with_enricher` | **DELETED @ RB-2B1** |
| `wire_long_running_scheduler(UnifiedTaskRunner)` | **DELETED @ RB-2B1** |
| `NexusLoopTaskExecutor` production path | **RETIRED** (NPSC-3C-C gates) |
| `NexusTaskExecutionAdapter` / `QueuedNexusExecutionAdapter` | **RETIRED** (NPSC-3G) |
| F-01 `intergrax → testing_support` imports | **CLOSED** (separate task) |
| UE-10R4.1 execution local imports | **CLOSED** (separate task) |

## Dependency / Ownership Impact

No module moves or import graph changes. Ownership unchanged: canonical root execution remains `HostTaskExecution` / `ExecutionRuntime`; harness scheduling remains `UnifiedTaskRunner` as non-production adapter.

## Pluginability Impact

**None.** No extension points removed; contract → configured implementation pattern preserved.

## Layer Boundary Assessment

No `runtime → testing_support`, no consolidation to concrete types, no service locator introduced. Cleanup scope did not touch composition roots beyond audit.

## Execution Engine Semantics Impact

**None.** Frozen components untouched; no behavior change.

## Tests

| Suite | Result |
| --- | --- |
| `test_ee_final_arch_*` (10 modules) | **PASS** |
| `test_platform_execution_unification_u5_final_zero_bypass.py` | **PASS** |
| `test_ue_10r41_execution_import_hygiene_gate.py` | **PASS** |
| `test_intergrax_no_testing_support_import_gate.py` | **PASS** |
| `tests/unit/testing_support/execution_qualification/` | **257 passed, 7 skipped** (live perf skips) |

## Architecture Gates

Mandatory gates above: **PASS** (logged under `.tmp/session/LEGACY-SURFACE-CLEANUP/mandatory-gates.log`).

## Static Quality

No production files changed. Documentation-only commit — `ruff` / `pyright` / `git diff --check` on changed doc at commit time.

## Runtime Regression

`uv run pytest tests/unit/runtime/architecture/ -q` — **2 collection errors** (pre-existing: `obs_diag_conformance` marker not registered on `test_obs_diag_conformance_*.py`). Not introduced by this task. Mandatory subset (257 tests) **PASS**.

## Changed Files

| Path | Change |
| --- | --- |
| `docs/project/maintainers/qualification/INTEGRAX_EXECUTION_LEGACY_REFERENCE_SURFACE_INVENTORY_AND_SAFE_CLEANUP.md` | **Added** — audit closure evidence |

## Remaining Debt

- P0 inventory P3 rows: EP-17, EP-20, EP-21 (non-production `UnifiedTaskRunner` / AW loop) — retained by design.
- `task_run_bridge` intake alias debt documented in NPSC-4.1 / enterprise verification (not dead).
- Optional future task: formal retirement of experiments/eval-only runner construction (requires compatibility review, not silent delete).

## Decision

Full inventory completed. All candidates classified. **No surface met 100% dead-code proof without breaking qualification, public lazy exports, or harness contracts.** Prior RB-2B1 already removed the only proven-dead harness APIs.

## Commit SHA

`a44631fd27e1571ced2582a4d3fef72d5be69b5a`

## Final Verdict

**LEGACY / REFERENCE SURFACE CLEANUP = PASS — NO SAFE REMOVALS REQUIRED**
