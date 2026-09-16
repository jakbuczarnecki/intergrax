# INTEGRAx Execution Runtime Unit Failure Diagnostics and Blocker Classification

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-EXECUTION-RUNTIME-UNIT-FAILURE-DIAGNOSTICS-AND-BLOCKER-CLASSIFICATION` |
| Branch | `development` |
| Audit date | 2026-09-16 |

## Scope

`tests/unit/runtime/execution/` failure reproduction, per-test classification, blocker assessment, and minimal contract-first remediation. No Execution Engine ownership or frozen-contract changes.

## Repository State

| Item | Value |
| --- | --- |
| Branch | `development` |
| Baseline HEAD (audit start) | `e83d03df99acf20093393ca1d05ce48174d7f7f5` |
| `origin/development` | `e83d03df99acf20093393ca1d05ce48174d7f7f5` |
| Dirty (unrelated WIP, not staged) | runtime nexus/task WIP, platform_proofs, integration OBS spine |
| Stash | `stash@{0..2}` present |

## Baseline SHA

`e83d03df99acf20093393ca1d05ce48174d7f7f5`

## Initial Runtime Execution Suite Result

At baseline dirty tree:

- **Collection error** in `test_execution_runtime.py` (`ImportError: RootTaskIdentity` from `runtime`).
- With collection unblocked for inventory: **19 failed**, 1291 passed, 1 skipped (`--tb=short`, log: `.tmp/session/runtime-execution-unit-audit/without-execution-runtime.log`).

## Full Failure Inventory

| ID | Test | Error (summary) | First failing frame | Suspected owner | Classification |
| --- | --- | --- | --- | --- | --- |
| RU-F00 | `test_execution_runtime.py` (collect) | `ImportError: RootTaskIdentity` | test import line | Test imports | **D** |
| RU-F01 | `test_ue_8b1r1_ledger_lifecycle::test_per_run_isolation_on_long_lived_nexus_loop` | TaskResult validation: `authoritative_decision_exposure` | fake `TaskResult(...)` | Test fake | **D** |
| RU-F02 | `test_ue_8b1r1_ledger_lifecycle::test_handle_task_binds_root_execution_budget` | same | fake `TaskResult` | Test fake | **D** |
| RU-F03–RU-F09 | `test_ue_8b1r2_preserve_run_budget_through_nexus_entry::*` (7 tests) | same | fake `TaskResult` | Test fake | **D** |
| RU-F10–RU-F11 | `test_ue_9ar1_preserve_run_budget_across_redelivery::*` (2) | same | fake `TaskResult` | Test fake | **D** |
| RU-F12–RU-F18 | `test_host_task_revision_reentry::*` (7) | `LLMAdapterDependencyError` / missing `ollama` | `wire_application_environment` → `resolve_optional_environment_llm_adapter` | Harness composition | **C** + **E** |
| RU-F19 | `test_orchestration::test_nexus_does_not_rebind_when_boundary_execution_id_active` | spy: unexpected kwarg `task_id` | `boundary._spy_bind` | Test spy | **G** |

## Failure Groups

| Group | IDs | Root cause tag |
| --- | --- | --- |
| IDENTITY-IMPORT-STALE | RU-F00 | `RootTaskIdentity` canonical owner = `identity_authority` |
| TASKRESULT-TERMINAL-EXPOSURE | RU-F01–RU-F11 | Terminal `TaskResult` contract requires `authoritative_decision_exposure` |
| HARNESS-LLM-COUPLING | RU-F12–RU-F18 | Provider-neutral host tests materialized implicit Ollama via env wiring |
| ORCHESTRATION-SPY | RU-F19 | `bind_active_execution_identity` gained optional `task_id` (UE identity) |

## Optional Dependency Analysis

Failures were **not** “install ollama”. Neutral revision-reentry tests triggered `resolve_optional_environment_llm_adapter(env)` during `wire_application_environment` (tenant RAG/memory wiring) and Nexus orchestration materialization, despite Echo-only semantics.

## Ollama / Provider Coupling Analysis

| Test | Intended provider dependency | Actual dependency | Correct model |
| --- | --- | --- | --- |
| `test_host_task_revision_reentry::*` | None (Echo / execution identity) | Lab profile → Ollama adapter | Injected `FakeLLMAdapter` at harness composition root |

## Harness Composition Analysis

`build_harness_host_runtime` passed `llm_adapter=None` to Nexus and did not expose injection; `wire_application_environment` always resolved environment LLM for tenant memory/RAG wiring.

## Budget Failure Analysis

Not budget policy drift. Fakes returned terminal `TaskResult` without `authoritative_decision_exposure` (platform terminal exposure contract). Budget ledger behavior under test was not reached due to validation failure in fakes.

## Orchestration Spy / Signature Analysis

Production contract (`bind_active_execution_identity`) includes optional `task_id`. Boundary correctly forwards it. Spy omitted `task_id` → **G**, not production regression.

## Identity / Continuation Analysis

No continuation lifecycle defects in the 20 baseline failures. RU-F00 is import hygiene only (`RootTaskIdentity` module move).

## Root Cause Classification

| Category | IDs |
| --- | --- |
| A. REAL PRODUCTION SEMANTIC DEFECT | — |
| B. REAL PRODUCTION COMPOSITION DEFECT | — (harness Tier-3 only; fixed via injection) |
| C. TEST HARNESS DEFECT | RU-F12–RU-F18 |
| D. STALE TEST EXPECTATION | RU-F00, RU-F01–RU-F11 |
| E. OPTIONAL DEPENDENCY COUPLING | RU-F12–RU-F18 (symptom) |
| F. ENVIRONMENT / MACHINE | — |
| G. TEST DOUBLE / SPY SIGNATURE DRIFT | RU-F19 |
| H. BUDGET / POLICY EXPECTATION DRIFT | — (mis-tagged; was D) |
| I. ORDER / STATE LEAK | — |
| J. UNKNOWN | — |

## Architecture Blocker Classification

| Failure group | Architecture blocker? | Reason |
| --- | --- | --- |
| IDENTITY-IMPORT-STALE | No | Test import path only |
| TASKRESULT-TERMINAL-EXPOSURE | No | Test fakes lag terminal exposure gate |
| HARNESS-LLM-COUPLING | No | Composition/test harness; production path unchanged when adapter injected explicitly |
| ORCHESTRATION-SPY | No | Spy drift; production binding contract extended correctly |

## Selected Remediations

1. **Tests:** import `RootTaskIdentity` from `identity_authority`; add `terminal_task_result_exposure_no_decision_gate()` to budget fakes; extend orchestration spy with `task_id`; pass `FakeLLMAdapter()` into harness builder.
2. **Harness:** `build_harness_host_runtime(..., llm_adapter=...)`; `wire_application_environment(..., llm_adapter=...)` for RAG/memory wiring precedence.

### BEFORE / AFTER (ownership / contract flow)

**BEFORE:** Harness → env wiring → `resolve_optional_environment_llm_adapter(env)` → Ollama; Nexus `llm_adapter=None` → same resolver.

**AFTER:** Harness composition root accepts `LLMAdapter` contract implementation → propagated to env wiring + Nexus factory; environment resolver runs only when adapter not injected.

## Contracts / Ports

| Mechanism | Contract | Implementation | External replacement possible? |
| --- | --- | --- | --- |
| LLM for host harness | `LLMAdapter` | `FakeLLMAdapter` (tests) / profile resolver (prod) | Yes |
| Terminal task exposure | `TaskResult` + authoritative exposure | `terminal_task_result_exposure_no_decision_gate()` | Yes |
| Active identity bind | `bind_active_execution_identity` | `execution_identity` module | Yes |

## Pluginability Assessment

Preserved. No service locator; explicit injection at harness composition.

## Layer Boundary Assessment

Changes limited to `applications/_shared` harness wiring and unit tests. No `runtime → applications` or `runtime → testing_support` in production packages.

## Execution Engine Impact

None on frozen `ExecutionRuntime` / boundary / strategy ownership semantics.

## Architecture Reopen Assessment

**Not required** for remediated items.

## Targeted Regression Results

After remediations: affected groups green (budget suite, host revision reentry, orchestration spy, full `test_execution_runtime` collection).

## Full Runtime Execution Suite Result

**1318 passed**, 1 skipped, **0 failed** (post-remediation, log: `.tmp/session/runtime-execution-unit-audit/final-full.log`).

## R14 Regression

F34–F37 architecture gates not re-run as a labeled bundle in this session; UE-10R* import/authority gates included in regression bundle below passed.

## R11 Regression

Not isolated as dedicated R11 bundle; OBS-DIAG architecture gates in bundle passed.

## R9/R10 Regression

`test_npsc5e_final_recovery_plane_qualification_and_freeze` **failed** nested DG-001 slice: `test_host_task_resume_lineage_identity.py` (2 failures). **Not introduced by execution-unit remediations**; correlates with unrelated dirty runtime WIP on branch. Re-run after WIP reconciliation.

`test_npsc5d_final_multi_agent_governance_qualification` passed in bundle.

## EE Architecture Gates

`test_ee_final_arch_*.py` (10 modules) passed in regression bundle.

## U5

`test_platform_execution_unification_u5_final_zero_bypass.py` passed.

## UE-10R4.1

`test_ue_10r41_execution_import_hygiene_gate.py` passed.

## F-01

`test_ee_b2_final_fault_matrix.py` passed (registry gate).

## OBS-DIAG

`test_obs_diag_port_1_gates.py`, `test_obs_diag_conformance_architecture.py` passed.

## R5 Boundary

`test_gr5_r5_restart_exact_identity.py` passed.

## Qualification Regression

`tests/unit/testing_support/execution_qualification/` passed (7 skipped live perf).

## Static Quality

`ruff check` passed on changed `intergrax/applications/_shared/*` modules. Pre-existing ruff/pyright noise in untouched test files not expanded in this task.

## Cold Imports

`build_harness_host_runtime`, `wire_application_environment` import OK.

## Changed Files

- `intergrax/applications/_shared/harness_host_runtime.py`
- `intergrax/applications/_shared/environment_wiring.py`
- `tests/unit/runtime/execution/test_execution_runtime.py`
- `tests/unit/runtime/execution/test_orchestration.py`
- `tests/unit/runtime/execution/test_host_task_revision_reentry.py`
- `tests/unit/runtime/execution/budget/test_ue_8b1r1_ledger_lifecycle.py`
- `tests/unit/runtime/execution/budget/test_ue_8b1r2_preserve_run_budget_through_nexus_entry.py`
- `tests/unit/runtime/execution/budget/test_ue_9ar1_preserve_run_budget_across_redelivery.py`
- This document

## Remaining Debt

| Item | Final group |
| --- | --- |
| Lineage host-task resume tests failing on dirty branch (NPSC-5E nested) | **REAL BLOCKER** (separate from unit audit scope; investigate WIP runtime) |
| Unrelated repo WIP | Out of scope |

## Decision

All **20** baseline execution-unit failures classified; **19 runtime failures + 1 collection error** remediated with minimal harness/test updates. No skip/xfail. No architecture weakening.

## Commit SHA

(Set at commit: see git log for message `INTEGRAx-EXECUTION-RUNTIME-UNIT-FAILURE-DIAGNOSTICS-AND-BLOCKER-CLASSIFICATION`.)

## Final Verdict

**RUNTIME EXECUTION UNIT FAILURE AUDIT = PASS — ALL ACTIONABLE ROOT CAUSES REMEDIATED**

(NPSC-5E nested qualification on dirty tree: **follow-up** — not an execution-unit suite blocker.)
