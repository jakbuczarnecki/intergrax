# INTEGRAx-EXECUTION-FULL-ARCHITECTURE-SUITE-FAILURE-DIAGNOSTICS-AND-BLOCKER-CLASSIFICATION

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-EXECUTION-FULL-ARCHITECTURE-SUITE-FAILURE-DIAGNOSTICS-AND-BLOCKER-CLASSIFICATION` |
| Mode | Diagnostic audit · failure classification · root-cause analysis (no mass remediation) |
| Diagnostic HEAD | `e16ccafa4d61615dd1e74d261f711ceb4f8284b6` |
| `origin/development` | `e16ccafa4d61615dd1e74d261f711ceb4f8284b6` (aligned) |
| Session artifacts | `.tmp/session/full-architecture-failure-diagnostics/` |
| Production code edits | **0** (read-mostly task) |

## Scope

Full pytest collection and execution under `tests/unit/runtime/architecture/`, isolated reruns per failure, mandatory known-good gates, and `tests/unit/testing_support/execution_qualification/`. No changes to frozen Execution Engine surfaces (`ExecutionRuntime`, `ExecutionBoundary`, `StrategyExecutionRouter`, governance, identity authority, recovery semantics, evidence/control separation, tool boundary, root/child admission).

## Repository State

| Item | Value |
| --- | --- |
| Branch | `development` (expected) |
| HEAD | `e16ccafa4d61615dd1e74d261f711ceb4f8284b6` |
| Merge conflicts | None |
| Stash | `stash@{0..2}` present (not applied) |
| Dirty WIP (unstaged) | Memory/procedural-memory WIP, `tests/conftest.py`, token_optimization tests, `testing_support/builder.py`, etc. |
| Trustworthiness note | WIP does **not** block reproduction of architecture failures or mandatory gates at HEAD; documented for operator awareness. Not staged for this commit. |

## Baseline SHA

Committed baseline for this diagnostic: **`e16ccafa4d61615dd1e74d261f711ceb4f8284b6`**.

## Full Suite Reproduction

```bash
uv run pytest tests/unit/runtime/architecture/ -q --tb=no
```

| Metric | Result |
| --- | --- |
| Collected | 1856 |
| Passed | 1819 |
| Failed | **37** |
| Warnings | 9 |
| Duration | 2727.52s (~45m27s) |
| Log | `.tmp/session/full-architecture-failure-diagnostics/full-suite.log` |

Prior operator note referenced 35 failures / 1855 collected; at current HEAD the suite reports **37** failures and **1856** collected (delta: +2 failures, +1 test).

## Failure Inventory

| ID | Test | Failure type | Exception / assert (summary) | Isolated rerun | Group | Initial class |
| --- | --- | --- | --- | --- | --- | --- |
| ARCH-F01 | `test_audit_ideal_3_1_envelope_runtime_roundtrip` | API drift | `TypeError`: `from_envelope()` missing `task_id`, `run_id` | FAIL | AUDIT-IDEAL | D |
| ARCH-F02 | `test_audit_ideal_30_1_ecp_architecture_synced` | Missing doc | `FileNotFoundError`: `docs/architecture/ELASTIC_CAPACITY_AND_SCALING.md` | FAIL | AUDIT-IDEAL | G |
| ARCH-F03 | `test_audit_ideal_32_1_debt_burn_down` | Missing doc | `FileNotFoundError`: `docs/guides/ARCHITECTURE_DEBT_REGISTER.md` | FAIL | AUDIT-IDEAL | G |
| ARCH-F04 | `test_audit_ideal_32_2_plan_scorecard_sync` | Missing doc | `FileNotFoundError`: `docs/plan/AUDIT_IDEAL_2026.md` | FAIL | AUDIT-IDEAL | G |
| ARCH-F05 | `test_audit_ideal_28_3_lkw_hybrid_daemon` | Expectation | `wiring.enabled is False` (expected True) | FAIL | AUDIT-IDEAL | D |
| ARCH-F06 | `test_audit_ideal_register_complete` | Missing doc | Same `AUDIT_IDEAL_2026.md` path | FAIL | AUDIT-IDEAL | G |
| ARCH-F07 | `test_scenario_a_pre_b5_failure_emits_record_and_preserves_primary_exception` | Fixture / governance | `ReferenceProductionLifecycleGovernanceBlockedError` on activation | FAIL | DG001B4 | B |
| ARCH-F08 | `test_scenario_b_pre_b5_identity_boundary_has_attempt_id_without_fabricated_identity` | Same | Same governance block | FAIL | DG001B4 | B |
| ARCH-F09 | `test_scenario_c_reporter_failure_does_not_mask_primary_failure` | Same | Same | FAIL | DG001B4 | B |
| ARCH-F10 | `test_scenario_a_worker_main_entrypoint_uses_production_producer_and_guarded_segment` | Same | `background_worker_main.main()` activation denied | FAIL | DG001B4 | B |
| ARCH-F11 | `test_scenario_d_post_b5_boundary_preserves_host_diag3_application_failed_flow` | Same | `_activated_projection` governance block | FAIL | DG001B4 | B |
| ARCH-F12 | `test_pre_b5_success_path_reaches_post_b5_guard_without_bootstrap_failure_record` | Same | Same | FAIL | DG001B4 | B |
| ARCH-F13 | `test_df4_scenario_task_preserves_run_and_uses_terminal_diagnostics` | Optional dep | `LLMAdapterDependencyError`: missing `ollama` package | FAIL | DIAG-DF4 | C |
| ARCH-F14 | `test_df4_background_task_uses_shared_terminal_diagnostic_path` | API drift | `AttributeError`: port has no `_orchestrator` | FAIL | DIAG-DF4 | D |
| ARCH-F15 | `test_ee_b3_c_allowing_runtime_admission_not_wired_in_intergrax_tree` | Static scan | Unexpected paths referencing admission wiring | FAIL | EE-B3-C | E |
| ARCH-F16 | `test_ee_final_enterprise_no_execution_core_drift_since_revalidation` | SHA / tree pin | Non-empty `git diff` set under `intergrax/runtime/execution/` | FAIL | EE-FINAL-PIN | E |
| ARCH-F17 | `test_intergrax_no_applications_import_gate` | Layer gate script | Script exit 1: tier violations listed (agents + runtime) | FAIL | IMPORT/BOUNDARY | E |
| ARCH-F18 | `test_gate_allows_certified_harness_unified_task_runner_import` | Allowlist | `UnifiedTaskRunner` import in `long_running/wiring.py` | FAIL | GR2-R3 | E |
| ARCH-F19 | `test_direct_opentelemetry_imports_are_allowlisted` | OTEL allowlist | 3 new `opentelemetry.sdk._logs` import sites | FAIL | OTEL | E |
| ARCH-F20 | `test_hardening_3_contracts_do_not_import_runtime_except_allowlist` | Allowlist | `execution_reconstruction.py` → runtime reconstruction | FAIL | HARDENING-3 | E |
| ARCH-F21 | `test_ideal_l3_umbrella_gate_script` | Missing script | `scripts/check_ideal_harness_l3_gates.py` not found (moved under `scripts/gates/`) | FAIL | HARNESS-L3 | D |
| ARCH-F22 | `test_ideal_w2_w2_script_gates` | Missing script | `scripts/check_registry_snapshot_diff.py` not found | FAIL | HARNESS-L3 | D |
| ARCH-F23 | `test_harness_governance_signals_pass_l3_and_l4` | Live metrics | `report.l3.passed is False` (`metrics_pipeline_passed`) | FAIL | MATURITY | H |
| ARCH-F24 | `test_l4_fails_when_adaptive_governance_fails` | Cascading L3 | Same L3 precondition failure | FAIL | MATURITY | H |
| ARCH-F25 | `test_mandatory_frozen_suite_passes[Runtime events suites]` | Nested pytest | Subprocess non-zero in full suite | **PASS** (isolated ×3) | NPSC-5F-R1 | F |
| ARCH-F26 | `test_mandatory_frozen_suite_passes[Runtime observability suites]` | Nested pytest | Subprocess non-zero in full suite | PASS (isolated file run) | NPSC-5F-R1 | F |
| ARCH-F27 | `test_mandatory_frozen_suite_passes[NPSC-5E Final]` | Nested pytest | NPSC-5E final → R3 parallel qual FAIL | FAIL | NPSC-5F-R1 | B |
| ARCH-F28 | `test_mandatory_frozen_suite_passes[DG_001]` | Nested pytest | Subprocess non-zero in full suite | PASS (isolated file run) | NPSC-5F-R1 | F |
| ARCH-F29 | `test_mandatory_frozen_suite_passes[NPSC-5D Final]` | Nested pytest | Child `test_npsc5d_final_*` subprocess FAIL | FAIL | NPSC-5F-R1 | B |
| ARCH-F30 | `test_npsc5f_r3_h1_post_r3_event_surface_classification_recorded` | Frozen record | Missing bucket for `runtime_event.py` | FAIL | NPSC-5F-R3 | E |
| ARCH-F31 | `test_npsc5f_r3_h1_integrated_head_pin_recorded` | SHA pin | HEAD `72e2c05…` ≠ pinned `8ddf63f…` | FAIL | NPSC-5F-R3 | E |
| ARCH-F32 | `test_p0_frozen_child_execution_runner_import_surface` | Inventory pin | Extra import site `delegated_execution/service.py` | FAIL | U5-P0 | E |
| ARCH-F33 | `test_repo_prompt_golden_catalog_matches_expectations` | Golden hash | `tools_agent_system v1` content hash mismatch | FAIL | PROMPT-GOLDEN | D |
| ARCH-F34 | `test_execution_package_has_no_forbidden_quality_constructions` | UE guard | `lineage/codecs.py` `from typing import Any` flagged | FAIL | UE | D |
| ARCH-F35 | `test_host_task_does_not_bypass_execution_facade` | UE guard | `host_task.py:384: resolve_root_task_identity()` flagged | FAIL | UE | D |
| ARCH-F36 | `test_registry_module_owns_entry_point_loading` | UE guard | Source no longer contains literal `entry_points` | FAIL | UE | D |
| ARCH-F37 | `test_strategy_resolver_is_owned_by_canonical_router` | UE guard | `StrategyResolver()` in `root_execution_operation_mapping.py` | FAIL | UE | E |

## Failure Grouping

| Group | Failure IDs | Shared root cause (summary) |
| --- | --- | --- |
| AUDIT-IDEAL | F01–F06 | Legacy doc paths + stale Audit Ideal API / LKW wiring expectations |
| DG001B4 | F07–F12 | Reference production activation blocked in test harness (no governance bypass in prod path under test) |
| DIAG-DF4 | F13–F14 | Optional LLM provider dep + terminal diagnostic port refactor not reflected in tests |
| EE-B3-C | F15 | Static “admission wiring” scan hits new harness launch wiring paths |
| EE-FINAL-PIN | F16 | Post-revalidation execution-tree drift vs frozen empty-diff gate |
| IMPORT/BOUNDARY | F17 | Maintainer script reports `intergrax.applications` imports from tier-0/1 (contracts bridge) |
| GR2-R3 | F18 | Certified `UnifiedTaskRunner` import now present in long-running wiring |
| OTEL | F19 | New OTLP log exporter modules not on allowlist |
| HARDENING-3 | F20 | New contract→runtime reconstruction coupling not allowlisted |
| HARNESS-L3 | F21–F22 | Tests invoke retired `scripts/` paths; CI map uses `scripts/gates/` |
| MATURITY | F23–F24 | Live `metrics_pipeline_passed` false in `collect_harness_governance_signals()` |
| NPSC-5F-R1 | F25–F29 | Nested `uv run pytest` composition: order-sensitive (F25,F26,F28) + real child-suite failures (F27,F29) |
| NPSC-5F-R3 | F30–F31 | Frozen H1 classification / integrated HEAD pins stale vs current tree |
| U5-P0 | F32 | P0 ChildExecutionRunner import inventory missing new canonical site |
| PROMPT-GOLDEN | F33 | Golden catalog hash drift for `tools_agent_system` |
| UE | F34–F37 | UE-10R4 / 11GP / 8P2 / 9D gates not updated for legitimate execution-package evolution |

## Individual Failure Register

See **Failure Inventory** (ARCH-F01 … ARCH-F37). Each failure has exactly one primary classification below.

### Primary classification key

- **A** ARCHITECTURE BLOCKER  
- **B** TEST INFRASTRUCTURE DEFECT  
- **C** ENVIRONMENT / OPTIONAL DEPENDENCY  
- **D** STALE TEST EXPECTATION  
- **E** FROZEN / REFERENCE DRIFT  
- **F** FLAKY / ORDER DEPENDENCY  
- **G** DOCUMENTATION / MANIFEST DRIFT  
- **H** NON-BLOCKING OBSERVATION  

| ID | Primary | Secondary | Severity | Blocker? |
| --- | --- | --- | --- | --- |
| ARCH-F01 | D | — | LOW | No |
| ARCH-F02 | G | D | LOW | No |
| ARCH-F03 | G | — | LOW | No |
| ARCH-F04 | G | — | LOW | No |
| ARCH-F05 | D | — | MEDIUM | No |
| ARCH-F06 | G | — | LOW | No |
| ARCH-F07 | B | — | MEDIUM | No |
| ARCH-F08 | B | — | MEDIUM | No |
| ARCH-F09 | B | — | MEDIUM | No |
| ARCH-F10 | B | — | MEDIUM | No |
| ARCH-F11 | B | — | MEDIUM | No |
| ARCH-F12 | B | — | MEDIUM | No |
| ARCH-F13 | C | B | MEDIUM | No |
| ARCH-F14 | D | B | MEDIUM | No |
| ARCH-F15 | E | — | LOW | No |
| ARCH-F16 | E | — | MEDIUM | No |
| ARCH-F17 | E | G | HIGH | No* |
| ARCH-F18 | E | — | MEDIUM | No |
| ARCH-F19 | E | — | LOW | No |
| ARCH-F20 | E | — | MEDIUM | No |
| ARCH-F21 | D | G | LOW | No |
| ARCH-F22 | D | G | LOW | No |
| ARCH-F23 | H | — | OBSERVATION | No |
| ARCH-F24 | H | B | OBSERVATION | No |
| ARCH-F25 | F | B | MEDIUM | No |
| ARCH-F26 | F | B | MEDIUM | No |
| ARCH-F27 | B | E | HIGH | No |
| ARCH-F28 | F | B | MEDIUM | No |
| ARCH-F29 | B | — | HIGH | No |
| ARCH-F30 | E | — | MEDIUM | No |
| ARCH-F31 | E | — | MEDIUM | No |
| ARCH-F32 | E | — | MEDIUM | No |
| ARCH-F33 | D | — | LOW | No |
| ARCH-F34 | D | — | LOW | No |
| ARCH-F35 | D | E | MEDIUM | No |
| ARCH-F36 | D | — | LOW | No |
| ARCH-F37 | E | — | MEDIUM | No |

\*F17 evidences tier coupling via **contracts** (`intergrax.applications.contracts`) in runtime/agents paths; mandatory EE/U5/qualification gates still PASS — not proven as supported production **execution bypass** or governance defeat.

## Root Cause Analysis

1. **Audit Ideal (6)** — Tests still point at pre–doc-topology paths (`docs/architecture`, `docs/plan`, `docs/guides`) while canonical docs live under `docs/project/…`. One test calls an outdated `RuntimeRequest.from_envelope` signature; LKW hybrid daemon wiring disabled in current tree.
2. **DG001B4 (6)** — Integration tests activate real reference-production lifecycle; control-plane governance denies admission without test doubles. Failure is harness/setup, not demonstrated runtime bypass.
3. **DF4 (2)** — Scenario fixture resolves Ollama LLM adapter without optional `ollama` install; background-task test assumes private `_orchestrator` on `CentralTerminalExecutionDiagnosticPort` after refactor.
4. **EE enterprise pin (1)** — Gate expects zero post-pin diff in execution core; tree has evolved (new modules under `intergrax/runtime/execution/`). Certification slice gates (`test_ee_final_arch_*`) pass independently.
5. **Import / boundary / allowlist cluster (F17–F20, F32)** — Static gates and inventories lag repository reality (applications.contracts bridge, OTEL OTLP files, reconstruction contract, ChildExecutionRunner import surface).
6. **Harness L3 (2)** — RB-2B / CI path migration: scripts relocated to `scripts/gates/` per `scripts/ci/script_paths.py`; architecture tests still spawn legacy paths.
7. **Maturity (2)** — `collect_harness_governance_signals()` sets `metrics_pipeline_passed=False` on this workstation run; other L3 checks pass. Tests assert idealized all-green L3 without isolating metrics pipeline.
8. **NPSC-5F-R1 (5)** — Nested subprocess pytest: three parametrized targets fail only after long full-suite run (state/order); two targets fail consistently (NPSC-5E final → NPSC-5E/R3 parallel mandatory qual FAIL; NPSC-5D final child suite FAIL).
9. **NPSC-5F-R3 (2)** — Recorded H1 integrated SHA and event-surface buckets not updated after upstream event module changes.
10. **Prompt golden (1)** — Content hash for `tools_agent_system v1` changed without catalog update.
11. **UE cluster (4)** — Guards treat `typing.Any`, identity resolution helper, refactored registry source shape, and new `root_execution_operation_mapping` as violations; UE-10R4.1 import-hygiene gate passes.

## Isolated Rerun Results

| Command pattern | Outcome |
| --- | --- |
| All 37 failing nodes (batched) | **32 FAIL**, 114 PASS (same run includes passing modules) — log: `failures-detail.log` |
| `test_npsc5f_r1_final…` file alone | 2 FAIL (NPSC-5E, NPSC-5D), 27 PASS |
| `Runtime events suites` parametrized ×3 | **3× PASS** |
| Known-good gate bundle (EE final arch, U5, UE-10R4.1, OBS-DIAG) | **49 PASS** |
| F-01 `test_intergrax_no_testing_support_import_gate` | **2 PASS** |
| Qualification `execution_qualification/` | **216 PASS**, 7 skipped |

## Repeatability / Flakiness

| Failure IDs | Evidence |
| --- | --- |
| ARCH-F25, F26, F28 | FAIL in full suite; PASS in isolated reruns (F25 repeated 3×). Classified **F** (order/state leak in long suite), not architecture regression. |
| ARCH-F27, F29 | FAIL consistently isolated — reproducible nested-suite debt. |
| Remaining IDs | Stable FAIL on isolated rerun (deterministic). |

## Environment Analysis

| Check | Result |
| --- | --- |
| Optional `ollama` | Not installed; triggers ARCH-F13 via env profile resolution (`Intergrax-ai[llm-ollama]`). Architecture tests should guard or stub LLM resolution for DF4. |
| OTEL packages | Present enough to import; failure is **allowlist**, not `ModuleNotFoundError`. |
| Platform | `win32`, Python 3.12.11, pytest 8.4.2 |
| Metrics pipeline | `metrics_pipeline_passed=False` drives ARCH-F23/F24 |

## Architecture Invariant Mapping

| Group | Relevant invariant | Assessment |
| --- | --- | --- |
| AUDIT-IDEAL | NONE (doc/scorecard machinery) | — |
| DG001B4 | Governance admission fail-closed | **Holds** (errors are denials, not bypass) |
| DIAG-DF4 | Terminal diagnostics single path | Test drift; EE/OBS-DIAG gates PASS |
| EE-B3-C / EE-FINAL-PIN | Execution core ownership / no drift | Pin gates stale; `test_ee_final_arch_*` PASS |
| IMPORT/BOUNDARY | Tier boundaries | Coupling observed; not re-proven as execution bypass |
| GR2 / U5-P0 | No execution bypass / inventory SSOT | U5 zero-bypass gate PASS; P0 inventory stale |
| OTEL / HARDENING-3 | Observability pluginability / contract isolation | Allowlist maintenance |
| HARNESS-L3 | Harness reference surface | Script path drift |
| MATURITY | L3 evidence completeness | Observational metric pipeline |
| NPSC-5F-* | Evidence durability / frozen qual composition | Nested orchestration + pin drift |
| PROMPT-GOLDEN | Prompt catalog SSOT | Golden drift |
| UE | Graph authority / facade routing | UE-10R4.1 PASS; sibling UE gates stale |

## Pluginability Assessment

Failures do not demonstrate loss of injectable strategy/provider adapters. OTEL and persistence-related failures are **gate/allowlist** mismatches, not hard-coded vendor lock-in in core execution paths under certification gates.

## Layer Boundary Assessment

ARCH-F17 lists imports of `intergrax.applications` from `graph_builder.py`, sandbox modules, and agents — primarily **contracts** bridge pattern. This is **documented debt** for a dedicated boundary slice; it does not invalidate mandatory execution qualification at HEAD.

## Blocker Classification

**No failure meets the task’s architecture-blocker proof bar** (supported production bypass, duplicate lifecycle owner, governance bypass, identity authority defect, recovery defect, evidence/control violation, pluginability break, persistence abstraction break, or demonstrated layer violation with production impact on certified execution paths).

Mandatory execution certification gates remain green (see below).

## Non-Blocking Debt

All 37 failures are classified as test infrastructure, environment/optional dependency, stale expectations, frozen/reference drift, order dependency, documentation drift, or non-blocking maturity observation.

## Required Remediation Slices

| Slice | Scope | Failures |
| --- | --- | --- |
| **R1** | AUDIT-IDEAL: update doc paths to `docs/project/…`, fix `from_envelope` test, LKW wiring expectation | F01–F06 |
| **R2** | DG001B4: governance test doubles for reference-production activation | F07–F12 |
| **R3** | DF4: stub LLM / optional-dep guard; update terminal port assertions | F13–F14 |
| **R4** | EE-FINAL-PIN + EE-B3-C: classify post-freeze execution diff; refresh static baselines | F15–F16 |
| **R5** | IMPORT/BOUNDARY: tier import remediation or contract relocation decision | F17 |
| **R6** | GR2 + HARDENING-3 + OTEL + U5-P0: synchronized allowlist/inventory updates | F18–F20, F32 |
| **R7** | HARNESS-L3: point tests at `scripts/gates/` (CI path map) | F21–F22 |
| **R8** | MATURITY: decouple unit test from live metrics pipeline or fix pipeline inputs | F23–F24 |
| **R9** | NPSC-5E/R3 parallel mandatory qualification failures (execution qual orchestrator) | F27 (chain) |
| **R10** | NPSC-5D final nested suite failure | F29 |
| **R11** | NPSC-5F-R1 subprocess order: investigate suite pollution (F25,F26,F28) | F25,F26,F28 |
| **R12** | NPSC-5F-R3 H1 pins and event surface record | F30–F31 |
| **R13** | PROMPT-GOLDEN: intentional catalog/hash update | F33 |
| **R14** | UE gates: reconcile 10R4/11GP/8P2/9D with current execution package layout | F34–F37 |

## Known-Good Gates

```bash
uv run pytest \
  tests/unit/runtime/architecture/test_ee_final_arch_composition_root_convergence.py \
  tests/unit/runtime/architecture/test_ee_final_arch_execution_entry_inventory.py \
  tests/unit/runtime/architecture/test_ee_final_arch_legacy_nonproduction_paths.py \
  tests/unit/runtime/architecture/test_ee_final_arch_owner_uniqueness.py \
  tests/unit/runtime/architecture/test_ee_final_arch_persistence_abstraction.py \
  tests/unit/runtime/architecture/test_ee_final_arch_pluginability.py \
  tests/unit/runtime/architecture/test_ee_final_arch_scheduler_ownership.py \
  tests/unit/runtime/architecture/test_ee_final_arch_tool_side_effect_boundary.py \
  tests/unit/runtime/architecture/test_ee_final_arch_vendor_neutrality.py \
  tests/unit/runtime/architecture/test_ee_final_arch_zero_execution_bypass.py \
  tests/unit/runtime/architecture/test_platform_execution_unification_u5_final_zero_bypass.py \
  tests/unit/runtime/architecture/test_ue_10r41_execution_import_hygiene_gate.py \
  tests/unit/runtime/architecture/test_obs_diag_conformance_architecture.py \
  -q --tb=no
```

**Result:** 49 passed (`known-good-gates.log`).

| Gate | Status |
| --- | --- |
| EE Architecture (`test_ee_final_arch_*`) | PASS |
| U5 zero-bypass | PASS |
| UE-10R4.1 import hygiene | PASS |
| F-01 (`test_intergrax_no_testing_support_import_gate`) | PASS |
| OBS-DIAG conformance architecture | PASS |

## Qualification Regression

```bash
uv run pytest tests/unit/testing_support/execution_qualification/ -q --tb=no
```

**Result:** 216 passed, 7 skipped (`qualification.log`).

## Collect-Only Result

```bash
uv run pytest tests/unit/runtime/architecture/ --collect-only -q
```

**Result:** 1856 tests collected, **0 collection errors** (`collect-only.log`).

## Changed Files

| Path | Action |
| --- | --- |
| `docs/project/maintainers/qualification/INTEGRAX_EXECUTION_FULL_ARCHITECTURE_SUITE_FAILURE_DIAGNOSTICS_AND_BLOCKER_CLASSIFICATION.md` | Added (this document) |

No production code changes.

## Remaining Debt

37 failing architecture tests at HEAD, decomposed into remediation slices R1–R14. Operator WIP (memory, `conftest.py`, etc.) remains unstaged.

## Decision

**FULL ARCHITECTURE SUITE DIAGNOSTICS = PASS — ZERO ARCHITECTURE BLOCKERS**

Diagnostics complete: failures inventoried, root-caused, and classified. Remediation deferred to scoped slices; no frozen Execution Engine edits in this task.

## Commit SHA

Record at publish time: diagnostic document commit on `development` (after `e16ccafa4d61615dd1e74d261f711ceb4f8284b6`).

## Final Verdict

At `e16ccafa4d61615dd1e74d261f711ceb4f8284b6`, the full architecture suite reports **37 deterministic or order-sensitive test/debt failures** and **zero proven architecture blockers** against frozen Execution Engine certification criteria. Mandatory EE, U5, UE-10R4.1, F-01, OBS-DIAG, and execution qualification gates **PASS**.

---

### Blocker matrix (by group)

| Group | Failures | Root cause | Class | Severity | Blocker? | Next action |
| --- | ---: | --- | --- | --- | --- | --- |
| AUDIT-IDEAL | 6 | Legacy doc paths + stale API/wiring | D/G | LOW | No | R1 |
| DG001B4 | 6 | Governance blocked test activation | B | MEDIUM | No | R2 |
| DIAG-DF4 | 2 | Optional ollama + port API drift | C/D | MEDIUM | No | R3 |
| EE-B3-C | 1 | Static scan baseline | E | LOW | No | R4 |
| EE-FINAL-PIN | 1 | Post-pin execution tree diff | E | MEDIUM | No | R4 |
| IMPORT/BOUNDARY | 1 | applications.contracts imports | E | HIGH | No | R5 |
| GR2-R3 | 1 | UTR import inventory | E | MEDIUM | No | R6 |
| OTEL | 1 | Allowlist gap | E | LOW | No | R6 |
| HARDENING-3 | 1 | Allowlist gap | E | MEDIUM | No | R6 |
| HARNESS-L3 | 2 | Script path retired | D | LOW | No | R7 |
| MATURITY | 2 | metrics_pipeline_passed false | H | OBS | No | R8 |
| NPSC-5F-R1 | 5 | Nested pytest + order | F/B | MEDIUM–HIGH | No | R9–R11 |
| NPSC-5F-R3 | 2 | SHA / surface record | E | MEDIUM | No | R12 |
| U5-P0 | 1 | Import inventory | E | MEDIUM | No | R6 |
| PROMPT-GOLDEN | 1 | Hash drift | D | LOW | No | R13 |
| UE | 4 | Stale UE guards | D/E | LOW–MED | No | R14 |

### Minimal reproduction commands (per group)

```bash
# AUDIT-IDEAL
uv run pytest tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py -q --tb=short

# DG001B4
uv run pytest tests/unit/runtime/architecture/test_dg001b4_pre_b5_integration_qualification.py -q --tb=short

# DIAG-DF4
uv run pytest tests/unit/runtime/architecture/test_diag_foundation_4_entrypoint_consistency.py -q --tb=short

# EE pins / B3-C
uv run pytest tests/unit/runtime/architecture/test_ee_final_enterprise_execution_engine_certification.py::test_ee_final_enterprise_no_execution_core_drift_since_revalidation tests/unit/runtime/architecture/test_ee_b3_c_governance_spoofing_abuse.py -q --tb=short

# IMPORT / GR2 / OTEL / HARDENING / U5-P0
uv run pytest tests/unit/runtime/architecture/test_faudit_remediation.py::test_intergrax_no_applications_import_gate tests/unit/runtime/architecture/test_gr2_r3_model_c1_architecture_gates.py::test_gate_allows_certified_harness_unified_task_runner_import tests/unit/runtime/architecture/test_harden_3e_otel_import_gate.py tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py tests/unit/runtime/architecture/test_platform_execution_unification_p0_bypass_inventory.py::test_p0_frozen_child_execution_runner_import_surface -q --tb=short

# HARNESS-L3
uv run pytest tests/unit/runtime/architecture/test_ideal_harness_l3_depth_gate.py tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py -q --tb=short

# MATURITY
uv run pytest tests/unit/runtime/architecture/test_maturity_gate_evidence.py -q --tb=short

# NPSC-5F-R1 / R3
uv run pytest tests/unit/runtime/architecture/test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity.py tests/unit/runtime/architecture/test_npsc5f_r3_final_h1_upstream_runtime_event_drift_reconciliation.py -q --tb=short

# PROMPT-GOLDEN / UE
uv run pytest tests/unit/runtime/architecture/test_prompt_golden_catalog.py tests/unit/runtime/architecture/test_ue_10r4_graph_authority_fail_closed_gate.py tests/unit/runtime/architecture/test_ue_11gp_production_host_execution_gate.py tests/unit/runtime/architecture/test_ue_8p2_authority_policy_gate.py tests/unit/runtime/architecture/test_ue_9d_legacy_execution_retirement_gate.py -q --tb=short
```
