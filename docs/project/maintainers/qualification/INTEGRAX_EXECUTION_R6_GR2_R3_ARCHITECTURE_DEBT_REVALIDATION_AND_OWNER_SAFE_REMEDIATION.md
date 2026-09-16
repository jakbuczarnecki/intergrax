# INTEGRAx Execution R6 — GR2-R3 Architecture Debt Revalidation and Owner-Safe Remediation

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-EXECUTION-R6-GR2-R3-ARCHITECTURE-DEBT-REVALIDATION-AND-OWNER-SAFE-REMEDIATION` |
| Group | GR2-R3 (ARCH-F18) |
| Qualification date | 2026-09-16 |
| Branch | `development` |
| Mode | CURRENT-STATE REVALIDATION + QUALIFICATION HARNESS REMEDIATION |

## Scope

Revalidate ARCH-F18 (`test_gate_allows_certified_harness_unified_task_runner_import`) on committed HEAD. Remediate only stale GR-2-R3 MODEL C1 gate expectations and tighten the closed-world `UnifiedTaskRunner` import allowlist. **No** edits to governance production policy, `intergrax/runtime/execution/**`, `intergrax/runtime/governance/**` evaluators, or frozen execution chain.

## Repository State

| Item | Value |
| --- | --- |
| Branch | `development` |
| Baseline SHA | `e9a07ed0e21d186a4b625905b9f27cb61fe8da9a` |
| `origin/development` | `e9a07ed0e21d186a4b625905b9f27cb61fe8da9a` |
| Dirty tracked (main tree, unrelated) | `applications/local_workspace_application/tests/*`, `intergrax/memory/contracts/provider_qualification.py`, others |
| Untracked (unrelated) | `tests/unit/memory/*` durable provider qualification |
| Stash | `stash@{0..2}` (not applied) |

F18 revalidation executed on main tree; WIP does not touch GR2-R3 scan targets or changed qualification files.

## Baseline SHA

`e9a07ed0e21d186a4b625905b9f27cb61fe8da9a` — platform HEAD at R6 start.

## Protected Surface Inventory

| Surface / subsystem | Dirty? | Parallel work? | R6 ownership? | Editable? |
| --- | ---: | ---: | ---: | ---: |
| `intergrax/runtime/governance/**` | no | yes | no | **PROTECTED** |
| `intergrax/runtime/execution/**` | no | yes | no | **PROTECTED** |
| `intergrax/runtime/long_running/**` | no | yes | no | **PROTECTED** (read-only; wiring no longer imports UTR) |
| `intergrax/runtime/nexus/**` | no | yes | no | **PROTECTED** |
| `intergrax/runtime/task/**` | no | yes | no | **PROTECTED** |
| `intergrax/applications/**` | partial (LKW tests) | yes | no | **PROTECTED** (production scan target only) |
| `intergrax/memory/**` | **yes** | yes | no | **PROTECTED** |
| `testing_support/**` | no | yes | no | **PROTECTED** |
| GR2-R3 architecture gates | R6 | no | qualification | **yes** |

## Parallel Workstream Risk Assessment

Memory / LKW application test WIP is unrelated to F18. R6 touches only `tests/unit/runtime/architecture/gr2_r3_model_c1_*` and this qualification artifact.

## Historical ARCH-F18

Documented in `INTEGRAX_EXECUTION_FULL_ARCHITECTURE_SUITE_FAILURE_DIAGNOSTICS_AND_BLOCKER_CLASSIFICATION.md`: production AST scan reported certified `UnifiedTaskRunner` import in `intergrax/runtime/long_running/wiring.py` outside an updated allowlist (classification **E**).

## Exact Historical Node Mapping

| ID | Group | Exact test node | Historical symptom |
| --- | --- | --- | --- |
| ARCH-F18 | GR2-R3 | `tests/unit/runtime/architecture/test_gr2_r3_model_c1_architecture_gates.py::test_gate_allows_certified_harness_unified_task_runner_import` | Allowlist / inventory drift: `UnifiedTaskRunner` import in `long_running/wiring.py` |

## Current Revalidation

```bash
uv run pytest tests/unit/runtime/architecture/test_gr2_r3_model_c1_architecture_gates.py::test_gate_allows_certified_harness_unified_task_runner_import -q
```

| Test | Run 1 (pre-fix) | Run 2 (pre-fix) | Run 1 (post-fix) | Run 2 (post-fix) |
| --- | --- | --- | --- | --- |
| F18 | FAIL | FAIL | PASS | PASS |

Pre-fix failure: `raw == []` — `wiring.py` no longer contains `UnifiedTaskRunner` import (line 24 is `TaskEnricher`).

## Actual Call Chain

```text
pytest (F18)
  → read committed `intergrax/.../task_control.py` (post-fix pin)
  → ast.parse
  → collect_forbidden_unified_task_runner_imports (gr2_r3_model_c1_ast)
  → assert single FORBIDDEN_LEGACY_UNIFIED_TASK_RUNNER_IMPORT at import site
  → _scan_production(..., LEGACY_UNIFIED_TASK_RUNNER_IMPORT_ALLOWLIST)
  → skip allowlisted modules; assert zero violations elsewhere
```

No governance evaluator, admission request, or `execute_root_task` invocation on this path.

## Failure Inventory

| ID | Exact test | Current result (pre-fix) | Root cause | Canonical owner | R6 action |
| --- | --- | --- | --- | --- | --- |
| ARCH-F18 | `test_gate_allows_certified_harness_unified_task_runner_import` | FAIL | Stale pin on `wiring.py` after HostTaskExecutionPort migration | GR-2-R3 qualification harness | Repin exemplar module; prune dead allowlist entries |

## Governance Request Audit

Not applicable — F18 is static AST import inventory, not runtime authorization.

| Field | Expected by contract? | Present? | Produced by |
| --- | ---: | ---: | --- |
| tenant_id | N/A | N/A | N/A |
| policy bundle | N/A | N/A | N/A |
| execution identity | N/A | N/A | N/A |

## Canonical Owner Mapping

| Responsibility | Owner before | Owner now | Changed? | Valid? |
| --- | --- | --- | ---: | ---: |
| Root execution admission | Governance + orchestration harness | unchanged | no | yes |
| UTR harness scheduling | `unified_task_runner.py` + allowlisted bridges | unchanged | no | yes |
| Long-running scheduler wiring | `long_running/wiring.py` (HostTaskExecutionPort) | unchanged | no | yes |
| GR2-R3 closed-world gates | `gr2_r3_model_c1_gate_policy.py` | qualification | no | yes |

## Contract Mapping

| Mechanism | Contract | Canonical owner | Default implementation | Replaceable? |
| --- | --- | --- | --- | ---: |
| UTR import ban (production scan) | `FORBIDDEN_LEGACY_UNIFIED_TASK_RUNNER_IMPORT` | GR-2-R3 AST gates | `collect_forbidden_unified_task_runner_imports` | yes (policy file) |
| Allowlisted harness imports | `LEGACY_UNIFIED_TASK_RUNNER_IMPORT_ALLOWLIST` | qualification policy | frozen set in gate_policy | yes (explicit edits only) |
| Runtime governance DENY/ALLOW | Decision → Governance → DEA | governance | fail-closed evaluators | no (frozen) |

## Layer Ownership Mapping

| Layer | F18 involvement |
| --- | --- |
| `tests/unit/runtime/architecture/` | Gate tests + policy (R6 edits) |
| `intergrax/runtime/long_running/wiring.py` | Read-only evidence: no UTR import |
| `intergrax/applications/_shared/task_control.py` | Certified harness exemplar (still imports UTR) |

## GR-2 Current Architecture

MODEL C1 AST gates enforce closed-world root construction, `execute_root_task` import surfaces, and `UnifiedTaskRunner` import surfaces with explicit allowlists (`gr2_r3_model_c1_gate_policy.py`). Root launcher tests live in `tests/unit/runtime/governance/test_gr2_r3_root_execution_launcher.py`.

## GR-3 Current Architecture

Root admission remains governance-gated before execution; qualification gates document certified internal harness bridges without weakening production scan.

## F18 Analysis

Historical failure assumed `wiring.py` still imported `UnifiedTaskRunner`. Committed tree shows migration to `HostTaskExecutionPort` / terminal services only. The gate test’s middle assertion (raw collector finds import in pinned file) became stale; production-wide scan (`test_production_has_no_unauthorized_unified_task_runner_imports`) already **PASS** pre-fix.

## Root Cause Classification

**Primary: C — STALE GR2-R3 TEST EXPECTATION**  
**Secondary: E — FROZEN / REFERENCE DRIFT** (dead allowlist paths for modules that no longer import UTR)

Not A (governance defect), B (execution engine defect), or F (cross-layer owner blocker).

## Dirty vs Clean Assessment

| Target | Dirty main tree | Clean committed semantics | Classification |
| --- | --- | --- | --- |
| F18 | FAIL pre-fix | Same (wiring.py on HEAD has no UTR import) | Qualification-only; unrelated WIP does not affect AST targets |

## Editable vs Protected Surfaces

| Surface | Protected? | Touched by R6? |
| --- | ---: | ---: |
| `intergrax/runtime/governance/**` | yes | no |
| `intergrax/runtime/execution/**` | yes | no |
| `intergrax/runtime/long_running/**` | yes | no |
| GR2-R3 gate policy/tests | no | **yes** |

## Selected Remediation

1. Pin F18 exemplar to `intergrax/applications/_shared/task_control.py` (line 51 import).
2. Remove allowlist entries for modules with no `UnifiedTaskRunner` import: `long_running/wiring.py`, `long_running/scheduler.py`, `harness_task_routes.py`, `task_control_wiring.py`.

## Deferred Cross-Layer Items

None for F18. Applications harness modules that still import UTR remain allowlisted; narrowing those is out of R6 scope (applications owner).

## Contracts / Ports

No new ports. Reused existing GR-2-R3 AST collectors and allowlist contract.

## Pluginability Assessment

Allowlist remains explicit closed-world set; no hardcoded evaluator wiring added.

## Governance Fail-Closed Assessment

**Unchanged.** No implicit ALLOW introduced.

## Authorization Boundary Assessment

**N/A** for F18 (static analysis only).

## Layer Boundary Assessment

No new dependencies; qualification tests only.

## Frozen Invariant Assessment

Decision → Governance → DecisionExecutionAuthorization → ExecutionRequest → ExecutionRuntime chain untouched.

## Execution Engine Impact

None (qualification-only).

## Architecture Reopen Assessment

**Not required.**

## Targeted Results

F18: **PASS ×2** post-fix (`.tmp/session/r6-gr2-r3/f18-run*.log`, post-fix runs).  
Full `test_gr2_r3_model_c1_architecture_gates.py`: **19 passed**.

## R1 Regression

F01–F04, F06 named tests in `test_audit_ideal_depth_gate.py`: **PASS** (batch1).

## R3 Regression

F13–F14 (`test_diag_foundation_4_entrypoint_consistency.py`): **PASS** (batch1).

## R4 Regression

F15–F16: **PASS** (batch1).

## R14 Regression

F34–F37 UE gates: **PASS** (batch1).

## R9 Regression

`test_npsc5e_final_recovery_plane_qualification_and_freeze.py`: **PASS** isolated (22 passed, `.tmp/session/r6-gr2-r3/r9-isolated.log`). Nested `test_mandatory_frozen_suite_passes[...]` failed once when co-scheduled with large batch (resource/order); isolated file run green — unrelated to R6 diff.

## R10 Regression

`test_npsc5d_final_multi_agent_governance_qualification.py`: **PASS** (batch2).

## R11 Regression

`tests/unit/runtime/events/` + `tests/unit/runtime/observability/`: **826 passed** (`.tmp/session/r6-gr2-r3/regression-r11.log`).

## Governance Regression Bundle

| Node | Result |
| --- | --- |
| `test_gr2_r3_model_c1_architecture_gates.py` (all) | PASS |
| `test_gr2_r3_root_execution_launcher.py` | PASS (with F18 double-run batch) |
| `test_production_has_no_unauthorized_unified_task_runner_imports` | PASS (pre-fix) |

## EE Architecture Gates

`test_ee_final_arch_*.py` (10 modules): **PASS** (batch2).

## U5

`test_platform_execution_unification_u5_final_zero_bypass.py`: **PASS** (batch2).

## UE-10R4.1

`test_ue_10r41_execution_import_hygiene_gate.py`: **PASS** (batch2).

## F-01

`test_ee_b2_final_fault_matrix.py`: **PASS** (batch2).

## OBS-DIAG

`test_obs_diag_conformance_architecture.py`, `test_obs_diag_port_1_gates.py`, `test_obs_diag_conformance_qualification.py`: **PASS** (batch2).

## R5 Boundary

`test_intergrax_no_applications_import_gate`: **PASS** (batch2).

## Qualification Regression

`tests/unit/testing_support/execution_qualification/`: **PASS** (7 skips live perf env only, batch2).

## Runtime Unit Regression

**N/A** — no production runtime edits.

## Static Quality

| Check | Result |
| --- | --- |
| `ruff check` (changed files) | PASS |
| `ruff format` (changed files) | PASS |
| `git diff --check` | PASS (unrelated CRLF warnings on other dirty files) |

## Cold Imports

**N/A** — no production module changes.

## Changed Files

| File | Layer | Reason | Contract changed? | Boundary-safe? |
| --- | --- | --- | ---: | ---: |
| `tests/unit/runtime/architecture/test_gr2_r3_model_c1_architecture_gates.py` | qualification | Repin F18 exemplar | no | yes |
| `tests/unit/runtime/architecture/gr2_r3_model_c1_gate_policy.py` | qualification | Remove stale allowlist paths | policy data only | yes |
| `docs/project/maintainers/qualification/INTEGRAX_EXECUTION_R6_GR2_R3_ARCHITECTURE_DEBT_REVALIDATION_AND_OWNER_SAFE_REMEDIATION.md` | docs | R6 artifact | no | yes |

## Untouched Protected Files

All `intergrax/runtime/governance/**`, `intergrax/runtime/execution/**`, and parallel WIP paths listed in Repository State.

## Remaining Debt

ARCH-F19+ (OTEL, HARDENING-3, etc.) unchanged. F18 closed for R6.

## Decision

Close F18 via qualification harness alignment; preserve fail-closed governance and execution freeze.

## Commit SHA

`1d05c6bd6b71bfd22e7ed8d931788e28da79f1a1`

## Final Verdict

```text
R6 GR2-R3 = PASS — QUALIFICATION DEBT REMEDIATED, GOVERNANCE SEMANTICS PRESERVED
```

## Required F18 table

| ID | Exact test | Current result | Root cause | Canonical owner | R6 action |
| --- | --- | --- | --- | --- | --- |
| ARCH-F18 | `test_gate_allows_certified_harness_unified_task_runner_import` | PASS | C/E stale pin + allowlist | GR-2-R3 qualification | Repin + allowlist prune |

## Required governance request table

See Governance Request Audit (N/A).

## Required contract table

See Contract Mapping.

## Required ownership table

See Canonical Owner Mapping.

## Required boundary table

| Dependency | Allowed? | Existing/new | Verdict |
| --- | ---: | --- | --- |
| qualification → `intergrax/**` (read AST) | yes | existing | OK |
| qualification → runtime governance impl | no | — | not introduced |

## Required protected table

See Editable vs Protected Surfaces.

## Required changed-files table

See Changed Files.
