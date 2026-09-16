# INTEGRAx Execution R8 — HARDENING-3 / HARNESS-L3 Architecture Debt Revalidation and Owner-Safe Remediation

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-EXECUTION-R8-HARDENING3-HARNESS-L3-ARCHITECTURE-DEBT-REVALIDATION-AND-OWNER-SAFE-REMEDIATION` |
| Groups | ARCH-F20 (HARDENING-3), ARCH-F21–F22 (HARNESS-L3) |
| Qualification date | 2026-09-16 |
| Branch | `development` |
| Mode | CURRENT-STATE REVALIDATION + QUALIFICATION / HARNESS PATH REMEDIATION |

## Scope

Revalidate ARCH-F20–F22 on baseline `5ba853635c024f60f877bf40556ed34e2987f1ca`. Remediate stale script path references (SSOT: `scripts/ci/script_paths.py`) and latent prompt-golden fixture drift blocking the L3 umbrella gate. **No** contracts→runtime coupling changes, **no** production reconstruction redesign, **no** compatibility wrapper scripts at legacy `scripts/*.py` roots.

## Repository State

| Item | Value |
| --- | --- |
| Branch | `development` |
| Baseline SHA | `5ba853635c024f60f877bf40556ed34e2987f1ca` |
| `origin/development` | `5ba853635c024f60f877bf40556ed34e2987f1ca` |
| Dirty tracked (parallel WIP, not staged) | `applications/**`, `agents/**`, `intergrax/contracts/runtime_event.py`, `intergrax/tools/**`, ME14 marketplace WIP, etc. |
| Untracked | provider invocation, ME14 C1 integration tests, tool acquisition modules |
| Stash | `stash@{0..2}` (not applied) |
| Clean proof | `.tmp/worktrees/r8-baseline` @ baseline SHA + R8 patches |

## Baseline SHA

`5ba853635c024f60f877bf40556ed34e2987f1ca`

## Protected Surface Inventory

| Surface / subsystem | Dirty? | Parallel work? | R8 ownership? | Editable? |
| --- | ---: | ---: | ---: | ---: |
| `intergrax/contracts/**` | yes (runtime_event WIP) | yes | contracts | **PROTECTED** |
| `intergrax/runtime/execution/**` | no | yes | execution | **PROTECTED** |
| `intergrax/runtime/reconstruction/**` | no | yes | observability/diagnostics | **PROTECTED** |
| `intergrax/runtime/observability/**` | no | yes | observability | **PROTECTED** |
| `intergrax/runtime/diagnostics/**` | no | yes | diagnostics | **PROTECTED** |
| `intergrax/runtime/nexus/**` | no | yes | nexus | **PROTECTED** |
| `intergrax/applications/**` | yes | yes | applications | **PROTECTED** |
| `scripts/gates/**` | R8 | no | CI / harness gates | **yes** |
| `scripts/ci/script_paths.py` | no | no | CI SSOT | **read-only** |
| `testing_support/**` | partial | yes | mixed | **PROTECTED** |
| `platform_proofs/**` | no | yes | proofs | **PROTECTED** |
| HARNESS-L3 qualification tests | R8 | no | qualification | **yes** |
| `tests/fixtures/prompt_golden/**` | R8 | no | qualification | **yes** |

## Parallel Workstream Risk Assessment

Parallel WIP on `runtime_event` / events catalog broke pytest sessionstart on the main dirty tree during regression; all mandatory regressions were executed in an isolated worktree at baseline SHA with only R8 files applied. R8 changes do not touch parallel surfaces.

## Historical ARCH-F20–F22

| ID | Historical symptom | Classification (diagnostics doc) |
| --- | --- | --- |
| F20 | `execution_reconstruction.py` → runtime reconstruction import | HARDENING-3 allowlist drift (historical) |
| F21 | `scripts/check_ideal_harness_l3_gates.py` missing | HARNESS-L3 path drift (D) |
| F22 | `scripts/check_registry_snapshot_diff.py` missing | HARNESS-L3 path drift (D) |

## Exact Historical Node Mapping

| ID | Exact test |
| --- | --- |
| F20 | `tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py::test_hardening_3_contracts_do_not_import_runtime_except_allowlist` |
| F21 | `tests/unit/runtime/architecture/test_ideal_harness_l3_depth_gate.py::test_ideal_l3_umbrella_gate_script` |
| F22 | `tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py::test_ideal_w2_w2_script_gates` |

## Current Revalidation

| ID | Run 1 | Run 2 | First failure (pre-fix) | Owner candidate |
| --- | --- | --- | --- | --- |
| F20 | PASS | PASS | — (historical not reproduced) | contracts / qualification gate |
| F21 | FAIL (ENOENT umbrella) → PASS post-fix | PASS | missing `scripts/check_ideal_harness_l3_gates.py` | harness L3 / CI gates |
| F22 | FAIL (ENOENT registry) → PASS post-fix | PASS | missing `scripts/check_registry_snapshot_diff.py` | harness L3 / maintenance scripts |

Evidence logs: `.tmp/session/R8-hardening-harness/`

## Failure Inventory

| ID | Pre-fix | Post-fix |
| --- | --- | --- |
| F20 | N/A (PASS on baseline) | PASS |
| F21 | Stale path + umbrella internal stale paths + latent prompt golden hash | PASS |
| F22 | Stale `scripts/` root paths | PASS |

## F20 Actual Dependency Chain

Historical `intergrax/contracts/execution_reconstruction.py` no longer imports `intergrax.runtime.*`. Reconstruction port is contract-only (`ExecutionReconstructionReader` Protocol + models under `execution_reconstruction_models.py`).

## F20 Contract Boundary Analysis

AST scan across `intergrax/contracts/**` reports **zero** non-allowlisted `intergrax.runtime` imports on baseline. No allowlist expansion performed.

### F20 dependency table

| Source | Symbol | Imported from | Layer direction | Verdict |
| --- | --- | --- | --- | --- |
| — | — | — | — | No violation on HEAD |

## F21 Script Path Analysis

| Historical path | Current canonical path | CI SSOT |
| --- | --- | --- |
| `scripts/check_ideal_harness_l3_gates.py` | `scripts/gates/check_ideal_harness_l3_gates.py` | `SCRIPT_PATHS["check_ideal_harness_l3_gates.py"]` |

Umbrella script invoked child gates via `REPO_ROOT / "scripts" / basename` — updated to `resolve_script(basename)`.

## F22 Script Path Analysis

| Basename | Canonical path |
| --- | --- |
| `check_registry_snapshot_diff.py` | `scripts/maintenance/check_registry_snapshot_diff.py` |
| Other W2 scripts | `scripts/maintenance/*` per `script_paths.py` |

Qualification tests now call `resolve_script()` instead of `scripts/<basename>`.

## CI / Gate Path Mapping

| Historical path | Current canonical path | Referenced by CI? | Action |
| --- | --- | ---: | --- |
| `scripts/check_ideal_harness_l3_gates.py` | `scripts/gates/check_ideal_harness_l3_gates.py` | yes (SSOT) | update tests + umbrella |
| `scripts/check_registry_snapshot_diff.py` | `scripts/maintenance/check_registry_snapshot_diff.py` | yes (SSOT) | update tests |

## Canonical Owner Mapping

| ID | Mechanism | Canonical owner | R8 editable? |
| --- | --- | --- | ---: |
| F20 | contracts/runtime import gate | execution qualification | tests only (none required) |
| F21 | L3 umbrella gate | harness / CI gates | yes (tests + `scripts/gates/`) |
| F22 | W2 script matrix gate | harness qualification | yes (tests) |

## Contract Mapping

| Mechanism | Contract | Canonical owner | Implementation | Replaceable? |
| --- | --- | --- | --- | ---: |
| Execution reconstruction read port | `ExecutionReconstructionReader` | contracts | runtime/diagnostics providers | yes |

## Layer Ownership Mapping

No layer ownership changes. Remediation confined to qualification tests, gate runner script, and prompt golden fixture.

## Root Cause Classification

| ID | Root cause |
| --- | --- |
| F20 | **C** — historical failure not reproducible on HEAD |
| F21 | **D** — stale script path + stale umbrella runner paths; **D** — stale `tools_agent_system` golden hash (latent, surfaced after path fix) |
| F22 | **D** — stale script path in qualification test |

## Dirty vs Clean Assessment

F20–F22 targeted nodes validated on baseline worktree with R8 patches. Main tree dirty WIP unrelated to R8 staged files.

## Editable vs Protected Surfaces

Only R8-owned qualification/gate files edited; all protected production surfaces untouched.

## Selected Remediation

1. `resolve_script()` in HARNESS-L3 depth tests (F21/F22).
2. `resolve_script()` in `check_ideal_harness_l3_gates.py` child invocations.
3. Sync `tests/fixtures/prompt_golden/expectations.json` SHA for `tools_agent_system v1` to match committed `prompts/tools_agent_system/1.yaml`.

## Deferred Cross-Layer Items

- `scripts/gates/check_audit_ideal_gates.py` still uses legacy `scripts/` root paths (out of F21 exact node scope; follow-up hygiene).
- ARCH-F33 prompt-golden gate ownership remains with PROMPT-GOLDEN track; R8 only synced fixture required for umbrella PASS.

## Contracts / Ports

No contract API changes.

## Pluginability Assessment

Unchanged — reconstruction remains port-based.

## Layer Boundary Assessment

No new `contracts → runtime` dependencies. HARDENING-3 gate PASS without allowlist edits.

## Compatibility Surface Assessment

No legacy wrapper scripts added at `scripts/*.py` roots.

## Harness SSOT Assessment

`scripts/ci/script_paths.py` is the canonical basename map; qualification aligned to `resolve_script()`.

## Frozen Invariant Assessment

Contract-first and evidence≠control preserved.

## Execution Engine Impact

**None** — qualification and gate runner path alignment only.

## Architecture Reopen Assessment

**Not required.**

## Targeted Results

F20/F21/F22 exact nodes: **PASS ×2** (worktree + post-import-order fix on main).

## HARDENING-3 Regression

`test_hardening_3_layer_boundary_gate.py` (full module): **22 passed** (worktree log).

## HARNESS-L3 Regression

`test_ideal_harness_l3_depth_gate.py`, `test_ideal_harness_l3_w2_depth_gate.py`: included in above (**22 passed**).

## R1 Regression

F01–F04, F06 nodes in `test_audit_ideal_depth_gate.py`: **PASS** (worktree `regression-worktree-main.log`).

## R3 Regression

F13–F14 in `test_diag_foundation_4_entrypoint_consistency.py`: **PASS** (same log).

## R4 Regression

F15–F16: **PASS** (same log).

## R6 Regression

F18: **PASS** (same log).

## R7 Regression

F19: **PASS** (same log).

## R14 Regression

F34–F37: **PASS** (same log).

## R9 Regression

`test_mandatory_frozen_suite_passes[NPSC-5E Final]`: **PASS** (worktree `regression-npsc-r11.log`).

## R10 Regression

`test_mandatory_frozen_suite_passes[NPSC-5D Final]`: **PASS** (same log).

## R11 Regression

`Runtime events suites`, `Runtime observability suites`, `DG_001` mandatory rows: **PASS** (same log).

## EE Architecture Gates

`-k test_ee_final_arch`: **29 passed** (`regression-ee-final-arch.log`).

## U5

`test_platform_execution_unification_u5_final_zero_bypass.py`: **PASS** (main regression log).

## UE-10R4.1

`test_ue_10r41_execution_import_hygiene_gate.py`: **PASS** (same log).

## F-01

`test_ee_b2_worker_fault_injection.py`: **PASS** (same log).

## OBS-DIAG

OBS-DIAG architecture / qualification / port gates: **PASS** (same log).

## R5 Boundary

`test_intergrax_no_applications_import_gate`: **PASS** (same log).

## Qualification Regression

`tests/unit/testing_support/execution_qualification/`: **PASS** (7 skips live perf env only).

## Contract Boundary Regression

HARDENING-3 module + `test_prompt_golden_catalog_matches_expectations`: **PASS**.

## Runtime Unit Regression

**N/A** — no production runtime edits.

## Static Quality

| Check | Result |
| --- | --- |
| `ruff check` `scripts/gates/check_ideal_harness_l3_gates.py` | PASS |
| `ruff format` R8 test files | PASS |
| `git diff --check` R8 files | PASS |

## Cold Imports

Gate script only; no new production module surface.

## Changed Files

| File | Layer | Reason |
| --- | --- | --- |
| `tests/unit/runtime/architecture/test_ideal_harness_l3_depth_gate.py` | qualification | F21 SSOT paths |
| `tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py` | qualification | F22 SSOT paths |
| `scripts/gates/check_ideal_harness_l3_gates.py` | CI gate | umbrella child path SSOT |
| `tests/fixtures/prompt_golden/expectations.json` | qualification fixture | latent golden hash on HEAD |

## Untouched Protected Files

All `intergrax/contracts/**` (production), `intergrax/runtime/**` production modules, `applications/**`, `agents/**` — **not modified by R8 commit**.

## Remaining Debt

- `check_audit_ideal_gates.py` legacy path literals (AUDIT-IDEAL umbrella; separate hygiene).
- Full architecture suite re-run deferred per R8 scope.

## Decision

Remediate HARNESS-L3 path drift via SSOT; sync prompt golden fixture required for umbrella gate; F20 closed as historical non-repro.

## Commit SHA

`8d718e19109f88cf0c5b91a13e76cbbdaf08959c`

## Final Verdict

**R8 HARDENING-3 / HARNESS-L3 = PASS — QUALIFICATION AND HARNESS DEBT REMEDIATED, PLATFORM BOUNDARIES PRESERVED**

### Required F20–F22 table

| ID | Exact test | Current result | Root cause | Canonical owner | R8 action |
| --- | --- | --- | --- | --- | --- |
| F20 | `test_hardening_3_contracts_do_not_import_runtime_except_allowlist` | PASS | Historical C | qualification | none |
| F21 | `test_ideal_l3_umbrella_gate_script` | PASS | D path + D golden | harness L3 | SSOT + fixture |
| F22 | `test_ideal_w2_w2_script_gates` | PASS | D path | harness L3 | SSOT in test |

### Required script mapping table

| ID | Historical path | Current path | Canonical? | Action |
| --- | --- | --- | ---: | --- |
| F21 | `scripts/check_ideal_harness_l3_gates.py` | `scripts/gates/check_ideal_harness_l3_gates.py` | yes | reference SSOT |
| F22 | `scripts/check_registry_snapshot_diff.py` | `scripts/maintenance/check_registry_snapshot_diff.py` | yes | reference SSOT |

### Required boundary table

| Dependency | Allowed? | Existing/new | Verdict |
| --- | ---: | --- | --- |
| contracts → runtime (F20) | no | none found | PASS |

### Required protected table

| Surface | Protected? | Touched by R8? |
| --- | ---: | ---: |
| contracts production | yes | no |
| runtime production | yes | no |
| scripts/gates (umbrella) | no | yes |

### Required changed-files table

| File | Layer | Reason | Contract changed? | Boundary-safe? |
| --- | --- | --- | ---: | ---: |
| (see Changed Files) | qualification / gates | path SSOT | no | yes |
