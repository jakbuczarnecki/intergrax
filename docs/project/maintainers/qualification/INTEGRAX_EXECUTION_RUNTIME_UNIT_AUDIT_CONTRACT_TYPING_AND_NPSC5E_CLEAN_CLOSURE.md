# INTEGRAx Execution Runtime Unit Audit — Contract Typing and NPSC-5E Clean Closure

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-EXECUTION-RUNTIME-UNIT-AUDIT-CONTRACT-TYPING-AND-NPSC5E-CLEAN-CLOSURE` |
| Branch | `development` |
| Closure date | 2026-09-16 |

## Scope

Contract-first typing for LLM adapter injection at Tier-3 harness composition boundaries; clean-tree reconfirmation of NPSC-5E / ARCH-F27 and resume-lineage leaf tests. No Execution Engine semantics change, no lineage remediation.

## Repository State

| Item | Value |
| --- | --- |
| Branch | `development` |
| Baseline HEAD (task start) | `c5908ab9d06b63086e432ea16f4b64da97858928` |
| `origin/development` | `41785532ba8775e733746b87303a3ef45306f610` |
| Dirty (unrelated WIP, not staged) | `identity_authority.py`, nexus/task/long_running WIP, OBS spine integration tests, `platform_proofs/`, untracked OBS harness helpers |
| Stash | `stash@{0}` temp · `stash@{1}` rebase3 · `stash@{2}` mem-ent-1r3-rebase-wip |

## Baseline SHA

`c5908ab9d06b63086e432ea16f4b64da97858928`

## Previous Runtime Unit Audit Relationship

Follows `INTEGRAx-EXECUTION-RUNTIME-UNIT-FAILURE-DIAGNOSTICS-AND-BLOCKER-CLASSIFICATION`, which closed functional remediation at **1318 passed / 0 failed** on committed code and introduced `llm_adapter=` injection with loose `Any` / `object` annotations. This task hardens those boundaries and formally closes the outstanding NPSC-5E dirty-tree ambiguity.

## LLMAdapter Contract Discovery

| Attribute | Value |
| --- | --- |
| Canonical module | `intergrax.llm_adapters.contracts.llm_adapter` |
| Canonical symbol | `LLMAdapter` (ABC, not ad-hoc Protocol) |
| Representative consumers | `llm_resolver.py`, `nexus_factory.py`, `rag_runtime_bridge.py`, `orchestration_wiring.py`, `testing_support.builder.FakeLLMAdapter` |

## Contract Ownership

`LLMAdapter` lives in Tier-0/1 neutral `intergrax/llm_adapters/contracts/` — not in `applications/`, `testing_support/`, or vendor adapters. Tier-3 wiring imports the contract only (same pattern as existing application LLM wiring modules).

## Before Typing

| Surface | Annotation |
| --- | --- |
| `build_harness_host_runtime(..., llm_adapter=...)` | `Any \| None` |
| `wire_application_environment(..., llm_adapter=...)` | `object \| None` |

## After Typing

| Surface | Annotation |
| --- | --- |
| `build_harness_host_runtime(..., llm_adapter=...)` | `LLMAdapter \| None` |
| `wire_application_environment(..., llm_adapter=...)` | `LLMAdapter \| None` |

## Required typing table

| Surface | Before | After | Canonical contract |
| --- | --- | --- | --- |
| `build_harness_host_runtime` | `Any \| None` | `LLMAdapter \| None` | `intergrax.llm_adapters.contracts.llm_adapter.LLMAdapter` |
| `wire_application_environment` | `object \| None` | `LLMAdapter \| None` | `intergrax.llm_adapters.contracts.llm_adapter.LLMAdapter` |

## Injection Precedence

Unchanged: explicit injected adapter wins; when `None`, `resolve_optional_environment_llm_adapter(env)` runs for tenant RAG/memory wiring (same control flow as post-audit remediation).

## Pluginability Assessment

Harness composition root → `LLMAdapter` contract → `FakeLLMAdapter` (tests) or profile resolver (production). No vendor types on signatures.

## Layer Boundary Assessment

No `runtime → applications`, no `contracts → runtime implementation`, no vendor adapter imports on injection boundaries.

## Static Quality

| Check | Result |
| --- | --- |
| `ruff check` (changed files) | PASS |
| `ruff format --check` (after format) | PASS |
| `git diff --check` (scoped files) | PASS (unrelated WIP elsewhere may warn) |
| `pyright` (`harness_host_runtime.py`, `environment_wiring.py`) | **Pre-existing** `integration_profile` argument error at `wire_integration_tool_context` (not introduced by LLM typing) |

## Cold Imports

| Module | Result |
| --- | --- |
| `intergrax.applications._shared.harness_host_runtime` | PASS |
| `intergrax.applications._shared.environment_wiring` | PASS |

## Runtime Execution Suite

| Context | Result |
| --- | --- |
| Dirty working tree | **6 failed**, 1312 passed, 1 skipped (resume / fresh-root expectations; see dirty WIP) |
| Clean committed tree (worktree @ baseline SHA) | **1318 passed**, 1 skipped |

## Dirty Tree State

Local modifications to `identity_authority.py` (and related runtime paths) cause resume to reuse checkpoint `root_execution_id` instead of minting a fresh segment root — consistent with **E — order/state pollution / WIP contamination**, not harness typing.

## Clean Validation Method

```text
git worktree add .tmp/session/npsc5e-clean-closure/worktree c5908ab9d06b63086e432ea16f4b64da97858928
```

Detached worktree at committed SHA; no `reset --hard` / no WIP discard.

## Clean Validation SHA

`c5908ab9d06b63086e432ea16f4b64da97858928` (pre-typing commit; NPSC-5E proof independent of typing diff). Post-commit SHA recorded in **Commit SHA** after landing.

## NPSC-5E Direct Result

`tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py` on clean worktree: **PASS** (24 tests, ~199s).

## ARCH-F27 Result

Nested mandatory wrapper `test_mandatory_frozen_suite_passes[NPSC-5E Final]` (same module): **PASS** on clean tree.

## Resume Lineage Leaf Result

`tests/unit/runtime/execution/lineage/test_host_task_resume_lineage_identity.py`: **PASS** (2/2) clean tree; **FAIL** (2/2) dirty tree.

## Dirty vs Clean Comparison

| Target | Working tree result | Clean tree result | Final classification |
| --- | --- | --- | --- |
| `tests/unit/runtime/execution/` | 6 FAIL | 1318 PASS | WIP contamination (resume identity) |
| `test_host_task_resume_lineage_identity.py` | 2 FAIL | 2 PASS | WIP contamination |
| `test_npsc5e_final_recovery_plane_qualification_and_freeze.py` | Not re-run dirty (nested cost) | PASS | Committed baseline healthy |
| `test_host_task_revision_reentry.py` (typing + harness) | 7 PASS | — | Typing compatible with `FakeLLMAdapter` |

## Root Cause of Previous NPSC-5E Failure

Previous nested NPSC-5E / lineage failures on a **dirty** tree align with uncommitted `identity_authority` / runtime WIP altering resume root minting. **Clean committed HEAD reproduces PASS** — not a committed lineage defect.

## Architecture Reopen Assessment

**Not required** for this closure. No change to frozen identity authority, resume semantics, or public execution contracts in committed scope.

## R14 Regression

F34–F37 targeted gates (UE-10R4, UE-11GP, UE-8P2, UE-9D): **PASS** (bundled pytest).

## R11 Regression

Runtime events / observability not re-run as full directories; OBS-DIAG conformance gate **PASS**. DG_001 lineage directory covered by clean execution suite + lineage leaf.

## R10 Regression

`test_npsc5d_final_multi_agent_governance_qualification.py` (NPSC-5D Final): **PASS**.

## EE Architecture Gates

All `test_ee_final_arch_*.py` modules in regression bundle: **PASS**.

## U5

`test_platform_execution_unification_u5_final_zero_bypass.py`: **PASS**.

## UE-10R4.1

`test_ue_10r41_execution_import_hygiene_gate.py`: **PASS**.

## F-01

`test_ee_b2_final_fault_matrix.py` (F-01..F-12 registry): **PASS**.

## OBS-DIAG

`test_obs_diag_conformance_qualification.py`: **PASS**.

## R5 Boundary

`test_intergrax_no_applications_import_gate` (`test_faudit_remediation.py`): **PASS**.

## Qualification Regression

`tests/unit/testing_support/execution_qualification/`: **PASS** (7 skipped live-perf optional).

## Required contract table

| Mechanism | Platform contract | Injected implementation | External replacement possible? |
| --- | --- | --- | --- |
| Host harness LLM | `LLMAdapter` | `FakeLLMAdapter` / env resolver | Yes |
| Environment tenant RAG/memory LLM | `LLMAdapter` | explicit param or `resolve_optional_environment_llm_adapter` | Yes |

## Changed Files

- `intergrax/applications/_shared/harness_host_runtime.py`
- `intergrax/applications/_shared/environment_wiring.py`
- `docs/project/maintainers/qualification/INTEGRAX_EXECUTION_RUNTIME_UNIT_AUDIT_CONTRACT_TYPING_AND_NPSC5E_CLEAN_CLOSURE.md`

## Remaining Debt

- Pre-existing `pyright` error on `wire_integration_tool_context(integration_profile=...)` in `environment_wiring.py` (orthogonal to LLM injection).
- Uncommitted runtime WIP must not be used for qualification signals until reverted or landed separately.

## Decision

Land contract typing + qualification artifact; treat NPSC-5E regression as **closed on committed tree**.

## Commit SHA

`069782642` (`INTEGRAx-EXECUTION-RUNTIME-UNIT-AUDIT-CONTRACT-TYPING-AND-NPSC5E-CLEAN-CLOSURE`)

## Final Verdict

`RUNTIME EXECUTION UNIT AUDIT CLOSURE = PASS — CONTRACT TYPING HARDENED AND NPSC-5E CLEAN-TREE REGRESSION CONFIRMED`
