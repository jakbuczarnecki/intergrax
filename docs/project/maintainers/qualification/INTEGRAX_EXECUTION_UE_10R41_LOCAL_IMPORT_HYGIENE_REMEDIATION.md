# INTEGRAX-EXECUTION-UE-10R41-LOCAL-IMPORT-HYGIENE-DIAGNOSTICS-AND-REMEDIATION

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-EXECUTION-UE-10R41-LOCAL-IMPORT-HYGIENE-DIAGNOSTICS-AND-REMEDIATION` |
| Domain | `runtime/execution` import hygiene (UE-10R4.1) |
| Change class | Class A — internal hygiene |

## Scope

Remove all function-scoped imports under `intergrax/runtime/execution/**/*.py` while preserving frozen Execution Engine semantics, pluginability, and layer boundaries.

## Repository State

Remediation applied on branch `development` (operator default).

## Baseline SHA

`1a2e1c1f3165e07517ee32122ed394c74d45dccf`

## Gate Failure

`tests/unit/runtime/architecture/test_ue_10r41_execution_import_hygiene_gate.py` reported **8** `from`-import violations (0 plain `import` violations).

## Full Local Import Inventory

| File | Line | Function | Import | Reason likely | Classification |
| --- | ---: | --- | --- | --- | --- |
| `decision_finalization_conformance.py` | 344 | `assert_concurrent_finalization_race` | `from concurrent.futures import ThreadPoolExecutor` | Habitual lazy stdlib import in conformance helper | **A** |
| `decision_finalization_conformance.py` | 384 | `assert_concurrent_idempotent_replay` | `from concurrent.futures import ThreadPoolExecutor` | Same | **A** |
| `delegated_execution/correlation_persistence.py` | 92 | `__init__` | `DelegatedCorrelationQueryCursorCodec` | Defensive local import (sibling module); no cycle on inspection | **A** |
| `failure_evidence/recording_delegate.py` | 42 | `_record_delegate_failure` | `peek_active_execution_evidence_context` | Split from module-level `validate_*` import | **A** |
| `host_task.py` | 130 | `task_result_from_agent_execution` | `terminal_task_result_exposure_no_decision_gate` | Duplicate / style debt | **A** |
| `host_task.py` | 493 | `execute` | `terminal_task_result_exposure_no_decision_gate` | Branch-local duplicate | **A** |
| `retry/service.py` | 124 | `transition_for_retry` | `ExecutionLineageAttemptClosureKind` | Conditional-block lazy import | **A** |
| `retry/service.py` | 125 | `transition_for_retry` | `seal_lineage_attempt` | Same | **A** |

**Note:** `decision_lifecycle_host.py`, `authority/registry.py`, and `budget/registry.py` use module-level `TYPE_CHECKING` imports only — not gate violations.

## Root Cause Classification

All eight violations: **A (accidental local import / style debt)**. None required cycle breaking, optional dependency masking, or ownership realignment.

## Dependency Cycles Found

None introduced or revealed by module-level promotion.

- `correlation_persistence` → `correlation_query_cursor` (one-way; cursor does not import persistence).
- `retry/service` → `lineage/seal` + contracts (one-way).
- `host_task` → `runtime.task.task_result_authoritative_exposure_defaults` (one-way).
- `recording_delegate` → `failure_evidence.active_context` (already imported at module level for `validate_*`).

## Ownership Decisions

No ownership moves. Symbols remain in existing modules; only import placement changed.

## Selected Remediation

Promote each local import to module-level `import` / `from … import …` in the same owning module.

## Before Dependency Graph

No problematic cycles; local imports did not reflect real `A ↔ B` ownership errors.

Example (perceived deferral only):

```text
correlation_persistence.__init__
  → (local) correlation_query_cursor
```

## After Dependency Graph

```text
correlation_persistence (module)
  → correlation_query_cursor
```

(Same directed edge, module-level.)

## Contracts / Ports Impact

None. No new protocols or ports.

## Pluginability Impact

None. Entry-point registries unchanged; no concrete provider imports added to core.

## Layer Boundary Assessment

No new `execution → integration implementation` edges. Existing `host_task → runtime.task` and contract imports unchanged in meaning.

## Execution Engine Semantics Impact

None. Import order only; runtime behavior unchanged.

## Import Cycle Assessment

Cold import of all five touched modules: **PASS**.

## Cold Import Results

```text
intergrax.runtime.execution.delegated_execution.correlation_persistence — OK
intergrax.runtime.execution.failure_evidence.recording_delegate — OK
intergrax.runtime.execution.host_task — OK
intergrax.runtime.execution.decision_finalization_conformance — OK
intergrax.runtime.execution.retry.service — OK
```

## Architecture Gates

- UE-10R4.1: **PASS**
- `test_ee_final_arch_*`: **PASS** (full `tests/unit/runtime/architecture/` run)
- U5 zero-bypass (`test_platform_execution_unification_u5_final_zero_bypass.py`): **PASS**

## Execution Regression

Full `tests/unit/runtime/execution/`: **1397 passed**, 8 skipped, **18 failed** — failures are `LLMAdapterDependencyError` (missing optional `ollama` package) in harness/nexus budget and `test_host_task_revision_reentry` tests; **not attributable to import hygiene changes**. Targeted suites for correlation, failure evidence, decision finalization, retry security gate: **109 passed**.

## Qualification Regression

`tests/unit/testing_support/execution_qualification/`: **PASS** (included in architecture batch run; no qualification failures).

## Static Quality

- `ruff format` applied to changed production files.
- `ruff check` on changed files: pre-existing `decision_finalization_conformance.py` F401/F811 (unchanged by this task).
- `pyright` on changed files: pre-existing `host_task` / conformance issues unrelated to import moves.
- `git diff --check`: **PASS** on staged scope.

## Changed Files

- `intergrax/runtime/execution/delegated_execution/correlation_persistence.py`
- `intergrax/runtime/execution/failure_evidence/recording_delegate.py`
- `intergrax/runtime/execution/host_task.py`
- `intergrax/runtime/execution/decision_finalization_conformance.py`
- `intergrax/runtime/execution/retry/service.py`
- `docs/project/maintainers/qualification/INTEGRAX_EXECUTION_UE_10R41_LOCAL_IMPORT_HYGIENE_REMEDIATION.md`

## Remaining Debt

- Environment: optional `ollama` dependency causes unrelated execution harness failures on dev machines without `Intergrax-ai[llm-ollama]`.
- Pre-existing ruff/pyright findings in touched files (not introduced here).

## Decision

Proceed with Class A module-level import promotion; no architecture reopen.

## Commit SHA

`4bef684afaff0a0b8855ea4260df74b0ac781aed`

## Final Verdict

**UE-10R4.1 LOCAL IMPORT HYGIENE REMEDIATION = PASS**
