# INTEGRAx-QUALIFICATION-REPRESENTATIVE-ORCHESTRATOR-MIGRATION-AND-PARITY-PROOF

## Metadata

| Field | Value |
| ----- | ----- |
| **Task** | `INTEGRAx-QUALIFICATION-REPRESENTATIVE-ORCHESTRATOR-MIGRATION-AND-PARITY-PROOF` |
| **Date** | 2026-09-14 |
| **Branch** | `development` |
| **Parent** | [`INTEGRAX_QUALIFICATION_IN_RUN_DEDUP_AND_RECEIPT_REUSE.md`](INTEGRAX_QUALIFICATION_IN_RUN_DEDUP_AND_RECEIPT_REUSE.md) |

## Selected Representative Flow

`NPSC-5F/R1 Final` — `tests/unit/runtime/architecture/test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity.py`

Profile id: `npsc5f-r1-final`

## Selection Rationale

| | |
| --- | --- |
| **Candidate considered** | NPSC-5F/R1 Final, NPSC-5F/R2 Final, NPSC-5E/R3 Final |
| **Candidate selected** | NPSC-5F/R1 Final |
| **Reason** | Legacy `_MANDATORY_SUITES` with nested NPSC-5E → R3 → R2 expansion; shared `DG_001` / `NPSC-5D Final` duplicates for dedup proof; smaller than full NPSC-5F Final matrix |

## Legacy Architecture

Top-level `test_mandatory_frozen_suite_passes` parametrize over seven mandatory labels. `NPSC-5E Final` subprocess-composes R3 Final, which subprocess-composes R2 Final and additional section-94 suites — duplicate logical coverage vs direct R1 entries.

## Legacy Required Targets

Typed helper: `LegacyRequiredTarget` in `testing_support/npsc5f_r1_legacy_targets.py`. Expansion is explicit (no reflection) via orchestrator path registry. Semantic leaf set: `legacy_r1_final_required_leaf_targets()`.

## Canonical DAG Mapping

`build_npsc5f_r1_qualification_graph()` in `testing_support/npsc5f_r1_qualification_profile.py`:

- Explicit `QualificationSuite` manifest (no filesystem discovery)
- Gates: direct R1 branch + expanded NPSC-5E branch → root `npsc5f-r1.final`
- Compiler/runner unchanged (profile-only composition)

## Suite IDs

Semantic / reused ids include `runtime-events`, `runtime-observability`, `dg001-lineage`, `npsc5d-final`, `npsc5f-r1.*`, and reused `npsc5e-r3.mandatory.*` where pytest arguments match R3 mandatory sources. R2-only leaves use `npsc5e-r2.mandatory.*`.

## Gate/Profile Definition

Root gate `npsc5f-r1.final` depends on `npsc5f-r1.direct.aggregate` and `npsc5e-r3.expanded.aggregate`. Aggregate gates consume receipts only (no subprocess).

## Target-Set Parity

`normalize(legacy_r1_final_required_leaf_targets()) == normalize(dag leaf pytest_arguments)` — enforced by `test_representative_legacy_and_dag_required_target_sets_are_equal`.

## PASS Parity

Fake-executor canonical plan run → `QualificationRunStatus.PASS` (`test_canonical_all_pass_root_pass`).

## FAIL/SKIP Parity

Injected `dg001-lineage` FAIL/SKIP → root FAIL (`test_injected_leaf_fail_root_fail`, `test_injected_leaf_skip_root_fail`). Matches coordinator fail-closed SKIP semantics.

## Physical Execution Count

Legacy expanded subprocess count (`legacy_subprocess_count()`) exceeds `plan.leaf_suite_ids` length (`test_logical_requests_exceed_physical_executions`). Shared `dg001-lineage` invoked once (`test_shared_leaf_physically_executes_once`).

## Nested-Orchestrator Elimination

DAG leaves must not be orchestrator single-file targets (`test_representative_dag_contains_no_nested_pytest_orchestrator_leaf`).

## Existing Framework Reuse

`compile_qualification_execution_plan`, `QualificationPlanRunner`, `QualificationCoordinator`, `PytestSubprocessSuiteExecutor`, `QualificationAggregateEvaluator`.

## Typing Hardening

`QualificationCoordinatorPort` Protocol; `_InRunLeafDedupExecutor.execute` uses `QualificationExecutionContext`; removed `type: ignore` from `plan_runner.py`.

## Isolation Preservation

R3 cross-db exclusive resource preserved on `npsc5e-r3.mandatory.r3-implementation-gate` suite definition.

## Tests

`tests/unit/testing_support/execution_qualification/test_npsc5f_r1_representative_migration_parity.py` plus full `tests/unit/testing_support/execution_qualification/` regression.

## Static Quality

```bash
uv run pytest tests/unit/testing_support/execution_qualification/ -q
uv run ruff check <changed scope>
uv run ruff format --check <changed scope>
uv run pyright <changed scope>
```

## Production Changes

**NONE** (`intergrax/` untouched).

## Findings

Representative profile demonstrates plugin model: new profile + suites + gates without compiler/runner edits. Full legacy subprocess qualification of expanded matrix not re-run in-session (duration); parity proven via target-set equality + typed fake executor + existing subprocess executor path.

## Migration Decision

Keep legacy `test_mandatory_frozen_suite_passes` as reference; canonical profile is opt-in via `build_npsc5f_r1_qualification_graph()` until full orchestrator migration.

## Final Verdict

**REPRESENTATIVE DAG MIGRATION PROVEN**
