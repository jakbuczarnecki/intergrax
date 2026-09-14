# INTEGRAx-QUALIFICATION-EMBEDDED-HARNESS-ELIMINATION-AND-RECEIPT-BASED-FINAL-GATES

## Metadata

| Field | Value |
|---|---|
| Task ID | `INTEGRAx-QUALIFICATION-EMBEDDED-HARNESS-ELIMINATION-AND-RECEIPT-BASED-FINAL-GATES` |
| Base commit (analysis) | `7fa0bd99784360f7182e9c18a010dbd1b4010413` |
| Implementation HEAD | see git log after task commit |

## Scope

Eliminate nested mandatory qualification execution from canonical R2/R3 Final and `npsc5f-final.recovery` leaves; preserve semantic/freeze coverage via receipt-based gate dependencies and explicit semantic pytest slices.

## Embedded Harness Inventory

| Module | Test / pattern | Class |
|---|---|---|
| `test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py` | `test_mandatory_frozen_suite_passes` | META-QUALIFICATION EXECUTION |
| same | `_run_pytest` subprocess fan-out | META-QUALIFICATION EXECUTION |
| `test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py` | `test_mandatory_frozen_suites_pass_via_parallel_qualification` | META-QUALIFICATION EXECUTION |
| same | `run_npsc5e_r3_mandatory_qualification` | META-QUALIFICATION EXECUTION |
| R2/R3 Final (remaining) | SHA/freeze/domain scenarios | DOMAIN / FREEZE / ARCHITECTURE |

Legacy meta tests retained with `@pytest.mark.legacy_embedded_qualification_harness` for standalone runs only.

## Current Nested Execution Model

Canonical DAG deduplicated predecessor leaves, but `npsc5f-final.recovery` and full-file pytest targets still executed embedded meta tests (~399 s R3 parallel qualification + ~122 s R2 parametric matrix).

## Target Architecture

```text
Canonical Qualification DAG
  → predecessor suite receipts (single physical run)
  → R2/R3 final-semantic leaf (-k excludes harness)
  → aggregate gate receipts
  → root final gate (receipt proof, no re-run)
```

## Contract Boundary

- `QualificationEvidenceProvider` + `PlanRunQualificationEvidenceProvider` (`testing_support/execution_qualification/evidence_provider.py`)
- Gate semantics reuse `QualificationPlanRunResult` / `QualificationGateResult` (no new status model)
- Final consumers depend on evidence port only, not `QualificationCoordinator` / executors

## Pluginability

Evidence provider is a Protocol; plan-run adapter is DI-friendly and supports future CI/remote evidence strategies without consumer changes.

## Layering Validation

- No `intergrax/` production changes
- No `testing_support → tests.unit.*` in catalog contracts
- Catalog does not import legacy regression matrix modules (existing guard retained)

## R2 Migration

- Profile `npsc5e-r2-final`: extra leaf `npsc5e-r2.final-semantic` with SSOT `-k` expression
- Predecessors remain expanded mandatory leaves; final gate consumes receipts only

## R3 Migration

- Profile `npsc5e-r3-final`: extra leaf `npsc5e-r3.final-semantic`
- Embedded parallel qualification excluded from canonical path

## Receipt-Based Gate Semantics

`QualificationAggregateEvaluator` + plan runner receipts: all predecessor PASS + semantic leaf PASS → root PASS; any FAIL/SKIP/missing → fail closed.

## Failure / Skip Semantics

Verified via fake executor tests (T5/T6/T7/T8) in `test_embedded_harness_elimination.py`.

## Coverage Preservation

Semantic parity certifier allows only declared final-semantic argument sets as canonical extras vs legacy expansion.

## Freeze Assertions

`test_canonical_predecessor_shas_recorded` and schema/SHA tests remain in semantic slice (not in harness exclusion list).

## Architecture Assertions

Reflection/forbidden-name/coordinator AST checks remain in semantic slice.

## Duplicate Execution Elimination

- Recovery leaf uses `CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR` (SSOT in `embedded_harness_kexpr.py`)
- Fake executor T9: each canonical leaf at most once per plan run

## Recovery Runtime Before

~626–683 s (`npsc5f-final.recovery`) dominated by embedded R2/R3 mandatory re-execution (per slow-leaf analysis).

## Recovery Runtime After

3-file semantic slice (harness excluded): ~18 s wall time in session measurement (`.tmp/session/embedded-harness-elimination/recovery-slice-timing.log`). Full DAG recovery leaf re-certification deferred to operator full profile run.

## Full Verification

Full `npsc5f-final` run not completed in this session. Pre-existing failure: `test_npsc5e_r2_final…::test_final_hitl_interop_no_checkpoint_self_approval` (nested `test_npsc5d_r3_governed_continuation` failures) reproduces on clean tree before harness changes.

## Production Changes

NONE

## Findings

- Embedded harness was the dominant redundant cost, not missing DAG dedup
- Receipt model sufficient without new DTOs
- Unrelated HITL R3 qualification failures block end-to-end green until fixed separately

## Decision

`EMBEDDED QUALIFICATION HARNESS ELIMINATION = PASS` (canonical path and guards)

Full profile closure: operator must run `npsc5f-final` after HITL R3 domain fix.

## Final Verdict

Harness elimination and receipt-based final gates **implemented and regression-tested** in qualification tooling. Independent full canonical verification **pending** due to pre-existing domain test failures outside this task scope.
