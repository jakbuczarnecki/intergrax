# INTEGRAx-QUALIFICATION-IN-RUN-DEDUP-AND-RECEIPT-REUSE

## Metadata

| Field | Value |
| ----- | ----- |
| **Task** | `INTEGRAx-QUALIFICATION-IN-RUN-DEDUP-AND-RECEIPT-REUSE` |
| **Date** | 2026-09-14 |
| **Branch** | `development` |
| **Parent (P1)** | [`INTEGRAX_CANONICAL_QUALIFICATION_DAG_CONTRACTS_AND_GRAPH_COMPILER.md`](INTEGRAX_CANONICAL_QUALIFICATION_DAG_CONTRACTS_AND_GRAPH_COMPILER.md) |
| **Design (P0)** | [`INTEGRAX_QUALIFICATION_EXECUTION_GRAPH_AUDIT_AND_OPTIMIZATION_DESIGN.md`](INTEGRAX_QUALIFICATION_EXECUTION_GRAPH_AUDIT_AND_OPTIMIZATION_DESIGN.md) |

## Scope

In scope:

```text
QualificationExecutionPlan hardening
→ QualificationPlanRunner (leaf dedup via coordinator)
→ suite receipt index + aggregate gate evaluation
→ QualificationPlanRunResult
```

Out of scope: NPSC orchestrator migration, cross-run cache, `intergrax/` runtime changes.

## Reused Architecture

- `QualificationCoordinator`, `QualificationRunManifest`, `QualificationSuite`
- `QualificationSuiteExecutor` / `PytestSubprocessSuiteExecutor`
- `ExecutionQualificationSuiteResult` as canonical leaf receipt
- `QualificationRunStatus` for plan run and gate pass/fail projection
- `compile_qualification_execution_plan` unchanged compiler entry point

## Plan Hardening

`QualificationExecutionNode` and `QualificationExecutionPlan` validate direct construction (unique ids, leaf/root/edge invariants, topological order, identity match). Failures: `QualificationExecutionPlanError` (extends `QualificationGraphError`).

## Physical Dedup Semantics

Runner builds manifest from `plan.leaf_suite_ids` only (one entry per physical leaf). Coordinator executes each suite at most once per plan run. `_InRunLeafDedupExecutor` fails closed on duplicate `suite_id` invocation.

## Receipt Model

- **Leaf:** `ExecutionQualificationSuiteResult` (immutable after execution)
- **Gate:** `QualificationGateResult` (`gate_id`, `status`, `consumed_node_ids`, `mandatory`, optional `failure_dependencies`)
- **Run:** `QualificationPlanRunResult` (`profile_id`, `status`, `suite_receipts`, `gate_receipts`, `root_gate_receipts`, `physical_leaf_count`, `gate_count`)

Receipt mapping validates expected/missing/duplicate/unexpected suite ids (`QualificationReceiptConflictError`).

## Aggregate Gate Evaluation

`QualificationAggregateEvaluator` walks `plan.ordered_nodes`; only `AGGREGATE_GATE` / `STATIC_GATE` nodes are evaluated (no subprocess). Gate **PASS** iff every dependency receipt is **PASS**; **FAIL** or **SKIP** on a dependency fails the gate. Each gate evaluated at most once per run; shared gates reused across multiple roots.

## Failure Semantics

- Collect-all leaf execution preserved (coordinator unchanged).
- Plan run **PASS** only when all root gate receipts are **PASS**.
- Invalid plan, receipt contract violation, or duplicate physical execution → fail closed.

## Coordinator Integration

```text
plan.leaf_suite_ids → QualificationRunManifest → QualificationCoordinator.run
→ suite_results indexed by suite_id → aggregate evaluation
```

## Isolation Preservation

Exclusive resources and scheduling remain owned by `QualificationCoordinator` / `QualificationSuite` (T7 regression via plan runner).

## Tests

- `test_plan_runner.py` — T1–T10, determinism, architecture import guards
- `test_plan_hardening.py` — invalid direct plan construction

## Regression

```bash
uv run pytest tests/unit/testing_support/execution_qualification/ -q
```

## Static Quality

Changed-scope: `ruff check` + `ruff format` on new/changed modules. Full-directory `pyright` reports one pre-existing `executor.py` environ typing issue (unchanged).

## Production Changes

**NONE**

## Findings

P2 delivers in-run dedup and receipt reuse without migrating NPSC orchestrators. `logical_leaf_request_count` deferred until profile migration exposes multiplicity.

## Final Verdict

```text
QUALIFICATION IN-RUN DEDUP + RECEIPT REUSE = PASS
```
