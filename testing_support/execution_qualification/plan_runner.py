# © Artur Czarnecki. All rights reserved.

"""Execute compiled qualification plans with in-run leaf dedup and receipt reuse."""

from __future__ import annotations

from testing_support.execution_qualification.aggregate import (
    QualificationAggregateEvaluator,
)
from testing_support.execution_qualification.contracts import (
    ExecutionQualificationRunResult,
    ExecutionQualificationSuiteResult,
    QualificationExecutionContext,
    QualificationRunConfig,
    QualificationRunManifest,
    QualificationRunStatus,
    QualificationSuite,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.coordinator import QualificationCoordinator
from testing_support.execution_qualification.coordinator_port import (
    QualificationCoordinatorPort,
)
from testing_support.execution_qualification.graph_contracts import (
    QualificationExecutionPlan,
    QualificationGateResult,
    QualificationNodeKind,
    QualificationPlanRunResult,
    QualificationReceiptConflictError,
)
from testing_support.execution_qualification.executor import (
    PytestSubprocessSuiteExecutor,
    QualificationSuiteExecutor,
)


class _InRunLeafDedupExecutor:
    """Fail closed if the same suite_id is physically executed twice in one plan run."""

    def __init__(self, inner: QualificationSuiteExecutor) -> None:
        self._inner = inner
        self._executed_suite_ids: set[str] = set()

    def execute(
        self,
        suite: QualificationSuite,
        context: QualificationExecutionContext,
    ) -> ExecutionQualificationSuiteResult:
        if suite.suite_id in self._executed_suite_ids:
            raise QualificationReceiptConflictError(
                f"duplicate physical execution of suite {suite.suite_id!r}"
            )
        self._executed_suite_ids.add(suite.suite_id)
        return self._inner.execute(suite, context)


def _leaf_manifest_from_plan(
    plan: QualificationExecutionPlan,
) -> QualificationRunManifest:
    suite_by_id: dict[str, QualificationSuite] = {}
    for node in plan.ordered_nodes:
        if node.kind is QualificationNodeKind.LEAF_SUITE and node.suite is not None:
            suite_by_id[node.node_id] = node.suite
    suites: list[QualificationSuite] = []
    for suite_id in plan.leaf_suite_ids:
        suite = suite_by_id.get(suite_id)
        if suite is None:
            raise QualificationReceiptConflictError(
                f"missing suite definition for leaf {suite_id!r}"
            )
        suites.append(suite)
    manifest = QualificationRunManifest(suites=tuple(suites))
    manifest_ids = tuple(s.suite_id for s in manifest.suites)
    if manifest_ids != plan.leaf_suite_ids:
        raise QualificationReceiptConflictError(
            "manifest suite IDs must match plan.leaf_suite_ids"
        )
    return manifest


def _index_suite_receipts(
    plan: QualificationExecutionPlan,
    run_result: ExecutionQualificationRunResult,
) -> dict[str, ExecutionQualificationSuiteResult]:
    index: dict[str, ExecutionQualificationSuiteResult] = {}
    for receipt in run_result.suite_results:
        if receipt.suite_id in index:
            raise QualificationReceiptConflictError(
                f"duplicate suite receipt for {receipt.suite_id!r}"
            )
        index[receipt.suite_id] = receipt

    expected = set(plan.leaf_suite_ids)
    actual = set(index.keys())
    if unexpected := actual - expected:
        raise QualificationReceiptConflictError(
            f"unexpected suite receipts: {sorted(unexpected)!r}"
        )
    if missing := expected - actual:
        raise QualificationReceiptConflictError(
            f"missing suite receipts: {sorted(missing)!r}"
        )
    return index


def _root_run_status(
    root_gate_receipts: tuple[QualificationGateResult, ...],
) -> QualificationRunStatus:
    if any(
        receipt.status is not QualificationSuiteStatus.PASS
        for receipt in root_gate_receipts
    ):
        return QualificationRunStatus.FAIL
    return QualificationRunStatus.PASS


class QualificationPlanRunner:
    def __init__(
        self,
        coordinator: QualificationCoordinatorPort | None = None,
        aggregate_evaluator: QualificationAggregateEvaluator | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._aggregate_evaluator = (
            aggregate_evaluator
            if aggregate_evaluator is not None
            else QualificationAggregateEvaluator()
        )

    def run(
        self,
        plan: QualificationExecutionPlan,
        config: QualificationRunConfig,
    ) -> QualificationPlanRunResult:
        manifest = _leaf_manifest_from_plan(plan)
        coordinator = self._resolve_coordinator()
        run_result = coordinator.run(manifest, config)
        suite_index = _index_suite_receipts(plan, run_result)
        suite_receipts = tuple(suite_index[sid] for sid in plan.leaf_suite_ids)

        gate_receipts = self._aggregate_evaluator.evaluate(plan, suite_index)
        gate_by_id = {receipt.gate_id: receipt for receipt in gate_receipts}
        root_gate_receipts = tuple(
            gate_by_id[root_id] for root_id in plan.root_gate_ids
        )
        status = _root_run_status(root_gate_receipts)
        gate_count = sum(
            1
            for node in plan.ordered_nodes
            if node.kind is not QualificationNodeKind.LEAF_SUITE
        )

        return QualificationPlanRunResult(
            profile_id=plan.profile_id,
            status=status,
            suite_receipts=suite_receipts,
            gate_receipts=gate_receipts,
            root_gate_receipts=root_gate_receipts,
            physical_leaf_count=len(plan.leaf_suite_ids),
            gate_count=gate_count,
        )

    def _resolve_coordinator(
        self,
    ) -> QualificationCoordinatorPort:
        if self._coordinator is not None:
            return self._coordinator
        inner = PytestSubprocessSuiteExecutor()
        guarded = _InRunLeafDedupExecutor(inner)
        return QualificationCoordinator(executor=guarded)


def run_qualification_execution_plan(
    plan: QualificationExecutionPlan,
    config: QualificationRunConfig,
    *,
    coordinator: QualificationCoordinatorPort | None = None,
) -> QualificationPlanRunResult:
    return QualificationPlanRunner(coordinator=coordinator).run(plan, config)
