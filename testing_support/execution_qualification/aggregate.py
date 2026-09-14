# © Artur Czarnecki. All rights reserved.

"""Aggregate and static gate evaluation from suite and gate receipts."""

from __future__ import annotations

from collections.abc import Mapping

from testing_support.execution_qualification.contracts import (
    ExecutionQualificationSuiteResult,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.graph_contracts import (
    QualificationAggregateEvaluationError,
    QualificationExecutionNode,
    QualificationExecutionPlan,
    QualificationGateResult,
    QualificationNodeKind,
)


def _dependency_status(
    dep_id: str,
    suite_receipts: Mapping[str, ExecutionQualificationSuiteResult],
    gate_receipts: Mapping[str, QualificationGateResult],
) -> QualificationSuiteStatus:
    if dep_id in suite_receipts:
        return suite_receipts[dep_id].status
    gate_result = gate_receipts.get(dep_id)
    if gate_result is not None:
        return gate_result.status
    raise QualificationAggregateEvaluationError(
        f"missing receipt for dependency {dep_id!r}"
    )


def evaluate_gate_node(
    node: QualificationExecutionNode,
    suite_receipts: Mapping[str, ExecutionQualificationSuiteResult],
    gate_receipts: Mapping[str, QualificationGateResult],
) -> QualificationGateResult:
    if node.gate is None:
        raise QualificationAggregateEvaluationError(
            f"gate node {node.node_id!r} has no gate definition"
        )
    failure_dependencies: list[str] = []
    for dep_id in node.dependencies:
        dep_status = _dependency_status(dep_id, suite_receipts, gate_receipts)
        if dep_status is not QualificationSuiteStatus.PASS:
            failure_dependencies.append(dep_id)

    status = (
        QualificationSuiteStatus.PASS
        if not failure_dependencies
        else QualificationSuiteStatus.FAIL
    )
    return QualificationGateResult(
        gate_id=node.node_id,
        status=status,
        consumed_node_ids=node.dependencies,
        mandatory=node.gate.mandatory,
        failure_dependencies=tuple(failure_dependencies),
    )


class QualificationAggregateEvaluator:
    """Evaluate gates in plan order; each gate at most once per run."""

    def evaluate(
        self,
        plan: QualificationExecutionPlan,
        suite_receipts: Mapping[str, ExecutionQualificationSuiteResult],
    ) -> tuple[QualificationGateResult, ...]:
        gate_by_id: dict[str, QualificationGateResult] = {}
        ordered: list[QualificationGateResult] = []

        for node in plan.ordered_nodes:
            if node.kind is QualificationNodeKind.LEAF_SUITE:
                continue
            if node.node_id in gate_by_id:
                continue
            result = evaluate_gate_node(node, suite_receipts, gate_by_id)
            gate_by_id[node.node_id] = result
            ordered.append(result)

        return tuple(ordered)

    def gate_receipts_by_id(
        self,
        plan: QualificationExecutionPlan,
        suite_receipts: Mapping[str, ExecutionQualificationSuiteResult],
    ) -> dict[str, QualificationGateResult]:
        evaluated = self.evaluate(plan, suite_receipts)
        return {receipt.gate_id: receipt for receipt in evaluated}
