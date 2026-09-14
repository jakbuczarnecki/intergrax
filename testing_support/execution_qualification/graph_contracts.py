# © Artur Czarnecki. All rights reserved.

"""Typed contracts for canonical qualification DAG definitions and compiled plans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from testing_support.execution_qualification.contracts import (
    ExecutionQualificationSuiteResult,
    QualificationRunManifest,
    QualificationRunStatus,
    QualificationSuite,
    QualificationSuiteStatus,
)

_SAFE_NODE_ID_RE = re.compile(r"^[\w.-]+$")


class QualificationNodeKind(StrEnum):
    LEAF_SUITE = "LEAF_SUITE"
    AGGREGATE_GATE = "AGGREGATE_GATE"
    STATIC_GATE = "STATIC_GATE"


class QualificationGraphError(ValueError):
    """Base class for graph definition / compilation failures."""


class QualificationExecutionPlanError(QualificationGraphError):
    """Invalid compiled execution plan or plan-run contract violation."""


@dataclass(frozen=True, slots=True)
class QualificationGateDefinition:
    """Aggregate or static gate — no pytest/subprocess configuration."""

    gate_id: str
    requires: tuple[str, ...]
    mandatory: bool = True
    declaration_index: int = 0
    description: str | None = None
    gate_kind: QualificationNodeKind = QualificationNodeKind.AGGREGATE_GATE

    def __post_init__(self) -> None:
        if not self.gate_id or not _SAFE_NODE_ID_RE.fullmatch(self.gate_id):
            raise ValueError(f"invalid gate_id: {self.gate_id!r}")
        if self.declaration_index < 0:
            raise ValueError("declaration_index must be non-negative")
        if self.gate_kind not in (
            QualificationNodeKind.AGGREGATE_GATE,
            QualificationNodeKind.STATIC_GATE,
        ):
            raise ValueError("gate_kind must be AGGREGATE_GATE or STATIC_GATE")


@dataclass(frozen=True, slots=True)
class QualificationProfile:
    profile_id: str
    root_gate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.profile_id or not _SAFE_NODE_ID_RE.fullmatch(self.profile_id):
            raise ValueError(f"invalid profile_id: {self.profile_id!r}")


@dataclass(frozen=True, slots=True)
class QualificationGraphDefinition:
    """Explicit graph inputs: manifest suites + gates + profile catalog."""

    run_manifest: QualificationRunManifest
    gates: tuple[QualificationGateDefinition, ...]
    profiles: tuple[QualificationProfile, ...] = ()


@dataclass(frozen=True, slots=True)
class QualificationExecutionNode:
    node_id: str
    kind: QualificationNodeKind
    dependencies: tuple[str, ...]
    declaration_index: int
    suite: QualificationSuite | None = None
    gate: QualificationGateDefinition | None = None

    def __post_init__(self) -> None:
        if not self.node_id or not _SAFE_NODE_ID_RE.fullmatch(self.node_id):
            raise QualificationExecutionPlanError(f"invalid node_id: {self.node_id!r}")
        if self.declaration_index < 0:
            raise QualificationExecutionPlanError(
                "declaration_index must be non-negative"
            )
        if len(self.dependencies) != len(set(self.dependencies)):
            raise QualificationExecutionPlanError(
                f"duplicate dependencies on node {self.node_id!r}"
            )
        if self.node_id in self.dependencies:
            raise QualificationExecutionPlanError(
                f"node {self.node_id!r} cannot depend on itself"
            )
        if self.kind is QualificationNodeKind.LEAF_SUITE:
            if self.suite is None or self.gate is not None:
                raise QualificationExecutionPlanError(
                    "LEAF_SUITE requires suite and no gate"
                )
            if self.suite.suite_id != self.node_id:
                raise QualificationExecutionPlanError(
                    f"LEAF_SUITE node_id {self.node_id!r} != suite.suite_id {self.suite.suite_id!r}"
                )
        else:
            if self.gate is None or self.suite is not None:
                raise QualificationExecutionPlanError(
                    "gate nodes require gate and no suite"
                )
            if self.gate.gate_id != self.node_id:
                raise QualificationExecutionPlanError(
                    f"gate node_id {self.node_id!r} != gate.gate_id {self.gate.gate_id!r}"
                )
            if self.gate.gate_kind is not self.kind:
                raise QualificationExecutionPlanError(
                    f"node kind {self.kind!r} does not match gate_kind {self.gate.gate_kind!r}"
                )


@dataclass(frozen=True, slots=True)
class QualificationExecutionPlan:
    profile_id: str
    ordered_nodes: tuple[QualificationExecutionNode, ...]
    leaf_suite_ids: tuple[str, ...]
    root_gate_ids: tuple[str, ...]
    dependency_edges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise QualificationExecutionPlanError("profile_id must be non-empty")
        _validate_qualification_execution_plan(self)


@dataclass(frozen=True, slots=True)
class QualificationGateResult:
    """Immutable aggregate/static gate receipt for one plan run."""

    gate_id: str
    status: QualificationSuiteStatus
    consumed_node_ids: tuple[str, ...]
    mandatory: bool
    failure_dependencies: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QualificationPlanRunResult:
    """Typed outcome of executing a compiled qualification plan."""

    profile_id: str
    status: QualificationRunStatus
    suite_receipts: tuple[ExecutionQualificationSuiteResult, ...]
    gate_receipts: tuple[QualificationGateResult, ...]
    root_gate_receipts: tuple[QualificationGateResult, ...]
    physical_leaf_count: int
    gate_count: int


def _validate_qualification_execution_plan(plan: QualificationExecutionPlan) -> None:
    node_ids: list[str] = []
    node_by_id: dict[str, QualificationExecutionNode] = {}
    for node in plan.ordered_nodes:
        if node.node_id in node_by_id:
            raise QualificationExecutionPlanError(
                f"duplicate node_id: {node.node_id!r}"
            )
        node_ids.append(node.node_id)
        node_by_id[node.node_id] = node

    if len(plan.leaf_suite_ids) != len(set(plan.leaf_suite_ids)):
        raise QualificationExecutionPlanError("duplicate leaf_suite_ids")

    leaf_id_set = set(plan.leaf_suite_ids)
    for leaf_id in plan.leaf_suite_ids:
        leaf_node = node_by_id.get(leaf_id)
        if leaf_node is None:
            raise QualificationExecutionPlanError(
                f"leaf_suite_id {leaf_id!r} not present in ordered_nodes"
            )
        if leaf_node.kind is not QualificationNodeKind.LEAF_SUITE:
            raise QualificationExecutionPlanError(
                f"leaf_suite_id {leaf_id!r} is not a LEAF_SUITE node"
            )

    for node in plan.ordered_nodes:
        if node.kind is QualificationNodeKind.LEAF_SUITE:
            if node.node_id not in leaf_id_set:
                raise QualificationExecutionPlanError(
                    f"LEAF_SUITE {node.node_id!r} missing from leaf_suite_ids"
                )

    for root_id in plan.root_gate_ids:
        root_node = node_by_id.get(root_id)
        if root_node is None:
            raise QualificationExecutionPlanError(f"unknown root gate: {root_id!r}")
        if root_node.kind is QualificationNodeKind.LEAF_SUITE:
            raise QualificationExecutionPlanError(
                f"root gate {root_id!r} cannot be a LEAF_SUITE"
            )

    edge_set: set[tuple[str, str]] = set()
    for dep, dependent in plan.dependency_edges:
        if dep not in node_by_id:
            raise QualificationExecutionPlanError(
                f"dependency edge references unknown node {dep!r}"
            )
        if dependent not in node_by_id:
            raise QualificationExecutionPlanError(
                f"dependency edge references unknown node {dependent!r}"
            )
        if dep == dependent:
            raise QualificationExecutionPlanError(f"self-edge on node {dep!r}")
        edge_key = (dep, dependent)
        if edge_key in edge_set:
            raise QualificationExecutionPlanError(
                f"duplicate dependency edge {dep!r} -> {dependent!r}"
            )
        edge_set.add(edge_key)

    order_index = {node_id: index for index, node_id in enumerate(node_ids)}
    for dep, dependent in plan.dependency_edges:
        if order_index[dep] >= order_index[dependent]:
            raise QualificationExecutionPlanError(
                f"dependency {dep!r} must appear before dependent {dependent!r}"
            )

    for node in plan.ordered_nodes:
        if node.kind is QualificationNodeKind.LEAF_SUITE:
            expected_deps: tuple[str, ...] = ()
        else:
            expected_deps = node.dependencies
        actual_deps = tuple(
            dep for dep, dependent in plan.dependency_edges if dependent == node.node_id
        )
        if tuple(actual_deps) != expected_deps:
            raise QualificationExecutionPlanError(
                f"node {node.node_id!r} dependencies {expected_deps!r} "
                f"do not match declared edges {actual_deps!r}"
            )


class QualificationManifestConflictError(QualificationGraphError):
    """Duplicate or conflicting manifest node definitions."""


class ConflictingQualificationSuiteDefinitionError(QualificationManifestConflictError):
    """Same suite_id with differing executable definitions."""


class MissingQualificationDependencyError(QualificationGraphError):
    """Required node id does not exist in the manifest or gate catalog."""


class QualificationDependencyCycleError(QualificationGraphError):
    """Cycle detected in the qualification dependency graph."""

    def __init__(self, cycle_path: tuple[str, ...]) -> None:
        self.cycle_path = cycle_path
        path = " -> ".join(cycle_path)
        super().__init__(f"qualification dependency cycle: {path}")


class QualificationProfileError(QualificationGraphError):
    """Invalid or inconsistent qualification profile."""


class QualificationReceiptConflictError(QualificationExecutionPlanError):
    """Missing, duplicate, or unexpected suite receipts during a plan run."""


class QualificationAggregateEvaluationError(QualificationGraphError):
    """Aggregate gate evaluation cannot proceed reliably."""
