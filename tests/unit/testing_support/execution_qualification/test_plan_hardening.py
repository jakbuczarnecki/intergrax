# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from testing_support.execution_qualification.contracts import QualificationSuite
from testing_support.execution_qualification.graph_contracts import (
    QualificationExecutionNode,
    QualificationExecutionPlan,
    QualificationExecutionPlanError,
    QualificationGateDefinition,
    QualificationNodeKind,
)


def _suite(suite_id: str) -> QualificationSuite:
    return QualificationSuite(
        suite_id=suite_id, pytest_arguments=("tests/unit/foo.py",)
    )


def _leaf_node(
    suite_id: str, *, declaration_index: int = 0
) -> QualificationExecutionNode:
    return QualificationExecutionNode(
        node_id=suite_id,
        kind=QualificationNodeKind.LEAF_SUITE,
        dependencies=(),
        declaration_index=declaration_index,
        suite=_suite(suite_id),
        gate=None,
    )


def _gate_node(
    gate_id: str,
    *requires: str,
    declaration_index: int = 1,
) -> QualificationExecutionNode:
    gate = QualificationGateDefinition(
        gate_id=gate_id,
        requires=requires,
        declaration_index=declaration_index,
    )
    return QualificationExecutionNode(
        node_id=gate_id,
        kind=QualificationNodeKind.AGGREGATE_GATE,
        dependencies=requires,
        declaration_index=declaration_index,
        suite=None,
        gate=gate,
    )


def _minimal_plan(
    *,
    ordered_nodes: tuple[QualificationExecutionNode, ...],
    leaf_suite_ids: tuple[str, ...],
    root_gate_ids: tuple[str, ...],
    dependency_edges: tuple[tuple[str, str], ...],
) -> QualificationExecutionPlan:
    return QualificationExecutionPlan(
        profile_id="cert",
        ordered_nodes=ordered_nodes,
        leaf_suite_ids=leaf_suite_ids,
        root_gate_ids=root_gate_ids,
        dependency_edges=dependency_edges,
    )


def test_invalid_duplicate_node_ids() -> None:
    with pytest.raises(QualificationExecutionPlanError, match="duplicate node_id"):
        _minimal_plan(
            ordered_nodes=(_leaf_node("leaf-x"), _leaf_node("leaf-x")),
            leaf_suite_ids=("leaf-x",),
            root_gate_ids=("root",),
            dependency_edges=(("leaf-x", "root"),),
        )


def test_invalid_duplicate_leaf_ids() -> None:
    leaf = _leaf_node("leaf-x")
    gate = _gate_node("root", "leaf-x")
    with pytest.raises(
        QualificationExecutionPlanError, match="duplicate leaf_suite_ids"
    ):
        QualificationExecutionPlan(
            profile_id="cert",
            ordered_nodes=(leaf, gate),
            leaf_suite_ids=("leaf-x", "leaf-x"),
            root_gate_ids=("root",),
            dependency_edges=(("leaf-x", "root"),),
        )


def test_invalid_unknown_root() -> None:
    leaf = _leaf_node("leaf-x")
    with pytest.raises(QualificationExecutionPlanError, match="unknown root"):
        _minimal_plan(
            ordered_nodes=(leaf,),
            leaf_suite_ids=("leaf-x",),
            root_gate_ids=("missing-root",),
            dependency_edges=(),
        )


def test_invalid_root_is_leaf() -> None:
    leaf = _leaf_node("leaf-x")
    with pytest.raises(QualificationExecutionPlanError, match="cannot be a LEAF_SUITE"):
        _minimal_plan(
            ordered_nodes=(leaf,),
            leaf_suite_ids=("leaf-x",),
            root_gate_ids=("leaf-x",),
            dependency_edges=(),
        )


def test_invalid_edge_unknown_node() -> None:
    leaf = _leaf_node("leaf-x")
    gate = _gate_node("root", "leaf-x")
    with pytest.raises(QualificationExecutionPlanError, match="unknown node"):
        _minimal_plan(
            ordered_nodes=(leaf, gate),
            leaf_suite_ids=("leaf-x",),
            root_gate_ids=("root",),
            dependency_edges=(("ghost", "root"),),
        )


def test_invalid_duplicate_edge() -> None:
    leaf = _leaf_node("leaf-x")
    gate = _gate_node("root", "leaf-x")
    with pytest.raises(
        QualificationExecutionPlanError, match="duplicate dependency edge"
    ):
        _minimal_plan(
            ordered_nodes=(leaf, gate),
            leaf_suite_ids=("leaf-x",),
            root_gate_ids=("root",),
            dependency_edges=(("leaf-x", "root"), ("leaf-x", "root")),
        )


def test_invalid_self_edge() -> None:
    gate = _gate_node("root")
    with pytest.raises(QualificationExecutionPlanError, match="self-edge"):
        _minimal_plan(
            ordered_nodes=(gate,),
            leaf_suite_ids=(),
            root_gate_ids=("root",),
            dependency_edges=(("root", "root"),),
        )


def test_invalid_dependency_after_dependent() -> None:
    leaf = _leaf_node("leaf-x")
    gate = _gate_node("root", "leaf-x")
    with pytest.raises(QualificationExecutionPlanError, match="must appear before"):
        _minimal_plan(
            ordered_nodes=(gate, leaf),
            leaf_suite_ids=("leaf-x",),
            root_gate_ids=("root",),
            dependency_edges=(("leaf-x", "root"),),
        )


def test_invalid_leaf_id_points_to_gate() -> None:
    gate = _gate_node("root")
    with pytest.raises(QualificationExecutionPlanError, match="not a LEAF_SUITE"):
        _minimal_plan(
            ordered_nodes=(gate,),
            leaf_suite_ids=("root",),
            root_gate_ids=("root",),
            dependency_edges=(),
        )


def test_invalid_node_identity_mismatch() -> None:
    with pytest.raises(QualificationExecutionPlanError, match="suite.suite_id"):
        QualificationExecutionNode(
            node_id="leaf-a",
            kind=QualificationNodeKind.LEAF_SUITE,
            dependencies=(),
            declaration_index=0,
            suite=_suite("leaf-b"),
            gate=None,
        )
