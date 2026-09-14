# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
import inspect
from unittest.mock import MagicMock

import pytest

from testing_support.execution_qualification.compiler import (
    QualificationGraphCompiler,
    compile_qualification_execution_plan,
    merge_qualification_suites,
)
from testing_support.execution_qualification.contracts import (
    QualificationRunManifest,
    QualificationSuite,
)
from testing_support.execution_qualification.graph_contracts import (
    ConflictingQualificationSuiteDefinitionError,
    MissingQualificationDependencyError,
    QualificationDependencyCycleError,
    QualificationGateDefinition,
    QualificationManifestConflictError,
    QualificationNodeKind,
    QualificationProfile,
    QualificationProfileError,
)


def _suite(suite_id: str, target: str = "tests/unit/foo.py") -> QualificationSuite:
    return QualificationSuite(suite_id=suite_id, pytest_arguments=(target,))


def _gate(
    gate_id: str,
    *requires: str,
    declaration_index: int = 0,
    gate_kind: QualificationNodeKind = QualificationNodeKind.AGGREGATE_GATE,
) -> QualificationGateDefinition:
    return QualificationGateDefinition(
        gate_id=gate_id,
        requires=requires,
        declaration_index=declaration_index,
        gate_kind=gate_kind,
    )


def _manifest(*suites: QualificationSuite) -> QualificationRunManifest:
    return QualificationRunManifest(suites=suites)


def _profile(*root_gate_ids: str, profile_id: str = "cert") -> QualificationProfile:
    return QualificationProfile(profile_id=profile_id, root_gate_ids=root_gate_ids)


def test_happy_path_root_to_one_leaf() -> None:
    plan = compile_qualification_execution_plan(
        _manifest(_suite("leaf-x")),
        (_gate("root", "leaf-x"),),
        _profile("root"),
    )
    assert plan.leaf_suite_ids == ("leaf-x",)
    assert [n.node_id for n in plan.ordered_nodes] == ["leaf-x", "root"]


def test_happy_path_dedup_shared_leaf() -> None:
    plan = compile_qualification_execution_plan(
        _manifest(_suite("leaf-x")),
        (
            _gate("gate-a", "leaf-x", declaration_index=1),
            _gate("gate-b", "leaf-x", declaration_index=2),
            _gate("root", "gate-a", "gate-b", declaration_index=3),
        ),
        _profile("root"),
    )
    assert plan.leaf_suite_ids == ("leaf-x",)
    assert sum(1 for n in plan.ordered_nodes if n.node_id == "leaf-x") == 1


def test_happy_path_multi_level_chain() -> None:
    plan = compile_qualification_execution_plan(
        _manifest(_suite("leaf")),
        (
            _gate("gate-b", "leaf", declaration_index=2),
            _gate("gate-a", "gate-b", declaration_index=1),
            _gate("root", "gate-a", declaration_index=0),
        ),
        _profile("root"),
    )
    ids = [n.node_id for n in plan.ordered_nodes]
    assert (
        ids.index("leaf")
        < ids.index("gate-b")
        < ids.index("gate-a")
        < ids.index("root")
    )


def test_unreachable_suite_excluded() -> None:
    plan = compile_qualification_execution_plan(
        _manifest(_suite("used"), _suite("orphan")),
        (_gate("root", "used"),),
        _profile("root"),
    )
    assert plan.leaf_suite_ids == ("used",)
    assert "orphan" not in {n.node_id for n in plan.ordered_nodes}


def test_determinism_independent_gate_collection_order() -> None:
    manifest = _manifest(_suite("leaf"))
    gates_a = (
        _gate("gate-a", "leaf", declaration_index=1),
        _gate("gate-b", "leaf", declaration_index=2),
        _gate("root", "gate-a", "gate-b", declaration_index=3),
    )
    gates_b = (gates_a[2], gates_a[0], gates_a[1])
    profile = _profile("root")
    plan_a = compile_qualification_execution_plan(manifest, gates_a, profile)
    plan_b = compile_qualification_execution_plan(manifest, gates_b, profile)
    assert plan_a == plan_b


def test_self_cycle() -> None:
    with pytest.raises(QualificationDependencyCycleError) as exc:
        compile_qualification_execution_plan(
            _manifest(_suite("unused")),
            (_gate("root", "root"),),
            _profile("root"),
        )
    assert exc.value.cycle_path[0] == "root"


def test_two_node_cycle() -> None:
    with pytest.raises(QualificationDependencyCycleError):
        compile_qualification_execution_plan(
            _manifest(_suite("unused")),
            (_gate("a", "b"), _gate("b", "a"), _gate("root", "a")),
            _profile("root"),
        )


def test_three_node_cycle() -> None:
    with pytest.raises(QualificationDependencyCycleError):
        compile_qualification_execution_plan(
            _manifest(_suite("unused")),
            (
                _gate("a", "b"),
                _gate("b", "c"),
                _gate("c", "a"),
                _gate("root", "a"),
            ),
            _profile("root"),
        )


def test_missing_gate_dependency() -> None:
    with pytest.raises(MissingQualificationDependencyError, match="missing-gate"):
        compile_qualification_execution_plan(
            _manifest(_suite("unused")),
            (_gate("root", "missing-gate"),),
            _profile("root"),
        )


def test_missing_suite_dependency() -> None:
    with pytest.raises(MissingQualificationDependencyError, match="missing-suite"):
        compile_qualification_execution_plan(
            _manifest(_suite("unused")),
            (_gate("root", "missing-suite"),),
            _profile("root"),
        )


def test_missing_profile_root() -> None:
    with pytest.raises(QualificationProfileError, match="unknown root"):
        compile_qualification_execution_plan(
            _manifest(_suite("unused")),
            (_gate("root"),),
            _profile("not-defined"),
        )


def test_empty_profile_roots() -> None:
    with pytest.raises(QualificationProfileError, match="at least one root"):
        compile_qualification_execution_plan(
            _manifest(_suite("x")),
            (_gate("root", "x"),),
            QualificationProfile(profile_id="empty", root_gate_ids=()),
        )


def test_suite_gate_id_collision() -> None:
    with pytest.raises(QualificationManifestConflictError, match="collides"):
        compile_qualification_execution_plan(
            _manifest(_suite("same-id")),
            (_gate("same-id"),),
            _profile("same-id"),
        )


def test_duplicate_gate_id() -> None:
    gate = _gate("dup")
    with pytest.raises(QualificationManifestConflictError, match="duplicate gate"):
        compile_qualification_execution_plan(
            _manifest(_suite("unused")),
            (gate, gate),
            _profile("dup"),
        )


def test_conflicting_suite_definitions() -> None:
    a = _suite("dup", "tests/a.py")
    b = _suite("dup", "tests/b.py")
    with pytest.raises(ConflictingQualificationSuiteDefinitionError):
        merge_qualification_suites((a,), (b,))


def test_topological_dependency_first() -> None:
    plan = compile_qualification_execution_plan(
        _manifest(_suite("s1"), _suite("s2")),
        (
            _gate("g1", "s1", declaration_index=1),
            _gate("g2", "s2", declaration_index=2),
            _gate("root", "g1", "g2", declaration_index=3),
        ),
        _profile("root"),
    )
    order = {n.node_id: idx for idx, n in enumerate(plan.ordered_nodes)}
    for edge in plan.dependency_edges:
        assert order[edge[0]] < order[edge[1]]


def test_stable_sibling_order_by_declaration_index() -> None:
    plan = compile_qualification_execution_plan(
        _manifest(_suite("s-low", "a.py"), _suite("s-high", "b.py")),
        (_gate("root", "s-low", "s-high", declaration_index=0),),
        _profile("root"),
    )
    leaf_order = [
        n.node_id
        for n in plan.ordered_nodes
        if n.kind is QualificationNodeKind.LEAF_SUITE
    ]
    assert leaf_order == ["s-low", "s-high"]


def test_compiler_does_not_invoke_coordinator() -> None:
    coordinator = MagicMock()
    compile_qualification_execution_plan(
        _manifest(_suite("leaf")),
        (_gate("root", "leaf"),),
        _profile("root"),
    )
    coordinator.run.assert_not_called()


def test_graph_compiler_facade() -> None:
    plan = QualificationGraphCompiler().compile(
        _manifest(_suite("leaf")),
        (_gate("root", "leaf"),),
        _profile("root"),
    )
    assert plan.leaf_suite_ids == ("leaf",)


def test_graph_compiler_has_no_subprocess_dependency() -> None:
    from testing_support.execution_qualification import compiler as compiler_module

    source = inspect.getsource(compiler_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "subprocess"
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "subprocess"


def test_graph_compiler_has_no_production_runtime_import() -> None:
    from testing_support.execution_qualification import compiler as compiler_module

    source = inspect.getsource(compiler_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("intergrax")
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("intergrax")
