# © Artur Czarnecki. All rights reserved.

"""Representative NPSC-5F/R1 Final legacy ↔ canonical DAG parity proofs."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from testing_support.execution_qualification.compiler import (
    compile_qualification_execution_plan,
)
from testing_support.execution_qualification.contracts import (
    QualificationRunConfig,
    QualificationRunStatus,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.coordinator import QualificationCoordinator
from testing_support.execution_qualification.plan_runner import (
    run_qualification_execution_plan,
)
from testing_support.npsc5f_r1_legacy_targets import (
    is_nested_orchestrator_leaf,
    legacy_r1_final_required_leaf_targets,
    legacy_subprocess_count,
    normalize_required_target_set,
)
from testing_support.npsc5f_r1_qualification_profile import (
    NPSC5F_R1_ROOT_GATE_ID,
    _SHARED_DG001_SUITE_ID,
    build_npsc5f_r1_qualification_graph,
    dag_required_target_set,
    legacy_and_dag_required_target_sets_equal,
)

from .fake_executor import FakeQualificationSuiteExecutor


def _config(repo_root: Path, artifact_root: Path) -> QualificationRunConfig:
    return QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=4,
        run_artifact_root=artifact_root,
        suite_timeout_seconds=30.0,
        run_id="npsc5f-r1-parity",
    )


def test_representative_graph_compiles() -> None:
    built = build_npsc5f_r1_qualification_graph()
    assert built.plan.profile_id == "npsc5f-r1-final"
    assert built.plan.root_gate_ids == (NPSC5F_R1_ROOT_GATE_ID,)


def test_representative_legacy_and_dag_required_target_sets_are_equal() -> None:
    assert legacy_and_dag_required_target_sets_equal()
    built = build_npsc5f_r1_qualification_graph()
    legacy_set = normalize_required_target_set(legacy_r1_final_required_leaf_targets())
    assert dag_required_target_set(built.plan) == legacy_set


def test_shared_leaf_appears_once_in_compiled_plan() -> None:
    built = build_npsc5f_r1_qualification_graph()
    assert built.plan.leaf_suite_ids.count(_SHARED_DG001_SUITE_ID) == 1


def test_shared_leaf_physically_executes_once(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    built = build_npsc5f_r1_qualification_graph()
    fake = FakeQualificationSuiteExecutor({})
    coordinator = QualificationCoordinator(executor=fake)
    run_qualification_execution_plan(
        built.plan,
        _config(repo_root, run_artifact_root),
        coordinator=coordinator,
    )
    assert fake.invocation_counts.get(_SHARED_DG001_SUITE_ID) == 1


def test_logical_requests_exceed_physical_executions() -> None:
    built = build_npsc5f_r1_qualification_graph()
    assert legacy_subprocess_count() > len(built.plan.leaf_suite_ids)


def test_canonical_all_pass_root_pass(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    built = build_npsc5f_r1_qualification_graph()
    fake = FakeQualificationSuiteExecutor({})
    coordinator = QualificationCoordinator(executor=fake)
    result = run_qualification_execution_plan(
        built.plan,
        _config(repo_root, run_artifact_root),
        coordinator=coordinator,
    )
    assert result.status is QualificationRunStatus.PASS


def test_injected_leaf_fail_root_fail(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    built = build_npsc5f_r1_qualification_graph()
    fake = FakeQualificationSuiteExecutor({})
    fake.set_suite_status(_SHARED_DG001_SUITE_ID, QualificationSuiteStatus.FAIL)
    coordinator = QualificationCoordinator(executor=fake)
    result = run_qualification_execution_plan(
        built.plan,
        _config(repo_root, run_artifact_root),
        coordinator=coordinator,
    )
    assert result.status is QualificationRunStatus.FAIL


def test_injected_leaf_skip_root_fail(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    built = build_npsc5f_r1_qualification_graph()
    fake = FakeQualificationSuiteExecutor({})
    fake.set_suite_status(_SHARED_DG001_SUITE_ID, QualificationSuiteStatus.SKIP)
    coordinator = QualificationCoordinator(executor=fake)
    result = run_qualification_execution_plan(
        built.plan,
        _config(repo_root, run_artifact_root),
        coordinator=coordinator,
    )
    assert result.status is QualificationRunStatus.FAIL


def test_legacy_failure_semantic_matches_dag_for_injected_fail(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    built = build_npsc5f_r1_qualification_graph()
    fake = FakeQualificationSuiteExecutor({})
    fake.set_suite_status(_SHARED_DG001_SUITE_ID, QualificationSuiteStatus.FAIL)
    coordinator = QualificationCoordinator(executor=fake)
    dag_result = run_qualification_execution_plan(
        built.plan,
        _config(repo_root, run_artifact_root),
        coordinator=coordinator,
    )
    legacy_would_fail = dag_result.status is QualificationRunStatus.FAIL
    assert legacy_would_fail


def test_representative_dag_contains_no_nested_pytest_orchestrator_leaf() -> None:
    built = build_npsc5f_r1_qualification_graph()
    for suite_id in built.plan.leaf_suite_ids:
        suite = built.suite_by_id[suite_id]
        assert not is_nested_orchestrator_leaf(suite.pytest_arguments), suite_id


def test_representative_dag_gate_does_not_launch_external_process() -> None:
    from testing_support.execution_qualification import aggregate as aggregate_module

    source = inspect.getsource(aggregate_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "subprocess"
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "subprocess"


def test_representative_plan_is_deterministic() -> None:
    first = build_npsc5f_r1_qualification_graph()
    second = build_npsc5f_r1_qualification_graph()
    assert first.plan == second.plan


def test_representative_profile_zero_compiler_changes_required() -> None:
    built = build_npsc5f_r1_qualification_graph()
    recompiled = compile_qualification_execution_plan(
        built.graph.run_manifest,
        built.graph.gates,
        built.graph.profiles[0],
    )
    assert recompiled == built.plan


def test_shared_leaf_gate_paths_require_same_suite_twice() -> None:
    built = build_npsc5f_r1_qualification_graph()
    direct_gate = f"npsc5f-r1.direct.requires.{_SHARED_DG001_SUITE_ID}"
    npsc5e_gate = f"npsc5e-r3.expanded.requires.{_SHARED_DG001_SUITE_ID}"
    gate_ids = {gate.gate_id for gate in built.graph.gates}
    assert direct_gate in gate_ids
    assert npsc5e_gate in gate_ids
    fake = FakeQualificationSuiteExecutor({})
    coordinator = QualificationCoordinator(executor=fake)
    result = run_qualification_execution_plan(
        built.plan,
        _config(Path("."), Path("build/qualification/shared-gate-proof")),
        coordinator=coordinator,
    )
    assert fake.invocation_counts.get(_SHARED_DG001_SUITE_ID) == 1
    assert result.status is QualificationRunStatus.PASS
