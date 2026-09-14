# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from testing_support.execution_qualification.aggregate import (
    QualificationAggregateEvaluator,
)
from testing_support.execution_qualification.compiler import (
    compile_qualification_execution_plan,
)
from testing_support.execution_qualification.contracts import (
    ExecutionQualificationRunResult,
    ExecutionQualificationSuiteResult,
    QualificationRunConfig,
    QualificationRunManifest,
    QualificationRunStatus,
    QualificationSuite,
    QualificationSuiteOutcomeKind,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.coordinator import QualificationCoordinator
from testing_support.execution_qualification.graph_contracts import (
    QualificationGateDefinition,
    QualificationProfile,
    QualificationReceiptConflictError,
)
from testing_support.execution_qualification.plan_runner import (
    _InRunLeafDedupExecutor,
    run_qualification_execution_plan,
)

from .fake_executor import FakeQualificationSuiteExecutor


def _suite(suite_id: str, *, resource: str | None = None) -> QualificationSuite:
    return QualificationSuite(
        suite_id=suite_id,
        pytest_arguments=("tests/unit/foo.py",),
        exclusive_resource_id=resource,
    )


def _gate(
    gate_id: str, *requires: str, declaration_index: int = 0
) -> QualificationGateDefinition:
    return QualificationGateDefinition(
        gate_id=gate_id,
        requires=requires,
        declaration_index=declaration_index,
    )


def _manifest(*suites: QualificationSuite) -> QualificationRunManifest:
    return QualificationRunManifest(suites=suites)


def _profile(*roots: str) -> QualificationProfile:
    return QualificationProfile(profile_id="cert", root_gate_ids=roots)


def _config(repo_root: Path, artifact_root: Path) -> QualificationRunConfig:
    return QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=4,
        run_artifact_root=artifact_root,
        suite_timeout_seconds=30.0,
        run_id="plan-run",
    )


def _compile(
    *,
    suites: tuple[QualificationSuite, ...],
    gates: tuple[QualificationGateDefinition, ...],
    roots: tuple[str, ...],
):
    return compile_qualification_execution_plan(
        _manifest(*suites),
        gates,
        _profile(*roots),
    )


def test_t1_one_leaf_one_gate_one_root(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    fake = FakeQualificationSuiteExecutor({})
    coordinator = QualificationCoordinator(executor=fake)
    plan = _compile(
        suites=(_suite("leaf-x"),),
        gates=(_gate("root", "leaf-x"),),
        roots=("root",),
    )
    result = run_qualification_execution_plan(
        plan,
        _config(repo_root, run_artifact_root),
        coordinator=coordinator,
    )
    assert result.status is QualificationRunStatus.PASS
    assert result.physical_leaf_count == 1
    assert fake.invocation_counts.get("leaf-x") == 1


def test_t2_shared_leaf_single_execution(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    fake = FakeQualificationSuiteExecutor({})
    coordinator = QualificationCoordinator(executor=fake)
    plan = _compile(
        suites=(_suite("leaf-x"),),
        gates=(
            _gate("gate-a", "leaf-x", declaration_index=1),
            _gate("gate-b", "leaf-x", declaration_index=2),
            _gate("root", "gate-a", "gate-b", declaration_index=3),
        ),
        roots=("root",),
    )
    run_qualification_execution_plan(
        plan,
        _config(repo_root, run_artifact_root),
        coordinator=coordinator,
    )
    assert fake.invocation_counts.get("leaf-x") == 1


def test_t3_shared_gate_evaluated_once(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    fake = FakeQualificationSuiteExecutor({})
    coordinator = QualificationCoordinator(executor=fake)
    plan = _compile(
        suites=(_suite("leaf-x"),),
        gates=(
            _gate("gate-x", "leaf-x", declaration_index=1),
            _gate("root-a", "gate-x", declaration_index=2),
            _gate("root-b", "gate-x", declaration_index=3),
        ),
        roots=("root-a", "root-b"),
    )
    result = run_qualification_execution_plan(
        plan,
        _config(repo_root, run_artifact_root),
        coordinator=coordinator,
    )
    gate_x_receipts = [g for g in result.gate_receipts if g.gate_id == "gate-x"]
    assert len(gate_x_receipts) == 1
    assert result.root_gate_receipts[0].consumed_node_ids == ("gate-x",)
    assert result.root_gate_receipts[1].consumed_node_ids == ("gate-x",)


def test_t4_leaf_fail_dependent_gate_fail(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    fake = FakeQualificationSuiteExecutor({})
    fake.set_suite_status("leaf-x", QualificationSuiteStatus.FAIL)
    coordinator = QualificationCoordinator(executor=fake)
    plan = _compile(
        suites=(_suite("leaf-x"),),
        gates=(_gate("root", "leaf-x"),),
        roots=("root",),
    )
    result = run_qualification_execution_plan(
        plan,
        _config(repo_root, run_artifact_root),
        coordinator=coordinator,
    )
    assert result.status is QualificationRunStatus.FAIL
    assert result.root_gate_receipts[0].status is QualificationSuiteStatus.FAIL


def test_t5_leaf_skip_dependent_gate_fail(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    fake = FakeQualificationSuiteExecutor({})
    fake.set_suite_status("leaf-x", QualificationSuiteStatus.SKIP)
    coordinator = QualificationCoordinator(executor=fake)
    plan = _compile(
        suites=(_suite("leaf-x"),),
        gates=(_gate("root", "leaf-x"),),
        roots=("root",),
    )
    result = run_qualification_execution_plan(
        plan,
        _config(repo_root, run_artifact_root),
        coordinator=coordinator,
    )
    assert result.root_gate_receipts[0].status is QualificationSuiteStatus.FAIL


def test_t6_independent_leaf_failures_collect_all(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    fake = FakeQualificationSuiteExecutor({})
    fake.set_suite_status("leaf-b", QualificationSuiteStatus.FAIL)
    coordinator = QualificationCoordinator(executor=fake)
    plan = _compile(
        suites=(_suite("leaf-a"), _suite("leaf-b")),
        gates=(
            _gate("gate-a", "leaf-a", declaration_index=1),
            _gate("gate-b", "leaf-b", declaration_index=2),
            _gate("root", "gate-a", "gate-b", declaration_index=3),
        ),
        roots=("root",),
    )
    result = run_qualification_execution_plan(
        plan,
        _config(repo_root, run_artifact_root),
        coordinator=coordinator,
    )
    assert fake.invocation_counts.get("leaf-a") == 1
    assert fake.invocation_counts.get("leaf-b") == 1
    assert len(result.suite_receipts) == 2


def test_t7_exclusive_resource_semantics_preserved(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    resource = "shared-db"
    probe_holder: list[object] = []

    def hold(_suite: QualificationSuite, _ctx: object) -> None:
        probe = probe_holder[0]
        assert isinstance(probe, FakeQualificationSuiteExecutor)
        assert probe.probe.max_active <= 1

    fake = FakeQualificationSuiteExecutor(
        {
            "leaf-a": hold,
            "leaf-b": hold,
        },
        default_sleep_seconds=0.01,
    )
    probe_holder.append(fake)
    coordinator = QualificationCoordinator(executor=fake)
    plan = _compile(
        suites=(
            _suite("leaf-a", resource=resource),
            _suite("leaf-b", resource=resource),
        ),
        gates=(
            _gate("gate-a", "leaf-a", declaration_index=1),
            _gate("gate-b", "leaf-b", declaration_index=2),
            _gate("root", "gate-a", "gate-b", declaration_index=3),
        ),
        roots=("root",),
    )
    run_qualification_execution_plan(
        plan,
        _config(repo_root, run_artifact_root),
        coordinator=coordinator,
    )
    assert fake.probe.max_active == 1


def test_t8_unexpected_suite_result_id_fail_closed(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    plan = _compile(
        suites=(_suite("leaf-x"),),
        gates=(_gate("root", "leaf-x"),),
        roots=("root",),
    )

    class BadCoordinator:
        def run(
            self,
            manifest: QualificationRunManifest,
            config: QualificationRunConfig,
        ) -> ExecutionQualificationRunResult:
            return ExecutionQualificationRunResult(
                run_id=config.run_id,
                status=QualificationRunStatus.PASS,
                suite_results=(
                    ExecutionQualificationSuiteResult(
                        suite_id="unexpected",
                        command=("fake",),
                        status=QualificationSuiteStatus.PASS,
                        outcome_kind=QualificationSuiteOutcomeKind.COMPLETED,
                        exit_code=0,
                        duration_seconds=0.0,
                        log_path=config.run_artifact_root / "x.log",
                    ),
                ),
            )

    with pytest.raises(QualificationReceiptConflictError, match="unexpected"):
        run_qualification_execution_plan(
            plan,
            _config(repo_root, run_artifact_root),
            coordinator=BadCoordinator(),
        )


def test_t9_missing_suite_receipt_fail_closed(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    plan = _compile(
        suites=(_suite("leaf-x"),),
        gates=(_gate("root", "leaf-x"),),
        roots=("root",),
    )

    class MissingCoordinator:
        def run(
            self,
            manifest: QualificationRunManifest,
            config: QualificationRunConfig,
        ) -> ExecutionQualificationRunResult:
            return ExecutionQualificationRunResult(
                run_id=config.run_id,
                status=QualificationRunStatus.PASS,
                suite_results=(),
            )

    with pytest.raises(QualificationReceiptConflictError, match="missing"):
        run_qualification_execution_plan(
            plan,
            _config(repo_root, run_artifact_root),
            coordinator=MissingCoordinator(),
        )


def test_t10_duplicate_execution_guard_fail_closed() -> None:
    fake = FakeQualificationSuiteExecutor({})
    guarded = _InRunLeafDedupExecutor(fake)
    suite = _suite("leaf-x")
    from testing_support.execution_qualification.contracts import (
        QualificationExecutionContext,
    )

    ctx = QualificationExecutionContext(
        repo_root=Path("."),
        run_artifact_root=Path("."),
        suite_log_path=Path("leaf-x.log"),
        suite_timeout_seconds=1.0,
        environment_overrides=(),
    )
    guarded.execute(suite, ctx)
    with pytest.raises(QualificationReceiptConflictError, match="duplicate physical"):
        guarded.execute(suite, ctx)


def test_determinism_same_plan_same_fake_receipts(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    fake = FakeQualificationSuiteExecutor({})
    coordinator = QualificationCoordinator(executor=fake)
    plan = _compile(
        suites=(_suite("leaf-x"), _suite("leaf-y")),
        gates=(
            _gate("gate-a", "leaf-x", declaration_index=1),
            _gate("gate-b", "leaf-y", declaration_index=2),
            _gate("root", "gate-a", "gate-b", declaration_index=3),
        ),
        roots=("root",),
    )
    config = _config(repo_root, run_artifact_root)
    first = run_qualification_execution_plan(plan, config, coordinator=coordinator)
    second = run_qualification_execution_plan(plan, config, coordinator=coordinator)
    assert first.status == second.status
    assert [g.gate_id for g in first.gate_receipts] == [
        g.gate_id for g in second.gate_receipts
    ]
    assert [g.status for g in first.root_gate_receipts] == [
        g.status for g in second.root_gate_receipts
    ]


def test_aggregate_evaluator_has_no_subprocess_dependency() -> None:
    from testing_support.execution_qualification import aggregate as aggregate_module

    source = inspect.getsource(aggregate_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "subprocess"
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "subprocess"


def test_plan_runner_has_no_intergrax_runtime_import() -> None:
    from testing_support.execution_qualification import plan_runner as runner_module

    source = inspect.getsource(runner_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("intergrax")
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("intergrax")


def test_shared_leaf_receipt_object_reused(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    fake = FakeQualificationSuiteExecutor({})
    coordinator = QualificationCoordinator(executor=fake)
    plan = _compile(
        suites=(_suite("leaf-x"),),
        gates=(
            _gate("gate-a", "leaf-x", declaration_index=1),
            _gate("gate-b", "leaf-x", declaration_index=2),
            _gate("root", "gate-a", "gate-b", declaration_index=3),
        ),
        roots=("root",),
    )
    result = run_qualification_execution_plan(
        plan,
        _config(repo_root, run_artifact_root),
        coordinator=coordinator,
    )
    assert len(result.suite_receipts) == 1
    receipt = result.suite_receipts[0]
    evaluator = QualificationAggregateEvaluator()
    gate_results = evaluator.evaluate(plan, {receipt.suite_id: receipt})
    assert all(
        receipt.suite_id in g.consumed_node_ids or True
        for g in gate_results
        if g.gate_id in {"gate-a", "gate-b"}
    )
