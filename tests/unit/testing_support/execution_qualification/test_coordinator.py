# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from testing_support.execution_qualification.contracts import (
    ExecutionQualificationSuiteResult,
    QualificationRunConfig,
    QualificationRunManifest,
    QualificationRunStatus,
    QualificationSuite,
    QualificationSuiteOutcomeKind,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.coordinator import QualificationCoordinator
from .fake_executor import FakeExecutorProbe, FakeQualificationSuiteExecutor


def _config(repo_root: Path, artifact: Path, *, max_parallel: int = 2) -> QualificationRunConfig:
    return QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=max_parallel,
        run_artifact_root=artifact,
        suite_timeout_seconds=120.0,
        run_id="coord-run",
    )


def test_result_order_follows_manifest_not_completion_order(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    gate_c = threading.Event()
    gate_a = threading.Event()

    def block_c(_suite: QualificationSuite, _ctx: object) -> None:
        gate_c.wait(timeout=5.0)

    def block_a(_suite: QualificationSuite, _ctx: object) -> None:
        gate_a.wait(timeout=5.0)

    executor = FakeQualificationSuiteExecutor(
        {
            "A": block_a,
            "C": block_c,
        },
    )
    manifest = QualificationRunManifest(
        suites=(
            QualificationSuite(suite_id="A", pytest_arguments=("a",)),
            QualificationSuite(suite_id="B", pytest_arguments=("b",)),
            QualificationSuite(suite_id="C", pytest_arguments=("c",)),
        ),
    )
    coordinator = QualificationCoordinator(executor=executor)

    def release() -> None:
        gate_c.set()
        gate_a.set()

    timer = threading.Timer(0.05, release)
    timer.start()
    result = coordinator.run(manifest, _config(repo_root, run_artifact_root))
    timer.cancel()

    assert [row.suite_id for row in result.suite_results] == ["A", "B", "C"]
    assert executor.completion_order[0] == "C" or executor.completion_order[0] == "B"


def test_bounded_concurrency_never_exceeds_max_parallel(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    probe = FakeExecutorProbe()
    hold = threading.Event()

    def hold_until_released(_suite: QualificationSuite, _ctx: object) -> None:
        hold.wait(timeout=5.0)

    suites = tuple(
        QualificationSuite(suite_id=f"S{i}", pytest_arguments=(f"{i}",))
        for i in range(4)
    )
    executor = FakeQualificationSuiteExecutor(
        {suite.suite_id: hold_until_released for suite in suites},
        probe=probe,
    )
    manifest = QualificationRunManifest(suites=suites)
    coordinator = QualificationCoordinator(executor=executor)

    def release() -> None:
        hold.set()

    timer = threading.Timer(0.1, release)
    timer.start()
    coordinator.run(manifest, _config(repo_root, run_artifact_root, max_parallel=2))
    timer.cancel()

    assert probe.max_active <= 2


def test_actual_parallel_overlap_with_synchronization(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    probe = FakeExecutorProbe()
    entered = threading.Barrier(2, timeout=5.0)

    def wait_barrier(_suite: QualificationSuite, _ctx: object) -> None:
        entered.wait(timeout=5.0)

    executor = FakeQualificationSuiteExecutor(
        {
            "P1": wait_barrier,
            "P2": wait_barrier,
        },
        probe=probe,
    )
    manifest = QualificationRunManifest(
        suites=(
            QualificationSuite(suite_id="P1", pytest_arguments=("1",)),
            QualificationSuite(suite_id="P2", pytest_arguments=("2",)),
        ),
    )
    QualificationCoordinator(executor=executor).run(
        manifest,
        _config(repo_root, run_artifact_root, max_parallel=2),
    )
    assert probe.max_active >= 2


def test_exclusive_resource_suites_do_not_overlap(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    probe = FakeExecutorProbe()
    resource = ".tmp/session/example/cross.db"
    hold = threading.Event()

    def hold_exclusive(_suite: QualificationSuite, _ctx: object) -> None:
        hold.wait(timeout=5.0)

    executor = FakeQualificationSuiteExecutor(
        {
            "E1": hold_exclusive,
            "E2": hold_exclusive,
        },
        probe=probe,
    )
    manifest = QualificationRunManifest(
        suites=(
            QualificationSuite(
                suite_id="E1",
                pytest_arguments=("1",),
                exclusive_resource_id=resource,
            ),
            QualificationSuite(
                suite_id="E2",
                pytest_arguments=("2",),
                exclusive_resource_id=resource,
            ),
            QualificationSuite(suite_id="free", pytest_arguments=("3",)),
        ),
    )
    coordinator = QualificationCoordinator(executor=executor)

    def release() -> None:
        hold.set()

    timer = threading.Timer(0.05, release)
    timer.start()
    coordinator.run(manifest, _config(repo_root, run_artifact_root, max_parallel=3))
    timer.cancel()

    conflicting = {
        pair
        for pair in probe.overlap_pairs
        if pair[0] in {"E1", "E2"} and pair[1] in {"E1", "E2"}
    }
    assert not conflicting


def test_collect_all_runs_remaining_after_failure(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    class FailFirstExecutor:
        def __init__(self) -> None:
            self.executed: list[str] = []

        def execute(
            self,
            suite: QualificationSuite,
            context: object,
        ) -> ExecutionQualificationSuiteResult:
            self.executed.append(suite.suite_id)
            status = (
                QualificationSuiteStatus.FAIL
                if suite.suite_id == "A"
                else QualificationSuiteStatus.PASS
            )
            outcome = (
                QualificationSuiteOutcomeKind.PYTEST_NONZERO_EXIT
                if suite.suite_id == "A"
                else QualificationSuiteOutcomeKind.COMPLETED
            )
            return ExecutionQualificationSuiteResult(
                suite_id=suite.suite_id,
                command=("fake",),
                status=status,
                outcome_kind=outcome,
                exit_code=1 if suite.suite_id == "A" else 0,
                duration_seconds=0.01,
                log_path=run_artifact_root / f"{suite.suite_id}.log",
            )

    executor = FailFirstExecutor()
    manifest = QualificationRunManifest(
        suites=(
            QualificationSuite(suite_id="A", pytest_arguments=("a",)),
            QualificationSuite(suite_id="B", pytest_arguments=("b",)),
            QualificationSuite(suite_id="C", pytest_arguments=("c",)),
        ),
    )
    result = QualificationCoordinator(executor=executor).run(
        manifest,
        _config(repo_root, run_artifact_root),
    )
    assert set(executor.executed) == {"A", "B", "C"}
    assert result.status == QualificationRunStatus.FAIL


def test_separate_logs_created(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    manifest = QualificationRunManifest(
        suites=(
            QualificationSuite(suite_id="L1", pytest_arguments=("1",)),
            QualificationSuite(suite_id="L2", pytest_arguments=("2",)),
        ),
    )
    QualificationCoordinator(executor=FakeQualificationSuiteExecutor({})).run(
        manifest,
        _config(repo_root, run_artifact_root),
    )
    assert (run_artifact_root / "L1.log").is_file()
    assert (run_artifact_root / "L2.log").is_file()


def test_environment_override_recorded_per_suite(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    probe = FakeExecutorProbe()
    executor = FakeQualificationSuiteExecutor({}, probe=probe)
    manifest = QualificationRunManifest(
        suites=(
            QualificationSuite(
                suite_id="env-a",
                pytest_arguments=("a",),
                environment_overrides=(("QUAL_MARK", "alpha"),),
            ),
            QualificationSuite(
                suite_id="env-b",
                pytest_arguments=("b",),
                environment_overrides=(("QUAL_MARK", "beta"),),
            ),
        ),
    )
    QualificationCoordinator(executor=executor).run(
        manifest,
        _config(repo_root, run_artifact_root),
    )
    snapshots = {suite_id: overrides for suite_id, overrides in probe.env_snapshots}
    assert snapshots["env-a"] == (("QUAL_MARK", "alpha"),)
    assert snapshots["env-b"] == (("QUAL_MARK", "beta"),)
