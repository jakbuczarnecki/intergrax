# © Artur Czarnecki. All rights reserved.

"""Bounded parallel qualification coordinator."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from testing_support.execution_qualification.contracts import (
    ExecutionQualificationRunResult,
    ExecutionQualificationSuiteResult,
    QualificationCoordinatorError,
    QualificationExecutionContext,
    QualificationManifestError,
    QualificationRunConfig,
    QualificationRunManifest,
    QualificationRunStatus,
    QualificationSuite,
    QualificationSuiteOutcomeKind,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.executor import (
    PytestSubprocessSuiteExecutor,
    QualificationSuiteExecutor,
    suite_log_path,
)


def _aggregate_run_status(
    results: tuple[ExecutionQualificationSuiteResult, ...],
) -> QualificationRunStatus:
    if any(result.status == QualificationSuiteStatus.FAIL for result in results):
        return QualificationRunStatus.FAIL
    if any(result.status == QualificationSuiteStatus.SKIP for result in results):
        return QualificationRunStatus.FAIL
    return QualificationRunStatus.PASS


def _validate_executor_suite_result(
    suite: QualificationSuite,
    result: ExecutionQualificationSuiteResult,
) -> None:
    if result.suite_id != suite.suite_id:
        raise QualificationCoordinatorError(
            "executor result suite_id "
            f"{result.suite_id!r} does not match requested {suite.suite_id!r}"
        )


def _not_started_result(
    suite: QualificationSuite,
    log_path: Path,
) -> ExecutionQualificationSuiteResult:
    return ExecutionQualificationSuiteResult(
        suite_id=suite.suite_id,
        command=(),
        status=QualificationSuiteStatus.SKIP,
        outcome_kind=QualificationSuiteOutcomeKind.NOT_STARTED,
        exit_code=None,
        duration_seconds=0.0,
        log_path=log_path,
    )


class QualificationCoordinator:
    def __init__(
        self,
        executor: QualificationSuiteExecutor | None = None,
    ) -> None:
        self._executor = executor if executor is not None else PytestSubprocessSuiteExecutor()

    def run(
        self,
        manifest: QualificationRunManifest,
        config: QualificationRunConfig,
    ) -> ExecutionQualificationRunResult:
        try:
            config.run_artifact_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise QualificationCoordinatorError(
                f"cannot prepare run artifact area: {config.run_artifact_root}"
            ) from exc

        suites = manifest.suites
        result_slots: list[ExecutionQualificationSuiteResult | None] = [None] * len(suites)
        log_paths = [suite_log_path(config.run_artifact_root, suite.suite_id) for suite in suites]

        lock = threading.Lock()
        active_count = 0
        started_indices: set[int] = set()
        held_exclusive: set[str] = set()
        done_count = 0
        done_condition = threading.Condition(lock)
        infrastructure_failure: QualificationCoordinatorError | None = None

        def try_schedule(executor: ThreadPoolExecutor) -> None:
            nonlocal active_count, done_count, infrastructure_failure
            while True:
                with lock:
                    if infrastructure_failure is not None:
                        return
                    if active_count >= config.max_parallel:
                        return
                    chosen_index: int | None = None
                    for index, suite in enumerate(suites):
                        if index in started_indices:
                            continue
                        exclusive = suite.exclusive_resource_id
                        if exclusive is not None and exclusive in held_exclusive:
                            continue
                        chosen_index = index
                        break
                    if chosen_index is None:
                        return
                    suite = suites[chosen_index]
                    started_indices.add(chosen_index)
                    active_count += 1
                    if suite.exclusive_resource_id is not None:
                        held_exclusive.add(suite.exclusive_resource_id)

                index = chosen_index
                suite = suites[index]

                def run_suite(
                    suite_index: int = index,
                    current_suite: QualificationSuite = suite,
                ) -> ExecutionQualificationSuiteResult:
                    context = QualificationExecutionContext(
                        repo_root=config.repo_root,
                        run_artifact_root=config.run_artifact_root,
                        suite_log_path=log_paths[suite_index],
                        suite_timeout_seconds=config.suite_timeout_seconds,
                        environment_overrides=current_suite.environment_overrides,
                    )
                    return self._executor.execute(current_suite, context)

                future = executor.submit(run_suite)

                def on_done(
                    fut: Future[ExecutionQualificationSuiteResult],
                    slot_index: int = index,
                ) -> None:
                    nonlocal active_count, done_count, infrastructure_failure
                    slot_error: QualificationCoordinatorError | None = None
                    result: ExecutionQualificationSuiteResult | None = None
                    try:
                        completed = fut.result()
                        _validate_executor_suite_result(suites[slot_index], completed)
                        result = completed
                    except QualificationCoordinatorError as exc:
                        slot_error = exc
                    except Exception as exc:
                        slot_error = QualificationCoordinatorError(
                            f"unexpected executor failure for suite "
                            f"{suites[slot_index].suite_id!r}"
                        )
                        slot_error.__cause__ = exc

                    with done_condition:
                        if slot_error is not None:
                            infrastructure_failure = slot_error
                        elif result is not None:
                            result_slots[slot_index] = result
                        active_count -= 1
                        done_count += 1
                        exclusive = suites[slot_index].exclusive_resource_id
                        if exclusive is not None:
                            held_exclusive.discard(exclusive)
                        done_condition.notify_all()
                    if infrastructure_failure is None:
                        try_schedule(executor)

                future.add_done_callback(on_done)

        with ThreadPoolExecutor(max_workers=config.max_parallel) as pool:
            try_schedule(pool)
            with done_condition:
                while True:
                    if done_count >= len(suites):
                        break
                    if infrastructure_failure is not None and active_count == 0:
                        break
                    done_condition.wait()

        if infrastructure_failure is not None:
            raise infrastructure_failure

        final_results: list[ExecutionQualificationSuiteResult] = []
        for index, suite in enumerate(suites):
            slot = result_slots[index]
            if slot is None:
                final_results.append(_not_started_result(suite, log_paths[index]))
            else:
                final_results.append(slot)

        result_tuple = tuple(final_results)
        return ExecutionQualificationRunResult(
            run_id=config.run_id,
            status=_aggregate_run_status(result_tuple),
            suite_results=result_tuple,
        )


def validate_and_run(
    manifest: QualificationRunManifest,
    config: QualificationRunConfig,
    *,
    executor: QualificationSuiteExecutor | None = None,
) -> ExecutionQualificationRunResult:
    """Validate manifest/config then run. Raises ``QualificationManifestError`` before launch."""
    try:
        QualificationRunManifest(suites=manifest.suites)
        QualificationRunConfig(
            repo_root=config.repo_root,
            max_parallel=config.max_parallel,
            run_artifact_root=config.run_artifact_root,
            suite_timeout_seconds=config.suite_timeout_seconds,
            run_id=config.run_id,
        )
    except ValueError as exc:
        raise QualificationManifestError(str(exc)) from exc

    coordinator = QualificationCoordinator(executor=executor)
    return coordinator.run(manifest, config)
