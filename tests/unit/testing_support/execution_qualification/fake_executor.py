# © Artur Czarnecki. All rights reserved.

"""Deterministic fake executor for coordinator unit tests."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from testing_support.execution_qualification.contracts import (
    ExecutionQualificationSuiteResult,
    QualificationExecutionContext,
    QualificationSuite,
    QualificationSuiteOutcomeKind,
    QualificationSuiteStatus,
)


@dataclass
class FakeExecutorProbe:
    active: int = 0
    max_active: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    overlap_pairs: list[tuple[str, str]] = field(default_factory=list)
    active_suite_ids: set[str] = field(default_factory=set)
    env_snapshots: list[tuple[str, tuple[tuple[str, str], ...]]] = field(default_factory=list)

    def enter(self, suite_id: str) -> None:
        with self.lock:
            for other in self.active_suite_ids:
                self.overlap_pairs.append((other, suite_id))
            self.active_suite_ids.add(suite_id)
            self.active += 1
            if self.active > self.max_active:
                self.max_active = self.active

    def leave(self, suite_id: str) -> None:
        with self.lock:
            self.active_suite_ids.discard(suite_id)
            self.active -= 1


class FakeQualificationSuiteExecutor:
    def __init__(
        self,
        behaviors: dict[str, Callable[[QualificationSuite, QualificationExecutionContext], None]],
        *,
        probe: FakeExecutorProbe | None = None,
        default_sleep_seconds: float = 0.0,
    ) -> None:
        self._behaviors = behaviors
        self._probe = probe if probe is not None else FakeExecutorProbe()
        self._default_sleep_seconds = default_sleep_seconds
        self.completion_order: list[str] = []

    @property
    def probe(self) -> FakeExecutorProbe:
        return self._probe

    def execute(
        self,
        suite: QualificationSuite,
        context: QualificationExecutionContext,
    ) -> ExecutionQualificationSuiteResult:
        self._probe.enter(suite.suite_id)
        started = time.monotonic()
        try:
            self._probe.env_snapshots.append(
                (suite.suite_id, context.environment_overrides),
            )
            behavior = self._behaviors.get(suite.suite_id)
            if behavior is not None:
                behavior(suite, context)
            elif self._default_sleep_seconds > 0:
                time.sleep(self._default_sleep_seconds)
        finally:
            self._probe.leave(suite.suite_id)
        self.completion_order.append(suite.suite_id)
        duration = time.monotonic() - started
        context.suite_log_path.parent.mkdir(parents=True, exist_ok=True)
        context.suite_log_path.write_text("", encoding="utf-8")
        return ExecutionQualificationSuiteResult(
            suite_id=suite.suite_id,
            command=("fake", suite.suite_id),
            status=QualificationSuiteStatus.PASS,
            outcome_kind=QualificationSuiteOutcomeKind.COMPLETED,
            exit_code=0,
            duration_seconds=duration,
            log_path=context.suite_log_path,
        )
