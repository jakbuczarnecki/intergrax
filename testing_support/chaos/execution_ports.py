# © Artur Czarnecki. All rights reserved.

"""ExecutionWorkPort fault adapters (deterministic, port-based)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Generic, TypeVar

from intergrax.contracts.execution_reliability import (
    ExecutionFailureContext,
    ExecutionFailureSemanticCategory,
)
from intergrax.contracts.resilience_policy import FailureClass
from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort
from intergrax.runtime.execution.reliability import default_execution_failure_classifier
from intergrax.runtime.execution.request import ExecutionRequest

InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")


class DeterministicWorkerFaultPort(
    ExecutionWorkPort[str, str, ResultT],
    Generic[ResultT],
):
    """Fail selected labels with a deterministic exception."""

    def __init__(
        self,
        *,
        fail_labels: frozenset[str],
        succeed: Callable[[str], ResultT],
        message: str = "ee_b2_worker_fault",
    ) -> None:
        self._fail_labels = fail_labels
        self._succeed = succeed
        self._message = message

    async def execute(self, request: ExecutionRequest[str, str]) -> ResultT:
        if request.input in self._fail_labels:
            raise RuntimeError(self._message)
        return self._succeed(request.input)


class DeterministicDependencyFaultPort(
    ExecutionWorkPort[str, str, ResultT],
    Generic[ResultT],
):
    """Simulate dependency outage via ConnectionError (provider-neutral at boundary)."""

    def __init__(
        self,
        *,
        fail_label: str,
        succeed: Callable[[str], ResultT],
    ) -> None:
        self._fail_label = fail_label
        self._succeed = succeed

    async def execute(self, request: ExecutionRequest[str, str]) -> ResultT:
        if request.input == self._fail_label:
            raise ConnectionError("upstream_unavailable")
        return self._succeed(request.input)


class InvocationCounterPort(
    ExecutionWorkPort[str, str, ResultT],
    Generic[ResultT],
):
    """Count invocations per label; optional hang until gate releases."""

    def __init__(
        self,
        *,
        succeed: Callable[[str], ResultT],
        hang_label: str | None = None,
        hang_event: asyncio.Event | None = None,
    ) -> None:
        self.counts: dict[str, int] = {}
        self._succeed = succeed
        self._hang_label = hang_label
        self._hang_event = hang_event

    async def execute(self, request: ExecutionRequest[str, str]) -> ResultT:
        label = request.input
        self.counts[label] = self.counts.get(label, 0) + 1
        if self._hang_label == label and self._hang_event is not None:
            await self._hang_event.wait()
        return self._succeed(label)


def classify_dependency_failure() -> ExecutionFailureSemanticCategory:
    classifier = default_execution_failure_classifier()
    decision = classifier.classify(
        ExecutionFailureContext(
            failure_class=FailureClass.DEPENDENCY_ERROR,
            dependency_unavailable=True,
            reason="upstream_unavailable",
        ),
    )
    return decision.category
