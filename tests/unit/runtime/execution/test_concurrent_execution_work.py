# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from intergrax.runtime.execution.concurrent_execution_work import (
    ConcurrentExecutionWorkDisposition,
    execute_concurrent_execution_work,
    execute_concurrent_execution_work_resilient,
)
from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort
from intergrax.runtime.execution.request import ExecutionRequest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True, slots=True)
class WorkResult:
    value: str


def _request(label: str) -> ExecutionRequest[str, WorkResult]:
    return ExecutionRequest(input=label, output_type=WorkResult)


class BarrierWorkPort(ExecutionWorkPort[str, WorkResult, WorkResult]):
    def __init__(
        self,
        *,
        release: asyncio.Event,
        started: list[str],
        fail_labels: frozenset[str] = frozenset(),
        completion_delays: dict[str, float] | None = None,
    ) -> None:
        self._release = release
        self._started = started
        self._fail_labels = fail_labels
        self._completion_delays = completion_delays or {}

    async def execute(
        self,
        request: ExecutionRequest[str, WorkResult],
    ) -> WorkResult:
        label = request.input
        self._started.append(label)
        await self._release.wait()
        delay = self._completion_delays.get(label, 0.0)
        if delay > 0.0:
            await asyncio.sleep(delay)
        if label in self._fail_labels:
            raise RuntimeError(f"failed: {label}")
        return WorkResult(value=label)


class ImmediateWorkPort(ExecutionWorkPort[str, WorkResult, WorkResult]):
    def __init__(self, *, fail_labels: frozenset[str] = frozenset()) -> None:
        self._fail_labels = fail_labels

    async def execute(
        self,
        request: ExecutionRequest[str, WorkResult],
    ) -> WorkResult:
        label = request.input
        if label in self._fail_labels:
            raise RuntimeError(f"failed: {label}")
        return WorkResult(value=label)


async def _wait_until_started(
    started: list[str],
    expected: frozenset[str],
) -> None:
    for _ in range(100):
        if frozenset(started) == expected:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected started={sorted(expected)}, got={started}")


@pytest.mark.asyncio
async def test_concurrent_execution_work_all_success() -> None:
    port = ImmediateWorkPort()
    results = await execute_concurrent_execution_work(
        port,
        (_request("a"), _request("b"), _request("c")),
    )
    assert tuple(result.value for result in results) == ("a", "b", "c")


@pytest.mark.asyncio
async def test_concurrent_execution_work_one_failure_raises() -> None:
    port = ImmediateWorkPort(fail_labels=frozenset({"b"}))
    with pytest.raises(RuntimeError, match="failed: b"):
        await execute_concurrent_execution_work(
            port,
            (_request("a"), _request("b"), _request("c")),
        )


@pytest.mark.asyncio
async def test_concurrent_execution_work_resilient_all_success() -> None:
    port = ImmediateWorkPort()
    outcomes = await execute_concurrent_execution_work_resilient(
        port,
        (_request("a"), _request("b"), _request("c")),
    )
    assert len(outcomes) == 3
    assert all(
        outcome.disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
        for outcome in outcomes
    )
    assert tuple(outcome.result.value for outcome in outcomes) == ("a", "b", "c")


@pytest.mark.asyncio
async def test_concurrent_execution_work_resilient_one_failure() -> None:
    port = ImmediateWorkPort(fail_labels=frozenset({"b"}))
    outcomes = await execute_concurrent_execution_work_resilient(
        port,
        (_request("a"), _request("b"), _request("c")),
    )
    assert outcomes[0].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
    assert outcomes[0].result is not None
    assert outcomes[0].result.value == "a"
    assert outcomes[1].disposition is ConcurrentExecutionWorkDisposition.FAILED
    assert outcomes[1].error is not None
    assert outcomes[2].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
    assert outcomes[2].result is not None
    assert outcomes[2].result.value == "c"


@pytest.mark.asyncio
async def test_concurrent_execution_work_resilient_multiple_failures() -> None:
    port = ImmediateWorkPort(fail_labels=frozenset({"a", "c"}))
    outcomes = await execute_concurrent_execution_work_resilient(
        port,
        (_request("a"), _request("b"), _request("c")),
    )
    assert outcomes[0].disposition is ConcurrentExecutionWorkDisposition.FAILED
    assert outcomes[1].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
    assert outcomes[1].result is not None
    assert outcomes[1].result.value == "b"
    assert outcomes[2].disposition is ConcurrentExecutionWorkDisposition.FAILED


@pytest.mark.asyncio
async def test_concurrent_execution_work_resilient_stable_ordering() -> None:
    release = asyncio.Event()
    started: list[str] = []
    port = BarrierWorkPort(
        release=release,
        started=started,
        completion_delays={"c": 0.0, "a": 0.01, "b": 0.02},
    )
    task = asyncio.create_task(
        execute_concurrent_execution_work_resilient(
            port,
            (_request("a"), _request("b"), _request("c")),
        ),
    )
    await _wait_until_started(started, frozenset({"a", "b", "c"}))
    release.set()
    outcomes = await task
    assert tuple(outcome.result.value for outcome in outcomes) == ("a", "b", "c")


@pytest.mark.asyncio
async def test_concurrent_execution_work_resilient_all_scheduled_concurrently() -> None:
    release = asyncio.Event()
    started: list[str] = []
    port = BarrierWorkPort(release=release, started=started)
    task = asyncio.create_task(
        execute_concurrent_execution_work_resilient(
            port,
            (_request("a"), _request("b"), _request("c")),
        ),
    )
    await _wait_until_started(started, frozenset({"a", "b", "c"}))
    release.set()
    outcomes = await task
    assert frozenset(started) == frozenset({"a", "b", "c"})
    assert len(outcomes) == 3


@pytest.mark.asyncio
async def test_concurrent_execution_work_resilient_cancellation_propagates() -> None:
    release = asyncio.Event()
    started: list[str] = []
    port = BarrierWorkPort(release=release, started=started)
    task = asyncio.create_task(
        execute_concurrent_execution_work_resilient(
            port,
            (_request("a"), _request("b")),
        ),
    )
    await _wait_until_started(started, frozenset({"a", "b"}))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
