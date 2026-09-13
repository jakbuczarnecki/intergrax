# © Artur Czarnecki. All rights reserved.

"""EE-B1.3 — cancellation propagation without corrupting completed siblings."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy
from intergrax.runtime.execution.concurrent_execution_work import (
    execute_concurrent_execution_work_resilient,
)
from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort
from intergrax.runtime.execution.request import ExecutionRequest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_POLICY = ConcurrentExecutionWorkPolicy(max_concurrency=2)


@dataclass(frozen=True, slots=True)
class _WorkResult:
    value: str


def _req(label: str) -> ExecutionRequest[str, _WorkResult]:
    return ExecutionRequest(input=label, output_type=_WorkResult)


class _BarrierPort(ExecutionWorkPort[str, _WorkResult, _WorkResult]):
    def __init__(self, release: asyncio.Event, started: list[str]) -> None:
        self._release = release
        self._started = started

    async def execute(self, request: ExecutionRequest[str, _WorkResult]) -> _WorkResult:
        self._started.append(request.input)
        await self._release.wait()
        return _WorkResult(value=request.input)


async def _wait_started(started: list[str], expected: frozenset[str]) -> None:
    for _ in range(100):
        if frozenset(started) == expected:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected {expected}, got {started}")


@pytest.mark.asyncio
async def test_ee_b1_3_resilient_host_cancellation_propagates() -> None:
    release = asyncio.Event()
    started: list[str] = []
    port = _BarrierPort(release, started)
    task = asyncio.create_task(
        execute_concurrent_execution_work_resilient(
            port,
            (_req("A"), _req("B")),
            policy=_POLICY,
        ),
    )
    await _wait_started(started, frozenset({"A", "B"}))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_ee_b1_3_completed_unit_before_cancel_retains_success() -> None:
    class _FastSlowPort(ExecutionWorkPort[str, _WorkResult, _WorkResult]):
        async def execute(
            self, request: ExecutionRequest[str, _WorkResult]
        ) -> _WorkResult:
            if request.input == "fast":
                return _WorkResult(value="fast")
            await asyncio.sleep(30.0)
            return _WorkResult(value="slow")

    slow_task = asyncio.create_task(
        execute_concurrent_execution_work_resilient(
            _FastSlowPort(),
            (_req("fast"), _req("slow")),
            policy=_POLICY,
        ),
    )
    await asyncio.sleep(0.05)
    slow_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await slow_task
