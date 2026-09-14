# © Artur Czarnecki. All rights reserved.

"""EE-B1.3 — timeout containment per worker without sibling cascade."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy
from intergrax.runtime.execution.concurrent_execution_work import (
    ConcurrentExecutionWorkDisposition,
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


class _TimeoutPort(ExecutionWorkPort[str, _WorkResult, _WorkResult]):
    async def execute(self, request: ExecutionRequest[str, _WorkResult]) -> _WorkResult:
        if request.input == "slow":
            await asyncio.sleep(10.0)
        return _WorkResult(value=request.input)


@pytest.mark.asyncio
async def test_ee_b1_3_timeout_isolated_sibling_unaffected() -> None:
    class _BoundedPort(ExecutionWorkPort[str, _WorkResult, _WorkResult]):
        async def execute(
            self, request: ExecutionRequest[str, _WorkResult]
        ) -> _WorkResult:
            if request.input == "slow":
                try:
                    await asyncio.wait_for(asyncio.sleep(10.0), timeout=0.05)
                except TimeoutError as exc:
                    raise TimeoutError("worker_operation_timeout") from exc
            return _WorkResult(value=request.input)

    outcomes = await execute_concurrent_execution_work_resilient(
        _BoundedPort(),
        (_req("slow"), _req("fast")),
        policy=_POLICY,
    )
    assert outcomes[0].disposition is ConcurrentExecutionWorkDisposition.FAILED
    assert isinstance(outcomes[0].error, TimeoutError)
    assert outcomes[1].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
    assert outcomes[1].result is not None
    assert outcomes[1].result.value == "fast"
