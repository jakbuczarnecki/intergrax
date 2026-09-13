# © Artur Czarnecki. All rights reserved.

"""EE-B2 — dependency failure and timeout fault injection via execution work port."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy
from intergrax.contracts.execution_reliability import ExecutionFailureSemanticCategory
from intergrax.runtime.execution.concurrent_execution_work import (
    ConcurrentExecutionWorkDisposition,
    execute_concurrent_execution_work_resilient,
)
from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort
from intergrax.runtime.execution.request import ExecutionRequest
from testing_support.chaos.execution_ports import (
    DeterministicDependencyFaultPort,
    classify_dependency_failure,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True, slots=True)
class _WorkResult:
    value: str


def _req(label: str) -> ExecutionRequest[str, _WorkResult]:
    return ExecutionRequest(input=label, output_type=_WorkResult)


@pytest.mark.asyncio
async def test_ee_b2_dependency_failure_typed_at_classifier() -> None:
    assert (
        classify_dependency_failure()
        is ExecutionFailureSemanticCategory.DEPENDENCY_FAILURE
    )


@pytest.mark.asyncio
async def test_ee_b2_dependency_port_failure_isolated() -> None:
    port: ExecutionWorkPort[str, _WorkResult, _WorkResult] = (
        DeterministicDependencyFaultPort(
            fail_label="dep",
            succeed=lambda label: _WorkResult(value=label),
        )
    )
    outcomes = await execute_concurrent_execution_work_resilient(
        port,
        (_req("dep"), _req("ok")),
        policy=ConcurrentExecutionWorkPolicy(max_concurrency=2),
    )
    assert outcomes[0].disposition is ConcurrentExecutionWorkDisposition.FAILED
    assert isinstance(outcomes[0].error, ConnectionError)
    assert outcomes[1].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED


class _TimeoutPort(ExecutionWorkPort[str, _WorkResult, _WorkResult]):
    async def execute(self, request: ExecutionRequest[str, _WorkResult]) -> _WorkResult:
        if request.input == "slow":
            try:
                await asyncio.wait_for(asyncio.sleep(10.0), timeout=0.05)
            except TimeoutError as exc:
                raise TimeoutError("dependency_timeout") from exc
        return _WorkResult(value=request.input)


@pytest.mark.asyncio
async def test_ee_b2_dependency_timeout_bounded_and_isolated() -> None:
    outcomes = await execute_concurrent_execution_work_resilient(
        _TimeoutPort(),
        (_req("slow"), _req("fast")),
        policy=ConcurrentExecutionWorkPolicy(max_concurrency=2),
    )
    assert outcomes[0].disposition is ConcurrentExecutionWorkDisposition.FAILED
    assert isinstance(outcomes[0].error, TimeoutError)
    assert outcomes[1].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
