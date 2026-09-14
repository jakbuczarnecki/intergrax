# © Artur Czarnecki. All rights reserved.

"""EE-B1.3 — failure storm: mixed outcomes, per-unit truth, no global crash."""

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

_N = 10
_POLICY = ConcurrentExecutionWorkPolicy(max_concurrency=4)


@dataclass(frozen=True, slots=True)
class _WorkResult:
    value: str


def _req(label: str) -> ExecutionRequest[str, _WorkResult]:
    return ExecutionRequest(input=label, output_type=_WorkResult)


class _StormPort(ExecutionWorkPort[str, _WorkResult, _WorkResult]):
    async def execute(self, request: ExecutionRequest[str, _WorkResult]) -> _WorkResult:
        label = request.input
        idx = int(label.split("-", 1)[1])
        if idx in {0, 1, 2}:
            raise RuntimeError(f"fail:{label}")
        if idx in {3, 4}:
            try:
                await asyncio.wait_for(asyncio.sleep(10.0), timeout=0.02)
            except TimeoutError as exc:
                raise TimeoutError(f"timeout:{label}") from exc
        if idx == 5:
            raise ValueError(f"abort:{label}")
        return _WorkResult(value=label)


@pytest.mark.asyncio
async def test_ee_b1_3_failure_storm_preserves_per_unit_truth() -> None:
    requests = tuple(_req(f"u-{i}") for i in range(_N))
    outcomes = await execute_concurrent_execution_work_resilient(
        _StormPort(),
        requests,
        policy=_POLICY,
    )
    assert len(outcomes) == _N
    failed = sum(
        1
        for o in outcomes
        if o.disposition is ConcurrentExecutionWorkDisposition.FAILED
    )
    succeeded = sum(
        1
        for o in outcomes
        if o.disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
    )
    assert failed == 6
    assert succeeded == 4
    for i in (6, 7, 8, 9):
        assert outcomes[i].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
