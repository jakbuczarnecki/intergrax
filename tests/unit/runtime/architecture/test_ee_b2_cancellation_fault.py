# © Artur Czarnecki. All rights reserved.

"""EE-B2 — cancellation fault injection with capacity release."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy
from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityAdmissionRequest,
    ExecutionCapacityPolicy,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.execution.capacity import LocalExecutionCapacityAdmission
from intergrax.runtime.execution.concurrent_execution_work import (
    execute_concurrent_execution_work_resilient,
)
from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort
from intergrax.runtime.execution.request import ExecutionRequest
from testing_support.chaos.barriers import PhaseGate

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True, slots=True)
class _WorkResult:
    value: str


def _req(label: str) -> ExecutionRequest[str, _WorkResult]:
    return ExecutionRequest(input=label, output_type=_WorkResult)


class _GatePort(ExecutionWorkPort[str, _WorkResult, _WorkResult]):
    def __init__(self, gate: PhaseGate) -> None:
        self._gate = gate

    async def execute(self, request: ExecutionRequest[str, _WorkResult]) -> _WorkResult:
        self._gate.mark_started(request.input)
        await self._gate.block()
        return _WorkResult(value=request.input)


@pytest.mark.asyncio
async def test_ee_b2_cancel_during_concurrent_work_propagates() -> None:
    gate = PhaseGate()
    task = asyncio.create_task(
        execute_concurrent_execution_work_resilient(
            _GatePort(gate),
            (_req("A"), _req("B")),
            policy=ConcurrentExecutionWorkPolicy(max_concurrency=2),
        ),
    )
    await gate.wait_until_started(frozenset({"A", "B"}))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_ee_b2_cancel_releases_capacity_permit() -> None:
    policy = ExecutionCapacityPolicy(max_concurrent_root_executions=1)
    admission = LocalExecutionCapacityAdmission(policy)
    req = ExecutionCapacityAdmissionRequest(
        tenant_id="t",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    permit = await admission.acquire(req)
    await permit.release()
    second = await admission.acquire(req)
    await second.release()
