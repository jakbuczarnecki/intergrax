# © Artur Czarnecki. All rights reserved.

"""EE-B2 — shutdown phase contract under active worker fault."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy
from intergrax.contracts.execution_reliability import (
    EXECUTION_RUNTIME_SHUTDOWN_PHASE_ORDER,
    ExecutionRuntimeShutdownPhase,
)
from intergrax.runtime.execution.concurrent_execution_work import (
    ConcurrentExecutionWorkDisposition,
    execute_concurrent_execution_work_resilient,
)
from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort
from intergrax.runtime.execution.request import ExecutionRequest
from testing_support.chaos.barriers import PhaseGate
from testing_support.chaos.execution_ports import DeterministicWorkerFaultPort

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True, slots=True)
class _WorkResult:
    value: str


def _req(label: str) -> ExecutionRequest[str, _WorkResult]:
    return ExecutionRequest(input=label, output_type=_WorkResult)


def test_ee_b2_shutdown_phase_order_canonical() -> None:
    assert EXECUTION_RUNTIME_SHUTDOWN_PHASE_ORDER[0] is (
        ExecutionRuntimeShutdownPhase.STOP_ACCEPTING_NEW_WORK
    )
    assert EXECUTION_RUNTIME_SHUTDOWN_PHASE_ORDER[-1] is (
        ExecutionRuntimeShutdownPhase.TERMINATE_WORKERS
    )


@pytest.mark.asyncio
async def test_ee_b2_drain_cancel_while_worker_fault_contained() -> None:
    """Simulate drain: cancel host while one worker would fail — no runtime crash."""
    gate = PhaseGate()

    class _DrainPort(ExecutionWorkPort[str, _WorkResult, _WorkResult]):
        async def execute(
            self, request: ExecutionRequest[str, _WorkResult]
        ) -> _WorkResult:
            if request.input == "block":
                gate.mark_started(request.input)
                await gate.block()
            return _WorkResult(value=request.input)

    fault = DeterministicWorkerFaultPort(
        fail_labels=frozenset({"fail"}),
        succeed=lambda label: _WorkResult(value=label),
    )
    drain_task = asyncio.create_task(
        execute_concurrent_execution_work_resilient(
            _DrainPort(),
            (_req("block"),),
            policy=ConcurrentExecutionWorkPolicy(max_concurrency=1),
        ),
    )
    await gate.wait_until_started(frozenset({"block"}))
    drain_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain_task
    outcomes = await execute_concurrent_execution_work_resilient(
        fault,
        (_req("fail"), _req("ok")),
        policy=ConcurrentExecutionWorkPolicy(max_concurrency=2),
    )
    assert outcomes[0].disposition is ConcurrentExecutionWorkDisposition.FAILED
    assert outcomes[1].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
