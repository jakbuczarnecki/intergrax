# © Artur Czarnecki. All rights reserved.

"""EE-B2 — worker failure fault injection (L1)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy
from intergrax.runtime.execution.concurrent_execution_work import (
    ConcurrentExecutionWorkDisposition,
    execute_concurrent_execution_work,
    execute_concurrent_execution_work_resilient,
)
from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort
from intergrax.runtime.execution.request import ExecutionRequest
from testing_support.chaos.execution_ports import (
    DeterministicWorkerFaultPort,
    InvocationCounterPort,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_POLICY = ConcurrentExecutionWorkPolicy(max_concurrency=3)


@dataclass(frozen=True, slots=True)
class _WorkResult:
    value: str


def _req(label: str) -> ExecutionRequest[str, _WorkResult]:
    return ExecutionRequest(input=label, output_type=_WorkResult)


def _ok(label: str) -> _WorkResult:
    return _WorkResult(value=label)


@pytest.mark.asyncio
async def test_ee_b2_worker_failure_siblings_continue_resilient() -> None:
    port: ExecutionWorkPort[str, _WorkResult, _WorkResult] = (
        DeterministicWorkerFaultPort(
            fail_labels=frozenset({"A"}),
            succeed=_ok,
        )
    )
    outcomes = await execute_concurrent_execution_work_resilient(
        port,
        (_req("A"), _req("B"), _req("C")),
        policy=_POLICY,
    )
    assert outcomes[0].disposition is ConcurrentExecutionWorkDisposition.FAILED
    assert outcomes[1].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
    assert outcomes[2].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED


@pytest.mark.asyncio
async def test_ee_b2_worker_failure_strict_fail_fast() -> None:
    port: ExecutionWorkPort[str, _WorkResult, _WorkResult] = (
        DeterministicWorkerFaultPort(
            fail_labels=frozenset({"B"}),
            succeed=_ok,
        )
    )
    with pytest.raises(RuntimeError, match="ee_b2_worker_fault"):
        await execute_concurrent_execution_work(
            port,
            (_req("A"), _req("B"), _req("C")),
            policy=_POLICY,
        )


@pytest.mark.asyncio
async def test_ee_b2_worker_fault_no_duplicate_invocation() -> None:
    counter: ExecutionWorkPort[str, _WorkResult, _WorkResult] = InvocationCounterPort(
        succeed=_ok,
    )
    await execute_concurrent_execution_work_resilient(
        counter,
        (_req("x"), _req("y")),
        policy=_POLICY,
    )
    assert counter.counts == {"x": 1, "y": 1}
