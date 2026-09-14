# © Artur Czarnecki. All rights reserved.

"""EE-B2-FINAL — deterministic chaos repeatability (3x identical outcomes)."""

from __future__ import annotations

import pytest

from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy
from intergrax.runtime.execution.concurrent_execution_work import (
    ConcurrentExecutionWorkDisposition,
    execute_concurrent_execution_work_resilient,
)
from intergrax.runtime.execution.request import ExecutionRequest
from testing_support.chaos.execution_ports import DeterministicWorkerFaultPort

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_POLICY = ConcurrentExecutionWorkPolicy(max_concurrency=3)
_RUNS = 3


def _req(label: str) -> ExecutionRequest[str, str]:
    return ExecutionRequest(input=label, output_type=str)


def _ok(label: str) -> str:
    return label


async def _chaos_slice() -> tuple[str, ...]:
    port = DeterministicWorkerFaultPort(
        fail_labels=frozenset({"mid"}),
        succeed=_ok,
    )
    outcomes = await execute_concurrent_execution_work_resilient(
        port,
        (_req("left"), _req("mid"), _req("right")),
        policy=_POLICY,
    )
    return tuple(o.disposition.name for o in outcomes)


@pytest.mark.asyncio
async def test_ee_b2_final_chaos_outcomes_repeatable_three_times() -> None:
    baseline = await _chaos_slice()
    for _ in range(_RUNS - 1):
        assert await _chaos_slice() == baseline
    assert baseline == (
        ConcurrentExecutionWorkDisposition.SUCCEEDED.name,
        ConcurrentExecutionWorkDisposition.FAILED.name,
        ConcurrentExecutionWorkDisposition.SUCCEEDED.name,
    )
