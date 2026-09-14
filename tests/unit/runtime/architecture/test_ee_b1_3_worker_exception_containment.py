# © Artur Czarnecki. All rights reserved.

"""EE-B1.3 — strict vs resilient exception containment."""

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

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_POLICY = ConcurrentExecutionWorkPolicy(max_concurrency=3)


@dataclass(frozen=True, slots=True)
class _WorkResult:
    value: str


def _req(label: str) -> ExecutionRequest[str, _WorkResult]:
    return ExecutionRequest(input=label, output_type=_WorkResult)


class _ImmediatePort(ExecutionWorkPort[str, _WorkResult, _WorkResult]):
    def __init__(self, *, fail: frozenset[str] = frozenset()) -> None:
        self._fail = fail

    async def execute(self, request: ExecutionRequest[str, _WorkResult]) -> _WorkResult:
        if request.input in self._fail:
            raise RuntimeError(f"boom:{request.input}")
        return _WorkResult(value=request.input)


@pytest.mark.asyncio
async def test_ee_b1_3_resilient_one_failure_siblings_succeed() -> None:
    requests = (_req("A"), _req("B"), _req("C"))
    outcomes = await execute_concurrent_execution_work_resilient(
        _ImmediatePort(fail=frozenset({"A"})),
        requests,
        policy=_POLICY,
    )
    assert outcomes[0].disposition is ConcurrentExecutionWorkDisposition.FAILED
    assert outcomes[1].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
    assert outcomes[1].result is not None
    assert outcomes[1].result.value == "B"
    assert outcomes[2].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
    assert outcomes[2].result is not None
    assert outcomes[2].result.value == "C"


@pytest.mark.asyncio
async def test_ee_b1_3_strict_mode_fail_fast_raises() -> None:
    with pytest.raises(RuntimeError, match="boom:B"):
        await execute_concurrent_execution_work(
            _ImmediatePort(fail=frozenset({"B"})),
            (_req("A"), _req("B"), _req("C")),
            policy=_POLICY,
        )


class _CountingPort(ExecutionWorkPort[str, _WorkResult, _WorkResult]):
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def execute(self, request: ExecutionRequest[str, _WorkResult]) -> _WorkResult:
        label = request.input
        self.counts[label] = self.counts.get(label, 0) + 1
        return _WorkResult(value=label)


@pytest.mark.asyncio
async def test_ee_b1_3_no_duplicate_execution_body_invocation() -> None:
    port = _CountingPort()
    await execute_concurrent_execution_work_resilient(
        port,
        (_req("x"), _req("y"), _req("z")),
        policy=_POLICY,
    )
    assert port.counts == {"x": 1, "y": 1, "z": 1}


@pytest.mark.asyncio
async def test_ee_b1_3_resilient_dependency_failure_not_swallowed() -> None:
    class _DepPort(ExecutionWorkPort[str, _WorkResult, _WorkResult]):
        async def execute(
            self, request: ExecutionRequest[str, _WorkResult]
        ) -> _WorkResult:
            if request.input == "dep":
                raise ConnectionError("upstream_unavailable")
            return _WorkResult(value=request.input)

    outcomes = await execute_concurrent_execution_work_resilient(
        _DepPort(),
        (_req("dep"), _req("ok")),
        policy=_POLICY,
    )
    assert outcomes[0].disposition is ConcurrentExecutionWorkDisposition.FAILED
    assert isinstance(outcomes[0].error, ConnectionError)
    assert outcomes[1].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
