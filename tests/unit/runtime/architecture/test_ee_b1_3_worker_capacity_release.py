# © Artur Czarnecki. All rights reserved.

"""EE-B1.3 — root capacity release when delegate hosts failing concurrent workers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
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
from intergrax.runtime.execution.runtime import ExecutionRuntime, RootExecutionContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_AUTHORITY = ParentExecutionAuthority.unrestricted_root()


@dataclass(frozen=True, slots=True)
class _Req:
    label: str


@dataclass(frozen=True, slots=True)
class _Res:
    label: str


@dataclass(frozen=True, slots=True)
class _Inner:
    value: str


def _ctx() -> RootExecutionContext:
    return RootExecutionContext(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        authority=_AUTHORITY,
        tenant_id="t",
        task_id=mint_task_id(),
    )


class _InnerPort(ExecutionWorkPort[str, _Inner, _Inner]):
    def __init__(self, *, fail: frozenset[str]) -> None:
        self._fail = fail

    async def execute(self, request: ExecutionRequest[str, _Inner]) -> _Inner:
        if request.input in self._fail:
            raise RuntimeError(f"fail:{request.input}")
        return _Inner(value=request.input)


class _ResilientHostDelegate:
    def __init__(self, *, fail: frozenset[str]) -> None:
        self._fail = fail

    async def execute(self, request: _Req) -> _Res:
        port = _InnerPort(fail=self._fail)
        inner_requests = tuple(
            ExecutionRequest(input=f"{request.label}-{i}", output_type=_Inner)
            for i in range(3)
        )
        await execute_concurrent_execution_work_resilient(
            port,
            inner_requests,
            policy=ConcurrentExecutionWorkPolicy(max_concurrency=3),
        )
        return _Res(label=request.label)


@pytest.mark.asyncio
async def test_ee_b1_3_capacity_zero_after_mixed_root_executions() -> None:
    policy = ExecutionCapacityPolicy(max_concurrent_root_executions=1)
    admission = LocalExecutionCapacityAdmission(policy)
    runtime = ExecutionRuntime(
        _ResilientHostDelegate(fail=frozenset({"x-0", "y-1"})),
        execution_capacity_admission=admission,
    )
    await runtime.execute(_Req(label="x"), _ctx())
    await runtime.execute(_Req(label="y"), _ctx())
    await runtime.execute(_Req(label="z"), _ctx())
    req = ExecutionCapacityAdmissionRequest(
        tenant_id="t",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    permit = await admission.acquire(req)
    await permit.release()


@pytest.mark.asyncio
async def test_ee_b1_3_double_release_permit_idempotent() -> None:
    admission = LocalExecutionCapacityAdmission(
        ExecutionCapacityPolicy(max_concurrent_root_executions=1),
    )
    req = ExecutionCapacityAdmissionRequest(
        tenant_id="t",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    permit = await admission.acquire(req)
    release_task = asyncio.create_task(permit.release())
    await asyncio.sleep(0)
    release_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await release_task
    await permit.release()
    second = await admission.acquire(req)
    await second.release()
