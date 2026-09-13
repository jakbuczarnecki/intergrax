# © Artur Czarnecki. All rights reserved.

"""EE-B1.2 — permit release on success, failure, and cancellation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

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
from intergrax.runtime.execution.runtime import ExecutionRuntime, RootExecutionContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_AUTHORITY = ParentExecutionAuthority.unrestricted_root()


@dataclass(frozen=True, slots=True)
class _Req:
    label: str


@dataclass(frozen=True, slots=True)
class _Res:
    label: str


class _OkDelegate:
    async def execute(self, request: _Req) -> _Res:
        return _Res(label=request.label)


class _SelectiveBoomDelegate:
    async def execute(self, request: _Req) -> _Res:
        if request.label == "a":
            raise RuntimeError(f"boom:{request.label}")
        return _Res(label=request.label)


class _HoldDelegate:
    def __init__(self) -> None:
        self._hold = asyncio.Event()

    async def execute(self, request: _Req) -> _Res:
        await self._hold.wait()
        return _Res(label=request.label)


def _ctx() -> RootExecutionContext:
    return RootExecutionContext(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        authority=_AUTHORITY,
        tenant_id="t",
        task_id=mint_task_id(),
    )


def _runtime(delegate: object) -> ExecutionRuntime[_Req, _Res]:
    policy = ExecutionCapacityPolicy(max_concurrent_root_executions=1)
    return ExecutionRuntime(
        delegate,
        execution_capacity_admission=LocalExecutionCapacityAdmission(policy),
    )


@pytest.mark.asyncio
async def test_ee_b1_2_release_on_success_no_leak() -> None:
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
    p1 = await admission.acquire(req)
    await p1.release()
    p2 = await admission.acquire(req)
    await p2.release()


@pytest.mark.asyncio
async def test_ee_b1_2_release_on_exception_allows_next() -> None:
    runtime = _runtime(_SelectiveBoomDelegate())
    with pytest.raises(RuntimeError, match="boom:a"):
        await runtime.execute(_Req(label="a"), _ctx())
    await runtime.execute(_Req(label="b"), _ctx())


@pytest.mark.asyncio
async def test_ee_b1_2_release_on_cancellation_allows_next() -> None:
    hold_delegate = _HoldDelegate()
    runtime = _runtime(hold_delegate)
    task = asyncio.create_task(runtime.execute(_Req(label="a"), _ctx()))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    hold_delegate._hold.set()
    await runtime.execute(_Req(label="b"), _ctx())


@pytest.mark.asyncio
async def test_ee_b1_2_double_release_idempotent_no_negative() -> None:
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
    await permit.release()
    await permit.release()
    second = await admission.acquire(req)
    await second.release()
