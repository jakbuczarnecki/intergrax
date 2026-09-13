# © Artur Czarnecki. All rights reserved.

"""EE-B1.2 — concurrent admission must never exceed configured capacity."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityOverloadMode,
    ExecutionCapacityPolicy,
)
from intergrax.runtime.execution.capacity import LocalExecutionCapacityAdmission
from intergrax.runtime.execution.runtime import ExecutionRuntime, RootExecutionContext
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_AUTHORITY = ParentExecutionAuthority.unrestricted_root()
_CAPACITY = 2


@dataclass(frozen=True, slots=True)
class _Req:
    n: int


@dataclass(frozen=True, slots=True)
class _Res:
    n: int


class _HoldDelegate:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self._hold = asyncio.Event()
        self._lock = asyncio.Lock()

    async def execute(self, request: _Req) -> _Res:
        async with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        await self._hold.wait()
        async with self._lock:
            self.active -= 1
        return _Res(n=request.n)


def _ctx() -> RootExecutionContext:
    return RootExecutionContext(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        authority=_AUTHORITY,
        tenant_id="t",
        task_id=mint_task_id(),
    )


@pytest.mark.asyncio
async def test_ee_b1_2_max_active_never_exceeds_capacity() -> None:
    delegate = _HoldDelegate()
    policy = ExecutionCapacityPolicy(
        max_concurrent_root_executions=_CAPACITY,
        overload_mode=ExecutionCapacityOverloadMode.WAIT_WITH_TIMEOUT,
        wait_timeout_seconds=5.0,
    )
    runtime = ExecutionRuntime(
        delegate,
        execution_capacity_admission=LocalExecutionCapacityAdmission(policy),
    )
    tasks = [
        asyncio.create_task(runtime.execute(_Req(n=i), _ctx()))
        for i in range(_CAPACITY + 3)
    ]
    await asyncio.sleep(0.08)
    assert delegate.max_active <= _CAPACITY
    delegate._hold.set()
    await asyncio.gather(*tasks)
    assert delegate.max_active <= _CAPACITY
    assert delegate.active == 0
