# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Process-local root execution capacity admission (W1-A)."""

from __future__ import annotations

import asyncio

from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityAdmissionPort,
    ExecutionCapacityAdmissionRequest,
    ExecutionCapacityAdmissionTimeoutError,
    ExecutionCapacityExceededError,
    ExecutionCapacityOverloadMode,
    ExecutionCapacityPermit,
    ExecutionCapacityPolicy,
)


class _LocalExecutionCapacityPermit:
    __slots__ = ("_admission", "_released", "_release_guard")

    def __init__(self, admission: LocalExecutionCapacityAdmission) -> None:
        self._admission = admission
        self._released = False
        self._release_guard = asyncio.Lock()

    async def release(self) -> None:
        async with self._release_guard:
            if self._released:
                return
            self._released = True
        await self._admission._release_slot()


class LocalExecutionCapacityAdmission(ExecutionCapacityAdmissionPort):
    """Bounded process-local root execution slots (single capacity state for all overload modes)."""

    __slots__ = ("_active", "_capacity", "_condition", "_policy")

    def __init__(self, policy: ExecutionCapacityPolicy) -> None:
        if type(policy) is not ExecutionCapacityPolicy:
            raise TypeError("policy must be ExecutionCapacityPolicy")
        self._policy = policy
        self._capacity = policy.max_concurrent_root_executions
        self._active = 0
        self._condition = asyncio.Condition()

    async def _release_slot(self) -> None:
        async with self._condition:
            if self._active > 0:
                self._active -= 1
            self._condition.notify()

    async def acquire(
        self,
        request: ExecutionCapacityAdmissionRequest,
    ) -> ExecutionCapacityPermit:
        if type(request) is not ExecutionCapacityAdmissionRequest:
            raise TypeError("request must be ExecutionCapacityAdmissionRequest")
        if self._policy.overload_mode is ExecutionCapacityOverloadMode.REJECT:
            async with self._condition:
                if self._active >= self._capacity:
                    raise ExecutionCapacityExceededError(
                        "root execution capacity saturated"
                    )
                self._active += 1
        else:
            assert self._policy.wait_timeout_seconds is not None
            try:
                await asyncio.wait_for(
                    self._acquire_when_available(),
                    timeout=self._policy.wait_timeout_seconds,
                )
            except TimeoutError as exc:
                raise ExecutionCapacityAdmissionTimeoutError(
                    "timed out waiting for root execution capacity"
                ) from exc
        return _LocalExecutionCapacityPermit(self)

    async def _acquire_when_available(self) -> None:
        async with self._condition:
            while self._active >= self._capacity:
                await self._condition.wait()
            self._active += 1
