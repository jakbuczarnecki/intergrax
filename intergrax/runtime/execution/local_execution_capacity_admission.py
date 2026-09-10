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
    __slots__ = ("_released", "_semaphore")

    def __init__(self, semaphore: asyncio.Semaphore) -> None:
        self._semaphore = semaphore
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._semaphore.release()


class LocalExecutionCapacityAdmission(ExecutionCapacityAdmissionPort):
    """Bounded process-local root execution slots via asyncio semaphore (implementation detail)."""

    __slots__ = ("_policy", "_semaphore")

    def __init__(self, policy: ExecutionCapacityPolicy) -> None:
        if type(policy) is not ExecutionCapacityPolicy:
            raise TypeError("policy must be ExecutionCapacityPolicy")
        self._policy = policy
        self._semaphore = asyncio.Semaphore(policy.max_concurrent_root_executions)

    async def acquire(
        self,
        request: ExecutionCapacityAdmissionRequest,
    ) -> ExecutionCapacityPermit:
        if type(request) is not ExecutionCapacityAdmissionRequest:
            raise TypeError("request must be ExecutionCapacityAdmissionRequest")
        if self._policy.overload_mode is ExecutionCapacityOverloadMode.REJECT:
            if self._semaphore.locked():
                raise ExecutionCapacityExceededError(
                    "root execution capacity saturated"
                )
            await self._semaphore.acquire()
        else:
            assert self._policy.wait_timeout_seconds is not None
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self._policy.wait_timeout_seconds,
                )
            except TimeoutError as exc:
                raise ExecutionCapacityAdmissionTimeoutError(
                    "timed out waiting for root execution capacity"
                ) from exc
        return _LocalExecutionCapacityPermit(self._semaphore)
