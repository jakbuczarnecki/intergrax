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


class _LocalRootCapacityState:
    """Shared acquire/release ledger for admission and permits (module-internal)."""

    __slots__ = ("_active", "_capacity", "_condition")

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._active = 0
        self._condition = asyncio.Condition()

    async def release_one(self) -> None:
        async with self._condition:
            if self._active > 0:
                self._active -= 1
            self._condition.notify()

    async def acquire_reject(self) -> None:
        async with self._condition:
            if self._active >= self._capacity:
                raise ExecutionCapacityExceededError(
                    "root execution capacity saturated"
                )
            self._active += 1

    async def acquire_when_available(self) -> None:
        async with self._condition:
            while self._active >= self._capacity:
                await self._condition.wait()
            self._active += 1


class _LocalExecutionCapacityPermit:
    __slots__ = ("_capacity_state", "_released", "_release_guard")

    def __init__(self, capacity_state: _LocalRootCapacityState) -> None:
        self._capacity_state = capacity_state
        self._released = False
        self._release_guard = asyncio.Lock()

    async def release(self) -> None:
        async with self._release_guard:
            if self._released:
                return
            slot_returned = False
            try:
                await asyncio.shield(self._capacity_state.release_one())
                slot_returned = True
            finally:
                if slot_returned:
                    self._released = True


class LocalExecutionCapacityAdmission(ExecutionCapacityAdmissionPort):
    """Bounded process-local root execution slots (single capacity state for all overload modes)."""

    __slots__ = ("_capacity_state", "_policy")

    def __init__(self, policy: ExecutionCapacityPolicy) -> None:
        if type(policy) is not ExecutionCapacityPolicy:
            raise TypeError("policy must be ExecutionCapacityPolicy")
        self._policy = policy
        self._capacity_state = _LocalRootCapacityState(
            policy.max_concurrent_root_executions
        )

    async def acquire(
        self,
        request: ExecutionCapacityAdmissionRequest,
    ) -> ExecutionCapacityPermit:
        if type(request) is not ExecutionCapacityAdmissionRequest:
            raise TypeError("request must be ExecutionCapacityAdmissionRequest")
        if self._policy.overload_mode is ExecutionCapacityOverloadMode.REJECT:
            await self._capacity_state.acquire_reject()
        else:
            assert self._policy.wait_timeout_seconds is not None
            try:
                await asyncio.wait_for(
                    self._capacity_state.acquire_when_available(),
                    timeout=self._policy.wait_timeout_seconds,
                )
            except TimeoutError as exc:
                raise ExecutionCapacityAdmissionTimeoutError(
                    "timed out waiting for root execution capacity"
                ) from exc
        return _LocalExecutionCapacityPermit(self._capacity_state)
