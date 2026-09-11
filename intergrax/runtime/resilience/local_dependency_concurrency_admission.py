# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Process-local dependency concurrency admission (W2-B1).

Each configured ``DependencyConcurrencyIdentity`` owns an isolated bounded capacity
state on the current process and asyncio event loop. Not safe to share one instance
across threads or different event loops.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionPort,
    DependencyConcurrencyAdmissionRequest,
    DependencyConcurrencyAdmissionTimeoutError,
    DependencyConcurrencyExceededError,
    DependencyConcurrencyIdentity,
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPermit,
    DependencyConcurrencyPolicy,
    DependencyConcurrencyPolicyMissingError,
)


class _LocalDependencyCapacityState:
    """Per-dependency acquire/release ledger (module-internal)."""

    __slots__ = ("_active", "_capacity", "_condition")

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._active = 0
        self._condition = asyncio.Condition()

    async def release_one(self) -> None:
        async with self._condition:
            if self._active <= 0:
                raise RuntimeError(
                    "dependency concurrency release without matching acquisition"
                )
            self._active -= 1
            self._condition.notify()

    async def acquire_reject(self) -> None:
        async with self._condition:
            if self._active >= self._capacity:
                raise DependencyConcurrencyExceededError(
                    "dependency concurrency saturated"
                )
            self._active += 1

    async def acquire_when_available(self) -> None:
        async with self._condition:
            while self._active >= self._capacity:
                await self._condition.wait()
            self._active += 1


@dataclass(frozen=True, slots=True)
class _DependencyAdmissionEntry:
    policy: DependencyConcurrencyPolicy
    state: _LocalDependencyCapacityState


class _LocalDependencyConcurrencyPermit:
    __slots__ = ("_capacity_state", "_released", "_release_guard")

    def __init__(self, capacity_state: _LocalDependencyCapacityState) -> None:
        self._capacity_state = capacity_state
        self._released = False
        self._release_guard = asyncio.Lock()

    async def _await_release_task(self, release_task: asyncio.Task[None]) -> None:
        while not release_task.done():
            try:
                await asyncio.shield(release_task)
            except asyncio.CancelledError:
                continue
        if release_task.cancelled():
            raise RuntimeError("invariant: dependency capacity release task cancelled")
        exc = release_task.exception()
        if exc is not None:
            raise exc

    async def release(self) -> None:
        async with self._release_guard:
            if self._released:
                return
            release_task = asyncio.create_task(
                self._capacity_state.release_one(),
            )
            cancelled = False
            try:
                await asyncio.shield(release_task)
            except asyncio.CancelledError:
                cancelled = True
            await self._await_release_task(release_task)
            self._released = True
            if cancelled:
                raise asyncio.CancelledError()


class LocalDependencyConcurrencyAdmission(DependencyConcurrencyAdmissionPort):
    """Bounded process-local dependency slots keyed by ``DependencyConcurrencyIdentity``."""

    __slots__ = ("_entries",)

    def __init__(
        self,
        policies: Mapping[
            DependencyConcurrencyIdentity,
            DependencyConcurrencyPolicy,
        ],
    ) -> None:
        if not isinstance(policies, Mapping):
            raise TypeError("policies must be a Mapping")
        snapshot = dict(policies)
        entries: dict[DependencyConcurrencyIdentity, _DependencyAdmissionEntry] = {}
        for identity, policy in snapshot.items():
            if type(identity) is not DependencyConcurrencyIdentity:
                raise TypeError("policy keys must be DependencyConcurrencyIdentity")
            if type(policy) is not DependencyConcurrencyPolicy:
                raise TypeError("policy values must be DependencyConcurrencyPolicy")
            entries[identity] = _DependencyAdmissionEntry(
                policy=policy,
                state=_LocalDependencyCapacityState(policy.max_concurrent_calls),
            )
        self._entries = entries

    async def acquire(
        self,
        request: DependencyConcurrencyAdmissionRequest,
    ) -> DependencyConcurrencyPermit:
        if type(request) is not DependencyConcurrencyAdmissionRequest:
            raise TypeError("request must be DependencyConcurrencyAdmissionRequest")
        identity = request.dependency
        entry = self._entries.get(identity)
        if entry is None:
            raise DependencyConcurrencyPolicyMissingError(
                f"no dependency concurrency policy for "
                f"{identity.kind.value}/{identity.value}"
            )
        policy = entry.policy
        state = entry.state
        if policy.overload_mode is DependencyConcurrencyOverloadMode.REJECT:
            try:
                await state.acquire_reject()
            except DependencyConcurrencyExceededError:
                raise DependencyConcurrencyExceededError(
                    f"dependency concurrency saturated for "
                    f"{identity.kind.value}/{identity.value}"
                ) from None
        else:
            assert policy.wait_timeout_seconds is not None
            try:
                await asyncio.wait_for(
                    state.acquire_when_available(),
                    timeout=policy.wait_timeout_seconds,
                )
            except TimeoutError as exc:
                raise DependencyConcurrencyAdmissionTimeoutError(
                    f"timed out waiting for dependency concurrency slot "
                    f"({identity.kind.value}/{identity.value})"
                ) from exc
        return _LocalDependencyConcurrencyPermit(state)
