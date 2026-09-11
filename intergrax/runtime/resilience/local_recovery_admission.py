# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Process-local recovery start admission (W3-C).

Each configured :class:`RecoveryKind` owns an isolated bounded capacity state on the
current process and asyncio event loop. Not safe to share one instance across threads
or different event loops.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

from intergrax.contracts.recovery_admission import (
    RecoveryAdmissionExceededError,
    RecoveryAdmissionOverloadMode,
    RecoveryAdmissionPermit,
    RecoveryAdmissionPolicy,
    RecoveryAdmissionPolicyMissingError,
    RecoveryAdmissionPort,
    RecoveryAdmissionRequest,
    RecoveryAdmissionTimeoutError,
    RecoveryKind,
)


class _LocalRecoveryCapacityState:
    """Per-recovery-kind acquire/release ledger (module-internal)."""

    __slots__ = ("_active", "_capacity", "_condition")

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._active = 0
        self._condition = asyncio.Condition()

    async def release_one(self) -> None:
        async with self._condition:
            if self._active <= 0:
                raise RuntimeError(
                    "recovery admission release without matching acquisition"
                )
            self._active -= 1
            self._condition.notify()

    async def acquire_reject(self) -> None:
        async with self._condition:
            if self._active >= self._capacity:
                raise RecoveryAdmissionExceededError("recovery start capacity saturated")
            self._active += 1

    async def acquire_when_available(self) -> None:
        async with self._condition:
            while self._active >= self._capacity:
                await self._condition.wait()
            self._active += 1


@dataclass(frozen=True, slots=True)
class _RecoveryAdmissionEntry:
    policy: RecoveryAdmissionPolicy
    state: _LocalRecoveryCapacityState


class _LocalRecoveryAdmissionPermit:
    __slots__ = ("_capacity_state", "_released", "_release_guard")

    def __init__(self, capacity_state: _LocalRecoveryCapacityState) -> None:
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
            raise RuntimeError("invariant: recovery capacity release task cancelled")
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


class LocalRecoveryAdmission(RecoveryAdmissionPort):
    """Bounded process-local recovery start slots keyed by :class:`RecoveryKind`."""

    __slots__ = ("_entries",)

    def __init__(
        self,
        policies: Mapping[RecoveryKind, RecoveryAdmissionPolicy],
    ) -> None:
        if not isinstance(policies, Mapping):
            raise TypeError("policies must be a Mapping")
        snapshot = dict(policies)
        entries: dict[RecoveryKind, _RecoveryAdmissionEntry] = {}
        for kind, policy in snapshot.items():
            if type(kind) is not RecoveryKind:
                raise TypeError("policy keys must be RecoveryKind")
            if type(policy) is not RecoveryAdmissionPolicy:
                raise TypeError("policy values must be RecoveryAdmissionPolicy")
            entries[kind] = _RecoveryAdmissionEntry(
                policy=policy,
                state=_LocalRecoveryCapacityState(policy.max_concurrent_recovery_starts),
            )
        self._entries = entries

    async def acquire(
        self,
        request: RecoveryAdmissionRequest,
    ) -> RecoveryAdmissionPermit:
        if type(request) is not RecoveryAdmissionRequest:
            raise TypeError("request must be RecoveryAdmissionRequest")
        kind = request.recovery_kind
        entry = self._entries.get(kind)
        if entry is None:
            raise RecoveryAdmissionPolicyMissingError(
                f"no recovery admission policy for {kind.value}"
            )
        policy = entry.policy
        state = entry.state
        if policy.overload_mode is RecoveryAdmissionOverloadMode.REJECT:
            try:
                await state.acquire_reject()
            except RecoveryAdmissionExceededError:
                raise RecoveryAdmissionExceededError(
                    f"recovery start capacity saturated for {kind.value}"
                ) from None
        else:
            assert policy.wait_timeout_seconds is not None
            try:
                await asyncio.wait_for(
                    state.acquire_when_available(),
                    timeout=policy.wait_timeout_seconds,
                )
            except TimeoutError as exc:
                raise RecoveryAdmissionTimeoutError(
                    f"timed out waiting for recovery start slot ({kind.value})"
                ) from exc
        return _LocalRecoveryAdmissionPermit(state)
