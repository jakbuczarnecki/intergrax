# © Artur Czarnecki. All rights reserved.

"""Enterprise Scale & Resilience W3-C — local recovery start admission."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.recovery_admission import (
    RecoveryAdmissionExceededError,
    RecoveryAdmissionOverloadMode,
    RecoveryAdmissionPermit,
    RecoveryAdmissionPolicy,
    RecoveryAdmissionRequest,
    RecoveryAdmissionTimeoutError,
    RecoveryKind,
)
from intergrax.runtime.resilience.local_recovery_admission import LocalRecoveryAdmission

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _request(
    kind: RecoveryKind,
    *,
    tenant_id: str | None = None,
    task_id: str | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> RecoveryAdmissionRequest:
    return RecoveryAdmissionRequest(
        tenant_id=tenant_id,
        task_id=mint_task_id() if task_id is None else task_id,
        run_id=mint_run_id() if run_id is None else run_id,
        attempt_id=mint_attempt_id() if attempt_id is None else attempt_id,
        recovery_kind=kind,
    )


def _reject_policy(capacity: int) -> RecoveryAdmissionPolicy:
    return RecoveryAdmissionPolicy(
        max_concurrent_recovery_starts=capacity,
        overload_mode=RecoveryAdmissionOverloadMode.REJECT,
        wait_timeout_seconds=None,
    )


def _wait_policy(capacity: int, timeout: float) -> RecoveryAdmissionPolicy:
    return RecoveryAdmissionPolicy(
        max_concurrent_recovery_starts=capacity,
        overload_mode=RecoveryAdmissionOverloadMode.WAIT_WITH_TIMEOUT,
        wait_timeout_seconds=timeout,
    )


@pytest.mark.asyncio
async def test_reject_concurrent_acquire_capacity_one_atomic() -> None:
    admission = LocalRecoveryAdmission(
        {RecoveryKind.TASK_RESUME: _reject_policy(1)},
    )
    contender_count = 32
    start_barrier = asyncio.Barrier(contender_count)
    permits: list[RecoveryAdmissionPermit] = []
    errors: list[RecoveryAdmissionExceededError] = []
    permits_lock = asyncio.Lock()

    async def contender() -> None:
        await start_barrier.wait()
        try:
            permit = await admission.acquire(_request(RecoveryKind.TASK_RESUME))
        except RecoveryAdmissionExceededError as exc:
            async with permits_lock:
                errors.append(exc)
            return
        async with permits_lock:
            permits.append(permit)

    tasks = [asyncio.create_task(contender()) for _ in range(contender_count)]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
    finally:
        for permit in permits:
            await permit.release()

    assert len(permits) == 1
    assert len(errors) == contender_count - 1


@pytest.mark.asyncio
async def test_reject_when_saturated() -> None:
    admission = LocalRecoveryAdmission(
        {RecoveryKind.TASK_RESUME: _reject_policy(1)},
    )
    holder = await admission.acquire(_request(RecoveryKind.TASK_RESUME))
    with pytest.raises(RecoveryAdmissionExceededError):
        await admission.acquire(_request(RecoveryKind.TASK_RESUME))
    await holder.release()


@pytest.mark.asyncio
async def test_wait_acquires_after_release() -> None:
    admission = LocalRecoveryAdmission(
        {RecoveryKind.TASK_RESUME: _wait_policy(1, 5.0)},
    )
    permit_a = await admission.acquire(_request(RecoveryKind.TASK_RESUME))
    acquired_b = asyncio.Event()

    async def waiter() -> None:
        permit_b = await admission.acquire(_request(RecoveryKind.TASK_RESUME))
        acquired_b.set()
        await permit_b.release()

    task_b = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    await permit_a.release()
    await asyncio.wait_for(task_b, timeout=2.0)
    assert acquired_b.is_set()


@pytest.mark.asyncio
async def test_wait_timeout() -> None:
    admission = LocalRecoveryAdmission(
        {RecoveryKind.TASK_RESUME: _wait_policy(1, 0.05)},
    )
    await admission.acquire(_request(RecoveryKind.TASK_RESUME))
    with pytest.raises(RecoveryAdmissionTimeoutError):
        await admission.acquire(_request(RecoveryKind.TASK_RESUME))


@pytest.mark.asyncio
async def test_cancel_waiting_does_not_consume_slot() -> None:
    admission = LocalRecoveryAdmission(
        {RecoveryKind.TASK_RESUME: _wait_policy(1, 30.0)},
    )
    permit_a = await admission.acquire(_request(RecoveryKind.TASK_RESUME))
    waiter = asyncio.create_task(admission.acquire(_request(RecoveryKind.TASK_RESUME)))
    await asyncio.sleep(0.05)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await permit_a.release()
    permit_c = await admission.acquire(_request(RecoveryKind.TASK_RESUME))
    await permit_c.release()


@pytest.mark.asyncio
async def test_permit_concurrent_double_release() -> None:
    admission = LocalRecoveryAdmission(
        {RecoveryKind.TASK_RESUME: _reject_policy(1)},
    )
    permit = await admission.acquire(_request(RecoveryKind.TASK_RESUME))
    await asyncio.gather(permit.release(), permit.release())
    replacement = await admission.acquire(_request(RecoveryKind.TASK_RESUME))
    await replacement.release()


@pytest.mark.asyncio
async def test_recovery_kind_isolation() -> None:
    policies: Mapping[RecoveryKind, RecoveryAdmissionPolicy] = {
        RecoveryKind.TASK_RESUME: _reject_policy(1),
        RecoveryKind.PARTIAL_TOPOLOGY: _reject_policy(1),
    }
    admission = LocalRecoveryAdmission(policies)
    task_resume = await admission.acquire(_request(RecoveryKind.TASK_RESUME))
    partial = await admission.acquire(_request(RecoveryKind.PARTIAL_TOPOLOGY))
    with pytest.raises(RecoveryAdmissionExceededError):
        await admission.acquire(_request(RecoveryKind.TASK_RESUME))
    with pytest.raises(RecoveryAdmissionExceededError):
        await admission.acquire(_request(RecoveryKind.PARTIAL_TOPOLOGY))
    await task_resume.release()
    await partial.release()
