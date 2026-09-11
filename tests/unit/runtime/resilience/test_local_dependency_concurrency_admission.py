# © Artur Czarnecki. All rights reserved.

"""Enterprise Scale & Resilience W2-B1 — local dependency concurrency admission."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping
from typing import cast

import pytest

from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionRequest,
    DependencyConcurrencyAdmissionTimeoutError,
    DependencyConcurrencyExceededError,
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPermit,
    DependencyConcurrencyPolicy,
    DependencyConcurrencyPolicyMissingError,
)
from intergrax.runtime.resilience.local_dependency_concurrency_admission import (
    LocalDependencyConcurrencyAdmission,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _identity(
    kind: DependencyConcurrencyKind,
    value: str,
) -> DependencyConcurrencyIdentity:
    return DependencyConcurrencyIdentity(kind=kind, value=value)


def _request(
    identity: DependencyConcurrencyIdentity,
    *,
    tenant_id: str | None = None,
) -> DependencyConcurrencyAdmissionRequest:
    return DependencyConcurrencyAdmissionRequest(
        dependency=identity,
        tenant_id=tenant_id,
    )


def _reject_policy(capacity: int) -> DependencyConcurrencyPolicy:
    return DependencyConcurrencyPolicy(
        max_concurrent_calls=capacity,
        overload_mode=DependencyConcurrencyOverloadMode.REJECT,
        wait_timeout_seconds=None,
    )


def _wait_policy(capacity: int, timeout: float) -> DependencyConcurrencyPolicy:
    return DependencyConcurrencyPolicy(
        max_concurrent_calls=capacity,
        overload_mode=DependencyConcurrencyOverloadMode.WAIT_WITH_TIMEOUT,
        wait_timeout_seconds=timeout,
    )


_JIRA = _identity(DependencyConcurrencyKind.TOOL, "jira.search")
_SLACK = _identity(DependencyConcurrencyKind.TOOL, "slack.search")
_LLM_JIRA = _identity(DependencyConcurrencyKind.LLM_PROVIDER, "jira")


@pytest.mark.asyncio
async def test_unknown_dependency_policy_missing_and_configured_unchanged() -> None:
    admission = LocalDependencyConcurrencyAdmission({_JIRA: _reject_policy(2)})
    with pytest.raises(DependencyConcurrencyPolicyMissingError):
        await admission.acquire(_request(_SLACK))
    permit = await admission.acquire(_request(_JIRA))
    await permit.release()


@pytest.mark.asyncio
async def test_reject_concurrent_acquire_capacity_one_atomic() -> None:
    admission = LocalDependencyConcurrencyAdmission({_JIRA: _reject_policy(1)})
    contender_count = 32
    start_barrier = asyncio.Barrier(contender_count)
    permits: list[DependencyConcurrencyPermit] = []
    errors: list[DependencyConcurrencyExceededError] = []
    permits_lock = asyncio.Lock()

    async def contender() -> None:
        await start_barrier.wait()
        try:
            permit = await admission.acquire(_request(_JIRA))
        except DependencyConcurrencyExceededError as exc:
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
async def test_reject_concurrent_acquire_capacity_three() -> None:
    admission = LocalDependencyConcurrencyAdmission({_JIRA: _reject_policy(3)})
    contender_count = 10
    start_barrier = asyncio.Barrier(contender_count)
    permits: list[DependencyConcurrencyPermit] = []
    errors: list[DependencyConcurrencyExceededError] = []
    permits_lock = asyncio.Lock()

    async def contender() -> None:
        await start_barrier.wait()
        try:
            permit = await admission.acquire(_request(_JIRA))
        except DependencyConcurrencyExceededError as exc:
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

    assert len(permits) == 3
    assert len(errors) == contender_count - 3


@pytest.mark.asyncio
async def test_reject_does_not_enqueue_waiter_when_saturated() -> None:
    admission = LocalDependencyConcurrencyAdmission({_JIRA: _reject_policy(1)})
    holder = await admission.acquire(_request(_JIRA))
    reject_count = 16
    start_barrier = asyncio.Barrier(reject_count)

    async def reject_attempt() -> None:
        await start_barrier.wait()
        with pytest.raises(DependencyConcurrencyExceededError):
            await admission.acquire(_request(_JIRA))

    tasks = [asyncio.create_task(reject_attempt()) for _ in range(reject_count)]
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
    await holder.release()
    replacement = await admission.acquire(_request(_JIRA))
    await replacement.release()


@pytest.mark.asyncio
async def test_wait_with_timeout_acquires_after_release() -> None:
    admission = LocalDependencyConcurrencyAdmission(
        {_JIRA: _wait_policy(1, 5.0)},
    )
    permit_a = await admission.acquire(_request(_JIRA))
    acquired_b = asyncio.Event()

    async def waiter() -> None:
        permit_b = await admission.acquire(_request(_JIRA))
        acquired_b.set()
        await permit_b.release()

    task_b = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    await permit_a.release()
    await asyncio.wait_for(task_b, timeout=2.0)
    assert acquired_b.is_set()


@pytest.mark.asyncio
async def test_wait_with_timeout_expires() -> None:
    admission = LocalDependencyConcurrencyAdmission(
        {_JIRA: _wait_policy(1, 0.05)},
    )
    await admission.acquire(_request(_JIRA))
    with pytest.raises(DependencyConcurrencyAdmissionTimeoutError):
        await admission.acquire(_request(_JIRA))


@pytest.mark.asyncio
async def test_wait_cancellation_does_not_consume_slot() -> None:
    admission = LocalDependencyConcurrencyAdmission(
        {_JIRA: _wait_policy(1, 30.0)},
    )
    permit_a = await admission.acquire(_request(_JIRA))
    waiter = asyncio.create_task(admission.acquire(_request(_JIRA)))
    await asyncio.sleep(0.05)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await permit_a.release()
    permit_c = await admission.acquire(_request(_JIRA))
    await permit_c.release()


@pytest.mark.asyncio
async def test_permit_idempotent_release() -> None:
    admission = LocalDependencyConcurrencyAdmission({_JIRA: _reject_policy(1)})
    permit = await admission.acquire(_request(_JIRA))
    await permit.release()
    await permit.release()
    replacement = await admission.acquire(_request(_JIRA))
    await replacement.release()


@pytest.mark.asyncio
async def test_permit_concurrent_double_release() -> None:
    admission = LocalDependencyConcurrencyAdmission({_JIRA: _reject_policy(1)})
    permit = await admission.acquire(_request(_JIRA))
    await asyncio.gather(permit.release(), permit.release())
    replacement = await admission.acquire(_request(_JIRA))
    await replacement.release()


_REAL_ASYNCIO_SHIELD = asyncio.shield


async def _shield_raises_cancelled_after_inner_completes(
    awaitable: asyncio.Task[None] | Coroutine[None, None, None],
) -> None:
    await _REAL_ASYNCIO_SHIELD(awaitable)
    raise asyncio.CancelledError()


@pytest.mark.asyncio
async def test_permit_release_cancel_during_lifecycle_frees_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = LocalDependencyConcurrencyAdmission({_JIRA: _reject_policy(1)})
    permit = await admission.acquire(_request(_JIRA))
    monkeypatch.setattr(
        asyncio,
        "shield",
        _shield_raises_cancelled_after_inner_completes,
    )
    with pytest.raises(asyncio.CancelledError):
        await permit.release()
    monkeypatch.setattr(asyncio, "shield", _REAL_ASYNCIO_SHIELD)
    replacement = await admission.acquire(_request(_JIRA))
    await replacement.release()


@pytest.mark.asyncio
async def test_cancelled_release_terminal_state_blocks_old_permit_double_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = LocalDependencyConcurrencyAdmission({_JIRA: _reject_policy(1)})
    permit_a = await admission.acquire(_request(_JIRA))
    monkeypatch.setattr(
        asyncio,
        "shield",
        _shield_raises_cancelled_after_inner_completes,
    )
    with pytest.raises(asyncio.CancelledError):
        await permit_a.release()
    monkeypatch.setattr(asyncio, "shield", _REAL_ASYNCIO_SHIELD)
    await permit_a.release()
    permit_b = await admission.acquire(_request(_JIRA))
    await permit_a.release()
    with pytest.raises(DependencyConcurrencyExceededError):
        await admission.acquire(_request(_JIRA))
    await permit_b.release()
    permit_c = await admission.acquire(_request(_JIRA))
    await permit_c.release()


@pytest.mark.asyncio
async def test_identity_isolation_across_tool_and_provider() -> None:
    openai_tool = _identity(DependencyConcurrencyKind.TOOL, "openai")
    openai_provider = _identity(DependencyConcurrencyKind.LLM_PROVIDER, "openai")
    admission = LocalDependencyConcurrencyAdmission(
        {
            _JIRA: _reject_policy(1),
            _SLACK: _reject_policy(1),
            openai_tool: _reject_policy(1),
            openai_provider: _reject_policy(1),
            _LLM_JIRA: _reject_policy(1),
        },
    )
    permit_jira = await admission.acquire(_request(_JIRA))
    permit_slack = await admission.acquire(_request(_SLACK))
    permit_llm_jira = await admission.acquire(_request(_LLM_JIRA))
    with pytest.raises(DependencyConcurrencyExceededError):
        await admission.acquire(_request(_JIRA))
    with pytest.raises(DependencyConcurrencyExceededError):
        await admission.acquire(_request(_SLACK))
    with pytest.raises(DependencyConcurrencyExceededError):
        await admission.acquire(_request(_LLM_JIRA))
    await permit_jira.release()
    await permit_slack.release()
    await permit_llm_jira.release()
    permit_openai_tool = await admission.acquire(_request(openai_tool))
    permit_openai_provider = await admission.acquire(_request(openai_provider))
    await permit_openai_tool.release()
    await permit_openai_provider.release()


@pytest.mark.asyncio
async def test_tenant_does_not_partition_capacity() -> None:
    admission = LocalDependencyConcurrencyAdmission({_JIRA: _reject_policy(1)})
    await admission.acquire(_request(_JIRA, tenant_id="tenant-a"))
    with pytest.raises(DependencyConcurrencyExceededError):
        await admission.acquire(_request(_JIRA, tenant_id="tenant-b"))


@pytest.mark.asyncio
async def test_multiple_policies_in_one_instance() -> None:
    identity_c = _identity(DependencyConcurrencyKind.INTEGRATION, "crm")
    admission = LocalDependencyConcurrencyAdmission(
        {
            _JIRA: _reject_policy(1),
            _SLACK: _reject_policy(2),
            identity_c: _wait_policy(1, 5.0),
        },
    )
    await admission.acquire(_request(_JIRA))
    with pytest.raises(DependencyConcurrencyExceededError):
        await admission.acquire(_request(_JIRA))
    await admission.acquire(_request(_SLACK))
    await admission.acquire(_request(_SLACK))
    with pytest.raises(DependencyConcurrencyExceededError):
        await admission.acquire(_request(_SLACK))

    holder = await admission.acquire(_request(identity_c))
    acquired = asyncio.Event()

    async def wait_for_c() -> None:
        permit = await admission.acquire(_request(identity_c))
        acquired.set()
        await permit.release()

    waiter = asyncio.create_task(wait_for_c())
    await asyncio.sleep(0.05)
    await holder.release()
    await asyncio.wait_for(waiter, timeout=2.0)
    assert acquired.is_set()


@pytest.mark.asyncio
async def test_source_policy_mapping_mutation_does_not_affect_instance() -> None:
    policies: dict[
        DependencyConcurrencyIdentity,
        DependencyConcurrencyPolicy,
    ] = {_JIRA: _reject_policy(1)}
    admission = LocalDependencyConcurrencyAdmission(policies)
    policies[_JIRA] = _reject_policy(99)
    policies[_SLACK] = _reject_policy(99)
    await admission.acquire(_request(_JIRA))
    with pytest.raises(DependencyConcurrencyExceededError):
        await admission.acquire(_request(_JIRA))
    with pytest.raises(DependencyConcurrencyPolicyMissingError):
        await admission.acquire(_request(_SLACK))


@pytest.mark.asyncio
async def test_missing_identity_no_magic_fallback_capacity() -> None:
    admission = LocalDependencyConcurrencyAdmission(
        {_JIRA: _reject_policy(8)},
    )
    unknown = _identity(DependencyConcurrencyKind.TOOL, "unknown.tool")
    for _ in range(3):
        with pytest.raises(DependencyConcurrencyPolicyMissingError):
            await admission.acquire(_request(unknown))


@pytest.mark.asyncio
async def test_permit_release_does_not_affect_other_identity() -> None:
    admission = LocalDependencyConcurrencyAdmission(
        {
            _JIRA: _reject_policy(1),
            _SLACK: _reject_policy(1),
        },
    )
    permit_jira = await admission.acquire(_request(_JIRA))
    await permit_jira.release()
    permit_slack = await admission.acquire(_request(_SLACK))
    await permit_slack.release()


@pytest.mark.asyncio
async def test_parallel_acquire_different_dependencies_not_globally_serialized() -> None:
    admission = LocalDependencyConcurrencyAdmission(
        {
            _JIRA: _reject_policy(1),
            _SLACK: _reject_policy(1),
        },
    )
    hold_jira = asyncio.Event()
    hold_slack = asyncio.Event()
    both_held = asyncio.Event()

    async def hold_identity(
        identity: DependencyConcurrencyIdentity,
        hold: asyncio.Event,
    ) -> None:
        permit = await admission.acquire(_request(identity))
        hold.set()
        if hold_jira.is_set() and hold_slack.is_set():
            both_held.set()
        await asyncio.sleep(0.2)
        await permit.release()

    await asyncio.wait_for(
        asyncio.gather(
            hold_identity(_JIRA, hold_jira),
            hold_identity(_SLACK, hold_slack),
        ),
        timeout=2.0,
    )
    assert both_held.is_set()


def test_constructor_rejects_non_mapping() -> None:
    invalid_policies = cast(
        Mapping[DependencyConcurrencyIdentity, DependencyConcurrencyPolicy],
        [],
    )
    with pytest.raises(TypeError, match="Mapping"):
        LocalDependencyConcurrencyAdmission(invalid_policies)


def test_constructor_rejects_invalid_key_type() -> None:
    invalid_policies = cast(
        Mapping[DependencyConcurrencyIdentity, DependencyConcurrencyPolicy],
        {"jira": _reject_policy(1)},
    )
    with pytest.raises(TypeError, match="DependencyConcurrencyIdentity"):
        LocalDependencyConcurrencyAdmission(invalid_policies)


def test_constructor_rejects_invalid_policy_type() -> None:
    invalid_policies = cast(
        Mapping[DependencyConcurrencyIdentity, DependencyConcurrencyPolicy],
        {_JIRA: {"max": 1}},
    )
    with pytest.raises(TypeError, match="DependencyConcurrencyPolicy"):
        LocalDependencyConcurrencyAdmission(invalid_policies)


@pytest.mark.asyncio
async def test_acquire_rejects_duck_typed_request() -> None:
    admission = LocalDependencyConcurrencyAdmission({_JIRA: _reject_policy(1)})

    class _FakeRequest:
        dependency = _JIRA
        tenant_id = None

    fake_request = cast(DependencyConcurrencyAdmissionRequest, _FakeRequest())
    with pytest.raises(TypeError, match="DependencyConcurrencyAdmissionRequest"):
        await admission.acquire(fake_request)


@pytest.mark.asyncio
async def test_empty_policy_map_all_acquire_missing() -> None:
    admission = LocalDependencyConcurrencyAdmission({})
    with pytest.raises(DependencyConcurrencyPolicyMissingError):
        await admission.acquire(_request(_JIRA))
