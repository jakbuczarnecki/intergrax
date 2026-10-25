# © Artur Czarnecki. All rights reserved.

"""TLX-TL11 OWN-A2 — canonical transport detachment supervision."""

from __future__ import annotations

import asyncio

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.execution_identity import mint_run_id
from intergrax.contracts.root_execution_transport_detachment_supervision import (
    RootExecutionTransportAdmissionRequest,
    RootExecutionTransportDetachmentHandoffDisposition,
    RootExecutionTransportDetachmentHandoffRefusalReason,
    RootExecutionTransportDetachmentHandoffRequest,
    RootExecutionTransportPreStartFinalizationDisposition,
    RootExecutionTransportPreStartFinalizationRequest,
)
from intergrax.runtime.execution.root_execution_transport_detachment_supervision import (
    build_host_root_execution_transport_detachment_supervisor,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-own-a2"
_OTHER = "tenant-other"


def _principal(*, tenant_id: str = _TENANT) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id="u1",
        principal_type=PrincipalType.USER,
        auth_subject="u1",
    )


@pytest.mark.asyncio
async def test_admission_supervision_without_active_binding_retains_host() -> None:
    supervisor = build_host_root_execution_transport_detachment_supervisor()
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    host_done = asyncio.Event()

    async def _host() -> str:
        await host_done.wait()
        return "ok"

    host_coro = asyncio.create_task(_host())
    await asyncio.sleep(0)
    admitted = await supervisor.admit_root_execution_supervision(
        RootExecutionTransportAdmissionRequest(
            task_id=task.task_id,
            verified_tenant_id=_TENANT,
            caller_request_identity=_principal(),
            host_asyncio_task=host_coro,
        )
    )
    assert admitted.disposition is RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED
    assert supervisor.holds_root_execution_supervision(
        task_id=task.task_id,
        host_asyncio_task=host_coro,
    )
    assert supervisor.supervised_count() == 1
    host_done.set()
    await asyncio.wait_for(host_coro, timeout=3.0)
    await asyncio.sleep(0.05)
    assert supervisor.supervised_count() == 0


@pytest.mark.asyncio
async def test_handoff_accepted_retains_strong_reference_until_completion() -> None:
    supervisor = build_host_root_execution_transport_detachment_supervisor()
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    host_done = asyncio.Event()

    async def _host() -> str:
        await host_done.wait()
        return "ok"

    host_coro = asyncio.create_task(_host())
    await asyncio.sleep(0)
    result = await supervisor.request_transport_detachment_handoff(
        RootExecutionTransportDetachmentHandoffRequest(
            task_id=task.task_id,
            run_id=run_id,
            verified_tenant_id=_TENANT,
            caller_request_identity=_principal(),
            host_asyncio_task=host_coro,
            governed_transport_cancel_admitted=True,
        )
    )
    assert result.disposition is RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED
    assert supervisor.supervised_count() == 1
    host_done.set()
    await asyncio.wait_for(host_coro, timeout=3.0)
    await asyncio.sleep(0.05)
    assert supervisor.supervised_count() == 0
    await ActiveTaskRegistry.unregister(task.task_id, run_id)


@pytest.mark.asyncio
async def test_handoff_refused_cross_tenant() -> None:
    supervisor = build_host_root_execution_transport_detachment_supervisor()
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    host_coro = asyncio.create_task(asyncio.sleep(3600))
    result = await supervisor.request_transport_detachment_handoff(
        RootExecutionTransportDetachmentHandoffRequest(
            task_id=task.task_id,
            run_id=run_id,
            verified_tenant_id=_TENANT,
            caller_request_identity=_principal(tenant_id=_OTHER),
            host_asyncio_task=host_coro,
            governed_transport_cancel_admitted=False,
        )
    )
    host_coro.cancel()
    with pytest.raises(asyncio.CancelledError):
        await host_coro
    assert result.disposition is RootExecutionTransportDetachmentHandoffDisposition.REFUSED
    assert (
        result.refusal_reason
        is RootExecutionTransportDetachmentHandoffRefusalReason.PRINCIPAL_TENANT_MISMATCH
    )
    await ActiveTaskRegistry.unregister(task.task_id, run_id)


@pytest.mark.asyncio
async def test_duplicate_handoff_idempotent_accepted() -> None:
    supervisor = build_host_root_execution_transport_detachment_supervisor()
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    host_coro = asyncio.create_task(asyncio.sleep(3600))
    req = RootExecutionTransportDetachmentHandoffRequest(
        task_id=task.task_id,
        run_id=run_id,
        verified_tenant_id=_TENANT,
        caller_request_identity=_principal(),
        host_asyncio_task=host_coro,
        governed_transport_cancel_admitted=True,
    )
    first = await supervisor.request_transport_detachment_handoff(req)
    second = await supervisor.request_transport_detachment_handoff(req)
    host_coro.cancel()
    with pytest.raises(asyncio.CancelledError):
        await host_coro
    assert first.disposition is RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED
    assert second.disposition is RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED
    assert supervisor.supervised_count() == 0
    await ActiveTaskRegistry.unregister(task.task_id, run_id)


@pytest.mark.asyncio
async def test_shutdown_refuses_new_handoff() -> None:
    supervisor = build_host_root_execution_transport_detachment_supervisor()
    await supervisor.shutdown_supervised_work(timeout_s=0.1)
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    host_coro = asyncio.create_task(asyncio.sleep(3600))
    result = await supervisor.request_transport_detachment_handoff(
        RootExecutionTransportDetachmentHandoffRequest(
            task_id=task.task_id,
            run_id=run_id,
            verified_tenant_id=_TENANT,
            caller_request_identity=_principal(),
            host_asyncio_task=host_coro,
            governed_transport_cancel_admitted=False,
        )
    )
    host_coro.cancel()
    with pytest.raises(asyncio.CancelledError):
        await host_coro
    assert result.disposition is RootExecutionTransportDetachmentHandoffDisposition.REFUSED
    assert (
        result.refusal_reason
        is RootExecutionTransportDetachmentHandoffRefusalReason.SHUTDOWN_IN_PROGRESS
    )
    await ActiveTaskRegistry.unregister(task.task_id, run_id)


@pytest.mark.asyncio
async def test_duplicate_handoff_conflicting_host_task_refused() -> None:
    supervisor = build_host_root_execution_transport_detachment_supervisor()
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    host_a = asyncio.create_task(asyncio.sleep(3600))
    host_b = asyncio.create_task(asyncio.sleep(3600))
    base = RootExecutionTransportDetachmentHandoffRequest(
        task_id=task.task_id,
        run_id=run_id,
        verified_tenant_id=_TENANT,
        caller_request_identity=_principal(),
        host_asyncio_task=host_a,
        governed_transport_cancel_admitted=True,
    )
    first = await supervisor.request_transport_detachment_handoff(base)
    second = await supervisor.request_transport_detachment_handoff(
        RootExecutionTransportDetachmentHandoffRequest(
            task_id=task.task_id,
            run_id=run_id,
            verified_tenant_id=_TENANT,
            caller_request_identity=_principal(),
            host_asyncio_task=host_b,
            governed_transport_cancel_admitted=True,
        )
    )
    host_a.cancel()
    host_b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await host_a
    with pytest.raises(asyncio.CancelledError):
        await host_b
    assert first.disposition is RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED
    assert second.disposition is RootExecutionTransportDetachmentHandoffDisposition.REFUSED
    assert (
        second.refusal_reason
        is RootExecutionTransportDetachmentHandoffRefusalReason.DUPLICATE_HANDOFF_CONFLICT
    )
    assert supervisor.supervised_count() == 0
    await ActiveTaskRegistry.unregister(task.task_id, run_id)


@pytest.mark.asyncio
async def test_shutdown_observation_does_not_cancel_host_execution() -> None:
    supervisor = build_host_root_execution_transport_detachment_supervisor()
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    host_release = asyncio.Event()
    cancelled_by_supervisor = False

    async def _host() -> str:
        nonlocal cancelled_by_supervisor
        try:
            await host_release.wait()
            return "ok"
        except asyncio.CancelledError:
            cancelled_by_supervisor = True
            raise

    host_coro = asyncio.create_task(_host())
    await asyncio.sleep(0)
    accepted = await supervisor.request_transport_detachment_handoff(
        RootExecutionTransportDetachmentHandoffRequest(
            task_id=task.task_id,
            run_id=run_id,
            verified_tenant_id=_TENANT,
            caller_request_identity=_principal(),
            host_asyncio_task=host_coro,
            governed_transport_cancel_admitted=False,
        )
    )
    assert accepted.disposition is RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED
    await supervisor.shutdown_supervised_work(timeout_s=0.05)
    assert not host_coro.done()
    assert not cancelled_by_supervisor
    host_release.set()
    await asyncio.wait_for(host_coro, timeout=3.0)
    await asyncio.sleep(0.05)
    assert supervisor.supervised_count() == 0
    await ActiveTaskRegistry.unregister(task.task_id, run_id)


@pytest.mark.asyncio
async def test_shutdown_repeat_is_idempotent() -> None:
    supervisor = build_host_root_execution_transport_detachment_supervisor()
    await supervisor.shutdown_supervised_work(timeout_s=0.01)
    await supervisor.shutdown_supervised_work(timeout_s=0.01)


@pytest.mark.asyncio
async def test_handoff_refused_when_host_completes_during_acceptance() -> None:
    supervisor = build_host_root_execution_transport_detachment_supervisor()
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)

    async def _quick() -> str:
        return "done"

    host_coro = asyncio.create_task(_quick())
    await host_coro
    result = await supervisor.request_transport_detachment_handoff(
        RootExecutionTransportDetachmentHandoffRequest(
            task_id=task.task_id,
            run_id=run_id,
            verified_tenant_id=_TENANT,
            caller_request_identity=_principal(),
            host_asyncio_task=host_coro,
            governed_transport_cancel_admitted=False,
        )
    )
    assert result.disposition is RootExecutionTransportDetachmentHandoffDisposition.REFUSED
    assert (
        result.refusal_reason
        is RootExecutionTransportDetachmentHandoffRefusalReason.HOST_TASK_ALREADY_DONE
    )
    await ActiveTaskRegistry.unregister(task.task_id, run_id)


@pytest.mark.asyncio
async def test_relinquish_admission_drops_supervision_without_cancelling_host() -> None:
    supervisor = build_host_root_execution_transport_detachment_supervisor()
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    host_done = asyncio.Event()

    async def _host() -> str:
        await host_done.wait()
        return "ok"

    host_coro = asyncio.create_task(_host())
    await asyncio.sleep(0)
    admitted = await supervisor.admit_root_execution_supervision(
        RootExecutionTransportAdmissionRequest(
            task_id=task.task_id,
            verified_tenant_id=_TENANT,
            caller_request_identity=_principal(),
            host_asyncio_task=host_coro,
        )
    )
    assert admitted.disposition is RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED
    relinquished = await supervisor.relinquish_root_execution_supervision_admission(
        task_id=task.task_id,
        host_asyncio_task=host_coro,
    )
    assert relinquished is True
    assert supervisor.supervised_count() == 0
    assert not host_coro.done()
    host_done.set()
    await asyncio.wait_for(host_coro, timeout=3.0)


@pytest.mark.asyncio
async def test_finalize_pre_start_retains_supervision_until_host_terminal() -> None:
    """Pre-start finalization must not relinquish supervision before the gated host is terminal."""
    supervisor = build_host_root_execution_transport_detachment_supervisor()
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    execution_start_gate = asyncio.Event()
    release_cancel = asyncio.Event()

    async def _gated_host() -> None:
        try:
            await execution_start_gate.wait()
        except asyncio.CancelledError:
            await release_cancel.wait()
            raise

    host_coro = asyncio.create_task(_gated_host())
    await asyncio.sleep(0)
    admitted = await supervisor.admit_root_execution_supervision(
        RootExecutionTransportAdmissionRequest(
            task_id=task.task_id,
            verified_tenant_id=_TENANT,
            caller_request_identity=_principal(),
            host_asyncio_task=host_coro,
        )
    )
    assert admitted.disposition is RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED

    finalize = asyncio.create_task(
        supervisor.finalize_root_execution_supervision_before_start(
            RootExecutionTransportPreStartFinalizationRequest(
                task_id=task.task_id,
                host_asyncio_task=host_coro,
                execution_start_gate=execution_start_gate,
                transport_waiter_will_continue=False,
                host_terminate_timeout_s=3.0,
            )
        )
    )
    await asyncio.sleep(0.05)
    assert supervisor.holds_root_execution_supervision(
        task_id=task.task_id,
        host_asyncio_task=host_coro,
    )
    release_cancel.set()
    result = await asyncio.wait_for(finalize, timeout=5.0)
    assert (
        result.disposition
        is RootExecutionTransportPreStartFinalizationDisposition.TERMINATED_UNDER_SUPERVISION
    )
    assert host_coro.done()
    assert supervisor.supervised_count() == 0
