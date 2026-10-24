# © Artur Czarnecki. All rights reserved.

"""TLX-TL11 OWN-A2 — canonical transport detachment supervision."""

from __future__ import annotations

import asyncio

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.execution_identity import mint_run_id
from intergrax.contracts.root_execution_transport_detachment_supervision import (
    RootExecutionTransportDetachmentHandoffDisposition,
    RootExecutionTransportDetachmentHandoffRefusalReason,
    RootExecutionTransportDetachmentHandoffRequest,
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
