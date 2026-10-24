# © Artur Czarnecki. All rights reserved.

"""TLX-TL11 CANCEL-A — MCP transport cancel → governed cooperative cancel."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_harness_control_plane_mutation_boundary,
)
from intergrax.applications._shared.mcp_nexus_server import execute_mcp_agent_task
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.execution_identity import mint_run_id
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.runtime.task.task import TaskResult

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-cancel-a"
_OTHER = "tenant-other"


def _principal(*, tenant_id: str = _TENANT) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id="operator-1",
        principal_type=PrincipalType.USER,
        auth_subject="operator-1",
    )


def _harness_boundary():
    env = ApplicationEnvironmentProfile(application_profile=ApplicationProfile.PRODUCT)
    return build_harness_control_plane_mutation_boundary(env)


@pytest.mark.asyncio
async def test_cancel_a_governed_cancel_invoked_on_asyncio_transport_cancel() -> None:
    hang_started = asyncio.Event()
    registered_task: Task | None = None
    run_id = mint_run_id()

    cancel_release = asyncio.Event()

    async def _hang_execute(task: Task, **_kwargs: object) -> TaskResult:
        nonlocal registered_task
        registered_task = task
        await ActiveTaskRegistry.register(task, run_id)
        hang_started.set()
        try:
            await cancel_release.wait()
        finally:
            await ActiveTaskRegistry.unregister(task.task_id, run_id)
        return TaskResult(
            task_id=task.task_id,
            run_id=run_id,
            state=TaskState.CANCELLED,
            answer="",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )

    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(side_effect=_hang_execute)
    boundary = _harness_boundary()

    call = asyncio.create_task(
        execute_mcp_agent_task(
            port,
            message="hang",
            capability="demo.cap",
            tenant_id=_TENANT,
            user_id="u1",
            caller_request_identity=_principal(),
            task_control_mutation_boundary=boundary,
        ),
    )
    await asyncio.wait_for(hang_started.wait(), timeout=3.0)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert registered_task is not None
    assert CancellationCoordinator.is_requested(registered_task.metadata)
    cancel_release.set()


@pytest.mark.asyncio
async def test_cancel_a_cross_tenant_governed_cancel_not_applied() -> None:
    """Forged tenant on tool args without verified principal does not reach governed cancel."""

    port = AsyncMock(spec=HostTaskExecutionPort)
    boundary = _harness_boundary()
    registered_task: Task | None = None
    run_id = mint_run_id()

    async def _other_tenant_hang(task: Task, **_kwargs: object) -> TaskResult:
        nonlocal registered_task
        registered_task = task
        await ActiveTaskRegistry.register(task, run_id)
        hang_started = asyncio.Event()
        hang_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await ActiveTaskRegistry.unregister(task.task_id, run_id)
        return TaskResult(
            task_id=task.task_id,
            run_id=run_id,
            state=TaskState.CANCELLED,
            answer="",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )

    port.execute = AsyncMock(side_effect=_other_tenant_hang)
    host_task = asyncio.create_task(
        execute_mcp_agent_task(
            port,
            message="x",
            capability="demo.cap",
            tenant_id=_OTHER,
            user_id="u",
            caller_request_identity=_principal(tenant_id=_TENANT),
            task_control_mutation_boundary=boundary,
        ),
    )
    await asyncio.sleep(0.15)
    host_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await host_task
    assert registered_task is not None
    assert not CancellationCoordinator.is_requested(registered_task.metadata)


@pytest.mark.asyncio
async def test_cancel_a_without_boundary_skips_governed_cancel_path() -> None:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(
        return_value=TaskResult(
            task_id="t",
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="ok",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        ),
    )
    with patch(
        "intergrax.applications._shared.mcp_host_execution_transport_cancel."
        "run_host_execution_with_mcp_transport_cancel",
        new_callable=AsyncMock,
    ) as mock_transport:
        await execute_mcp_agent_task(
            port,
            message="m",
            capability="demo.cap",
            tenant_id=_TENANT,
            user_id="u",
            caller_request_identity=None,
            task_control_mutation_boundary=None,
        )
        mock_transport.assert_not_called()
    port.execute.assert_awaited_once()
