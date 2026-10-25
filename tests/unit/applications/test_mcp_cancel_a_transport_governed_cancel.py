# © Artur Czarnecki. All rights reserved.

"""TLX-TL11 CANCEL-A — MCP transport cancel → governed cooperative cancel."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import AsyncMock, patch

import pytest
import uvicorn
from fastapi import FastAPI

from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_harness_control_plane_mutation_boundary,
)
from intergrax.applications._shared.mcp_host_execution_transport_cancel import (
    McpHostExecutionTransportCancelContext,
    McpHostExecutionTransportOwnershipInvariantError,
    McpProtectedHostExecutionAdmissionError,
    create_mcp_protected_host_execution_launch,
    run_host_execution_with_mcp_transport_cancel,
)
from intergrax.applications._shared.mcp_nexus_server import (
    build_nexus_mcp_server,
    execute_mcp_agent_task,
)
from intergrax.applications._shared.task_control import TaskControlResult
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.execution.root_execution_transport_detachment_supervision import (
    build_host_root_execution_transport_detachment_supervisor,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.runtime.task.task import TaskResult

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-cancel-a"
_OTHER = "tenant-other"
_API_KEY = "cancel-a-qual-key"


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


def _transport_supervision():
    return build_host_root_execution_transport_detachment_supervisor()


@pytest.fixture(autouse=True)
def fast_mcp_transport_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "intergrax.applications._shared.mcp_host_execution_transport_cancel."
        "_HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S",
        0.25,
    )


def _cancelled_result(task: Task, run_id: str) -> TaskResult:
    return TaskResult(
        task_id=task.task_id,
        run_id=run_id,
        state=TaskState.CANCELLED,
        answer="",
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )


@contextmanager
def _uvicorn_mcp_base_url(app: FastAPI) -> Iterator[str]:
    bind = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bind.bind(("127.0.0.1", 0))
    port = bind.getsockname()[1]
    bind.close()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        raise RuntimeError("uvicorn_mcp_qualification_startup_timeout")
    try:
        yield f"http://127.0.0.1:{port}/mcp/"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


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
            while not CancellationCoordinator.is_requested(task.metadata):
                await asyncio.sleep(0.02)
            await cancel_release.wait()
        finally:
            await ActiveTaskRegistry.unregister(task.task_id, run_id)
        return _cancelled_result(task, run_id)

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
            transport_detachment_supervision=_transport_supervision(),
        ),
    )
    await asyncio.wait_for(hang_started.wait(), timeout=3.0)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(call, timeout=12.0)

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
    release = asyncio.Event()

    async def _other_tenant_hang(task: Task, **_kwargs: object) -> TaskResult:
        nonlocal registered_task
        registered_task = task
        await ActiveTaskRegistry.register(task, run_id)
        try:
            await release.wait()
        finally:
            await ActiveTaskRegistry.unregister(task.task_id, run_id)
        return _cancelled_result(task, run_id)

    port.execute = AsyncMock(side_effect=_other_tenant_hang)
    with pytest.raises(McpProtectedHostExecutionAdmissionError):
        await execute_mcp_agent_task(
            port,
            message="x",
            capability="demo.cap",
            tenant_id=_OTHER,
            user_id="u",
            caller_request_identity=_principal(tenant_id=_TENANT),
            task_control_mutation_boundary=boundary,
            transport_detachment_supervision=_transport_supervision(),
        )
    assert registered_task is None
    port.execute.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_cancel_a_governed_rejection_does_not_certify_admission() -> None:
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary = _harness_boundary()
    cancel_ctx = McpHostExecutionTransportCancelContext(
        mutation_boundary=boundary,
        caller_request_identity=_principal(),
    )
    host_done = asyncio.Event()

    async def _host() -> TaskResult:
        await host_done.wait()
        return _cancelled_result(task, run_id)

    launch = create_mcp_protected_host_execution_launch(_host)
    waiter = asyncio.create_task(
        run_host_execution_with_mcp_transport_cancel(
            launch.host_task,
            task=task,
            cancel_ctx=cancel_ctx,
            transport_detachment_supervision=_transport_supervision(),
            execution_start_gate=launch.execution_start_gate,
        ),
    )
    await asyncio.sleep(0.05)
    with patch(
        "intergrax.applications._shared.mcp_host_execution_transport_cancel."
        "governed_cancel_active_task",
        new_callable=AsyncMock,
        return_value=TaskControlResult(
            task_id=task.task_id,
            action="cancel",
            accepted=False,
            detail="task_not_cancellable",
        ),
    ):
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, timeout=12.0)
    assert not CancellationCoordinator.is_requested(task.metadata)
    host_done.set()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(launch.host_task, timeout=3.0)
    await ActiveTaskRegistry.unregister(task.task_id, run_id)


@pytest.mark.asyncio
async def test_cancel_a_host_completes_during_transport_cancel_drain() -> None:
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary = _harness_boundary()
    cancel_ctx = McpHostExecutionTransportCancelContext(
        mutation_boundary=boundary,
        caller_request_identity=_principal(),
    )
    completed = TaskResult(
        task_id=task.task_id,
        run_id=run_id,
        state=TaskState.COMPLETED,
        answer="done",
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )

    async def _host() -> TaskResult:
        await asyncio.sleep(0.05)
        return completed

    launch = create_mcp_protected_host_execution_launch(_host)
    waiter = asyncio.create_task(
        run_host_execution_with_mcp_transport_cancel(
            launch.host_task,
            task=task,
            cancel_ctx=cancel_ctx,
            transport_detachment_supervision=_transport_supervision(),
            execution_start_gate=launch.execution_start_gate,
        ),
    )
    await asyncio.sleep(0.01)
    waiter.cancel()
    result = await asyncio.wait_for(waiter, timeout=12.0)
    assert result == completed
    await ActiveTaskRegistry.unregister(task.task_id, run_id)


@pytest.mark.asyncio
async def test_cancel_a_pre_admission_transport_cancel_no_false_cooperative_signal() -> None:
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    boundary = _harness_boundary()
    cancel_ctx = McpHostExecutionTransportCancelContext(
        mutation_boundary=boundary,
        caller_request_identity=_principal(),
    )
    admission = asyncio.Event()

    async def _slow_host() -> TaskResult:
        await admission.wait()
        run_id = mint_run_id()
        await ActiveTaskRegistry.register(task, run_id)
        return TaskResult(
            task_id=task.task_id,
            run_id=run_id,
            state=TaskState.COMPLETED,
            answer="ok",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )

    supervision = _transport_supervision()
    launch = create_mcp_protected_host_execution_launch(_slow_host)
    waiter = asyncio.create_task(
        run_host_execution_with_mcp_transport_cancel(
            launch.host_task,
            task=task,
            cancel_ctx=cancel_ctx,
            transport_detachment_supervision=supervision,
            execution_start_gate=launch.execution_start_gate,
        ),
    )
    await asyncio.sleep(0.05)
    waiter.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(waiter, timeout=12.0)
    assert not CancellationCoordinator.is_requested(task.metadata)
    assert supervision.supervised_count() == 1
    admission.set()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(launch.host_task, timeout=5.0)


@pytest.mark.asyncio
async def test_cancel_a_transport_cleanup_bounded_when_host_suppresses_cancellation() -> None:
    """OWN-A2: transport waiter ends after accepted handoff; host stays supervised."""
    supervision = _transport_supervision()
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary = _harness_boundary()
    cancel_ctx = McpHostExecutionTransportCancelContext(
        mutation_boundary=boundary,
        caller_request_identity=_principal(),
    )
    host_release = asyncio.Event()

    async def _suppress_cancel_host() -> TaskResult:
        try:
            while not host_release.is_set():
                if CancellationCoordinator.is_requested(task.metadata):
                    break
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    pass
        finally:
            await ActiveTaskRegistry.unregister(task.task_id, run_id)
        return _cancelled_result(task, run_id)

    launch = create_mcp_protected_host_execution_launch(_suppress_cancel_host)
    waiter = asyncio.create_task(
        run_host_execution_with_mcp_transport_cancel(
            launch.host_task,
            task=task,
            cancel_ctx=cancel_ctx,
            transport_detachment_supervision=supervision,
            execution_start_gate=launch.execution_start_gate,
        ),
    )
    await asyncio.sleep(0.05)
    started = time.monotonic()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, timeout=12.0)
    assert time.monotonic() - started < 2.0
    assert not launch.host_task.done()
    assert supervision.supervised_count() == 1
    assert CancellationCoordinator.is_requested(task.metadata)
    host_release.set()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(launch.host_task, timeout=5.0)


@pytest.mark.asyncio
async def test_cancel_a_http_disconnect_observed_admits_governed_cancel() -> None:
    """Unit monitor path: disconnect observation routes to governed cancel (not L3 physical proof)."""
    hang_started = asyncio.Event()
    registered: Task | None = None
    run_id = mint_run_id()
    disconnect_seen = asyncio.Event()

    async def _hang(task: Task, **_kwargs: object) -> TaskResult:
        nonlocal registered
        registered = task
        await ActiveTaskRegistry.register(task, run_id)
        hang_started.set()
        try:
            while not CancellationCoordinator.is_requested(task.metadata):
                await asyncio.sleep(0.02)
        finally:
            await ActiveTaskRegistry.unregister(task.task_id, run_id)
        return _cancelled_result(task, run_id)

    async def _disconnected(_request: object) -> bool:
        return disconnect_seen.is_set()

    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(side_effect=_hang)
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
            transport_detachment_supervision=_transport_supervision(),
        ),
    )
    with patch(
        "intergrax.applications._shared.mcp_host_execution_transport_cancel._client_disconnected",
        side_effect=_disconnected,
    ):
        await asyncio.wait_for(hang_started.wait(), timeout=3.0)
        disconnect_seen.set()
        await call
    assert registered is not None
    assert CancellationCoordinator.is_requested(registered.metadata)


@pytest.mark.asyncio
async def test_cancel_a_l3_streamable_http_run_agent_with_task_control_boundary() -> None:
    """Physical streamable HTTP/uvicorn: MCP run_agent with coupled task-control boundary."""
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(
        return_value=TaskResult(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="l3-ok",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        ),
    )
    api = FastAPI()
    api.state.harness_auth = HarnessAuthState(
        require_api_key=True,
        resolved_api_key=_API_KEY,
        api_key_principal_tenant_id=_TENANT,
    )
    boundary = _harness_boundary()
    mcp = build_nexus_mcp_server(
        name="cancel-a-l3",
        host_execution=port,
        registry=AgentRegistry(),
        default_capability="demo.cap",
        task_control_mutation_boundary=boundary,
    )
    supervision = _transport_supervision()
    app = couple_fastapi_with_mcp(
        api,
        mcp,
        mount_path="/mcp",
        task_control_mutation_boundary=boundary,
        transport_detachment_supervision=supervision,
    )
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    with _uvicorn_mcp_base_url(app) as base_url:
        async with streamablehttp_client(base_url, headers={"X-Api-Key": _API_KEY}) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await asyncio.wait_for(
                    session.call_tool(
                        "run_agent",
                        {"message": "m", "capability": "demo.cap"},
                    ),
                    timeout=12.0,
                )
    assert not result.isError
    port.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_a_l3_physical_http_client_abandon_admits_governed_cancel() -> None:
    """Streamable HTTP: real client transport teardown → server disconnect → governed cancel."""
    hang_started = threading.Event()
    governed_cancel_seen = threading.Event()
    run_finished = threading.Event()
    run_id = mint_run_id()

    async def _hang_execute(task: Task, **_kwargs: object) -> TaskResult:
        await ActiveTaskRegistry.register(task, run_id)
        hang_started.set()
        try:
            while not CancellationCoordinator.is_requested(task.metadata):
                await asyncio.sleep(0.02)
            governed_cancel_seen.set()
        finally:
            await ActiveTaskRegistry.unregister(task.task_id, run_id)
            run_finished.set()
        return _cancelled_result(task, run_id)

    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(side_effect=_hang_execute)
    boundary = _harness_boundary()
    api = FastAPI()
    api.state.harness_auth = HarnessAuthState(
        require_api_key=True,
        resolved_api_key=_API_KEY,
        api_key_principal_tenant_id=_TENANT,
    )
    mcp = build_nexus_mcp_server(
        name="cancel-a-l3-abandon",
        host_execution=port,
        registry=AgentRegistry(),
        default_capability="demo.cap",
        task_control_mutation_boundary=boundary,
    )
    supervision = _transport_supervision()
    app = couple_fastapi_with_mcp(
        api,
        mcp,
        mount_path="/mcp",
        task_control_mutation_boundary=boundary,
        transport_detachment_supervision=supervision,
    )
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def _abandon_client(base_url: str) -> None:
        async with streamablehttp_client(base_url, headers={"X-Api-Key": _API_KEY}) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                call = asyncio.create_task(
                    session.call_tool(
                        "run_agent",
                        {"message": "m", "capability": "demo.cap"},
                    ),
                )
                deadline = time.monotonic() + 8.0
                while not hang_started.is_set() and time.monotonic() < deadline:
                    await asyncio.sleep(0.03)
                assert hang_started.is_set()
                call.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait_for(call, timeout=3.0)

    with _uvicorn_mcp_base_url(app) as base_url:
        await asyncio.wait_for(_abandon_client(base_url), timeout=15.0)

    deadline = time.monotonic() + 10.0
    while not governed_cancel_seen.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert governed_cancel_seen.is_set()
    deadline = time.monotonic() + 10.0
    while not run_finished.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert run_finished.is_set()


@pytest.mark.asyncio
async def test_own_a2_refused_handoff_raises_not_silent_transport_release() -> None:
    """OWN-FINAL-01: shutdown refuses admission — no live execution without accountable owner."""
    supervision = _transport_supervision()
    await supervision.shutdown_supervised_work(timeout_s=0.01)
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary = _harness_boundary()
    cancel_ctx = McpHostExecutionTransportCancelContext(
        mutation_boundary=boundary,
        caller_request_identity=_principal(),
    )
    launch = create_mcp_protected_host_execution_launch(lambda: asyncio.sleep(3600))
    waiter = asyncio.create_task(
        run_host_execution_with_mcp_transport_cancel(
            launch.host_task,
            task=task,
            cancel_ctx=cancel_ctx,
            transport_detachment_supervision=supervision,
            execution_start_gate=launch.execution_start_gate,
        ),
    )
    with pytest.raises(McpProtectedHostExecutionAdmissionError):
        await asyncio.wait_for(waiter, timeout=12.0)
    assert launch.host_task.done()
    assert supervision.supervised_count() == 0
    await ActiveTaskRegistry.unregister(task.task_id, run_id)


@pytest.mark.asyncio
async def test_own_a2_protected_http_missing_supervision_fail_closed_before_execution() -> None:
    """OWN-FINAL-02: coupled HTTP MCP with governance boundary but no supervision rejects before execute."""
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_cancelled_result(Task(tenant_id=_TENANT, user_id="u", message="m"), mint_run_id()))
    boundary = _harness_boundary()
    api = FastAPI()
    api.state.harness_auth = HarnessAuthState(
        require_api_key=True,
        resolved_api_key=_API_KEY,
        api_key_principal_tenant_id=_TENANT,
    )
    mcp = build_nexus_mcp_server(
        name="own-a2-admission",
        host_execution=port,
        registry=AgentRegistry(),
        default_capability="demo.cap",
        task_control_mutation_boundary=boundary,
    )
    app = couple_fastapi_with_mcp(
        api,
        mcp,
        mount_path="/mcp",
        task_control_mutation_boundary=boundary,
    )
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    with _uvicorn_mcp_base_url(app) as base_url:
        async with streamablehttp_client(base_url, headers={"X-Api-Key": _API_KEY}) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await asyncio.wait_for(
                    session.call_tool(
                        "run_agent",
                        {"message": "m", "capability": "demo.cap"},
                    ),
                    timeout=12.0,
                )
    assert result.isError
    port.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_a_governed_admitted_without_supervision_raises_ownership_invariant() -> None:
    """OWN-FINAL-01: governed cancel admission without supervision cannot release a live host."""
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    boundary = _harness_boundary()
    cancel_ctx = McpHostExecutionTransportCancelContext(
        mutation_boundary=boundary,
        caller_request_identity=_principal(),
    )
    host_release = asyncio.Event()

    async def _host() -> TaskResult:
        await host_release.wait()
        return _cancelled_result(task, run_id)

    host_coro = asyncio.create_task(_host())
    with patch(
        "intergrax.applications._shared.mcp_host_execution_transport_cancel."
        "governed_cancel_active_task",
        new_callable=AsyncMock,
        return_value=TaskControlResult(
            task_id=task.task_id,
            action="cancel",
            accepted=True,
            detail="ok",
        ),
    ):
        waiter = asyncio.create_task(
            run_host_execution_with_mcp_transport_cancel(
                host_coro,
                task=task,
                cancel_ctx=cancel_ctx,
                transport_detachment_supervision=None,
            ),
        )
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(McpHostExecutionTransportOwnershipInvariantError):
            await asyncio.wait_for(waiter, timeout=12.0)
    assert not host_coro.done()
    host_release.set()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(host_coro, timeout=3.0)
    await ActiveTaskRegistry.unregister(task.task_id, run_id)


@pytest.mark.asyncio
async def test_adm01_business_execution_blocked_until_supervision_accepts() -> None:
    """ADM-01: host business work cannot start before admit_root_execution_supervision ACCEPT."""
    supervision = _transport_supervision()
    task = Task(tenant_id=_TENANT, user_id="u", message="m")
    boundary = _harness_boundary()
    cancel_ctx = McpHostExecutionTransportCancelContext(
        mutation_boundary=boundary,
        caller_request_identity=_principal(),
    )
    business_started = asyncio.Event()
    admit_release = asyncio.Event()
    original_admit = supervision.admit_root_execution_supervision

    async def _slow_admit(request: object) -> object:
        await admit_release.wait()
        return await original_admit(request)

    async def _host() -> str:
        business_started.set()
        return "ok"

    supervision.admit_root_execution_supervision = _slow_admit  # type: ignore[method-assign]
    launch = create_mcp_protected_host_execution_launch(_host)
    waiter = asyncio.create_task(
        run_host_execution_with_mcp_transport_cancel(
            launch.host_task,
            task=task,
            cancel_ctx=cancel_ctx,
            transport_detachment_supervision=supervision,
            execution_start_gate=launch.execution_start_gate,
        ),
    )
    await asyncio.sleep(0.05)
    assert not business_started.is_set()
    admit_release.set()
    result = await asyncio.wait_for(waiter, timeout=12.0)
    assert result == "ok"
    assert business_started.is_set()


@pytest.mark.asyncio
async def test_adm01_refused_admission_never_starts_business_execution() -> None:
    """ADM-02: REFUSED admission terminates gated host without running business work."""
    supervision = _transport_supervision()
    await supervision.shutdown_supervised_work(timeout_s=0.01)
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock()
    boundary = _harness_boundary()
    with pytest.raises(McpProtectedHostExecutionAdmissionError):
        await execute_mcp_agent_task(
            port,
            message="m",
            capability="demo.cap",
            tenant_id=_TENANT,
            user_id="u",
            caller_request_identity=_principal(),
            task_control_mutation_boundary=boundary,
            transport_detachment_supervision=supervision,
        )
    port.execute.assert_not_awaited()
