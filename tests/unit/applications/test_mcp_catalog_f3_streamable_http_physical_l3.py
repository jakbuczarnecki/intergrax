# © Artur Czarnecki. All rights reserved.

"""F3 physical L3 — streamable HTTP/uvicorn, canonical harness host, OWN-A2, cross-loop ingress."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator
from unittest.mock import AsyncMock, patch

import pytest
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_harness_control_plane_mutation_boundary,
)
from intergrax.applications._shared.mcp_catalog_invocation_bridge import (
    _METADATA_STEP_ID,
    _serialize_tool_result,
)
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.applications._shared.mcp_nexus_server import task_from_mcp_agent_intake
from intergrax.applications._shared.production_catalog_tool_child_execution_wiring import (
    build_production_catalog_tool_child_execution_port,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.tool_execution_outcome import ToolExecutionResult
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.execution.host_task import TaskBoundAgenticDelegate
from intergrax.runtime.execution.root_execution_child_work_ingress import (
    RootExecutionChildWorkIngress,
)
from intergrax.runtime.execution.root_execution_transport_detachment_supervision import (
    build_host_root_execution_transport_detachment_supervisor,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.runtime.task.task import TaskResult
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from tests.fixtures.harness_host_task_execution import build_harness_host_task_execution

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.network]

_TENANT = "tenant-f3-http-l3"
_USER = "harness-api-key"
_STEP = "f3.http.catalog.step"
_API_KEY = "f3-http-l3-qual-key"
_CAP = "f3.http.cap"


class _In(BaseModel):
    query: str = "x"


class _Out(BaseModel):
    value: str = "invoked"


@dataclass
class _RecordingCatalogInvoker:
    results: list[ExecutionBoundCatalogToolInvokeRequest] = field(default_factory=list)
    delay_s: float = 0.35
    started: asyncio.Event = field(default_factory=asyncio.Event)

    def invoke(self, request: ExecutionBoundCatalogToolInvokeRequest) -> ToolExecutionResult[_Out]:
        self.results.append(request)
        self.started.set()
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        return ToolExecutionResult.ok(_Out(value="child-http-done"))


def _contract(tool_id: str) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description="f3 http l3",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        category="f3",
    )


def _handler() -> ToolHandler[_In, _Out]:
    async def _run(_input: _In, _ctx: object) -> _Out:
        return _Out(value="unused")

    return _run


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
    deadline = time.monotonic() + 12.0
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


def _task_with_catalog_step(**kwargs: object) -> Task:
    task = task_from_mcp_agent_intake(**kwargs)  # type: ignore[arg-type]
    task.metadata[_METADATA_STEP_ID] = _STEP
    task.metadata["workspace_id"] = "ws-f3-http"
    task.agent_id = "agent-f3-http"
    return task


def _tool_payload(result: object) -> dict[str, object]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    return {}


def _build_http_mcp_app(
    invoker: _RecordingCatalogInvoker,
) -> FastAPI:
    registry = AgentRegistry()
    nexus = NexusLoop(registry)
    child_port = build_production_catalog_tool_child_execution_port()
    ingress = RootExecutionChildWorkIngress(
        catalog_child_execution=child_port,
        serialize_tool_result=_serialize_tool_result,
    )
    host = build_harness_host_task_execution(
        nexus,
        orchestration_triggers=frozenset(),
        root_child_work_ingress=ingress,
    )
    tool_registry = ToolRegistry()
    tool_registry.register(_contract("f3.http.ok"), _handler())
    env = ApplicationEnvironmentProfile(application_profile=ApplicationProfile.PRODUCT)
    boundary = build_harness_control_plane_mutation_boundary(env)
    supervision = build_host_root_execution_transport_detachment_supervisor()
    mcp = build_nexus_mcp_server(
        name="f3-streamable-http-l3",
        host_execution=host,
        registry=registry,
        default_capability=_CAP,
        default_tenant_id=_TENANT,
        default_user_id=_USER,
        tool_registry=tool_registry,
        catalog_tool_invoker=invoker,
        task_control_mutation_boundary=boundary,
        catalog_root_child_work_ingress=ingress,
        enable_cross_loop_catalog_child=True,
        transport_detachment_supervision=supervision,
        auto_wire_catalog_transport_supervision=False,
    )
    api = FastAPI()
    api.state.harness_auth = HarnessAuthState(
        require_api_key=True,
        resolved_api_key=_API_KEY,
        api_key_principal_tenant_id=_TENANT,
    )
    return couple_fastapi_with_mcp(
        api,
        mcp,
        mount_path="/mcp",
        task_control_mutation_boundary=boundary,
        transport_detachment_supervision=supervision,
    )


@pytest.fixture(autouse=True)
async def _clear_tasks() -> None:
    ActiveTaskRegistry.clear_for_tests()
    yield
    ActiveTaskRegistry.clear_for_tests()


@pytest.fixture(autouse=True)
def _fast_transport_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "intergrax.applications._shared.mcp_host_execution_transport_cancel."
        "_HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S",
        0.25,
    )


@pytest.fixture(autouse=True)
def _catalog_step_on_mcp_intake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "intergrax.applications._shared.mcp_nexus_server.task_from_mcp_agent_intake",
        _task_with_catalog_step,
    )


async def _streamable_call_tool(
    base_url: str,
    tool: str,
    arguments: dict[str, object],
    *,
    timeout_s: float = 30.0,
) -> object:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def _run() -> object:
        async with streamablehttp_client(
            base_url,
            headers={"X-Api-Key": _API_KEY},
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool, arguments)

    return await asyncio.wait_for(_run(), timeout=timeout_s)


@pytest.mark.asyncio
async def test_f3_streamable_http_catalog_child_under_canonical_root() -> None:
    root_hang = threading.Event()
    invoker = _RecordingCatalogInvoker(delay_s=0.4)
    app = _build_http_mcp_app(invoker)

    async def _slow_root(_request: object) -> TaskResult:
        task_ids = await ActiveTaskRegistry.list_ids()
        binding = await ActiveTaskRegistry.get(task_ids[0])
        assert binding is not None
        task = binding.task
        run_id = binding.run_id
        root_hang.set()
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if CancellationCoordinator.is_requested(dict(task.metadata)):
                break
            await asyncio.sleep(0.03)
        return TaskResult(
            task_id=task.task_id,
            run_id=str(run_id),
            state=TaskState.COMPLETED,
            answer="root-http-done",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )

    with patch.object(
        TaskBoundAgenticDelegate,
        "execute",
        new_callable=AsyncMock,
        side_effect=_slow_root,
    ):
        with _uvicorn_mcp_base_url(app) as base_url:
            run_task = asyncio.create_task(
                _streamable_call_tool(
                    base_url,
                    "run_agent",
                    {"message": "f3-root", "capability": _CAP},
                ),
            )
            if not await asyncio.to_thread(root_hang.wait, 8.0):
                raise TimeoutError("root execution did not reach slow delegate")
            task_ids = await ActiveTaskRegistry.list_ids()
            assert len(task_ids) == 1
            binding = await ActiveTaskRegistry.get(task_ids[0])
            assert binding is not None
            catalog = await _streamable_call_tool(
                base_url,
                "invoke_catalog_tool",
                {
                    "tool_id": "f3.http.ok",
                    "task_id": str(binding.task_id),
                    "run_id": str(binding.run_id),
                    "step_id": _STEP,
                    "input": {"query": "hi"},
                },
            )
            run_result = await asyncio.wait_for(run_task, timeout=20.0)
    catalog_payload = _tool_payload(catalog)
    assert invoker.results, catalog_payload
    assert catalog_payload.get("success") is True
    assert not run_result.isError
    run_payload = _tool_payload(run_result)
    assert str(run_payload.get("task_id")) == str(binding.task_id)


@pytest.mark.asyncio
async def test_f3_streamable_http_parent_cancel_during_catalog_child() -> None:
    invoker = _RecordingCatalogInvoker(delay_s=0.55)
    app = _build_http_mcp_app(invoker)
    root_hang = threading.Event()

    async def _slow_root(_request: object) -> TaskResult:
        task_ids = await ActiveTaskRegistry.list_ids()
        binding = await ActiveTaskRegistry.get(task_ids[0])
        assert binding is not None
        task = binding.task
        run_id = binding.run_id
        root_hang.set()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if CancellationCoordinator.is_requested(dict(task.metadata)):
                return TaskResult(
                    task_id=task.task_id,
                    run_id=str(run_id),
                    state=TaskState.CANCELLED,
                    answer="",
                    authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
                )
            await asyncio.sleep(0.03)
        return TaskResult(
            task_id=task.task_id,
            run_id=str(run_id),
            state=TaskState.COMPLETED,
            answer="done",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )

    with patch.object(
        TaskBoundAgenticDelegate,
        "execute",
        new_callable=AsyncMock,
        side_effect=_slow_root,
    ):
        with _uvicorn_mcp_base_url(app) as base_url:
            run_task = asyncio.create_task(
                _streamable_call_tool(
                    base_url,
                    "run_agent",
                    {"message": "f3-cancel", "capability": _CAP},
                ),
            )
            if not await asyncio.to_thread(root_hang.wait, 8.0):
                raise TimeoutError("root execution did not reach slow delegate")
            binding = await ActiveTaskRegistry.get((await ActiveTaskRegistry.list_ids())[0])
            assert binding is not None

            async def _request_parent_cancel() -> None:
                await asyncio.sleep(0.06)
                refreshed = await ActiveTaskRegistry.get(binding.task_id)
                assert refreshed is not None
                CancellationCoordinator.request(
                    refreshed.task,
                    reason="f3_http_parent_cancel_l3",
                )

            cancel_task = asyncio.create_task(_request_parent_cancel())
            catalog = await _streamable_call_tool(
                base_url,
                "invoke_catalog_tool",
                {
                    "tool_id": "f3.http.ok",
                    "task_id": str(binding.task_id),
                    "run_id": str(binding.run_id),
                },
            )
            await cancel_task
            await asyncio.wait_for(run_task, timeout=25.0)

    catalog_payload = _tool_payload(catalog)
    assert catalog_payload.get("success") is not True
    assert catalog_payload.get("error") in {
        "execution_cancelled",
        "execution_admission_denied",
    }


@pytest.mark.asyncio
async def test_f3_streamable_http_transport_disconnect_during_catalog_child() -> None:
    """HTTP client abandons in-flight catalog invoke; root supervision still drains."""
    invoker = _RecordingCatalogInvoker(delay_s=0.6)
    app = _build_http_mcp_app(invoker)
    root_hang = threading.Event()
    invoker_started = threading.Event()

    async def _slow_root(_request: object) -> TaskResult:
        task_ids = await ActiveTaskRegistry.list_ids()
        binding = await ActiveTaskRegistry.get(task_ids[0])
        assert binding is not None
        task = binding.task
        run_id = binding.run_id
        root_hang.set()
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            if CancellationCoordinator.is_requested(dict(task.metadata)):
                break
            await asyncio.sleep(0.03)
        return TaskResult(
            task_id=task.task_id,
            run_id=str(run_id),
            state=TaskState.COMPLETED,
            answer="root-after-disconnect",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )

    original_invoke = invoker.invoke

    def _invoke_track(request: ExecutionBoundCatalogToolInvokeRequest) -> ToolExecutionResult[_Out]:
        invoker_started.set()
        return original_invoke(request)

    invoker.invoke = _invoke_track  # type: ignore[method-assign]

    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def _abandon_catalog_client(base_url: str) -> None:
        async with streamablehttp_client(
            base_url,
            headers={"X-Api-Key": _API_KEY},
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                run_call = asyncio.create_task(
                    session.call_tool(
                        "run_agent",
                        {"message": "f3-disc", "capability": _CAP},
                    ),
                )
                if not await asyncio.to_thread(root_hang.wait, 8.0):
                    raise TimeoutError("root did not start")
                binding = await ActiveTaskRegistry.get((await ActiveTaskRegistry.list_ids())[0])
                assert binding is not None
                catalog_call = asyncio.create_task(
                    session.call_tool(
                        "invoke_catalog_tool",
                        {
                            "tool_id": "f3.http.ok",
                            "task_id": str(binding.task_id),
                            "run_id": str(binding.run_id),
                            "step_id": _STEP,
                        },
                    ),
                )
                deadline = time.monotonic() + 8.0
                while not invoker_started.is_set() and time.monotonic() < deadline:
                    await asyncio.sleep(0.03)
                assert invoker_started.is_set()
                catalog_call.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait_for(catalog_call, timeout=3.0)
                await asyncio.wait_for(run_call, timeout=20.0)

    with patch.object(
        TaskBoundAgenticDelegate,
        "execute",
        new_callable=AsyncMock,
        side_effect=_slow_root,
    ):
        with _uvicorn_mcp_base_url(app) as base_url:
            await asyncio.wait_for(_abandon_catalog_client(base_url), timeout=30.0)

    assert invoker.results
    assert await ActiveTaskRegistry.list_ids() == []
