# © Artur Czarnecki. All rights reserved.

"""MCP-RC MRC-3 — call bound as a governed-cancel trigger, admission slots and resilience (R-P7-04, OD-MCP-RC-02)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from intergrax.applications._shared.mcp_catalog_invocation_bridge import (
    McpCatalogInvocationComposition,
    execute_mcp_catalog_tool_invocation,
)
from intergrax.applications._shared.mcp_engine_concurrency import (
    McpConcurrentToolCallLimitExceeded,
    McpEngineConcurrentCallAdmission,
)
from intergrax.applications._shared.mcp_harness_principal import McpRunAgentIdentityResolution
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.execution.root_execution_transport_detachment_supervision import (
    build_host_root_execution_transport_detachment_supervisor,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from tests.qualification.mcp_server_x.test_mcp_server_x_p7_failure_injection_l2 import (
    _boundary,
    _HangingHost,
    _identity_resolution,
    _reference_server,
    _terminal_result,
    _wait_supervision_drained,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_NEXUS = "intergrax.applications._shared.mcp_nexus_server"
_TRANSPORT = "intergrax.applications._shared.mcp_host_execution_transport_cancel"
_T = 12.0


@pytest.fixture(autouse=True)
def _clean_registry_and_fast_drain(monkeypatch: pytest.MonkeyPatch):
    ActiveTaskRegistry.clear_for_tests()
    monkeypatch.setattr(f"{_TRANSPORT}._HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S", 0.25)
    yield
    ActiveTaskRegistry.clear_for_tests()


class _StubbornHost:
    """Host double that ignores the governed cancel until ``release`` (non-cooperative work)."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancel_seen = False
        self.tasks: list[Task] = []
        self.port = AsyncMock(spec=HostTaskExecutionPort)
        self.port.execute = AsyncMock(side_effect=self._execute)

    async def _execute(self, task: Task, **_kwargs: object) -> TaskResult:
        from intergrax.contracts.execution_identity import mint_run_id

        run_id = mint_run_id()
        self.tasks.append(task)
        await ActiveTaskRegistry.register(task, run_id)
        self.started.set()
        try:
            while not self.release.is_set():
                self.cancel_seen = self.cancel_seen or CancellationCoordinator.is_requested(task.metadata)
                await asyncio.sleep(0.02)
        finally:
            await ActiveTaskRegistry.unregister(task.task_id, run_id)
        return _terminal_result(task, str(run_id), TaskState.COMPLETED, "late")


def _patched(supervision):  # type: ignore[no-untyped-def]
    return (
        patch(f"{_NEXUS}.resolve_mcp_run_agent_identity", return_value=_identity_resolution()),
        patch(f"{_NEXUS}.resolve_transport_detachment_supervision_from_request", return_value=supervision),
    )


# --- deadline as a governed-cancel trigger -------------------------------------------------------------


async def test_call_bound_requests_governed_cancel_and_reports_deadline_exceeded() -> None:
    host = _HangingHost()
    supervision = build_host_root_execution_transport_detachment_supervisor()
    mcp = _reference_server(
        host.port, task_control_mutation_boundary=_boundary(), max_call_seconds=0.3, max_concurrent_tool_calls=1
    )
    p1, p2 = _patched(supervision)
    with p1, p2:
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="MCP_DEADLINE_EXCEEDED") as excinfo:
                await asyncio.wait_for(client.call_tool("run_agent", {"message": "hang"}), timeout=_T)
            assert "outcome=cancelled" in str(excinfo.value)
            # the governed cancel reached the task through the canonical coordinator
            (task,) = host.tasks
            assert CancellationCoordinator.is_requested(task.metadata)
            assert host.governed_cancel_seen is True
            assert host.outcomes == [TaskState.CANCELLED]
            # the admission slot (limit 1) was released: the next call is admitted
            again = await client.call_tool("qual_echo", {"message": "slot-free"})
            assert again.data["echo"] == "slot-free"
    await _wait_supervision_drained(supervision)


async def test_non_cooperative_host_is_never_reported_as_success_and_stays_supervised() -> None:
    host = _StubbornHost()
    supervision = build_host_root_execution_transport_detachment_supervisor()
    mcp = _reference_server(host.port, task_control_mutation_boundary=_boundary(), max_call_seconds=0.2)
    p1, p2 = _patched(supervision)
    with p1, p2:
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="MCP_DEADLINE_EXCEEDED") as excinfo:
                await asyncio.wait_for(client.call_tool("run_agent", {"message": "stubborn"}), timeout=_T)
        text = str(excinfo.value)
        assert "outcome=outcome_unknown" in text or "outcome=cancelled" in text
        # the governed cancel was requested even though the host ignored it
        (task,) = host.tasks
        assert CancellationCoordinator.is_requested(task.metadata)
        # the work outlives the MCP call: it is owned by the canonical supervisor, not silently dropped
        assert host.port.execute.await_count == 1
        host.release.set()
    await _wait_supervision_drained(supervision)


async def test_without_a_call_bound_a_hung_call_is_not_cancelled_by_a_timer() -> None:
    host = _HangingHost()
    supervision = build_host_root_execution_transport_detachment_supervisor()
    mcp = _reference_server(host.port, task_control_mutation_boundary=_boundary())
    p1, p2 = _patched(supervision)
    with p1, p2:
        call = asyncio.create_task(mcp.call_tool("run_agent", {"message": "hang"}))
        await asyncio.wait_for(host.started.wait(), timeout=_T)
        await asyncio.sleep(0.6)
        assert not call.done()
        assert not CancellationCoordinator.is_requested(host.tasks[0].metadata)
        host.release.set()
        result = await asyncio.wait_for(call, timeout=_T)
    assert result.structured_content["state"] == TaskState.COMPLETED.value
    await _wait_supervision_drained(supervision)


async def test_a_completed_run_inside_the_bound_is_returned_unchanged() -> None:
    port = AsyncMock(spec=HostTaskExecutionPort)

    async def _fast(task: Task, **_kwargs: object) -> TaskResult:
        from intergrax.contracts.execution_identity import mint_run_id

        return _terminal_result(task, str(mint_run_id()), TaskState.COMPLETED, "quick")

    port.execute = AsyncMock(side_effect=_fast)
    mcp = _reference_server(port, max_call_seconds=5.0)
    async with Client(mcp) as client:
        result = await client.call_tool("run_agent", {"message": "x"})
    assert result.data["answer"] == "quick"


async def test_local_dev_path_without_cancel_context_also_honours_the_bound() -> None:
    gate = asyncio.Event()
    port = AsyncMock(spec=HostTaskExecutionPort)

    async def _slow(task: Task, **_kwargs: object) -> TaskResult:
        await gate.wait()
        raise AssertionError("must have been cancelled by the bound")

    port.execute = AsyncMock(side_effect=_slow)
    mcp = _reference_server(port, max_call_seconds=0.2)
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="MCP_DEADLINE_EXCEEDED"):
            await asyncio.wait_for(client.call_tool("run_agent", {"message": "x"}), timeout=_T)


# --- admission ---------------------------------------------------------------------------------------


async def test_over_limit_call_raises_the_declared_concurrency_code_and_keeps_the_legacy_string() -> None:
    release = asyncio.Event()

    async def _hold(message: str = "") -> dict[str, object]:
        await release.wait()
        return {"echo": message}

    mcp = _reference_server(AsyncMock(spec=HostTaskExecutionPort), custom_echo=_hold, max_concurrent_tool_calls=1)
    async with Client(mcp) as client:
        first = asyncio.create_task(client.call_tool("qual_echo", {"message": "a"}))
        await asyncio.sleep(0.1)
        with pytest.raises(ToolError, match="MCP_CONCURRENT_CALL_LIMIT_EXCEEDED.*mcp_concurrent_call_limit_exceeded"):
            await client.call_tool("qual_echo", {"message": "b"})
        release.set()
        await first


async def test_slot_is_released_after_a_handler_failure() -> None:
    state = {"fail": True}

    async def _flaky(message: str = "") -> dict[str, object]:
        if state["fail"]:
            raise RuntimeError("boom")
        return {"echo": message}

    mcp = _reference_server(AsyncMock(spec=HostTaskExecutionPort), custom_echo=_flaky, max_concurrent_tool_calls=1)
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="MCP_INTERNAL_ERROR"):
            await client.call_tool("qual_echo", {"message": "a"})
        state["fail"] = False
        ok = await client.call_tool("qual_echo", {"message": "b"})
    assert ok.data["echo"] == "b"


async def test_admission_release_is_cancellation_safe() -> None:
    admission = McpEngineConcurrentCallAdmission(2)
    entered = asyncio.Event()

    async def _holder() -> None:
        async with admission:
            entered.set()
            await asyncio.sleep(60)

    tasks = [asyncio.create_task(_holder()) for _ in range(2)]
    await asyncio.sleep(0.05)
    assert admission.inflight == 2
    with pytest.raises(McpConcurrentToolCallLimitExceeded):
        async with admission:
            pass
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert admission.inflight == 0


async def test_admission_cap_holds_under_a_burst_of_concurrent_entries() -> None:
    admission = McpEngineConcurrentCallAdmission(3)
    release = asyncio.Event()
    admitted = 0
    refused = 0

    async def _enter() -> None:
        nonlocal admitted, refused
        try:
            async with admission:
                admitted += 1
                await release.wait()
        except McpConcurrentToolCallLimitExceeded:
            refused += 1

    burst = [asyncio.create_task(_enter()) for _ in range(20)]
    await asyncio.sleep(0.1)
    assert admitted == 3 and refused == 17
    release.set()
    await asyncio.gather(*burst)
    assert admission.inflight == 0


# --- catalog invoke bound ---------------------------------------------------------------------------


async def test_catalog_invoke_exceeding_the_bound_is_outcome_unknown_not_success() -> None:
    from tests.qualification.mcp_server_x.test_mcp_server_x_p7_l2_security_trace import (
        _AGENT,
        _SERVICE_A,
        _STEP,
        _TENANT_A,
        _contract,
        _RecordingCatalogInvoker,
        _register_binding,
        _tool_registry,
    )

    class _HangingChild:
        async def execute(self, **_kwargs: object) -> object:
            await asyncio.sleep(60)

    task, run_id = await _register_binding(tenant=_TENANT_A, user_id=_SERVICE_A)
    identity = RequestIdentity(
        tenant_id=_TENANT_A, user_id=_SERVICE_A, principal_type=PrincipalType.USER, auth_subject=_SERVICE_A
    )
    resolution = McpRunAgentIdentityResolution(
        tenant_id=_TENANT_A, user_id=_SERVICE_A, principal=None, request_identity=identity
    )
    composition = McpCatalogInvocationComposition(
        catalog_invoker=_RecordingCatalogInvoker(),
        registry=_tool_registry(_contract("p7.ok")),
        catalog_tool_child_execution=_HangingChild(),
        invoke_timeout_s=0.2,
    )
    bridge = "intergrax.applications._shared.mcp_catalog_invocation_bridge"
    with (
        patch(f"{bridge}.resolve_mcp_run_agent_identity", return_value=resolution),
        patch(f"{bridge}._in_process_parent_matches_binding", return_value=True),
    ):
        with pytest.raises(ToolError, match="MCP_DEADLINE_EXCEEDED.*outcome_unknown"):
            await asyncio.wait_for(
                execute_mcp_catalog_tool_invocation(
                    composition,
                    tool_id="p7.ok",
                    task_id=str(task.task_id),
                    run_id=run_id,
                    step_id=_STEP,
                ),
                timeout=_T,
            )
    assert _AGENT  # imported fixture constant used by the shared helpers


# --- a slow host-provided registry must not stall the server event loop --------------------------------


async def test_slow_catalog_registry_read_does_not_block_other_calls() -> None:
    import time

    from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
    from intergrax.tools.registry.runtime import ToolRegistry
    from pydantic import BaseModel
    from tests.qualification.mcp_server_x.reference_app.build import (
        build_mcp_server_x_reference_qualification_server,
    )

    class _In(BaseModel):
        query: str = "x"

    class _Out(BaseModel):
        value: str = "ok"

    async def _handler(_input: _In, _ctx: object) -> _Out:
        return _Out()

    registry = ToolRegistry()
    registry.register(
        ToolContract(
            tool_id="rc.slow",
            name="rc.slow",
            description="slow registry read",
            input_schema=_In,
            output_schema=_Out,
            error_mapping={},
            side_effects=False,
            risk_level=ToolRiskLevel.LOW,
            category="rc",
        ),
        _handler,
    )
    original_list = ToolRegistry.list

    def _slow_list(self: ToolRegistry):  # type: ignore[no-untyped-def]
        if getattr(self, "_rc_slow", False):
            time.sleep(1.0)
        return original_list(self)

    mcp = build_mcp_server_x_reference_qualification_server(
        host_execution=AsyncMock(spec=HostTaskExecutionPort),
        registry=__import__("intergrax.runtime.registry.agent_registry", fromlist=["AgentRegistry"]).AgentRegistry(),
        tool_registry=registry,
    )
    with patch.object(ToolRegistry, "list", _slow_list):
        view_registries = [r for r in [registry] if r is not None]
        for reg in view_registries:
            reg._rc_slow = True  # type: ignore[attr-defined]
        async with Client(mcp) as client:
            started = time.monotonic()
            slow_call = asyncio.create_task(client.call_tool("list_catalog_tools", {}))
            await asyncio.sleep(0.1)
            echoed = await asyncio.wait_for(client.call_tool("qual_echo", {"message": "free"}), timeout=_T)
            quick = time.monotonic() - started  # client and server share this loop: a blocked loop delays this
            await slow_call
    assert echoed.data["echo"] == "free"
    assert quick < 0.8, f"another call was stalled for {quick:.2f}s by a slow registry read"
