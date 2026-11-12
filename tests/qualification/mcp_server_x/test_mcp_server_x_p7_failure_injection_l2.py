# © Artur Czarnecki. All rights reserved.

"""MCP-SERVER-X P7 (MCP-P7-05, ticket T-IGX-035-H) - failure injection, level L2.

Level: **L2 - in-process, not a physical transport test.** The reference MCP server is composed
exactly as in production (``compose_mcp_server_from_definition``) and driven through
``fastmcp.Client(mcp)`` (in-memory transport) or the server-side ``mcp.call_tool`` coroutine.
There is no socket, no HTTP and no client disconnect on the wire here. Where ``run_agent`` needs a
verified caller (the production wiring takes it from the HTTP request), the module injects the
principal and the transport-detachment supervisor by patching the two resolver call sites, so
these tests prove the governed-cancel *logic*, not the transport. Physical streamable HTTP
evidence for the same scenarios is in ``test_mcp_server_x_p7_failure_injection_l3.py``; an L2
result is never counted as L3.

Scenarios (MCP-ACC-13 cancel, MCP-ACC-14 timeout, MCP-ACC-15 disconnect, plus limit exceeded and
unknown tool of AC-P7-04). Every scenario asserts a terminal state and runs under a deadline, so
no call can hang the suite.

MCP-ACC-14 at L2: ``run_agent`` has no timeout/deadline parameter and the MCP transport module is
documented as *not* the root Execution deadline authority. The owning API
(``ExecutionDeadlineAuthorityResolver``, policy-driven by ``RunBudget.max_wall_time_seconds``)
exists in Execution but is not wired into the reference host composition and not reachable from
the MCP surface, so a canonical-execution deadline through MCP is recorded as **not applicable**
by ``test_acc_14_canonical_execution_deadline_is_not_reachable_from_run_agent`` (a passing test
that pins those facts), and the caller-side deadline is exercised instead.
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from intergrax.applications._shared import mcp_host_execution_transport_cancel as transport_cancel_module
from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_harness_control_plane_mutation_boundary,
)
from intergrax.applications._shared.mcp_harness_principal import McpRunAgentIdentityResolution
from intergrax.applications._shared.mcp_host_execution_transport_cancel import (
    request_governed_cancel_for_bound_mcp_task,
)
from intergrax.applications._shared.task_control import governed_cancel_active_task
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.execution import deadline_authority
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.execution.root_execution_transport_detachment_supervision import (
    build_host_root_execution_transport_detachment_supervisor,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.tools.providers.provider_resource_enforcement import DEFAULT_MCP_MAX_SERIALIZED_BYTES
from tests.fixtures.harness_host_task_execution import build_harness_host_task_execution
from tests.qualification.mcp_server_x.reference_app.build import (
    build_mcp_server_x_reference_qualification_server,
)
from tests.qualification.mcp_server_x.reference_app.variants import (
    CATALOG_TOOL_ID,
    RecordingCatalogInvoker,
    build_catalog_registry,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_TENANT = "p7-fi-l2-tenant"
_OTHER_TENANT = "p7-fi-l2-other-tenant"
_USER = "p7-fi-l2-user"
_DEADLINE_S = 12.0
_ENGINE = "intergrax.applications._shared.mcp_server_engine"
_NEXUS_SERVER = "intergrax.applications._shared.mcp_nexus_server"
_BRIDGE = "intergrax.applications._shared.mcp_catalog_invocation_bridge"
_TRANSPORT = "intergrax.applications._shared.mcp_host_execution_transport_cancel"


@pytest.fixture(autouse=True)
def _clean_active_tasks_and_fast_drain(monkeypatch: pytest.MonkeyPatch):
    ActiveTaskRegistry.clear_for_tests()
    monkeypatch.setattr(f"{_TRANSPORT}._HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S", 0.25)
    yield
    ActiveTaskRegistry.clear_for_tests()


# --------------------------------------------------------------------------- helpers


def _principal(*, tenant_id: str = _TENANT) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=_USER,
        principal_type=PrincipalType.USER,
        auth_subject=_USER,
    )


def _identity_resolution(*, tenant_id: str = _TENANT) -> McpRunAgentIdentityResolution:
    return McpRunAgentIdentityResolution(
        tenant_id=tenant_id,
        user_id=_USER,
        principal=None,
        request_identity=_principal(tenant_id=tenant_id),
    )


def _boundary():
    env = ApplicationEnvironmentProfile(application_profile=ApplicationProfile.PRODUCT)
    return build_harness_control_plane_mutation_boundary(env)


def _terminal_result(task: Task, run_id: str, state: TaskState, answer: str = "") -> TaskResult:
    return TaskResult(
        task_id=task.task_id,
        run_id=run_id,
        state=state,
        answer=answer,
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )


class _HangingHost:
    """Host execution port double: registers an active binding, then runs until a governed cancel."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.governed_cancel_seen = False
        self.tasks: list[Task] = []
        self.outcomes: list[TaskState] = []
        self.release = asyncio.Event()  # safety valve so a failing test cannot leave a live task
        self.port = AsyncMock(spec=HostTaskExecutionPort)
        self.port.execute = AsyncMock(side_effect=self._execute)

    async def _execute(self, task: Task, **_kwargs: object) -> TaskResult:
        run_id = mint_run_id()
        self.tasks.append(task)
        await ActiveTaskRegistry.register(task, run_id)
        self.started.set()
        try:
            while not CancellationCoordinator.is_requested(task.metadata) and not self.release.is_set():
                await asyncio.sleep(0.02)
            self.governed_cancel_seen = CancellationCoordinator.is_requested(task.metadata)
        finally:
            await ActiveTaskRegistry.unregister(task.task_id, run_id)
            self.finished.set()
        state = TaskState.CANCELLED if self.governed_cancel_seen else TaskState.COMPLETED
        self.outcomes.append(state)
        return _terminal_result(task, str(run_id), state)


def _reference_server(host_port: object, **kwargs: object):
    return build_mcp_server_x_reference_qualification_server(
        host_execution=host_port,  # type: ignore[arg-type]
        registry=kwargs.pop("registry", AgentRegistry()),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


class _SlowTool:
    """Custom reference tool that holds an admission slot until ``delay_s`` elapses or ``release``."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.done = asyncio.Event()
        self.release = asyncio.Event()
        self.outcomes: list[str] = []

    async def handler(self, delay_s: float = 0.0) -> dict[str, object]:
        self.started.set()
        try:
            await asyncio.wait_for(self.release.wait(), timeout=delay_s)
            self.outcomes.append("released")
        except TimeoutError:
            self.outcomes.append("completed")
        except asyncio.CancelledError:
            self.outcomes.append("cancelled")
            raise
        finally:
            self.done.set()
        return {"slept": delay_s}


async def _wait_supervision_drained(supervision, *, timeout_s: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while supervision.supervised_count() > 0 and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.02)
    assert supervision.supervised_count() == 0


# --------------------------------------------------------------------------- MCP-ACC-13 (cancel)


async def test_acc_13_cancel_of_in_flight_run_agent_reaches_governed_cancel() -> None:
    """The server-side ``run_agent`` coroutine is cancelled mid-invoke; a governed cancel is admitted."""
    host = _HangingHost()
    supervision = build_host_root_execution_transport_detachment_supervisor()
    mcp = _reference_server(host.port, task_control_mutation_boundary=_boundary())
    with (
        patch(f"{_ENGINE}.resolve_mcp_run_agent_identity", return_value=_identity_resolution()),
        patch(f"{_NEXUS_SERVER}.resolve_transport_detachment_supervision_from_request", return_value=supervision),
    ):
        call = asyncio.create_task(mcp.call_tool("run_agent", {"message": "hang"}))
        await asyncio.wait_for(host.started.wait(), timeout=_DEADLINE_S)
        call.cancel()
        # The transport waiter admits the governed cancel and drains the cooperative host; the host finishes
        # inside the bounded drain, so the cancelled call ends with the host's terminal result (it is not
        # re-raised as CancelledError - that only happens when the host outlives the drain bound).
        result = await asyncio.wait_for(call, timeout=_DEADLINE_S)
        assert result.structured_content["state"] == TaskState.CANCELLED.value
        # terminal state: the governed cancel reached the task, the host finished as CANCELLED, nothing is left bound
        await asyncio.wait_for(host.finished.wait(), timeout=_DEADLINE_S)
    (task,) = host.tasks
    assert CancellationCoordinator.is_requested(task.metadata)
    assert host.governed_cancel_seen is True
    assert host.outcomes == [TaskState.CANCELLED]
    assert await ActiveTaskRegistry.get(task.task_id) is None
    await _wait_supervision_drained(supervision)


async def test_acc_13_governed_cancel_for_unknown_task_id_is_fail_closed() -> None:
    """Unknown task id: no active binding, so no cancel is admitted and nothing is mutated."""
    unknown = Task(tenant_id=_TENANT, user_id=_USER, message="never registered")
    accepted = await request_governed_cancel_for_bound_mcp_task(
        task=unknown,
        principal=_principal(),
        mutation_boundary=_boundary(),
    )
    assert accepted is False
    assert not CancellationCoordinator.is_requested(unknown.metadata)

    result = await governed_cancel_active_task(
        task_id=str(mint_task_id()),
        run_id=str(mint_run_id()),
        mutation_id="p7-fi-unknown-task",
        principal=_principal(),
        mutation_boundary=_boundary(),
    )
    assert result.accepted is False
    assert result.detail == "task_not_active"


async def test_acc_13_invoke_with_unknown_task_id_is_missing_execution_binding_over_the_reference_server() -> None:
    """``invoke_catalog_tool`` for an unknown task id is a typed fail-closed error; the invoker never runs."""
    invoker = RecordingCatalogInvoker()
    mcp = _reference_server(
        AsyncMock(spec=HostTaskExecutionPort),
        tool_registry=build_catalog_registry(),
        catalog_tool_invoker=invoker,
        task_control_mutation_boundary=_boundary(),
    )
    with patch(f"{_BRIDGE}.resolve_mcp_run_agent_identity", return_value=_identity_resolution()):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="missing_execution_binding"):
                await asyncio.wait_for(
                    client.call_tool(
                        "invoke_catalog_tool",
                        {
                            "tool_id": CATALOG_TOOL_ID,
                            "task_id": str(mint_task_id()),
                            "run_id": str(mint_run_id()),
                        },
                    ),
                    timeout=_DEADLINE_S,
                )
    assert invoker.requests == []


async def test_acc_13_cancel_by_another_tenant_principal_is_not_applied() -> None:
    """A principal of another tenant cannot cancel a bound task: rejected, task cancellation not requested."""
    host = _HangingHost()
    supervision = build_host_root_execution_transport_detachment_supervisor()
    mcp = _reference_server(host.port, task_control_mutation_boundary=_boundary())
    with (
        patch(f"{_ENGINE}.resolve_mcp_run_agent_identity", return_value=_identity_resolution()),
        patch(f"{_NEXUS_SERVER}.resolve_transport_detachment_supervision_from_request", return_value=supervision),
    ):
        call = asyncio.create_task(mcp.call_tool("run_agent", {"message": "hang"}))
        await asyncio.wait_for(host.started.wait(), timeout=_DEADLINE_S)
        (task,) = host.tasks
        binding = await ActiveTaskRegistry.get(task.task_id)
        assert binding is not None
        result = await governed_cancel_active_task(
            task_id=str(task.task_id),
            run_id=str(binding.run_id),
            mutation_id="p7-fi-cross-tenant-cancel",
            principal=_principal(tenant_id=_OTHER_TENANT),
            mutation_boundary=_boundary(),
        )
        assert result.accepted is False
        assert result.detail == "tenant_authority_mismatch"
        assert not CancellationCoordinator.is_requested(task.metadata)
        # clean up through the sanctioned owner: its own tenant cancels, the run reaches its terminal state
        own = await governed_cancel_active_task(
            task_id=str(task.task_id),
            run_id=str(binding.run_id),
            mutation_id="p7-fi-own-tenant-cancel",
            principal=_principal(),
            mutation_boundary=_boundary(),
        )
        assert own.accepted is True
        done = await asyncio.wait_for(call, timeout=_DEADLINE_S)
    assert done.structured_content["state"] == TaskState.CANCELLED.value
    assert host.outcomes == [TaskState.CANCELLED]
    await _wait_supervision_drained(supervision)


# --------------------------------------------------------------------------- MCP-ACC-14 (timeout)


async def test_acc_14_caller_deadline_cancels_in_flight_handler_and_releases_admission_slot() -> None:
    """A caller deadline on a slow reference tool ends the call in a terminal (cancelled) state."""
    slow = _SlowTool()
    mcp = _reference_server(
        AsyncMock(spec=HostTaskExecutionPort),
        extra_custom_handlers={"qual_slow": slow.handler},
        max_concurrent_tool_calls=1,
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(mcp.call_tool("qual_slow", {"delay_s": 30.0}), timeout=0.4)
    assert slow.started.is_set()
    await asyncio.wait_for(slow.done.wait(), timeout=_DEADLINE_S)
    assert slow.outcomes == ["cancelled"]
    # the single admission slot was released: a following call is admitted and completes
    follow_up = await asyncio.wait_for(mcp.call_tool("qual_echo", {"message": "after-timeout"}), timeout=_DEADLINE_S)
    assert follow_up.structured_content["echo"] == "after-timeout"


async def test_acc_14_client_deadline_through_in_memory_client_ends_terminal_and_client_stays_usable() -> None:
    """In-memory client: the client-side deadline fires, the handler reaches a terminal state, no hang."""
    slow = _SlowTool()
    mcp = _reference_server(
        AsyncMock(spec=HostTaskExecutionPort),
        extra_custom_handlers={"qual_slow": slow.handler},
    )
    async with Client(mcp) as client:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(client.call_tool("qual_slow", {"delay_s": 1.0}), timeout=0.3)
        # terminal handler state is "completed" (the in-memory client does not cancel the server) or
        # "cancelled"; either way it ends within a bound and never hangs
        await asyncio.wait_for(slow.done.wait(), timeout=_DEADLINE_S)
        assert slow.outcomes[0] in {"completed", "cancelled"}
        echoed = await asyncio.wait_for(client.call_tool("qual_echo", {"message": "still-up"}), timeout=_DEADLINE_S)
    assert echoed.data["echo"] == "still-up"


async def test_acc_14_canonical_execution_deadline_is_not_reachable_from_run_agent() -> None:
    """N/A record: no per-request canonical-execution deadline can be set through MCP ``run_agent``.

    Facts pinned (each failing would mean the N/A reasoning changed and this record must be revisited):
    ``run_agent`` takes no timeout/deadline argument; the reference host composition carries no
    ``ExecutionDeadlineAuthorityResolver``; the MCP transport module declares it is not the deadline
    authority; the owning API exists in Execution.
    """
    mcp = _reference_server(AsyncMock(spec=HostTaskExecutionPort))
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    properties = tools["run_agent"].inputSchema["properties"]
    assert not [name for name in properties if "timeout" in name.lower() or "deadline" in name.lower()], properties

    host = build_harness_host_task_execution(NexusLoop(AgentRegistry()), orchestration_triggers=frozenset())
    assert host._deadline_authority_resolver is None

    assert "not root Execution deadline authority" in inspect.getsource(transport_cancel_module)
    assert inspect.isclass(deadline_authority.ExecutionDeadlineAuthorityResolver)


# --------------------------------------------------------------------------- MCP-ACC-15 (disconnect)


async def test_acc_15_disconnect_observation_admits_governed_cancel_in_process() -> None:
    """L2 only (supplements the L3 module): a monitor-observed disconnect routes to a governed cancel.

    The disconnect is injected by patching the transport module's ``_client_disconnected`` probe.
    No socket is involved, so this is not a physical disconnect proof.
    """
    host = _HangingHost()
    supervision = build_host_root_execution_transport_detachment_supervisor()
    disconnected = asyncio.Event()

    async def _probe(_request: object) -> bool:
        return disconnected.is_set()

    mcp = _reference_server(host.port, task_control_mutation_boundary=_boundary())
    with (
        patch(f"{_ENGINE}.resolve_mcp_run_agent_identity", return_value=_identity_resolution()),
        patch(f"{_NEXUS_SERVER}.resolve_transport_detachment_supervision_from_request", return_value=supervision),
        patch(f"{_TRANSPORT}._client_disconnected", side_effect=_probe),
    ):
        call = asyncio.create_task(mcp.call_tool("run_agent", {"message": "hang"}))
        await asyncio.wait_for(host.started.wait(), timeout=_DEADLINE_S)
        disconnected.set()
        result = await asyncio.wait_for(call, timeout=_DEADLINE_S)
    # the call itself ended in a terminal state (cancelled), delivered as a normal result
    assert result.structured_content["state"] == TaskState.CANCELLED.value
    (task,) = host.tasks
    assert CancellationCoordinator.is_requested(task.metadata)
    assert host.outcomes == [TaskState.CANCELLED]
    assert await ActiveTaskRegistry.get(task.task_id) is None
    await _wait_supervision_drained(supervision)


# --------------------------------------------------------------------------- AC-P7-04: limit exceeded


async def test_p7_05_limit_exceeded_second_concurrent_call_is_rejected_and_slot_is_released() -> None:
    """``max_concurrent_tool_calls=1``: the 2nd in-flight call is a typed error, never a hang or a queue."""
    slow = _SlowTool()
    mcp = _reference_server(
        AsyncMock(spec=HostTaskExecutionPort),
        extra_custom_handlers={"qual_slow": slow.handler},
        max_concurrent_tool_calls=1,
    )
    async with Client(mcp) as client:
        first = asyncio.create_task(client.call_tool("qual_slow", {"delay_s": 30.0}))
        await asyncio.wait_for(slow.started.wait(), timeout=_DEADLINE_S)
        # injected on a host builtin and on the custom reference tool (the custom wrapper applies admission too)
        with pytest.raises(ToolError, match="mcp_concurrent_call_limit_exceeded"):
            await asyncio.wait_for(client.call_tool("list_agents", {}), timeout=_DEADLINE_S)
        with pytest.raises(ToolError, match="mcp_concurrent_call_limit_exceeded"):
            await asyncio.wait_for(client.call_tool("qual_echo", {"message": "over-limit"}), timeout=_DEADLINE_S)
        slow.release.set()
        held = await asyncio.wait_for(first, timeout=_DEADLINE_S)
        assert held.data["slept"] == 30.0
        assert slow.outcomes == ["released"]
        again = await asyncio.wait_for(client.call_tool("qual_echo", {"message": "slot-free"}), timeout=_DEADLINE_S)
    assert again.data["echo"] == "slot-free"


async def test_p7_05_limit_exceeded_result_bound_on_list_agents_is_a_typed_error_not_an_empty_catalog() -> None:
    """A roster larger than the serialized byte ceiling raises a typed error; it is never emitted as ``[]``."""

    class _Contract:
        id = "agent-big"
        name = "x" * (DEFAULT_MCP_MAX_SERIALIZED_BYTES + 1024)
        capabilities = ("cap",)

    class _Agent:
        def get_contract(self) -> _Contract:
            return _Contract()

    class _BigRegistry:
        def list_agent_ids(self) -> list[str]:
            return ["agent-big"]

        def get(self, _agent_id: str) -> _Agent:
            return _Agent()

    mcp = _reference_server(AsyncMock(spec=HostTaskExecutionPort), registry=_BigRegistry())
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="mcp_payload_byte_limit_exceeded"):
            await asyncio.wait_for(client.call_tool("list_agents", {}), timeout=_DEADLINE_S)


async def test_p7_05_limit_exceeded_result_bound_on_run_agent_is_a_typed_error_payload() -> None:
    """An over-large ``run_agent`` answer is replaced by a typed error payload (terminal, bounded)."""
    port = AsyncMock(spec=HostTaskExecutionPort)

    async def _huge(task: Task, **_kwargs: object) -> TaskResult:
        return _terminal_result(
            task, str(mint_run_id()), TaskState.COMPLETED, "y" * (DEFAULT_MCP_MAX_SERIALIZED_BYTES + 1024)
        )

    port.execute = AsyncMock(side_effect=_huge)
    mcp = _reference_server(port)
    async with Client(mcp) as client:
        result = await asyncio.wait_for(client.call_tool("run_agent", {"message": "big"}), timeout=_DEADLINE_S)
    assert result.data == {"error": "mcp_payload_byte_limit_exceeded"}


async def test_p7_05_limit_exceeded_result_bound_is_not_applied_to_the_custom_tool_qual_echo() -> None:
    """RECORDED GAP (not a pass of a bound): the custom-tool wrapper applies concurrency admission only.

    An over-large ``qual_echo`` result is returned in full instead of a typed ``mcp_payload_byte_limit_exceeded``
    error, unlike ``list_agents`` / ``run_agent``. This pins current behaviour so the gap stays visible; if the
    engine later bounds custom tool results, this test must be updated to assert the typed error.
    """
    oversize = DEFAULT_MCP_MAX_SERIALIZED_BYTES + 1024
    mcp = _reference_server(AsyncMock(spec=HostTaskExecutionPort))
    async with Client(mcp) as client:
        result = await asyncio.wait_for(client.call_tool("qual_echo", {"message": "z" * oversize}), timeout=_DEADLINE_S)
    assert len(result.data["echo"]) == oversize
    assert "error" not in result.data


# --------------------------------------------------------------------------- AC-P7-04: unknown tool


async def test_p7_05_unknown_tool_is_an_error_result_and_the_server_stays_usable() -> None:
    """Calling a tool that is not published is an error (no redirect, no fallback tool), then service continues."""
    port = AsyncMock(spec=HostTaskExecutionPort)
    mcp = _reference_server(port)
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="(?i)unknown tool"):
            await asyncio.wait_for(client.call_tool("no_such_tool", {}), timeout=_DEADLINE_S)
        raw = await asyncio.wait_for(client.call_tool("no_such_tool", {}, raise_on_error=False), timeout=_DEADLINE_S)
        assert raw.is_error is True
        ok = await asyncio.wait_for(client.call_tool("qual_echo", {"message": "after-unknown"}), timeout=_DEADLINE_S)
    assert ok.data["echo"] == "after-unknown"
    port.execute.assert_not_awaited()


async def test_p7_05_unknown_tool_unpublished_builtin_is_not_redirected_to_another_tool() -> None:
    """An unpublished builtin name (``invoke_catalog_tool`` without a wired invoker) is unknown, not redirected."""
    mcp = _reference_server(AsyncMock(spec=HostTaskExecutionPort), tool_registry=build_catalog_registry())
    async with Client(mcp) as client:
        names = {tool.name for tool in await client.list_tools()}
        assert "invoke_catalog_tool" not in names
        with pytest.raises(ToolError, match="(?i)unknown tool"):
            await asyncio.wait_for(client.call_tool("invoke_catalog_tool", {}), timeout=_DEADLINE_S)
