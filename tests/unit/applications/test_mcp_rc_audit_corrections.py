# © Artur Czarnecki. All rights reserved.

"""MCP-RC corrective loop after the independent audit (verdict PARTIAL): AF-01, AF-02, AF-03, AF-04, AF-10.

Each test pins one audit finding so that the corresponding auditor mutant is killed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastmcp import Client
from fastmcp.exceptions import ToolError
from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.mcp_catalog_invocation_bridge import (
    McpCatalogInvocationComposition,
    execute_mcp_catalog_tool_invocation,
)
from intergrax.applications._shared.mcp_harness_principal import (
    McpRunAgentIdentityResolution,
    resolve_mcp_run_agent_identity,
)
from intergrax.applications._shared.mcp_identity_mode import McpIdentityMode
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.root_execution_child_work_ingress import RootCatalogChildWorkTerminal
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.execution.root_execution_transport_detachment_supervision import (
    build_host_root_execution_transport_detachment_supervisor,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.registry.runtime import ToolRegistry
from pydantic import BaseModel
from tests.qualification.mcp_server_x.reference_app.build import (
    build_mcp_server_x_reference_qualification_server,
)
from tests.qualification.mcp_server_x.test_mcp_server_x_p7_failure_injection_l2 import (
    _boundary,
    _identity_resolution,
    _reference_server,
    _terminal_result,
    _wait_supervision_drained,
)
from tests.unit.applications.test_mcp_trust_a_principal_binding import (
    _completed,
    _streamable_mcp_call_tool,
    _uvicorn_mcp_base_url,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_LOCAL = McpIdentityMode.LOCAL_DEV_EXPLICIT
_NEXUS = "intergrax.applications._shared.mcp_nexus_server"
_TRANSPORT = "intergrax.applications._shared.mcp_host_execution_transport_cancel"
_BRIDGE = "intergrax.applications._shared.mcp_catalog_invocation_bridge"
_T = 12.0
_SECRET = "SECRET-HOST-FAILURE-hunter2 /srv/app/db.py:42"


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    ActiveTaskRegistry.clear_for_tests()
    monkeypatch.setattr(f"{_TRANSPORT}._HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S", 0.25)
    yield
    ActiveTaskRegistry.clear_for_tests()


def _host() -> AsyncMock:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed())
    return port


# --- AF-01 / AF-10: no raw exception text on any tool path ---------------------------------------------


async def test_run_agent_host_exception_text_is_not_echoed() -> None:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(side_effect=RuntimeError(_SECRET))
    mcp = build_nexus_mcp_server(
        name="af01", host_execution=port, registry=AgentRegistry(), default_capability="c", identity_mode=_LOCAL
    )
    async with Client(mcp) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("run_agent", {"message": "x"})
    text = str(excinfo.value)
    assert "MCP_INTERNAL_ERROR" in text and "RuntimeError" in text
    assert "SECRET" not in text and "hunter2" not in text and "db.py" not in text


async def test_list_agents_registry_exception_text_is_not_echoed() -> None:
    class _BrokenRegistry:
        def list_agent_ids(self) -> list[str]:
            raise ValueError(_SECRET)

    mcp = build_nexus_mcp_server(
        name="af01b", host_execution=_host(), registry=_BrokenRegistry(), default_capability="c", identity_mode=_LOCAL
    )
    async with Client(mcp) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("list_agents", {})
    assert "MCP_INTERNAL_ERROR" in str(excinfo.value) and "SECRET" not in str(excinfo.value)


class _In(BaseModel):
    query: str = "x"


class _Out(BaseModel):
    value: str = "ok"


def _contract(tool_id: str = "af.tool") -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description=tool_id,
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        category="af",
    )


async def _noop(_input: _In, _ctx: object) -> _Out:
    return _Out()


def _catalog_server(identity_mode: McpIdentityMode | None = None):
    registry = ToolRegistry()
    registry.register(_contract(), _noop)
    return build_nexus_mcp_server(
        name="af-cat",
        host_execution=_host(),
        registry=AgentRegistry(),
        default_capability="c",
        tool_registry=registry,
        identity_mode=identity_mode,
    ), registry


async def test_catalog_tools_registry_exception_text_is_not_echoed() -> None:
    mcp, registry = _catalog_server(_LOCAL)

    def _boom(self: ToolRegistry, *_a: object, **_k: object):  # type: ignore[no-untyped-def]
        raise ValueError(_SECRET)

    async with Client(mcp) as client:
        with patch.object(ToolRegistry, "list", _boom):
            with pytest.raises(ToolError) as lst:
                await client.call_tool("list_catalog_tools", {})
        with patch.object(ToolRegistry, "has", _boom):
            with pytest.raises(ToolError) as desc:
                await client.call_tool("describe_catalog_tool", {"tool_id": "af.tool"})
    for err in (lst, desc):
        assert "MCP_INTERNAL_ERROR" in str(err.value) and "SECRET" not in str(err.value)


@pytest.mark.parametrize("shape", ["circular", "deep"])
async def test_unserializable_custom_tool_results_are_typed_not_raw(shape: str) -> None:
    async def _bad(message: str = "") -> dict[str, object]:
        if shape == "circular":
            cyclic: dict[str, object] = {}
            cyclic["self"] = cyclic
            return cyclic
        nested: dict[str, object] = {}
        cursor = nested
        for _ in range(5000):
            cursor["n"] = {}
            cursor = cursor["n"]  # type: ignore[assignment]
        return nested

    mcp = build_mcp_server_x_reference_qualification_server(
        host_execution=_host(), registry=AgentRegistry(), custom_echo=_bad, identity_mode=_LOCAL
    )
    async with Client(mcp) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("qual_echo", {"message": "m"})
    assert str(excinfo.value).startswith(("Error calling tool", "MCP_")) or "MCP_" in str(excinfo.value)
    assert "MCP_INTERNAL_ERROR" in str(excinfo.value) or "MCP_RESULT_BOUND_EXCEEDED" in str(excinfo.value)


# --- AF-02: the host ends cancelled or failed after the call bound ------------------------------------


class _ExplodingHost:
    """Registers a binding, waits for the governed cancel, then ends with ``outcome`` instead of a result."""

    def __init__(self, outcome: str) -> None:
        self.outcome = outcome
        self.started = asyncio.Event()
        self.tasks: list[Task] = []
        self.port = AsyncMock(spec=HostTaskExecutionPort)
        self.port.execute = AsyncMock(side_effect=self._execute)

    async def _execute(self, task: Task, **_kwargs: object):  # type: ignore[no-untyped-def]
        from intergrax.contracts.execution_identity import mint_run_id

        run_id = mint_run_id()
        self.tasks.append(task)
        await ActiveTaskRegistry.register(task, run_id)
        self.started.set()
        try:
            for _ in range(500):
                if CancellationCoordinator.is_requested(task.metadata):
                    break
                await asyncio.sleep(0.02)
        finally:
            await ActiveTaskRegistry.unregister(task.task_id, run_id)
        if self.outcome == "cancelled":
            raise asyncio.CancelledError
        raise RuntimeError(_SECRET)
        return _terminal_result(task, str(run_id), None)  # pragma: no cover


@pytest.mark.parametrize(("outcome", "expected"), [("cancelled", "outcome=cancelled"), ("failed", "outcome=failed")])
async def test_host_ending_cancelled_or_failed_after_the_bound_is_deadline_exceeded(
    outcome: str, expected: str
) -> None:
    host = _ExplodingHost(outcome)
    supervision = build_host_root_execution_transport_detachment_supervisor()
    mcp = _reference_server(host.port, task_control_mutation_boundary=_boundary(), max_call_seconds=0.2)
    with (
        patch(f"{_NEXUS}.resolve_mcp_run_agent_identity", return_value=_identity_resolution()),
        patch(f"{_NEXUS}.resolve_transport_detachment_supervision_from_request", return_value=supervision),
    ):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="MCP_DEADLINE_EXCEEDED") as excinfo:
                await asyncio.wait_for(client.call_tool("run_agent", {"message": "x"}), timeout=_T)
    assert expected in str(excinfo.value)
    assert "SECRET" not in str(excinfo.value)
    await _wait_supervision_drained(supervision)


# --- AF-03: custom host tools sit behind the principal gate by default ---------------------------------


async def test_custom_tool_is_refused_without_a_principal_in_the_default_mode_and_never_runs() -> None:
    calls: list[str] = []

    async def _echo(message: str = "") -> dict[str, object]:
        calls.append(message)
        return {"echo": message}

    mcp = build_mcp_server_x_reference_qualification_server(
        host_execution=_host(),
        registry=AgentRegistry(),
        custom_echo=_echo,
        identity_mode=McpIdentityMode.HARNESS_BOUND,
    )
    combined = couple_fastapi_with_mcp(FastAPI(), mcp, mount_path="/mcp")
    with _uvicorn_mcp_base_url(combined) as base_url:
        refused = await _streamable_mcp_call_tool(base_url, "qual_echo", {"message": "m"})
    assert refused.isError is True and "MCP_PRINCIPAL_REQUIRED" in refused.content[0].text
    assert calls == []


async def test_custom_tool_runs_for_a_verified_principal_and_for_an_explicit_public_opt_out() -> None:
    async def _echo(message: str = "") -> dict[str, object]:
        return {"echo": message}

    api = FastAPI()
    api.state.harness_auth = HarnessAuthState(
        require_api_key=True, resolved_api_key="ck", api_key_principal_tenant_id="custom-t"
    )
    gated = build_mcp_server_x_reference_qualification_server(
        host_execution=_host(),
        registry=AgentRegistry(),
        custom_echo=_echo,
        identity_mode=McpIdentityMode.HARNESS_BOUND,
    )
    with _uvicorn_mcp_base_url(couple_fastapi_with_mcp(api, gated, mount_path="/mcp")) as base_url:
        ok = await _streamable_mcp_call_tool(base_url, "qual_echo", {"message": "m"}, headers={"X-Api-Key": "ck"})
    assert ok.isError is False

    public = build_mcp_server_x_reference_qualification_server(
        host_execution=_host(),
        registry=AgentRegistry(),
        custom_echo=_echo,
        identity_mode=McpIdentityMode.HARNESS_BOUND,
        public_custom_tool_names=frozenset({"qual_echo"}),
    )
    with _uvicorn_mcp_base_url(couple_fastapi_with_mcp(FastAPI(), public, mount_path="/mcp")) as base_url:
        opened = await _streamable_mcp_call_tool(base_url, "qual_echo", {"message": "m"})
    assert opened.isError is False


# --- AF-04: catalog discovery gates and the LOCAL_DEV downgrade guard -----------------------------------


async def test_catalog_discovery_tools_refuse_without_a_principal_in_the_default_mode() -> None:
    mcp, _registry = _catalog_server()
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="MCP_PRINCIPAL_REQUIRED"):
            await client.call_tool("list_catalog_tools", {})
        with pytest.raises(ToolError, match="MCP_PRINCIPAL_REQUIRED"):
            await client.call_tool("describe_catalog_tool", {"tool_id": "af.tool"})
    open_mcp, _registry = _catalog_server(_LOCAL)
    async with Client(open_mcp) as client:
        assert (await client.call_tool("list_catalog_tools", {})).is_error is False
        assert (await client.call_tool("describe_catalog_tool", {"tool_id": "af.tool"})).is_error is False


def test_local_dev_mode_never_downgrades_configured_auth_for_in_process_callers() -> None:
    configured = HarnessAuthState(require_api_key=True, resolved_api_key="k", api_key_principal_tenant_id="t")
    with pytest.raises(ToolError, match="MCP_PRINCIPAL_REQUIRED"):
        resolve_mcp_run_agent_identity(
            harness_auth=configured,
            asserted_tenant_id="forged",
            asserted_user_id=None,
            default_tenant_id="mcp",
            default_user_id="mcp-user",
            identity_mode=_LOCAL,
        )


class _FakeIngress:
    def __init__(self, terminal: RootCatalogChildWorkTerminal) -> None:
        self.terminal = terminal

    async def submit_catalog_child_work(
        self, _ticket: object, *, wait_timeout_s: float
    ) -> RootCatalogChildWorkTerminal:
        return self.terminal


async def _cross_loop_call(terminal: RootCatalogChildWorkTerminal) -> dict[str, object]:
    from tests.qualification.mcp_server_x.test_mcp_server_x_p7_l2_security_trace import (
        _SERVICE_A,
        _STEP,
        _TENANT_A,
        _RecordingCatalogInvoker,
        _register_binding,
        _tool_registry,
    )
    from tests.qualification.mcp_server_x.test_mcp_server_x_p7_l2_security_trace import (
        _contract as trace_contract,
    )

    task, run_id = await _register_binding(tenant=_TENANT_A, user_id=_SERVICE_A)
    identity = RequestIdentity(
        tenant_id=_TENANT_A, user_id=_SERVICE_A, principal_type=PrincipalType.USER, auth_subject=_SERVICE_A
    )
    resolution = McpRunAgentIdentityResolution(
        tenant_id=_TENANT_A, user_id=_SERVICE_A, principal=None, request_identity=identity
    )
    composition = McpCatalogInvocationComposition(
        catalog_invoker=_RecordingCatalogInvoker(),
        registry=_tool_registry(trace_contract("p7.ok")),
        catalog_tool_child_execution=object(),
        root_child_work_ingress=_FakeIngress(terminal),
        enable_cross_loop_catalog_child=True,
    )
    with (
        patch(f"{_BRIDGE}.resolve_mcp_run_agent_identity", return_value=resolution),
        patch(f"{_BRIDGE}._in_process_parent_matches_binding", return_value=False),
    ):
        return await execute_mcp_catalog_tool_invocation(
            composition, tool_id="p7.ok", task_id=str(task.task_id), run_id=run_id, step_id=_STEP
        )


async def test_cross_loop_success_payload_over_the_bound_is_a_typed_tool_error() -> None:
    huge = RootCatalogChildWorkTerminal(success=True, payload={"success": True, "output": {"blob": "x" * 1_200_000}})
    with pytest.raises(ToolError, match="MCP_RESULT_BOUND_EXCEEDED"):
        await _cross_loop_call(huge)


async def test_cross_loop_failure_payload_is_bounded_and_does_not_leak_exception_text() -> None:
    leaky = RootCatalogChildWorkTerminal(success=False, payload={"error": "internal_error", "detail": _SECRET})
    payload = await _cross_loop_call(leaky)
    assert payload["mcp_code"] == "MCP_INTERNAL_ERROR"
    assert "SECRET" not in str(payload)
    oversize = RootCatalogChildWorkTerminal(success=False, payload={"error": "tool_failed", "detail": "y" * 1_200_000})
    with pytest.raises(ToolError, match="MCP_RESULT_BOUND_EXCEEDED"):
        await _cross_loop_call(oversize)


async def test_cross_loop_success_payload_within_the_bound_carries_the_invocation_id() -> None:
    ok = RootCatalogChildWorkTerminal(success=True, payload={"success": True, "output": {"v": 1}})
    payload = await _cross_loop_call(ok)
    assert payload["success"] is True and payload["mcp_invocation_id"].startswith("mcpi-")
