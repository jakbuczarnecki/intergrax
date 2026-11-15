# © Artur Czarnecki. All rights reserved.

"""MCP-RC MRC-1 — fail-closed MCP identity (R-P8-01, R-P7-06, G-ID-01..09)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.server.context import RequestContext, request_ctx
from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.mcp_harness_principal import resolve_mcp_run_agent_identity
from intergrax.applications._shared.mcp_identity_mode import (
    McpIdentityMode,
    coerce_mcp_identity_mode,
    is_loopback_client,
)
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry
from starlette.requests import Request
from tests.unit.applications.test_mcp_trust_a_principal_binding import (
    _completed,
    _streamable_mcp_call_tool,
    _uvicorn_mcp_base_url,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_LOCAL = McpIdentityMode.LOCAL_DEV_EXPLICIT


@contextmanager
def _request(*, client_host: str = "127.0.0.1", api_key: str | None = None) -> Iterator[Request]:
    headers: list[tuple[bytes, bytes]] = []
    if api_key is not None:
        headers.append((b"x-api-key", api_key.encode()))
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": headers,
        "query_string": b"",
        "client": (client_host, 12345),
        "app": FastAPI(),
    }

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    ctx = RequestContext(request_id="rc-req", meta=None, session=None, lifespan_context=None, request=request)
    token = request_ctx.set(ctx)
    try:
        yield request
    finally:
        request_ctx.reset(token)


def _resolve(
    auth: HarnessAuthState | None,
    *,
    tenant: str | None = None,
    user: str | None = None,
    mode: McpIdentityMode | None = None,
):
    return resolve_mcp_run_agent_identity(
        harness_auth=auth,
        asserted_tenant_id=tenant,
        asserted_user_id=user,
        default_tenant_id="mcp",
        default_user_id="mcp-user",
        identity_mode=mode,
    )


def _host() -> AsyncMock:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed())
    return port


def _server(port: AsyncMock, **kwargs: object):
    return build_nexus_mcp_server(
        name="rc-identity",
        host_execution=port,
        registry=AgentRegistry(),
        default_capability="demo.cap",
        **kwargs,
    )


# --- mode contract -----------------------------------------------------------------------------------


def test_default_mode_is_fail_closed_and_not_environment_selectable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_MCP_IDENTITY_MODE", "local_dev_explicit")
    assert coerce_mcp_identity_mode(None) is McpIdentityMode.HARNESS_BOUND
    assert coerce_mcp_identity_mode("local_dev_explicit") is _LOCAL
    with pytest.raises(ValueError):
        coerce_mcp_identity_mode("trust_everything")


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("localhost", True),
        ("testclient", True),
        ("10.1.2.3", False),
        ("203.0.113.9", False),
        ("", False),
    ],
)
def test_loopback_classification(host: str, expected: bool) -> None:
    with _request(client_host=host) as request:
        assert is_loopback_client(request) is expected
    assert is_loopback_client(None) is True


# --- run_agent: missing / empty / forged ------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_mode_rejects_in_process_run_agent_without_principal() -> None:
    port = _host()
    async with Client(_server(port)) as client:
        with pytest.raises(ToolError, match="MCP_PRINCIPAL_REQUIRED"):
            await client.call_tool("run_agent", {"message": "x", "tenant_id": "victim"})
    port.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_dev_explicit_in_process_admits_asserted_identity() -> None:
    port = _host()
    async with Client(_server(port, identity_mode=_LOCAL)) as client:
        await client.call_tool("run_agent", {"message": "x", "tenant_id": "dev-t", "user_id": "dev-u"})
    task = port.execute.await_args.args[0]
    assert (task.tenant_id, task.user_id) == ("dev-t", "dev-u")


@pytest.mark.asyncio
async def test_asserted_local_dev_identity_is_labelled_unverified_on_the_task() -> None:
    port = _host()
    async with Client(_server(port, identity_mode=_LOCAL)) as client:
        await client.call_tool("run_agent", {"message": "x", "tenant_id": "dev-t"})
    assert port.execute.await_args.args[0].metadata.get("mcp_identity_assertion") == "unverified"


@pytest.mark.asyncio
async def test_verified_principal_task_is_not_labelled_unverified() -> None:
    port = _host()
    combined = couple_fastapi_with_mcp(_api_with_key(), _server(port), mount_path="/mcp")
    with _uvicorn_mcp_base_url(combined) as base_url:
        await _streamable_mcp_call_tool(base_url, "run_agent", {"message": "x"}, headers={"X-Api-Key": "vk"})
    assert "mcp_identity_assertion" not in port.execute.await_args.args[0].metadata


def _api_with_key() -> FastAPI:
    api = FastAPI()
    api.state.harness_auth = HarnessAuthState(
        require_api_key=True, resolved_api_key="vk", api_key_principal_tenant_id="verified-t"
    )
    return api


def test_default_mode_rejects_empty_harness_auth_state_with_http_request() -> None:
    with _request():
        with pytest.raises(ToolError, match="MCP_PRINCIPAL_REQUIRED"):
            _resolve(HarnessAuthState(), tenant="victim")


def test_default_mode_rejects_missing_harness_auth() -> None:
    with _request():
        with pytest.raises(ToolError, match="MCP_PRINCIPAL_REQUIRED"):
            _resolve(None, tenant="victim")


def test_local_dev_explicit_admits_loopback_empty_state_but_refuses_remote_client() -> None:
    with _request(client_host="127.0.0.1"):
        assert _resolve(HarnessAuthState(), tenant="dev", mode=_LOCAL).tenant_id == "dev"
    with _request(client_host="203.0.113.9"):
        with pytest.raises(ToolError, match="MCP_PRINCIPAL_REQUIRED"):
            _resolve(HarnessAuthState(), tenant="dev", mode=_LOCAL)


def test_local_dev_explicit_never_downgrades_configured_auth() -> None:
    auth = HarnessAuthState(require_api_key=True, resolved_api_key="k", api_key_principal_tenant_id="t")
    with _request():
        with pytest.raises(ToolError, match="MCP_PRINCIPAL_REQUIRED"):
            _resolve(auth, tenant="forged", mode=_LOCAL)


@pytest.mark.parametrize("field", ["tenant", "user"])
def test_forged_identity_maps_to_tenant_assertion_conflict(field: str) -> None:
    auth = HarnessAuthState(require_api_key=True, resolved_api_key="k", api_key_principal_tenant_id="bound")
    with _request(api_key="k"):
        with pytest.raises(ToolError, match="MCP_TENANT_ASSERTION_CONFLICT"):
            _resolve(auth, **{field: "forged"})


def test_wrong_key_maps_to_principal_required() -> None:
    auth = HarnessAuthState(require_api_key=True, resolved_api_key="k", api_key_principal_tenant_id="bound")
    with _request(api_key="wrong"):
        with pytest.raises(ToolError, match="MCP_PRINCIPAL_REQUIRED"):
            _resolve(auth)


def test_verified_principal_wins_over_default_tenant_in_both_modes() -> None:
    auth = HarnessAuthState(require_api_key=True, resolved_api_key="k", api_key_principal_tenant_id="bound")
    for mode in (None, _LOCAL):
        with _request(api_key="k"):
            assert _resolve(auth, mode=mode).tenant_id == "bound"


# --- discovery tools carry the same gate -----------------------------------------------------------


@pytest.mark.asyncio
async def test_list_agents_requires_principal_by_default_and_opens_only_in_local_dev() -> None:
    async with Client(_server(_host())) as client:
        with pytest.raises(ToolError, match="MCP_PRINCIPAL_REQUIRED"):
            await client.call_tool("list_agents", {})
    async with Client(_server(_host(), identity_mode=_LOCAL)) as client:
        result = await client.call_tool("list_agents", {})
        assert result.is_error is False


# --- composition order (G-ID-02 / G-ID-09) over real streamable HTTP --------------------------------


def _coupled_without_inner_auth(port: AsyncMock) -> FastAPI:
    return couple_fastapi_with_mcp(FastAPI(), _server(port), mount_path="/mcp")


@pytest.mark.asyncio
async def test_http_host_without_auth_rejects_asserted_tenant_before_execution() -> None:
    port = _host()
    with _uvicorn_mcp_base_url(_coupled_without_inner_auth(port)) as base_url:
        result = await _streamable_mcp_call_tool(base_url, "run_agent", {"message": "x", "tenant_id": "victim"})
    assert result.isError is True
    assert "MCP_PRINCIPAL_REQUIRED" in result.content[0].text
    port.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_identity_wired_after_coupling_still_binds_and_conflicts() -> None:
    port = _host()
    combined = _coupled_without_inner_auth(port)
    combined.state.harness_auth = HarnessAuthState(
        require_api_key=True, resolved_api_key="late-key", api_key_principal_tenant_id="late-tenant"
    )
    with _uvicorn_mcp_base_url(combined) as base_url:
        ok = await _streamable_mcp_call_tool(base_url, "run_agent", {"message": "x"}, headers={"X-Api-Key": "late-key"})
        forged = await _streamable_mcp_call_tool(
            base_url, "run_agent", {"message": "x", "tenant_id": "victim"}, headers={"X-Api-Key": "late-key"}
        )
    assert ok.isError is False
    assert port.execute.await_args.args[0].tenant_id == "late-tenant"
    assert forged.isError is True and "MCP_TENANT_ASSERTION_CONFLICT" in forged.content[0].text
    assert port.execute.await_count == 1


# --- research pipeline (G-ID-04) and facades ----------------------------------------------------------


@pytest.mark.asyncio
async def test_research_pipeline_tool_cannot_bypass_identity() -> None:
    from research_application.mcp.server import build_research_mcp_server

    port = _host()
    server = build_research_mcp_server(host_execution=port, registry=AgentRegistry(), route_prefix="/v1")
    async with Client(server) as client:
        with pytest.raises(ToolError, match="MCP_PRINCIPAL_REQUIRED"):
            await client.call_tool("run_research_pipeline", {"message": "x", "tenant_id": "victim"})
    port.execute.assert_not_awaited()

    dev_port = _host()
    dev = build_research_mcp_server(
        host_execution=dev_port, registry=AgentRegistry(), route_prefix="/v1", identity_mode=_LOCAL
    )
    async with Client(dev) as client:
        await client.call_tool("run_research_pipeline", {"message": "x"})
    assert dev_port.execute.await_args.args[0].tenant_id == "research"


@pytest.mark.asyncio
async def test_research_pipeline_over_http_forged_tenant_conflicts_with_principal() -> None:
    from research_application.mcp.server import build_research_mcp_server

    port = _host()
    api = FastAPI()
    api.state.harness_auth = HarnessAuthState(
        require_api_key=True, resolved_api_key="rk", api_key_principal_tenant_id="research-bound"
    )
    server = build_research_mcp_server(host_execution=port, registry=AgentRegistry(), route_prefix="/v1")
    combined = couple_fastapi_with_mcp(api, server, mount_path="/mcp")
    with _uvicorn_mcp_base_url(combined) as base_url:
        forged = await _streamable_mcp_call_tool(
            base_url, "run_research_pipeline", {"message": "x", "tenant_id": "victim"}, headers={"X-Api-Key": "rk"}
        )
        ok = await _streamable_mcp_call_tool(
            base_url, "run_research_pipeline", {"message": "x"}, headers={"X-Api-Key": "rk"}
        )
    assert forged.isError is True and "MCP_TENANT_ASSERTION_CONFLICT" in forged.content[0].text
    assert ok.isError is False
    assert port.execute.await_args.args[0].tenant_id == "research-bound"


@pytest.mark.parametrize(
    "builder_path",
    [
        "applications.dispute_sim_application.mcp.server:build_dispute_sim_mcp_server",
        "applications.governed_contractor_application.mcp.server:build_governed_contractor_mcp_server",
        "applications.intergrax_assistant_application.mcp.server:build_intergrax_assistant_mcp_server",
        "applications.legal_application.mcp.server:build_legal_mcp_server",
        "applications.local_workspace_application.mcp.server:build_local_workspace_mcp_server",
        "applications.poc_template_application.mcp.server:build_poc_template_mcp_server",
        "applications.lab_application.mcp.server:build_lab_mcp_server",
        "applications.research_application.mcp.server:build_research_mcp_server",
    ],
)
@pytest.mark.asyncio
async def test_every_facade_defaults_to_fail_closed_and_accepts_explicit_local_dev(builder_path: str) -> None:
    import importlib

    module_name, symbol = builder_path.split(":")
    module = importlib.import_module(module_name.removeprefix("applications."))
    build = getattr(module, symbol)

    port = _host()
    closed = build(host_execution=port, registry=AgentRegistry(), route_prefix="/v1")
    async with Client(closed) as client:
        with pytest.raises(ToolError, match="MCP_PRINCIPAL_REQUIRED"):
            await client.call_tool("run_agent", {"message": "x", "tenant_id": "victim"})
    port.execute.assert_not_awaited()

    dev_port = _host()
    opened = build(host_execution=dev_port, registry=AgentRegistry(), route_prefix="/v1", identity_mode=_LOCAL)
    async with Client(opened) as client:
        await client.call_tool("run_agent", {"message": "x", "tenant_id": "dev-t"})
    assert dev_port.execute.await_args.args[0].tenant_id == "dev-t"


# --- catalog allowlist at invoke (G-ID-07) ------------------------------------------------------------


def test_invoke_catalog_composition_receives_the_allowlist_filtered_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.applications._shared import mcp_server_engine as engine
    from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
    from intergrax.tools.registry.runtime import ToolRegistry
    from pydantic import BaseModel

    class _In(BaseModel):
        query: str = "x"

    class _Out(BaseModel):
        value: str = "ok"

    async def _handler(_input: _In, _ctx: object) -> _Out:
        return _Out()

    registry = ToolRegistry()
    for tool_id in ("allowed.tool", "other.tool"):
        registry.register(
            ToolContract(
                tool_id=tool_id,
                name=tool_id,
                description=tool_id,
                input_schema=_In,
                output_schema=_Out,
                error_mapping={},
                side_effects=False,
                risk_level=ToolRiskLevel.LOW,
                category="rc",
            ),
            _handler,
        )
    seen: dict[str, object] = {}
    real = engine.McpCatalogInvocationComposition

    def _recording(**kwargs: object) -> object:
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(engine, "McpCatalogInvocationComposition", _recording)
    definition = engine.nexus_legacy_server_definition(
        server_id="rc-allow",
        display_name="rc",
        default_capability="c",
        default_tenant_id="t",
        default_user_id="u",
        catalog_tool_ids=["allowed.tool"],
        include_catalog_surface=True,
    )
    engine.compose_mcp_server_from_definition(
        definition,
        engine.McpServerCompositionDeps(
            host_execution=_host(),
            registry=AgentRegistry(),
            tool_registry=registry,
            catalog_tool_invoker=AsyncMock(),
        ),
    )
    invoked_registry = seen["registry"]
    assert invoked_registry.has("allowed.tool") is True
    assert invoked_registry.has("other.tool") is False
