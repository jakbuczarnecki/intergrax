# © Artur Czarnecki. All rights reserved.

"""MCP-SERVER-X P7 (MCP-P7-03, ticket T-IGX-035-H) - publication and discovery, level L3.

Level: **L3 - physical streamable HTTP on loopback**. Every test here talks to a real ``uvicorn``
socket on ``127.0.0.1`` (OS-assigned port) through the official MCP streamable HTTP client or
raw HTTP. Nothing in this module is an in-process (L2) check, and L2 evidence is never counted
as L3 evidence.

Covers MCP-ACC-01 (reference server), MCP-ACC-05 (publication / transport), MCP-ACC-06
(HTTP client initialize), MCP-ACC-07 (discovery) and MCP-ACC-08 (schema over HTTP).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version as _dist_version

import fastmcp
import httpx
import pytest
from intergrax.tools.exporters.mcp import to_mcp_tools
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS
from mcp.types import LATEST_PROTOCOL_VERSION
from packaging.version import Version
from tests.qualification.mcp_server_x.reference_app.build import (
    REFERENCE_CAPABILITY,
    REFERENCE_DISPLAY_NAME,
    REFERENCE_SERVER_ID,
    build_mcp_server_x_reference_definition,
    build_mcp_server_x_reference_qualification_server,
)
from tests.qualification.mcp_server_x.reference_app.http_serve import (
    ServedMcp,
    build_reference_host_parts,
    call_tool,
    completed_task_result,
    couple_reference_host,
    error_text,
    jsonrpc_post,
    list_tool_names,
    patched_task_delegate,
    served_mcp,
    tool_payload,
    with_session,
)
from tests.qualification.mcp_server_x.reference_app.variants import (
    CATALOG_TOOL_ID,
    QUAL_API_KEY,
    QUAL_API_KEY_TENANT,
    RecordingCatalogInvoker,
    api_key_auth,
    build_catalog_registry,
    catalog_contract,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.network, pytest.mark.qualification]

_AUTH = {"X-Api-Key": QUAL_API_KEY}

_FULL_REFERENCE_TOOL_SET = frozenset(
    {
        "list_agents",
        "run_agent",
        "qual_echo",
        "list_catalog_tools",
        "describe_catalog_tool",
        "invoke_catalog_tool",
    },
)


@contextmanager
def _serve(**build_kwargs: object) -> Iterator[ServedMcp]:
    parts = build_reference_host_parts()
    mcp = build_mcp_server_x_reference_qualification_server(
        host_execution=parts.host_execution,
        registry=parts.registry,
        **build_kwargs,  # type: ignore[arg-type]
    )
    with served_mcp(couple_reference_host(mcp, auth=api_key_auth())) as served:
        yield served


@pytest.fixture(scope="module")
def reference_server() -> Iterator[ServedMcp]:
    """Reference server with host builtins, qual echo and the full catalog surface (API-key auth)."""
    with _serve(
        tool_registry=build_catalog_registry(),
        catalog_tool_invoker=RecordingCatalogInvoker(),
    ) as served:
        yield served


# --------------------------------------------------------------------------- MCP-ACC-01


async def test_acc_01_reference_server_exact_tool_set_over_http(reference_server: ServedMcp) -> None:
    names = await list_tool_names(reference_server.url, headers=_AUTH)
    assert len(names) == len(set(names)), f"duplicate tool names: {names}"
    assert set(names) == _FULL_REFERENCE_TOOL_SET


async def test_acc_01_authorized_run_agent_over_http_with_auth(reference_server: ServedMcp) -> None:
    """Authorized ``run_agent`` through the canonical host task execution (delegate stubbed)."""
    with patched_task_delegate(completed_task_result(task_id="task-acc01", run_id="run-acc01")) as delegate:
        result = await call_tool(
            reference_server.url,
            "run_agent",
            {"message": "hello", "capability": REFERENCE_CAPABILITY, "tenant_id": QUAL_API_KEY_TENANT},
            headers=_AUTH,
        )
    assert result.isError is not True, error_text(result)
    payload = tool_payload(result)
    assert payload["state"] == "completed"
    assert payload["task_id"] == "task-acc01"
    assert payload["run_id"] == "run-acc01"
    delegate.assert_awaited_once()


async def test_acc_01_custom_echo_binding_over_http(reference_server: ServedMcp) -> None:
    result = await call_tool(reference_server.url, "qual_echo", {"message": "acc-01"}, headers=_AUTH)
    assert result.isError is not True, error_text(result)
    assert tool_payload(result) == {"echo": "acc-01", "server_id": REFERENCE_SERVER_ID}


# --------------------------------------------------------------------------- MCP-ACC-05


def test_acc_05_only_streamable_http_mount_is_published(reference_server: ServedMcp) -> None:
    """The served ASGI app mounts MCP only at ``/mcp`` (streamable HTTP) and the host app at ``/``."""
    mounts = {route.path for route in reference_server.app.routes if type(route).__name__ == "Mount"}
    assert mounts == {"/mcp", ""}
    for route in reference_server.app.routes:
        path = getattr(route, "path", "") or ""
        assert "sse" not in path.lower()
        assert "stdio" not in path.lower()


def test_acc_05_streamable_http_initialize_opens_a_session(reference_server: ServedMcp) -> None:
    response = jsonrpc_post(
        reference_server.url,
        "initialize",
        {
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "p7-l3-raw", "version": "0"},
        },
        headers=_AUTH,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers.get("mcp-session-id")


@pytest.mark.parametrize("path", ["/sse", "/mcp/sse", "/messages/", "/mcp/messages/", "/stdio", "/mcp/stdio"])
def test_acc_05_sse_and_stdio_style_paths_are_not_served(reference_server: ServedMcp, path: str) -> None:
    """Legacy SSE / message / stdio-style paths do not exist (valid credentials: 404, not masked by 401)."""
    response = httpx.get(
        f"http://127.0.0.1:{reference_server.port}{path}",
        headers={**_AUTH, "Accept": "text/event-stream"},
        timeout=10.0,
    )
    assert response.status_code == 404


# --------------------------------------------------------------------------- MCP-ACC-06


async def test_acc_06_client_initialize_result_authenticated(reference_server: ServedMcp) -> None:
    async def _action(_session: object, init: object) -> object:
        return init

    init = await with_session(reference_server.url, _action, headers=_AUTH)
    assert init.serverInfo.name == REFERENCE_DISPLAY_NAME
    installed = Version(_dist_version("fastmcp"))
    assert init.serverInfo.version == str(installed) == fastmcp.__version__
    assert Version("3.3.1") <= installed < Version("4")  # matrix: fastmcp>=3.3.1,<4
    assert init.protocolVersion in SUPPORTED_PROTOCOL_VERSIONS
    assert init.protocolVersion == LATEST_PROTOCOL_VERSION
    assert init.capabilities.tools is not None


def test_acc_06_initialize_without_credentials_is_rejected_at_http_level(reference_server: ServedMcp) -> None:
    response = jsonrpc_post(
        reference_server.url,
        "initialize",
        {
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "p7-l3-raw", "version": "0"},
        },
    )
    assert response.status_code == 401
    assert "mcp-session-id" not in response.headers


# --------------------------------------------------------------------------- MCP-ACC-07


async def test_acc_07_discovery_set_equals_definition_exposure_over_http(reference_server: ServedMcp) -> None:
    from intergrax.applications._shared.mcp_server_engine import exposed_builtin_tool_names

    definition = build_mcp_server_x_reference_definition(catalog_tool_ids=[CATALOG_TOOL_ID])
    expected = set(exposed_builtin_tool_names(definition))
    assert set(await list_tool_names(reference_server.url, headers=_AUTH)) == expected


@pytest.mark.parametrize(
    ("variant", "build_kwargs", "absent", "expected"),
    [
        pytest.param(
            "retired",
            {"retired_binding_ids": ["qual_echo"]},
            "qual_echo",
            {"list_agents", "run_agent"},
            id="retired",
        ),
        pytest.param(
            "disabled",
            {"omit_from_allowlist": ["run_agent"]},
            "run_agent",
            {"list_agents", "qual_echo"},
            id="disabled-not-allowlisted",
        ),
        pytest.param(
            "hidden",
            {"hidden_binding_ids": ["list_agents"]},
            "list_agents",
            {"run_agent", "qual_echo"},
            id="hidden",
        ),
    ],
)
async def test_acc_07_non_exposed_bindings_are_absent_and_not_callable(
    variant: str,
    build_kwargs: dict[str, object],
    absent: str,
    expected: set[str],
) -> None:
    """Retired, disabled (not allowlisted) and hidden bindings are not discoverable and not callable."""
    with _serve(**build_kwargs) as served:
        names = await list_tool_names(served.url, headers=_AUTH)
        assert set(names) == expected, variant
        assert absent not in names
        result = await call_tool(served.url, absent, {}, headers=_AUTH)
        assert result.isError is True
        assert "unknown tool" in (error_text(result) or "").lower()


async def test_acc_07_exposure_policy_disabled_publishes_no_tools() -> None:
    from intergrax.applications._shared.mcp_developer_api import McpExposurePolicy

    with _serve(exposure_policy=McpExposurePolicy(enabled=False, allowed_binding_ids=("run_agent",))) as served:
        assert await list_tool_names(served.url, headers=_AUTH) == []


async def test_acc_07_invoke_catalog_tool_is_absent_without_a_wired_invoker() -> None:
    """Characterization: the allowlisted ``invoke_catalog_tool`` builtin is mounted only with an invoker.

    The definition still lists ``invoke_catalog_tool`` as exposed, but discovery omits it when no
    ``ExecutionBoundCatalogToolInvoker`` is composed (fail-closed in effect). Pinned so a change is visible.
    """
    with _serve(tool_registry=build_catalog_registry()) as served:
        names = set(await list_tool_names(served.url, headers=_AUTH))
        assert names == _FULL_REFERENCE_TOOL_SET - {"invoke_catalog_tool"}


# --------------------------------------------------------------------------- MCP-ACC-08


async def _tools_by_name(url: str) -> dict[str, object]:
    async def _action(session: object, _init: object) -> dict[str, object]:
        listed = await session.list_tools()  # type: ignore[attr-defined]
        return {tool.name: tool for tool in listed.tools}

    return await with_session(url, _action, headers=_AUTH)


async def test_acc_08_input_schema_of_qual_echo_over_http(reference_server: ServedMcp) -> None:
    tool = (await _tools_by_name(reference_server.url))["qual_echo"]
    assert tool.inputSchema == {
        "type": "object",
        "properties": {"message": {"type": "string", "default": ""}},
        "additionalProperties": False,
    }


async def test_acc_08_input_schema_of_run_agent_over_http(reference_server: ServedMcp) -> None:
    schema = (await _tools_by_name(reference_server.url))["run_agent"].inputSchema
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["message"]
    assert set(schema["properties"]) == {"message", "capability", "tenant_id", "user_id", "session_id", "intent"}
    assert schema["properties"]["message"] == {"type": "string"}
    assert schema["properties"]["capability"] == {"type": "string", "default": REFERENCE_CAPABILITY}
    for optional in ("tenant_id", "user_id", "session_id", "intent"):
        assert schema["properties"][optional] == {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
        }


async def test_acc_08_describe_catalog_tool_payload_matches_tool_contract(reference_server: ServedMcp) -> None:
    result = await call_tool(
        reference_server.url,
        "describe_catalog_tool",
        {"tool_id": CATALOG_TOOL_ID},
        headers=_AUTH,
    )
    assert result.isError is not True, error_text(result)
    payload = tool_payload(result)
    expected = json.loads(json.dumps(to_mcp_tools([catalog_contract()])[0], default=str))
    assert payload == expected
    assert payload["name"] == CATALOG_TOOL_ID
    assert payload["inputSchema"]["properties"]["query"]["type"] == "string"


async def test_acc_08_describe_catalog_tool_unknown_id_is_a_typed_error_payload(reference_server: ServedMcp) -> None:
    result = await call_tool(
        reference_server.url,
        "describe_catalog_tool",
        {"tool_id": "does.not.exist"},
        headers=_AUTH,
    )
    assert tool_payload(result) == {"error": "unknown_tool:does.not.exist"}


async def test_acc_08_list_catalog_tools_matches_contract_over_http(reference_server: ServedMcp) -> None:
    result = await call_tool(reference_server.url, "list_catalog_tools", {}, headers=_AUTH)
    listed = tool_payload(result)["result"]
    assert [item["name"] for item in listed] == [CATALOG_TOOL_ID]
    assert listed[0] == json.loads(json.dumps(to_mcp_tools([catalog_contract()])[0], default=str))
