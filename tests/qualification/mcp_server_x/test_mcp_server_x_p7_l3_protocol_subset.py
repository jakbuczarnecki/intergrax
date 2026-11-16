# © Artur Czarnecki. All rights reserved.

"""MCP-SERVER-X P7 (MCP-P7-04, ticket T-IGX-035-H) - announced protocol subset matrix, level L3.

Level: **L3 - physical streamable HTTP on loopback** (``uvicorn`` on an OS-assigned port, official
MCP client plus a raw JSON-RPC-over-HTTP session). No in-process checks.

MCP-ACC-18: record what the reference server **checks** (tools) versus what it does **not announce**
(tasks, completions, resource subscriptions, stdio/SSE), and keep UNSUPPORTED as UNSUPPORTED.

Documentation contradiction D-P7-01 (corrected in the matrix at P7): ``MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md``
said Resources and Prompts are "Not advertised" (OD-MCP-02), but FastMCP 3.3.1 *does* advertise the
``resources`` and ``prompts`` capabilities in the initialize result. They are always empty
(``resources/list``, ``resources/templates/list``, ``prompts/list`` return empty lists; reading or
getting an unknown item fails). ``test_acc_18_resources_prompts_capability_is_advertised_but_empty``
pins the observed behaviour so the divergence is explicit and a FastMCP change is visible.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from mcp.shared.exceptions import McpError
from mcp.types import LATEST_PROTOCOL_VERSION
from tests.qualification.mcp_server_x.reference_app.build import (
    REFERENCE_SERVER_ID,
    build_mcp_server_x_reference_qualification_server,
)
from tests.qualification.mcp_server_x.reference_app.http_serve import (
    ServedMcp,
    build_reference_host_parts,
    couple_reference_host,
    served_mcp,
    tool_payload,
    with_session,
)
from tests.qualification.mcp_server_x.reference_app.variants import QUAL_API_KEY, api_key_auth

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.network, pytest.mark.qualification]

_AUTH = {"X-Api-Key": QUAL_API_KEY}

CHECKED = "checked"
EMPTY_BUT_ADVERTISED = "advertised_empty"
NOT_ANNOUNCED = "not_announced"
ANNOUNCED_UNDOCUMENTED = "announced_undocumented"
UNSUPPORTED_BY_DESIGN = "unsupported_by_design"
_STATUSES = frozenset(
    {CHECKED, EMPTY_BUT_ADVERTISED, NOT_ANNOUNCED, ANNOUNCED_UNDOCUMENTED, UNSUPPORTED_BY_DESIGN},
)

# method / feature -> (status, name of the test function in this module that checks it)
PROTOCOL_SUBSET_MATRIX: dict[str, tuple[str, str]] = {
    "initialize": (CHECKED, "test_acc_18_announced_capabilities_snapshot"),
    "tools/list": (CHECKED, "test_acc_18_tools_list_and_call_work_over_http"),
    "tools/call": (CHECKED, "test_acc_18_tools_list_and_call_work_over_http"),
    "resources/list": (EMPTY_BUT_ADVERTISED, "test_acc_18_resources_and_prompts_lists_are_empty"),
    "resources/templates/list": (EMPTY_BUT_ADVERTISED, "test_acc_18_resources_and_prompts_lists_are_empty"),
    "prompts/list": (EMPTY_BUT_ADVERTISED, "test_acc_18_resources_and_prompts_lists_are_empty"),
    "resources/read": (UNSUPPORTED_BY_DESIGN, "test_acc_18_unknown_resource_and_prompt_requests_fail"),
    "prompts/get": (UNSUPPORTED_BY_DESIGN, "test_acc_18_unknown_resource_and_prompt_requests_fail"),
    "resources/subscribe": (NOT_ANNOUNCED, "test_acc_18_unannounced_methods_return_method_not_found"),
    "tasks/list": (NOT_ANNOUNCED, "test_acc_18_unannounced_methods_return_method_not_found"),
    "completion/complete": (NOT_ANNOUNCED, "test_acc_18_unannounced_methods_return_method_not_found"),
    "logging/setLevel": (ANNOUNCED_UNDOCUMENTED, "test_acc_18_logging_capability_is_announced_and_accepted"),
    "transport:stdio": (UNSUPPORTED_BY_DESIGN, "test_acc_18_only_streamable_http_transport_is_reachable"),
    "transport:sse": (UNSUPPORTED_BY_DESIGN, "test_acc_18_only_streamable_http_transport_is_reachable"),
}


@pytest.fixture(scope="module")
def served() -> Iterator[ServedMcp]:
    parts = build_reference_host_parts()
    mcp = build_mcp_server_x_reference_qualification_server(
        host_execution=parts.host_execution,
        registry=parts.registry,
    )
    with served_mcp(couple_reference_host(mcp, auth=api_key_auth())) as server:
        yield server


class _RawStreamableSession:
    """Minimal raw streamable HTTP MCP session (initialize, notify, arbitrary JSON-RPC request)."""

    def __init__(self, url: str, protocol_version: str = LATEST_PROTOCOL_VERSION) -> None:
        self._url = url
        self._headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            **_AUTH,
        }
        self._next_id = 1
        self.negotiated_version = ""
        self._initialize(protocol_version)

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[len("data:") :].strip())
        return json.loads(response.text)

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        return httpx.post(self._url, json=body, headers=self._headers, timeout=10.0)

    def _initialize(self, protocol_version: str) -> None:
        response = self._post(
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": protocol_version,
                    "capabilities": {},
                    "clientInfo": {"name": "p7-l3-raw", "version": "0"},
                },
            },
        )
        assert response.status_code == 200, response.text
        self.negotiated_version = self._decode(response)["result"]["protocolVersion"]
        self._headers["mcp-session-id"] = response.headers["mcp-session-id"]
        self._headers["mcp-protocol-version"] = self.negotiated_version
        notified = self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert notified.status_code == 202

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        response = self._post(
            {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}},
        )
        assert response.status_code == 200, response.text
        return self._decode(response)


async def _initialize_result(url: str) -> Any:
    async def _action(_session: Any, init: Any) -> Any:
        return init

    return await with_session(url, _action, headers=_AUTH)


# --------------------------------------------------------------------------- MCP-ACC-18


async def test_acc_18_announced_capabilities_snapshot(served: ServedMcp) -> None:
    """Exact announced-capability snapshot of the reference server at the installed FastMCP."""
    init = await _initialize_result(served.url)
    assert init.capabilities.model_dump(exclude_none=True) == {
        "experimental": {},
        "logging": {},
        "prompts": {"listChanged": True},
        "resources": {"subscribe": False, "listChanged": True},
        "tools": {"listChanged": True},
        "extensions": {"io.modelcontextprotocol/ui": {}},
    }
    assert init.capabilities.tasks is None
    assert init.capabilities.completions is None


async def test_acc_18_tools_list_and_call_work_over_http(served: ServedMcp) -> None:
    async def _action(session: Any, _init: Any) -> tuple[list[str], dict[str, Any]]:
        listed = await session.list_tools()
        called = await session.call_tool("qual_echo", {"message": "acc-18"})
        return [tool.name for tool in listed.tools], tool_payload(called)

    names, payload = await with_session(served.url, _action, headers=_AUTH)
    assert set(names) == {"list_agents", "run_agent", "qual_echo"}
    assert payload == {"echo": "acc-18", "server_id": REFERENCE_SERVER_ID}


async def test_acc_18_resources_and_prompts_lists_are_empty(served: ServedMcp) -> None:
    async def _action(session: Any, _init: Any) -> tuple[list[Any], list[Any], list[Any]]:
        resources = await session.list_resources()
        templates = await session.list_resource_templates()
        prompts = await session.list_prompts()
        return resources.resources, templates.resourceTemplates, prompts.prompts

    resources, templates, prompts = await with_session(served.url, _action, headers=_AUTH)
    assert resources == []
    assert templates == []
    assert prompts == []


async def test_acc_18_resources_prompts_capability_is_advertised_but_empty(served: ServedMcp) -> None:
    """Pins the documented-vs-observed contradiction (see module docstring): advertised, yet empty.

    ``MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md`` states Resources/Prompts are "Not advertised".
    The initialize result nevertheless carries both capabilities. This test asserts the observed
    behaviour on purpose; it is a documentation finding, not a product regression.
    """
    init = await _initialize_result(served.url)
    assert init.capabilities.resources is not None
    assert init.capabilities.prompts is not None
    assert init.capabilities.resources.subscribe is False


async def test_acc_18_unknown_resource_and_prompt_requests_fail(served: ServedMcp) -> None:
    async def _action(session: Any, _init: Any) -> None:
        with pytest.raises(McpError, match="Unknown resource"):
            await session.read_resource("file:///p7/does-not-exist")
        with pytest.raises(McpError, match="Unknown prompt"):
            await session.get_prompt("p7-does-not-exist")

    await with_session(served.url, _action, headers=_AUTH)


def test_acc_18_unannounced_methods_return_method_not_found(served: ServedMcp) -> None:
    """Methods whose capability is not announced answer JSON-RPC ``-32601`` (UNSUPPORTED stays UNSUPPORTED)."""
    session = _RawStreamableSession(served.url)
    requests: dict[str, dict[str, Any]] = {
        "tasks/list": {},
        "completion/complete": {
            "ref": {"type": "ref/prompt", "name": "p7"},
            "argument": {"name": "a", "value": "b"},
        },
        "resources/subscribe": {"uri": "file:///p7/x"},
    }
    for method, params in requests.items():
        reply = session.request(method, params)
        assert reply["error"]["code"] == -32601, (method, reply)
        assert reply["error"]["message"] == "Method not found", (method, reply)
        assert "result" not in reply


def test_acc_18_logging_capability_is_announced_and_accepted(served: ServedMcp) -> None:
    """Undocumented in the matrix: FastMCP announces ``logging`` and accepts ``logging/setLevel``."""
    session = _RawStreamableSession(served.url)
    reply = session.request("logging/setLevel", {"level": "info"})
    assert reply["result"] == {}


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        pytest.param("2025-03-26", "2025-03-26", id="older-supported-version-is-honoured"),
        pytest.param("1999-01-01", LATEST_PROTOCOL_VERSION, id="unknown-version-gets-latest"),
    ],
)
def test_acc_18_protocol_version_negotiation_is_recorded(served: ServedMcp, requested: str, expected: str) -> None:
    """Records SDK negotiation (REQ-MCP-P-02 is DEFERRED to the SDK; this is observation, not a contract)."""
    assert _RawStreamableSession(served.url, protocol_version=requested).negotiated_version == expected


def test_acc_18_only_streamable_http_transport_is_reachable(served: ServedMcp) -> None:
    """stdio and SSE stay UNSUPPORTED_BY_DESIGN: only the streamable HTTP endpoint answers MCP traffic."""
    base = f"http://127.0.0.1:{served.port}"
    for path in ("/sse", "/mcp/sse", "/messages/", "/mcp/messages/", "/stdio"):
        response = httpx.get(base + path, headers={**_AUTH, "Accept": "text/event-stream"}, timeout=10.0)
        assert response.status_code == 404, path
    assert _RawStreamableSession(served.url).negotiated_version


def test_acc_18_matrix_rows_reference_existing_tests() -> None:
    """The recorded matrix is complete and every row points at a real test in this module."""
    module_tests = {name for name, obj in globals().items() if name.startswith("test_acc_18_") and callable(obj)}
    assert {status for status, _ in PROTOCOL_SUBSET_MATRIX.values()} <= _STATUSES
    for method, (status, test_name) in PROTOCOL_SUBSET_MATRIX.items():
        assert test_name in module_tests, (method, status, test_name)
    checked = {m for m, (s, _) in PROTOCOL_SUBSET_MATRIX.items() if s == CHECKED}
    not_announced = {m for m, (s, _) in PROTOCOL_SUBSET_MATRIX.items() if s == NOT_ANNOUNCED}
    assert {"tools/list", "tools/call"} <= checked
    assert {"tasks/list", "completion/complete", "resources/subscribe"} == not_announced
