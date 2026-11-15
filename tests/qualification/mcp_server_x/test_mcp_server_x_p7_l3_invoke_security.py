# © Artur Czarnecki. All rights reserved.

"""MCP-SERVER-X P7 (MCP-P7-03, ticket T-IGX-035-H) - authorized and denied invoke, level L3.

Level: **L3 - physical streamable HTTP on loopback** (``uvicorn`` on an OS-assigned port,
official MCP streamable HTTP client or raw HTTP). The host task execution port behind the
server is an ``AsyncMock`` of ``HostTaskExecutionPort`` so a test can assert exactly what reached
execution (and, for every denial, that nothing did). The identity provider is a **contract stub**
(``StubIdentityProvider``), not vendor integration evidence. L2 checks of the same behaviour live
in other modules and are never counted here.

Covers MCP-ACC-09 (authorized invoke with tenant assertion), MCP-ACC-10 (L3 deny: invalid key,
forged tenant/user, missing credential) and MCP-ACC-11 (L3 cross-tenant deny on one server and
across two hosts that share a registry).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock

import httpx
import pytest
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry
from tests.qualification.mcp_server_x.reference_app.build import (
    build_mcp_server_x_reference_qualification_server,
)
from tests.qualification.mcp_server_x.reference_app.http_serve import (
    ServedMcp,
    call_tool,
    completed_task_result,
    couple_reference_host,
    error_text,
    jsonrpc_post,
    list_tool_names,
    served_mcp,
    tool_payload,
)
from tests.qualification.mcp_server_x.reference_app.variants import (
    QUAL_API_KEY,
    QUAL_API_KEY_TENANT,
    QUAL_TENANT_B,
    api_key_auth,
    identity_provider_auth,
    two_tenant_identity_provider,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.network, pytest.mark.qualification]

_API_KEY_SERVICE_USER = "harness-api-key"
_KEY_HEADERS = {"X-Api-Key": QUAL_API_KEY}


@contextmanager
def _serve_with_port(
    auth: HarnessAuthState,
    *,
    registry: AgentRegistry | None = None,
) -> Iterator[tuple[ServedMcp, AsyncMock]]:
    """Reference server on loopback whose host execution is an inspectable ``AsyncMock`` port."""
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=completed_task_result())
    mcp = build_mcp_server_x_reference_qualification_server(
        host_execution=port,
        registry=registry if registry is not None else AgentRegistry(),
    )
    with served_mcp(couple_reference_host(mcp, auth=auth)) as served:
        yield served, port


def _leaves(group: BaseException) -> list[BaseException]:
    if isinstance(group, BaseExceptionGroup):
        found: list[BaseException] = []
        for item in group.exceptions:
            found.extend(_leaves(item))
        return found
    return [group]


async def _http_status_failure(url: str, headers: dict[str, str]) -> list[int]:
    """Run an MCP client session that must fail at HTTP level; return the HTTP status codes seen."""
    try:
        await list_tool_names(url, headers=headers)
    except BaseExceptionGroup as group:
        leaves = _leaves(group)
    except httpx.HTTPStatusError as exc:
        leaves = [exc]
    else:
        raise AssertionError("the client session was expected to fail with an HTTP status error")
    assert leaves, "no leaf exception"
    for leaf in leaves:
        assert isinstance(leaf, httpx.HTTPStatusError), repr(leaf)
    return [leaf.response.status_code for leaf in leaves]  # type: ignore[attr-defined]


def _executed_tasks(port: AsyncMock) -> list[object]:
    return [call.args[0] for call in port.execute.await_args_list]


# --------------------------------------------------------------------------- MCP-ACC-09


async def test_acc_09_authorized_invoke_without_assertion_binds_principal_tenant() -> None:
    with _serve_with_port(api_key_auth()) as (served, port):
        result = await call_tool(served.url, "run_agent", {"message": "hi"}, headers=_KEY_HEADERS)
    assert result.isError is not True, error_text(result)
    assert tool_payload(result)["state"] == "completed"
    (task,) = _executed_tasks(port)
    assert task.tenant_id == QUAL_API_KEY_TENANT
    assert task.user_id == _API_KEY_SERVICE_USER


async def test_acc_09_authorized_invoke_with_matching_tenant_and_user_assertion() -> None:
    with _serve_with_port(api_key_auth()) as (served, port):
        result = await call_tool(
            served.url,
            "run_agent",
            {"message": "hi", "tenant_id": QUAL_API_KEY_TENANT, "user_id": _API_KEY_SERVICE_USER},
            headers=_KEY_HEADERS,
        )
    assert result.isError is not True, error_text(result)
    (task,) = _executed_tasks(port)
    assert (task.tenant_id, task.user_id) == (QUAL_API_KEY_TENANT, _API_KEY_SERVICE_USER)


async def test_acc_09_authorized_invoke_via_bearer_identity_binds_token_tenant_and_user() -> None:
    with _serve_with_port(identity_provider_auth(two_tenant_identity_provider())) as (served, port):
        result = await call_tool(
            served.url,
            "run_agent",
            {"message": "hi", "tenant_id": QUAL_TENANT_B},
            headers={"Authorization": "Bearer token-b"},
        )
    assert result.isError is not True, error_text(result)
    (task,) = _executed_tasks(port)
    assert (task.tenant_id, task.user_id) == (QUAL_TENANT_B, "user-b")


# --------------------------------------------------------------------------- MCP-ACC-10


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({"X-Api-Key": "wrong-key"}, id="wrong-x-api-key"),
        pytest.param({"Authorization": "Bearer wrong-key"}, id="wrong-bearer"),
        pytest.param({"X-Api-Key": ""}, id="empty-x-api-key"),
    ],
)
async def test_acc_10_invalid_credentials_are_rejected_with_http_401_before_execution(
    headers: dict[str, str],
) -> None:
    with _serve_with_port(api_key_auth()) as (served, port):
        assert await _http_status_failure(served.url, headers) == [401]
        raw = jsonrpc_post(served.url, "tools/list", headers=headers)
    assert raw.status_code == 401
    assert raw.json() == {"detail": "Invalid or missing harness credentials"}
    port.execute.assert_not_awaited()


async def test_acc_10_missing_credential_is_rejected_with_http_401_and_leaks_no_tool_names() -> None:
    with _serve_with_port(api_key_auth()) as (served, port):
        assert await _http_status_failure(served.url, {}) == [401]
        listed = jsonrpc_post(served.url, "tools/list")
        called = jsonrpc_post(served.url, "tools/call", {"name": "run_agent", "arguments": {"message": "x"}})
    for response in (listed, called):
        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid or missing harness credentials"}
        assert "run_agent" not in response.text
    port.execute.assert_not_awaited()


async def test_acc_10_forged_tenant_assertion_is_denied_before_execution() -> None:
    with _serve_with_port(api_key_auth()) as (served, port):
        result = await call_tool(
            served.url,
            "run_agent",
            {"message": "x", "tenant_id": "forged-tenant"},
            headers=_KEY_HEADERS,
        )
    assert result.isError is True
    text = error_text(result) or ""
    assert text.startswith("MCP_TENANT_ASSERTION_CONFLICT:") and "identity_conflict:" in text
    assert "tenant_id" in text
    port.execute.assert_not_awaited()


async def test_acc_10_forged_user_assertion_is_denied_before_execution() -> None:
    with _serve_with_port(api_key_auth()) as (served, port):
        result = await call_tool(
            served.url,
            "run_agent",
            {"message": "x", "user_id": "forged-user"},
            headers=_KEY_HEADERS,
        )
    assert result.isError is True
    text = error_text(result) or ""
    assert text.startswith("MCP_TENANT_ASSERTION_CONFLICT:") and "identity_conflict:" in text
    assert "user_id" in text
    port.execute.assert_not_awaited()


async def test_acc_10_unknown_bearer_token_is_rejected_by_identity_provider_server() -> None:
    with _serve_with_port(identity_provider_auth(two_tenant_identity_provider())) as (served, port):
        headers = {"Authorization": "Bearer not-a-known-token"}
        assert await _http_status_failure(served.url, headers) == [401]
    port.execute.assert_not_awaited()


# --------------------------------------------------------------------------- MCP-ACC-11


async def test_acc_11_principal_cannot_assert_another_tenant_on_one_server() -> None:
    """Tenant A's token asserting tenant B is denied; tenant B's own token with the same assertion works."""
    with _serve_with_port(identity_provider_auth(two_tenant_identity_provider())) as (served, port):
        denied = await call_tool(
            served.url,
            "run_agent",
            {"message": "x", "tenant_id": QUAL_TENANT_B},
            headers={"Authorization": "Bearer token-a"},
        )
        assert denied.isError is True
        assert (error_text(denied) or "").startswith("MCP_TENANT_ASSERTION_CONFLICT:")
        port.execute.assert_not_awaited()
        allowed = await call_tool(
            served.url,
            "run_agent",
            {"message": "x", "tenant_id": QUAL_TENANT_B},
            headers={"Authorization": "Bearer token-b"},
        )
    assert allowed.isError is not True, error_text(allowed)
    (task,) = _executed_tasks(port)
    assert (task.tenant_id, task.user_id) == (QUAL_TENANT_B, "user-b")


async def test_acc_11_concurrent_principals_stay_isolated_on_one_server() -> None:
    with _serve_with_port(identity_provider_auth(two_tenant_identity_provider())) as (served, port):
        results = await asyncio.gather(
            *(
                call_tool(
                    served.url,
                    "run_agent",
                    {"message": f"{token}-{index}"},
                    headers={"Authorization": f"Bearer {token}"},
                )
                for index in range(3)
                for token in ("token-a", "token-b")
            ),
        )
    assert all(result.isError is not True for result in results)
    observed = {(task.user_id, task.tenant_id) for task in _executed_tasks(port)}
    assert observed == {("user-a", QUAL_API_KEY_TENANT), ("user-b", QUAL_TENANT_B)}
    assert len(_executed_tasks(port)) == 6


async def test_acc_11_two_hosts_sharing_a_registry_do_not_accept_each_others_credentials() -> None:
    shared_registry = AgentRegistry()
    auth_a = api_key_auth(api_key="key-host-a", tenant_id="tenant-host-a")
    auth_b = api_key_auth(api_key="key-host-b", tenant_id="tenant-host-b")
    with (
        _serve_with_port(auth_a, registry=shared_registry) as (host_a, port_a),
        _serve_with_port(auth_b, registry=shared_registry) as (host_b, port_b),
    ):
        assert await _http_status_failure(host_b.url, {"X-Api-Key": "key-host-a"}) == [401]
        assert await _http_status_failure(host_a.url, {"X-Api-Key": "key-host-b"}) == [401]
        port_a.execute.assert_not_awaited()
        port_b.execute.assert_not_awaited()
        ok_a = await call_tool(host_a.url, "run_agent", {"message": "a"}, headers={"X-Api-Key": "key-host-a"})
        ok_b = await call_tool(host_b.url, "run_agent", {"message": "b"}, headers={"X-Api-Key": "key-host-b"})
        forged = await call_tool(
            host_a.url,
            "run_agent",
            {"message": "a", "tenant_id": "tenant-host-b"},
            headers={"X-Api-Key": "key-host-a"},
        )
    assert ok_a.isError is not True and ok_b.isError is not True
    assert forged.isError is True
    assert [task.tenant_id for task in _executed_tasks(port_a)] == ["tenant-host-a"]
    assert [task.tenant_id for task in _executed_tasks(port_b)] == ["tenant-host-b"]
