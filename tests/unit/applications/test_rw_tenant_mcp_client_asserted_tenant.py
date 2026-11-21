# © Artur Czarnecki. All rights reserved.
"""T-IGX-051-A Part F: MCP fails_closed composition guard and asserted-tenant refusal."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.server.context import RequestContext, request_ctx
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.mcp_harness_principal import (
    McpIdentityCompositionError,
    assert_mcp_identity_composition,
    resolve_mcp_run_agent_identity,
)
from intergrax.applications._shared.mcp_identity_mode import McpIdentityMode
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry
from starlette.requests import Request
from tests.unit.applications.test_mcp_identity_mode_fail_closed import _completed

pytestmark = pytest.mark.unit

_LOCAL = McpIdentityMode.LOCAL_DEV_EXPLICIT


@contextmanager
def _request() -> Iterator[Request]:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "app": FastAPI(),
    }

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    ctx = RequestContext(request_id="ten-mcp", meta=None, session=None, lifespan_context=None, request=request)
    token = request_ctx.set(ctx)
    try:
        yield request
    finally:
        request_ctx.reset(token)


def _host() -> AsyncMock:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed())
    return port


def _server(port: AsyncMock, **kwargs: object):
    return build_nexus_mcp_server(
        name="rw-tenant-mcp",
        host_execution=port,
        registry=AgentRegistry(),
        default_capability="demo.cap",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_ten_07a_default_mode_refuses_asserted_tenant() -> None:
    port = _host()
    async with Client(_server(port)) as client:
        with pytest.raises(ToolError, match="MCP_PRINCIPAL_REQUIRED"):
            await client.call_tool("run_agent", {"message": "x", "tenant_id": "victim"})
    port.execute.assert_not_awaited()


def test_ten_07b_local_dev_mode_refused_when_fails_closed() -> None:
    with pytest.raises(McpIdentityCompositionError):
        assert_mcp_identity_composition(
            identity_mode=_LOCAL,
            harness_auth=None,
            fails_closed=True,
        )


def test_ten_07c_no_harness_auth_refused_when_fails_closed() -> None:
    with pytest.raises(McpIdentityCompositionError):
        assert_mcp_identity_composition(
            identity_mode=McpIdentityMode.HARNESS_BOUND,
            harness_auth=None,
            fails_closed=True,
        )


def test_ten_07d_assertion_branch_refused_when_fails_closed() -> None:
    with _request():
        with pytest.raises(ToolError, match="MCP_PRINCIPAL_REQUIRED"):
            resolve_mcp_run_agent_identity(
                harness_auth=None,
                asserted_tenant_id="victim",
                asserted_user_id="victim-u",
                default_tenant_id="mcp",
                default_user_id="mcp-user",
                identity_mode=_LOCAL,
                fails_closed=True,
            )

    with _request():
        resolved = resolve_mcp_run_agent_identity(
            harness_auth=None,
            asserted_tenant_id="dev-t",
            asserted_user_id="dev-u",
            default_tenant_id="mcp",
            default_user_id="mcp-user",
            identity_mode=_LOCAL,
            fails_closed=False,
        )
    assert resolved.tenant_id == "dev-t"


def test_ten_07e_conflicting_assertion_refused() -> None:
    auth = HarnessAuthState(
        require_api_key=True,
        resolved_api_key="vk",
        api_key_principal_tenant_id="verified-t",
    )

    @contextmanager
    def _request_with_key() -> Iterator[Request]:
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [(b"x-api-key", b"vk")],
            "query_string": b"",
            "client": ("127.0.0.1", 12345),
            "app": FastAPI(),
        }

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b"", "more_body": False}

        request = Request(scope, receive)
        ctx = RequestContext(request_id="ten-mcp-key", meta=None, session=None, lifespan_context=None, request=request)
        token = request_ctx.set(ctx)
        try:
            yield request
        finally:
            request_ctx.reset(token)

    with _request_with_key():
        with pytest.raises(ToolError, match="identity_conflict"):
            resolve_mcp_run_agent_identity(
                harness_auth=auth,
                asserted_tenant_id="forged-tenant",
                asserted_user_id=None,
                default_tenant_id="mcp",
                default_user_id="mcp-user",
            )
