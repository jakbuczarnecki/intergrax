"""Red-first NT-TEN probes using MCP identity APIs present at FRZ-TEN-07 prefix."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from fastmcp.exceptions import ToolError
from fastmcp.server.context import RequestContext, request_ctx
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.mcp_harness_principal import resolve_mcp_run_agent_identity
from starlette.requests import Request

pytestmark = pytest.mark.unit


@contextmanager
def _request(*, client_host: str = "127.0.0.1") -> Iterator[Request]:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [],
        "query_string": b"",
        "client": (client_host, 12345),
        "app": FastAPI(),
    }

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    ctx = RequestContext(request_id="nt-ten-07", meta=None, session=None, lifespan_context=None, request=request)
    token = request_ctx.set(ctx)
    try:
        yield request
    finally:
        request_ctx.reset(token)


def test_nt_ten_07_mcp_client_asserted_tenant_identity_must_fail_closed() -> None:
    """NT-TEN-07 / FRZ-TEN-07: MCP client-asserted tenant identity must fail closed."""
    with _request():
        try:
            resolved = resolve_mcp_run_agent_identity(
                harness_auth=HarnessAuthState(),
                asserted_tenant_id="victim",
                asserted_user_id=None,
                default_tenant_id="mcp",
                default_user_id="mcp-user",
            )
        except ToolError:
            return
        assert resolved.tenant_id != "victim", (
            "NT-TEN-07 MCP client-asserted tenant identity must fail closed: "
            "client-asserted tenant must not establish truth without verified harness principal"
        )
