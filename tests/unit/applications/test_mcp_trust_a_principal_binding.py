# © Artur Czarnecki. All rights reserved.

"""TLX-TL11 TRUST-A — MCP harness principal binding and adversarial negatives."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.server.context import RequestContext, request_ctx
from starlette.requests import Request

from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.mcp_harness_principal import resolve_mcp_run_agent_identity
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _completed() -> TaskResult:
    return TaskResult(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        state=TaskState.COMPLETED,
        answer="ok",
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )


@contextmanager
def _fake_mcp_http_request(
    *,
    api_key: str | None = None,
    authorization: str | None = None,
) -> Iterator[Request]:
    headers: list[tuple[bytes, bytes]] = []
    if api_key is not None:
        headers.append((b"x-api-key", api_key.encode()))
    if authorization is not None:
        headers.append((b"authorization", authorization.encode()))
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": headers,
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
    }

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    ctx = RequestContext(
        request_id="qual-req",
        meta=None,
        session=None,
        lifespan_context=None,
        request=request,
    )
    token = request_ctx.set(ctx)
    try:
        yield request
    finally:
        request_ctx.reset(token)


def test_resolve_mcp_identity_api_key_binds_authoritative_tenant() -> None:
    auth = HarnessAuthState(
        require_api_key=True,
        resolved_api_key="qual-secret",
        api_key_principal_tenant_id="bound-tenant",
    )
    with _fake_mcp_http_request(api_key="qual-secret"):
        resolved = resolve_mcp_run_agent_identity(
            harness_auth=auth,
            asserted_tenant_id=None,
            asserted_user_id=None,
            default_tenant_id="mcp",
            default_user_id="mcp-user",
        )
    assert resolved.tenant_id == "bound-tenant"
    assert resolved.principal is not None
    assert resolved.principal.auth_mode == "api_key"


def test_resolve_mcp_identity_rejects_forged_tenant() -> None:
    auth = HarnessAuthState(
        require_api_key=True,
        resolved_api_key="qual-secret",
        api_key_principal_tenant_id="bound-tenant",
    )
    with _fake_mcp_http_request(api_key="qual-secret"):
        with pytest.raises(ToolError, match="identity_conflict"):
            resolve_mcp_run_agent_identity(
                harness_auth=auth,
                asserted_tenant_id="forged-tenant",
                asserted_user_id=None,
                default_tenant_id="mcp",
                default_user_id="mcp-user",
            )


def test_resolve_mcp_identity_rejects_forged_user() -> None:
    auth = HarnessAuthState(
        require_api_key=True,
        resolved_api_key="qual-secret",
        api_key_principal_tenant_id="bound-tenant",
    )
    with _fake_mcp_http_request(api_key="qual-secret"):
        with pytest.raises(ToolError, match="identity_conflict"):
            resolve_mcp_run_agent_identity(
                harness_auth=auth,
                asserted_tenant_id=None,
                asserted_user_id="forged-user",
                default_tenant_id="mcp",
                default_user_id="mcp-user",
            )


def test_resolve_mcp_identity_fail_closed_without_credentials() -> None:
    auth = HarnessAuthState(
        require_api_key=True,
        resolved_api_key="qual-secret",
        api_key_principal_tenant_id="bound-tenant",
    )
    with _fake_mcp_http_request():
        with pytest.raises(ToolError, match="unauthorized"):
            resolve_mcp_run_agent_identity(
                harness_auth=auth,
                asserted_tenant_id=None,
                asserted_user_id=None,
                default_tenant_id="mcp",
                default_user_id="mcp-user",
            )


@pytest.mark.asyncio
async def test_mcp_lab_inprocess_without_auth_preserves_client_tenant() -> None:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed())
    mcp = build_nexus_mcp_server(
        name="trust-a-lab",
        host_execution=port,
        registry=AgentRegistry(),
        default_capability="demo.cap",
    )
    async with Client(mcp) as client:
        await client.call_tool(
            "run_agent",
            {"message": "lab", "tenant_id": "lab-tenant", "user_id": "lab-user"},
        )
    task = port.execute.await_args.args[0]
    assert task.tenant_id == "lab-tenant"
    assert task.user_id == "lab-user"


def test_mcp_identity_resolution_isolated_per_request_context() -> None:
    auth_a = HarnessAuthState(
        require_api_key=True,
        resolved_api_key="key-a",
        api_key_principal_tenant_id="tenant-a",
    )
    auth_b = HarnessAuthState(
        require_api_key=True,
        resolved_api_key="key-b",
        api_key_principal_tenant_id="tenant-b",
    )
    with _fake_mcp_http_request(api_key="key-a"):
        tenant_a = resolve_mcp_run_agent_identity(
            harness_auth=auth_a,
            asserted_tenant_id=None,
            asserted_user_id=None,
            default_tenant_id="mcp",
            default_user_id="mcp-user",
        ).tenant_id
    with _fake_mcp_http_request(api_key="key-b"):
        tenant_b = resolve_mcp_run_agent_identity(
            harness_auth=auth_b,
            asserted_tenant_id=None,
            asserted_user_id=None,
            default_tenant_id="mcp",
            default_user_id="mcp-user",
        ).tenant_id
    assert tenant_a == "tenant-a"
    assert tenant_b == "tenant-b"


def test_mcp_http_ingress_still_rejects_missing_credentials() -> None:
    api = FastAPI()
    api.state.harness_auth = HarnessAuthState(
        require_api_key=True,
        resolved_api_key="qual-secret",
        api_key_principal_tenant_id="bound-tenant",
    )
    port = AsyncMock(spec=HostTaskExecutionPort)
    mcp = build_nexus_mcp_server(
        name="ingress",
        host_execution=port,
        registry=AgentRegistry(),
        default_capability="demo.cap",
    )
    combined = couple_fastapi_with_mcp(api, mcp, mount_path="/mcp")
    with TestClient(combined) as client:
        response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert response.status_code == 401
