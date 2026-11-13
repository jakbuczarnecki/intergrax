# © Artur Czarnecki. All rights reserved.

"""MCP-SERVER-X P7 lane L2: denied invoke, tenant isolation, principal binding, correlation keys.

Level: **L2** (in-process). Tools are invoked through the FastMCP server object
(``FastMCP.call_tool`` / in-memory ``Client``) with a Starlette ``Request`` placed into the
FastMCP request context to emulate what the HTTP ingress would provide. This module is
**not a physical transport test**: no socket, no uvicorn, no streamable HTTP session. It is
never counted as L3. Physical HTTP counterparts live in the L3 lane files.

Scenarios (ticket ``T-IGX-035-H``, no product-path change):

* MCP-ACC-10 denied invoke (side-effect port deny, invalid / missing credentials).
* MCP-ACC-11 cross-tenant deny.
* MCP-ACC-12 principal binding.
* MCP-ACC-17 pin the existing correlation keys ``task_id`` / ``run_id`` / ``execution_id``.
  ``REQ-MCP-OBS-01`` (a dedicated MCP invocation trace id) is a documented residual (DR-1):
  nothing here asserts such an id, and nothing here asserts its absence either.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.server.context import RequestContext, request_ctx
from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.harness_principal import (
    api_key_service_harness_principal,
    harness_principal_to_request_identity,
)
from intergrax.applications._shared.mcp_catalog_invocation_bridge import _METADATA_STEP_ID
from intergrax.applications._shared.mcp_developer_api import (
    McpServerCompositionDeps,
    compose_mcp_server_from_definition,
)
from intergrax.applications._shared.mcp_identity_mode import McpIdentityMode
from intergrax.applications._shared.strict_production_serving_http_execution_principal import (
    get_strict_serving_http_caller_principal,
)
from intergrax.contracts.agent_run import canonical_principal_id_from_request_identity
from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_id,
    peek_active_execution_identity,
    require_active_execution_id,
    reset_active_execution_identity,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.mcp_server_definition import (
    McpCatalogToolBinding,
    McpHostBuiltinBinding,
    McpHostBuiltinId,
    build_mcp_server_definition,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.tool_execution_outcome import ToolExecutionResult
from intergrax.integrations.contracts.identity_provider import IdentityUser
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.registry.runtime import ToolRegistry
from pydantic import BaseModel
from starlette.requests import Request
from tests.fixtures.harness_host_task_execution import build_harness_host_task_execution
from tests.qualification.mcp_server_x.reference_app.build import (
    REFERENCE_CAPABILITY,
    REFERENCE_TENANT,
    REFERENCE_USER,
    build_mcp_server_x_reference_qualification_server,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_API_KEY = "p7-l2-secret"
_TENANT_A = "p7-tenant-a"
_TENANT_B = "p7-tenant-b"
_SERVICE_A = "svc-p7-a"
_SERVICE_B = "svc-p7-b"
_STEP = "p7.l2.catalog.step"
_AGENT = "agent-p7-l2"


# --------------------------------------------------------------------------------------
# Helpers (file-local; the shared reference_app fixtures are consumed read-only)
# --------------------------------------------------------------------------------------


def _auth_app(state: HarnessAuthState) -> FastAPI:
    app = FastAPI()
    app.state.harness_auth = state
    return app


def _api_key_state(
    *,
    tenant: str | None = _TENANT_A,
    service: str = _SERVICE_A,
    key: str = _API_KEY,
    **extra: Any,
) -> HarnessAuthState:
    return HarnessAuthState(
        require_api_key=True,
        resolved_api_key=key,
        api_key_principal_tenant_id=tenant,
        api_key_principal_service_id=service,
        **extra,
    )


@contextlib.contextmanager
def _mcp_http_request(
    app: FastAPI,
    *,
    api_key: str | None = None,
    authorization: str | None = None,
) -> Iterator[Request]:
    """Place a Starlette request (as the HTTP ingress would) into the FastMCP request context."""
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
        "app": app,
    }

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    ctx = RequestContext(
        request_id="p7-l2-req",
        meta=None,
        session=SimpleNamespace(),  # FastMCP caches a state prefix on the session object
        lifespan_context=None,
        request=request,
    )
    token = request_ctx.set(ctx)
    try:
        yield request
    finally:
        request_ctx.reset(token)


def _completed_result() -> TaskResult:
    return TaskResult(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        state=TaskState.COMPLETED,
        answer="p7-l2-done",
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )


def _recording_host() -> AsyncMock:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed_result())
    return port


def _reference_server(host: object, **kwargs: object):  # type: ignore[no-untyped-def]
    return build_mcp_server_x_reference_qualification_server(
        host_execution=host,  # type: ignore[arg-type]
        registry=AgentRegistry(),
        **kwargs,  # type: ignore[arg-type]
    )


async def _call(mcp: object, name: str, arguments: dict[str, object]) -> object:
    """Invoke a tool through the FastMCP server object; ``structured_content`` is the payload."""
    result = await mcp.call_tool(name, arguments)  # type: ignore[attr-defined]
    return result.structured_content


class _In(BaseModel):
    query: str = "x"


class _Out(BaseModel):
    value: str = "ok"


def _contract(tool_id: str, *, side_effects: bool = False) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description="p7 l2 security",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=side_effects,
        risk_level=ToolRiskLevel.LOW,
        category="p7-l2",
    )


def _tool_registry(*contracts: ToolContract) -> ToolRegistry:
    registry = ToolRegistry()
    for contract in contracts:

        async def _run(_input: _In, _ctx: object) -> _Out:
            return _Out(value="should-not-run")

        registry.register(contract, _run)
    return registry


@dataclass
class _RecordingCatalogInvoker:
    caller_agent_id: str = _AGENT
    results: list[ExecutionBoundCatalogToolInvokeRequest] = field(default_factory=list)

    def invoke(self, request: ExecutionBoundCatalogToolInvokeRequest) -> ToolExecutionResult[_Out]:
        self.results.append(request)
        return ToolExecutionResult.ok(_Out(value="invoked"))


@dataclass
class _MsePort:
    permitted: bool
    seen: list[CollaborativeWorkEnforcementRequest] = field(default_factory=list)

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str = "platform.orchestration.tool_invocation",
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        _ = source_agent_id, source_step_id
        self.seen.append(request)
        action = PolicyAction.ALLOW if self.permitted else PolicyAction.DENY
        return MeaningfulSideEffectAuthorizationResult(
            permitted=self.permitted,
            decision=PolicyDecision(action=action, reason="p7-l2"),
            enforcement_result=MagicMock(),
            requires_governed_continuation=False,
        )


def _catalog_server(
    tool_registry: ToolRegistry,
    invoker: _RecordingCatalogInvoker,
    *,
    mse_port: _MsePort | None,
):  # type: ignore[no-untyped-def]
    bindings: list = [
        McpHostBuiltinBinding(
            binding_id="list_catalog_tools",
            builtin=McpHostBuiltinId.LIST_CATALOG_TOOLS,
        ),
        McpHostBuiltinBinding(
            binding_id="describe_catalog_tool",
            builtin=McpHostBuiltinId.DESCRIBE_CATALOG_TOOL,
        ),
        McpHostBuiltinBinding(
            binding_id="invoke_catalog_tool",
            builtin=McpHostBuiltinId.INVOKE_CATALOG_TOOL,
        ),
    ]
    bindings.extend(
        McpCatalogToolBinding(binding_id=f"catalog:{item.contract.tool_id}", tool_id=item.contract.tool_id)
        for item in tool_registry.list()
    )
    definition = build_mcp_server_definition(
        server_id="p7-l2-catalog-security",
        display_name="P7 L2 catalog security",
        default_capability="p7.l2",
        default_tenant_id=_TENANT_A,
        default_user_id=_SERVICE_A,
        bindings=tuple(bindings),
    )
    deps = McpServerCompositionDeps(
        identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT,
        host_execution=_recording_host(),
        registry=AgentRegistry(),
        tool_registry=tool_registry,
        catalog_tool_invoker=invoker,
        meaningful_side_effect_authorization=mse_port,
    )
    return compose_mcp_server_from_definition(definition, deps)


async def _register_binding(*, tenant: str, user_id: str) -> tuple[Task, str]:
    task_id = mint_task_id()
    run_id = mint_run_id()
    task = Task(
        task_id=task_id,
        tenant_id=tenant,
        user_id=user_id,
        message="p7-l2",
        context=TaskContext(capability="p7.l2"),
        metadata={_METADATA_STEP_ID: _STEP},
        agent_id=_AGENT,
    )
    await ActiveTaskRegistry.register(task, run_id)
    return task, str(run_id)


@contextlib.contextmanager
def _parent_execution_context(task: Task, run_id: str, *, principal_id: str) -> Iterator[str]:
    parent_execution_id = mint_execution_id()
    authority_token = bind_active_execution_authority(ParentExecutionAuthority.unrestricted_root())
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=parent_execution_id,
        task_id=task.task_id,
    )
    governance_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=task.tenant_id,
            workspace_id="ws-p7-l2",
            principal_id=principal_id,
        ),
    )
    task.metadata["workspace_id"] = "ws-p7-l2"
    try:
        yield str(parent_execution_id)
    finally:
        reset_active_execution_governance_identity(governance_token)
        reset_active_execution_identity(identity_token)
        reset_active_execution_authority(authority_token)


def _principal_id_for(state: HarnessAuthState) -> str:
    principal = api_key_service_harness_principal(
        tenant_id=state.api_key_principal_tenant_id,
        service_id=state.api_key_principal_service_id,
    )
    return canonical_principal_id_from_request_identity(harness_principal_to_request_identity(principal))


@pytest.fixture(autouse=True)
async def _clear_active_tasks() -> Any:
    ActiveTaskRegistry.clear_for_tests()
    yield
    ActiveTaskRegistry.clear_for_tests()


# --------------------------------------------------------------------------------------
# MCP-ACC-10: denied invoke
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acc_10_side_effect_port_deny_blocks_catalog_invoke() -> None:
    state = _api_key_state()
    invoker = _RecordingCatalogInvoker()
    port = _MsePort(permitted=False)
    mcp = _catalog_server(_tool_registry(_contract("p7.fx", side_effects=True)), invoker, mse_port=port)
    task, run_id = await _register_binding(tenant=_TENANT_A, user_id=_SERVICE_A)
    with _mcp_http_request(_auth_app(state), api_key=_API_KEY):
        with _parent_execution_context(task, run_id, principal_id=_principal_id_for(state)):
            with pytest.raises(ToolError, match="side_effect_authorization_denied"):
                await _call(
                    mcp,
                    "invoke_catalog_tool",
                    {"tool_id": "p7.fx", "task_id": str(task.task_id), "run_id": run_id, "step_id": _STEP},
                )
    assert len(port.seen) == 1, "the deny decision must come from the injected port, once"
    assert invoker.results == [], "a denied side effect must never reach the tool invoker"


@pytest.mark.asyncio
async def test_acc_10_side_effect_port_permit_control_reaches_invoker() -> None:
    """Control for the deny test: the same path succeeds when the port permits (not vacuous)."""
    state = _api_key_state()
    invoker = _RecordingCatalogInvoker()
    port = _MsePort(permitted=True)
    mcp = _catalog_server(_tool_registry(_contract("p7.fx", side_effects=True)), invoker, mse_port=port)
    task, run_id = await _register_binding(tenant=_TENANT_A, user_id=_SERVICE_A)
    with _mcp_http_request(_auth_app(state), api_key=_API_KEY):
        with _parent_execution_context(task, run_id, principal_id=_principal_id_for(state)):
            payload = await _call(
                mcp,
                "invoke_catalog_tool",
                {"tool_id": "p7.fx", "task_id": str(task.task_id), "run_id": run_id, "step_id": _STEP},
            )
    assert payload["success"] is True
    assert payload["tool_id"] == "p7.fx"
    assert len(invoker.results) == 1


@pytest.mark.asyncio
async def test_acc_10_side_effect_tool_without_authorization_port_is_denied() -> None:
    state = _api_key_state()
    invoker = _RecordingCatalogInvoker()
    mcp = _catalog_server(_tool_registry(_contract("p7.fx", side_effects=True)), invoker, mse_port=None)
    task, run_id = await _register_binding(tenant=_TENANT_A, user_id=_SERVICE_A)
    with _mcp_http_request(_auth_app(state), api_key=_API_KEY):
        with _parent_execution_context(task, run_id, principal_id=_principal_id_for(state)):
            with pytest.raises(ToolError, match="unauthorized_side_effect_tool"):
                await _call(
                    mcp,
                    "invoke_catalog_tool",
                    {"tool_id": "p7.fx", "task_id": str(task.task_id), "run_id": run_id, "step_id": _STEP},
                )
    assert invoker.results == []


@pytest.mark.asyncio
async def test_acc_10_invalid_api_key_is_unauthorized_before_execution() -> None:
    host = _recording_host()
    mcp = _reference_server(host)
    with _mcp_http_request(_auth_app(_api_key_state()), api_key="wrong-key"):
        with pytest.raises(ToolError, match="unauthorized"):
            await _call(mcp, "run_agent", {"message": "x"})
    host.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_acc_10_invalid_bearer_key_is_unauthorized_before_execution() -> None:
    host = _recording_host()
    mcp = _reference_server(host)
    with _mcp_http_request(_auth_app(_api_key_state()), authorization="Bearer wrong-key"):
        with pytest.raises(ToolError, match="unauthorized"):
            await _call(mcp, "run_agent", {"message": "x"})
    host.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_acc_10_missing_credentials_is_unauthorized_before_execution() -> None:
    host = _recording_host()
    mcp = _reference_server(host)
    with _mcp_http_request(_auth_app(_api_key_state())):
        with pytest.raises(ToolError, match="unauthorized"):
            await _call(mcp, "run_agent", {"message": "x"})
    host.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_acc_10_rejected_identity_provider_token_is_unauthorized() -> None:
    class _RejectingProvider:
        def verify_token(self, token: str) -> IdentityUser:
            raise PermissionError(f"bad token {token}")

        def userinfo(self, token: str) -> IdentityUser:
            raise PermissionError(token)

        def list_tenants(self, *, limit: int = 50) -> list[object]:
            return []

    host = _recording_host()
    mcp = _reference_server(host)
    state = HarnessAuthState(identity_provider=_RejectingProvider())
    with _mcp_http_request(_auth_app(state), authorization="Bearer some-token"):
        with pytest.raises(ToolError, match="unauthorized"):
            await _call(mcp, "run_agent", {"message": "x"})
    host.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_acc_10_catalog_invoke_without_verified_caller_is_missing_caller_identity() -> None:
    invoker = _RecordingCatalogInvoker()
    mcp = _catalog_server(_tool_registry(_contract("p7.ok")), invoker, mse_port=None)
    task, run_id = await _register_binding(tenant=_TENANT_A, user_id=_SERVICE_A)
    # In-process client, no harness auth: lab identity carries no verified principal.
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="missing_caller_identity"):
            await client.call_tool(
                "invoke_catalog_tool",
                {"tool_id": "p7.ok", "task_id": str(task.task_id), "run_id": run_id},
            )
    assert invoker.results == []


@pytest.mark.asyncio
async def test_acc_10_same_tenant_other_user_is_not_the_execution_owner() -> None:
    state = _api_key_state(service=_SERVICE_B)
    invoker = _RecordingCatalogInvoker()
    mcp = _catalog_server(_tool_registry(_contract("p7.ok")), invoker, mse_port=None)
    task, run_id = await _register_binding(tenant=_TENANT_A, user_id=_SERVICE_A)
    with _mcp_http_request(_auth_app(state), api_key=_API_KEY):
        with pytest.raises(ToolError, match="unauthorized_execution_caller"):
            await _call(
                mcp,
                "invoke_catalog_tool",
                {"tool_id": "p7.ok", "task_id": str(task.task_id), "run_id": run_id, "step_id": _STEP},
            )
    assert invoker.results == []


def test_acc_10_asgi_ingress_rejects_missing_and_invalid_credentials_with_401() -> None:
    """In-process ASGI (``TestClient``), not a physical transport test."""
    host = _recording_host()
    api = FastAPI()
    api.state.harness_auth = _api_key_state()
    combined = couple_fastapi_with_mcp(api, _reference_server(host), mount_path="/mcp")
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    with TestClient(combined) as client:
        missing = client.post("/mcp", json=body)
        invalid = client.post("/mcp", json=body, headers={"X-Api-Key": "wrong-key"})
        invalid_bearer = client.post("/mcp", json=body, headers={"Authorization": "Bearer wrong-key"})
    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert invalid_bearer.status_code == 401
    host.execute.assert_not_awaited()


# --------------------------------------------------------------------------------------
# MCP-ACC-11: cross-tenant deny
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acc_11_catalog_invoke_for_other_tenants_execution_is_wrong_tenant() -> None:
    state = _api_key_state(tenant=_TENANT_B, service=_SERVICE_B)
    invoker = _RecordingCatalogInvoker()
    mcp = _catalog_server(_tool_registry(_contract("p7.ok")), invoker, mse_port=None)
    task, run_id = await _register_binding(tenant=_TENANT_A, user_id=_SERVICE_A)
    with _mcp_http_request(_auth_app(state), api_key=_API_KEY):
        with pytest.raises(ToolError, match="wrong_tenant"):
            await _call(
                mcp,
                "invoke_catalog_tool",
                {"tool_id": "p7.ok", "task_id": str(task.task_id), "run_id": run_id, "step_id": _STEP},
            )
    assert invoker.results == []


@pytest.mark.asyncio
async def test_acc_11_forged_tenant_on_catalog_invoke_is_identity_conflict() -> None:
    state = _api_key_state(tenant=_TENANT_A, service=_SERVICE_A)
    invoker = _RecordingCatalogInvoker()
    mcp = _catalog_server(_tool_registry(_contract("p7.ok")), invoker, mse_port=None)
    task, run_id = await _register_binding(tenant=_TENANT_B, user_id=_SERVICE_B)
    with _mcp_http_request(_auth_app(state), api_key=_API_KEY):
        with pytest.raises(ToolError, match="identity_conflict"):
            await _call(
                mcp,
                "invoke_catalog_tool",
                {
                    "tool_id": "p7.ok",
                    "task_id": str(task.task_id),
                    "run_id": run_id,
                    "step_id": _STEP,
                    "tenant_id": _TENANT_B,
                },
            )
    assert invoker.results == []


@pytest.mark.asyncio
async def test_acc_11_forged_tenant_on_run_agent_is_rejected_before_execution() -> None:
    host = _recording_host()
    mcp = _reference_server(host)
    with _mcp_http_request(_auth_app(_api_key_state(tenant=_TENANT_A)), api_key=_API_KEY):
        with pytest.raises(ToolError, match="identity_conflict"):
            await _call(mcp, "run_agent", {"message": "x", "tenant_id": _TENANT_B})
    host.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_acc_11_two_servers_keep_tenants_separate() -> None:
    host_a, host_b = _recording_host(), _recording_host()
    mcp_a, mcp_b = _reference_server(host_a), _reference_server(host_b)
    app_a = _auth_app(_api_key_state(tenant=_TENANT_A, service=_SERVICE_A, key="key-a"))
    app_b = _auth_app(_api_key_state(tenant=_TENANT_B, service=_SERVICE_B, key="key-b"))
    with _mcp_http_request(app_a, api_key="key-a"):
        await _call(mcp_a, "run_agent", {"message": "a"})
    with _mcp_http_request(app_b, api_key="key-b"):
        await _call(mcp_b, "run_agent", {"message": "b"})
    # A credential of one tenant does not authenticate against the other server.
    with _mcp_http_request(app_b, api_key="key-a"):
        with pytest.raises(ToolError, match="unauthorized"):
            await _call(mcp_b, "run_agent", {"message": "cross"})
    assert host_a.execute.await_args.args[0].tenant_id == _TENANT_A
    assert host_b.execute.await_count == 1
    assert host_b.execute.await_args.args[0].tenant_id == _TENANT_B


@pytest.mark.asyncio
async def test_acc_11_concurrent_requests_do_not_bleed_tenants() -> None:
    host = _recording_host()
    mcp = _reference_server(host)
    app_a = _auth_app(_api_key_state(tenant=_TENANT_A, service=_SERVICE_A, key="key-a"))
    app_b = _auth_app(_api_key_state(tenant=_TENANT_B, service=_SERVICE_B, key="key-b"))

    async def _as(app: FastAPI, key: str, message: str) -> None:
        with _mcp_http_request(app, api_key=key):
            await asyncio.sleep(0)  # force interleaving of the two request contexts
            await _call(mcp, "run_agent", {"message": message})

    jobs = [_as(app_a, "key-a", f"a{i}") if i % 2 == 0 else _as(app_b, "key-b", f"b{i}") for i in range(6)]
    await asyncio.gather(*jobs)
    seen = {call.args[0].message: call.args[0].tenant_id for call in host.execute.await_args_list}
    assert len(seen) == 6
    for message, tenant in seen.items():
        assert tenant == (_TENANT_A if message.startswith("a") else _TENANT_B)


# --------------------------------------------------------------------------------------
# MCP-ACC-12: principal binding
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acc_12_api_key_principal_binds_tenant_and_user_into_the_task() -> None:
    host = _recording_host()
    mcp = _reference_server(host)
    with _mcp_http_request(_auth_app(_api_key_state()), api_key=_API_KEY):
        await _call(mcp, "run_agent", {"message": "bind me"})
    task = host.execute.await_args.args[0]
    assert task.tenant_id == _TENANT_A
    assert task.user_id == _SERVICE_A
    assert task.tenant_id != REFERENCE_TENANT, "the definition default must not win over the principal"
    assert task.user_id != REFERENCE_USER


@pytest.mark.asyncio
async def test_acc_12_bearer_api_key_binds_the_same_principal() -> None:
    host = _recording_host()
    mcp = _reference_server(host)
    with _mcp_http_request(_auth_app(_api_key_state()), authorization=f"Bearer {_API_KEY}"):
        await _call(mcp, "run_agent", {"message": "bearer"})
    task = host.execute.await_args.args[0]
    assert (task.tenant_id, task.user_id) == (_TENANT_A, _SERVICE_A)


@pytest.mark.asyncio
async def test_acc_12_identity_provider_principal_binds_tenant_and_user() -> None:
    class _Provider:
        def verify_token(self, token: str) -> IdentityUser:
            if token != "good-token":
                raise PermissionError(token)
            return IdentityUser(user_id="alice", tenant_id="tenant-idp")

        def userinfo(self, token: str) -> IdentityUser:
            return self.verify_token(token)

        def list_tenants(self, *, limit: int = 50) -> list[object]:
            return []

    host = _recording_host()
    mcp = _reference_server(host)
    state = HarnessAuthState(identity_provider=_Provider())
    with _mcp_http_request(_auth_app(state), authorization="Bearer good-token"):
        await _call(mcp, "run_agent", {"message": "idp"})
    task = host.execute.await_args.args[0]
    assert (task.tenant_id, task.user_id) == ("tenant-idp", "alice")


@pytest.mark.asyncio
async def test_acc_12_execution_runs_under_the_verified_caller_principal() -> None:
    seen: list[object] = []

    async def _execute(task: Task) -> TaskResult:
        seen.append(get_strict_serving_http_caller_principal())
        return _completed_result()

    host = AsyncMock(spec=HostTaskExecutionPort)
    host.execute = AsyncMock(side_effect=_execute)
    mcp = _reference_server(host)
    with _mcp_http_request(_auth_app(_api_key_state()), api_key=_API_KEY):
        await _call(mcp, "run_agent", {"message": "principal"})
    assert len(seen) == 1
    caller = seen[0]
    assert caller.tenant_id == _TENANT_A  # type: ignore[attr-defined]
    assert caller.user_id == _SERVICE_A  # type: ignore[attr-defined]
    assert caller.auth_subject == _SERVICE_A  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_acc_12_matching_assertion_is_accepted_and_changes_nothing() -> None:
    host = _recording_host()
    mcp = _reference_server(host)
    with _mcp_http_request(_auth_app(_api_key_state()), api_key=_API_KEY):
        await _call(mcp, "run_agent", {"message": "m", "tenant_id": _TENANT_A, "user_id": _SERVICE_A})
    task = host.execute.await_args.args[0]
    assert (task.tenant_id, task.user_id) == (_TENANT_A, _SERVICE_A)


@pytest.mark.asyncio
async def test_acc_12_forged_user_assertion_is_rejected_before_execution() -> None:
    host = _recording_host()
    mcp = _reference_server(host)
    with _mcp_http_request(_auth_app(_api_key_state()), api_key=_API_KEY):
        with pytest.raises(ToolError, match="identity_conflict"):
            await _call(mcp, "run_agent", {"message": "m", "user_id": "intruder"})
    host.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_acc_12_tenant_required_without_principal_tenant_is_forbidden() -> None:
    host = _recording_host()
    mcp = _reference_server(host)
    state = _api_key_state(tenant=None, tenant_required=True)
    with _mcp_http_request(_auth_app(state), api_key=_API_KEY):
        with pytest.raises(ToolError, match="forbidden"):
            await _call(mcp, "run_agent", {"message": "m"})
    host.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_acc_12_explicit_local_dev_mode_trusts_client_assertion_for_loopback_only() -> None:
    """``LOCAL_DEV_EXPLICIT`` (code opt-in) with the empty local-dev state admits assertions from loopback."""
    host = _recording_host()
    mcp = _reference_server(host, identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT)
    with _mcp_http_request(_auth_app(HarnessAuthState())):
        await _call(mcp, "run_agent", {"message": "m", "tenant_id": "lab-tenant", "user_id": "lab-user"})
    task = host.execute.await_args.args[0]
    assert (task.tenant_id, task.user_id) == ("lab-tenant", "lab-user")


@pytest.mark.asyncio
async def test_acc_12_default_mode_refuses_local_dev_state_without_credentials() -> None:
    """R-P8-01: the fail-closed default never lets client assertions establish tenant or user identity."""
    host = _recording_host()
    mcp = _reference_server(host, identity_mode=McpIdentityMode.HARNESS_BOUND)
    with _mcp_http_request(_auth_app(HarnessAuthState())):
        with pytest.raises(ToolError, match="MCP_PRINCIPAL_REQUIRED"):
            await _call(mcp, "run_agent", {"message": "m", "tenant_id": "victim", "user_id": "victim-user"})
    host.execute.assert_not_awaited()


# --------------------------------------------------------------------------------------
# MCP-ACC-17: existing correlation keys (task_id / run_id / execution_id)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acc_17_run_agent_payload_carries_valid_task_and_run_ids() -> None:
    expected = _completed_result()
    host = AsyncMock(spec=HostTaskExecutionPort)
    host.execute = AsyncMock(return_value=expected)
    mcp = _reference_server(host)
    async with Client(mcp) as client:
        result = await client.call_tool("run_agent", {"message": "corr"})
    payload = result.data
    assert set(payload) == {"task_id", "run_id", "state", "answer", "agent_id", "metadata"}
    assert validate_task_id(payload["task_id"]) == expected.task_id
    assert validate_run_id(payload["run_id"]) == expected.run_id
    assert payload["state"] == TaskState.COMPLETED.value


@pytest.mark.asyncio
async def test_acc_17_each_invocation_runs_under_its_own_platform_execution_id() -> None:
    registry = AgentRegistry()
    host = build_harness_host_task_execution(NexusLoop(registry), orchestration_triggers=frozenset())
    mcp = build_mcp_server_x_reference_qualification_server(host_execution=host, registry=registry)
    observed: list[tuple[str, str, str]] = []

    async def _capture_execute(self: Any, request: Any) -> TaskResult:
        identity = peek_active_execution_identity()
        assert identity is not None
        run_id, attempt_id = identity
        observed.append((str(require_active_execution_id()), str(run_id), str(attempt_id)))
        return _completed_result()

    with patch.object(StrategyExecutionRouter, "execute", _capture_execute):
        async with Client(mcp) as client:
            await client.call_tool("run_agent", {"message": "first", "capability": REFERENCE_CAPABILITY})
            await client.call_tool("run_agent", {"message": "second", "capability": REFERENCE_CAPABILITY})

    assert len(observed) == 2
    for execution_id, run_id, _attempt in observed:
        validate_execution_id(execution_id)
        validate_run_id(run_id)
    assert observed[0][0] != observed[1][0], "execution_id is minted per invocation"
    assert observed[0][1] != observed[1][1], "run_id is minted per invocation"
    assert peek_active_execution_id() is None, "no execution identity may leak out of the call"


@pytest.mark.asyncio
async def test_acc_17_catalog_invoke_correlates_task_run_and_child_execution_ids() -> None:
    state = _api_key_state()
    invoker = _RecordingCatalogInvoker()
    port = _MsePort(permitted=True)
    mcp = _catalog_server(_tool_registry(_contract("p7.fx", side_effects=True)), invoker, mse_port=port)
    task, run_id = await _register_binding(tenant=_TENANT_A, user_id=_SERVICE_A)
    with _mcp_http_request(_auth_app(state), api_key=_API_KEY):
        with _parent_execution_context(task, run_id, principal_id=_principal_id_for(state)) as parent_exec:
            await _call(
                mcp,
                "invoke_catalog_tool",
                {"tool_id": "p7.fx", "task_id": str(task.task_id), "run_id": run_id, "step_id": _STEP},
            )
    assert len(invoker.results) == 1
    request = invoker.results[0]
    assert str(request.task_id) == str(task.task_id)
    assert str(request.run_id) == run_id
    assert request.tenant_id == _TENANT_A
    assert request.step_id == _STEP
    assert request.agent_id == _AGENT

    assert len(port.seen) == 1
    side_effect = port.seen[0].meaningful_side_effect_request
    assert side_effect is not None
    assert str(side_effect.task_id) == str(task.task_id)
    assert str(side_effect.run_id) == run_id
    validate_execution_id(str(side_effect.execution_id))
    assert str(side_effect.execution_id) != parent_exec, "the child execution id is distinct from the parent"


@pytest.mark.asyncio
async def test_acc_17_stale_run_id_is_rejected_instead_of_silently_correlated() -> None:
    state = _api_key_state()
    invoker = _RecordingCatalogInvoker()
    mcp = _catalog_server(_tool_registry(_contract("p7.ok")), invoker, mse_port=None)
    task, _run_id = await _register_binding(tenant=_TENANT_A, user_id=_SERVICE_A)
    with _mcp_http_request(_auth_app(state), api_key=_API_KEY):
        with pytest.raises(ToolError, match="stale_run_binding"):
            await _call(
                mcp,
                "invoke_catalog_tool",
                {
                    "tool_id": "p7.ok",
                    "task_id": str(task.task_id),
                    "run_id": str(mint_run_id()),
                    "step_id": _STEP,
                },
            )
    assert invoker.results == []
