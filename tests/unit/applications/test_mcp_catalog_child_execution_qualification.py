# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X R3.1 F1 — in-process catalog child execution via ChildExecutionRunner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

import contextlib

from intergrax.applications._shared.mcp_harness_principal import McpRunAgentIdentityResolution
from intergrax.applications._shared.mcp_catalog_invocation_bridge import (
    McpCatalogInvocationComposition,
    _METADATA_STEP_ID,
    execute_mcp_catalog_tool_invocation,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    reset_active_execution_identity,
)
from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.tool_execution_outcome import ToolEffectCertainty, ToolExecutionResult
from fastmcp.exceptions import ToolError
from unittest.mock import MagicMock
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.applications._shared.production_catalog_tool_child_execution_wiring import (
    build_production_catalog_tool_child_execution_port,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-child"
_USER = "operator-child"
_STEP = "child.catalog.step"


class _In(BaseModel):
    query: str = "x"


class _Out(BaseModel):
    value: str = "ok"


@dataclass
class _RecordingCatalogInvoker:
    caller_agent_id: str = "agent-child"
    results: list[ExecutionBoundCatalogToolInvokeRequest] = field(default_factory=list)

    def invoke(self, request: ExecutionBoundCatalogToolInvokeRequest) -> ToolExecutionResult[_Out]:
        self.results.append(request)
        return ToolExecutionResult.ok(_Out(value="child-invoked"))


def _contract(tool_id: str, *, side_effects: bool = False) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description="child qual",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=side_effects,
        risk_level=ToolRiskLevel.LOW,
        category="child",
    )


def _handler() -> ToolHandler[_In, _Out]:
    async def _run(_input: _In, _ctx: object) -> _Out:
        return _Out(value="should-not-run")

    return _run


def _principal() -> RequestIdentity:
    return RequestIdentity(
        tenant_id=_TENANT,
        user_id=_USER,
        principal_type=PrincipalType.USER,
        auth_subject=_USER,
    )


async def _register_binding() -> tuple[Task, str]:
    task_id = mint_task_id()
    run_id = mint_run_id()
    metadata: dict[str, Any] = {_METADATA_STEP_ID: _STEP}
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id=_USER,
        message="child",
        context=TaskContext(capability="child.cap"),
        metadata=metadata,
        agent_id="agent-child",
    )
    await ActiveTaskRegistry.register(task, run_id)
    return task, str(run_id)


def _host_composition(
    registry: ToolRegistry,
    invoker: _RecordingCatalogInvoker,
    **kwargs: object,
) -> McpCatalogInvocationComposition:
    return McpCatalogInvocationComposition(
        catalog_invoker=invoker,
        registry=registry,
        catalog_tool_child_execution=build_production_catalog_tool_child_execution_port(),
        **kwargs,
    )


@contextlib.contextmanager
def _parent_execution_context(
    task: Task,
    run_id: str,
    *,
    principal_id: str = _USER,
    include_task_id: bool = True,
):
    parent_execution_id = mint_execution_id()
    authority_token = bind_active_execution_authority(
        ParentExecutionAuthority.unrestricted_root(),
    )
    identity_kwargs: dict[str, object] = {
        "run_id": run_id,
        "attempt_id": mint_attempt_id(),
        "execution_id": parent_execution_id,
    }
    if include_task_id:
        identity_kwargs["task_id"] = task.task_id
    identity_token = bind_active_execution_identity(**identity_kwargs)
    governance_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id="ws-child",
            principal_id=principal_id,
        ),
    )
    task.metadata["workspace_id"] = "ws-child"
    try:
        yield parent_execution_id
    finally:
        reset_active_execution_governance_identity(governance_token)
        reset_active_execution_identity(identity_token)
        reset_active_execution_authority(authority_token)


@pytest.fixture(autouse=True)
async def _clear_tasks() -> None:
    ActiveTaskRegistry.clear_for_tests()
    yield
    ActiveTaskRegistry.clear_for_tests()


@pytest.mark.asyncio
async def test_catalog_child_invoke_mints_child_execution_and_invokes() -> None:
    registry = ToolRegistry()
    registry.register(_contract("child.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    composition = _host_composition(registry, invoker)
    task, run_id = await _register_binding()
    resolution = McpRunAgentIdentityResolution(
        tenant_id=_TENANT,
        user_id=_USER,
        principal=None,
        request_identity=_principal(),
    )
    with _parent_execution_context(task, run_id):
        with patch(
            "intergrax.applications._shared.mcp_catalog_invocation_bridge."
            "resolve_mcp_run_agent_identity",
            return_value=resolution,
        ):
            outcome = await execute_mcp_catalog_tool_invocation(
                composition,
                tool_id="child.ok",
                task_id=str(task.task_id),
                run_id=run_id,
                step_id=_STEP,
                asserted_tenant_id=_TENANT,
            )
    assert outcome.get("success") is True
    assert len(invoker.results) == 1
    assert invoker.results[0].run_id == run_id


@pytest.mark.asyncio
async def test_catalog_child_http_path_denied_without_parent_context() -> None:
    registry = ToolRegistry()
    registry.register(_contract("child.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    composition = McpCatalogInvocationComposition(
        catalog_invoker=invoker,
        registry=registry,
    )
    task, run_id = await _register_binding()
    resolution = McpRunAgentIdentityResolution(
        tenant_id=_TENANT,
        user_id=_USER,
        principal=None,
        request_identity=_principal(),
    )
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge."
        "resolve_mcp_run_agent_identity",
        return_value=resolution,
    ):
        outcome = await execute_mcp_catalog_tool_invocation(
            composition,
            tool_id="child.ok",
            task_id=str(task.task_id),
            run_id=run_id,
            step_id=_STEP,
            asserted_tenant_id=_TENANT,
        )
    assert outcome.get("error") == "execution_admission_denied"
    assert invoker.results == []


@pytest.mark.asyncio
async def test_catalog_child_mse_deny_prevents_invoker() -> None:
    class _DenyPort:
        def authorize(
            self,
            request: CollaborativeWorkEnforcementRequest,
            *,
            source_agent_id: str = "platform.orchestration.tool_invocation",
            source_step_id: str | None = None,
        ) -> MeaningfulSideEffectAuthorizationResult:
            _ = request, source_agent_id, source_step_id
            return MeaningfulSideEffectAuthorizationResult(
                permitted=False,
                decision=PolicyDecision(action=PolicyAction.DENY, reason="deny"),
                enforcement_result=MagicMock(),
                requires_governed_continuation=False,
            )

    registry = ToolRegistry()
    registry.register(_contract("child.fx", side_effects=True), _handler())
    invoker = _RecordingCatalogInvoker()
    composition = _host_composition(
        registry,
        invoker,
        meaningful_side_effect_authorization=_DenyPort(),
    )
    task, run_id = await _register_binding()
    resolution = McpRunAgentIdentityResolution(
        tenant_id=_TENANT,
        user_id=_USER,
        principal=None,
        request_identity=_principal(),
    )
    with _parent_execution_context(task, run_id):
        with patch(
            "intergrax.applications._shared.mcp_catalog_invocation_bridge."
            "resolve_mcp_run_agent_identity",
            return_value=resolution,
        ):
            with pytest.raises(ToolError, match="side_effect_authorization_denied"):
                await execute_mcp_catalog_tool_invocation(
                    composition,
                    tool_id="child.fx",
                    task_id=str(task.task_id),
                    run_id=run_id,
                    step_id=_STEP,
                    asserted_tenant_id=_TENANT,
                )
    assert invoker.results == []


@pytest.mark.asyncio
async def test_catalog_child_denied_without_host_child_port() -> None:
    registry = ToolRegistry()
    registry.register(_contract("child.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    composition = McpCatalogInvocationComposition(
        catalog_invoker=invoker,
        registry=registry,
        catalog_tool_child_execution=None,
    )
    task, run_id = await _register_binding()
    resolution = McpRunAgentIdentityResolution(
        tenant_id=_TENANT,
        user_id=_USER,
        principal=None,
        request_identity=_principal(),
    )
    with _parent_execution_context(task, run_id):
        with patch(
            "intergrax.applications._shared.mcp_catalog_invocation_bridge."
            "resolve_mcp_run_agent_identity",
            return_value=resolution,
        ):
            outcome = await execute_mcp_catalog_tool_invocation(
                composition,
                tool_id="child.ok",
                task_id=str(task.task_id),
                run_id=run_id,
                step_id=_STEP,
                asserted_tenant_id=_TENANT,
            )
    assert "not configured at host composition root" in str(outcome.get("detail", ""))
    assert invoker.results == []


@pytest.mark.asyncio
async def test_catalog_child_denied_when_active_task_id_missing() -> None:
    registry = ToolRegistry()
    registry.register(_contract("child.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    composition = _host_composition(registry, invoker)
    task, run_id = await _register_binding()
    resolution = McpRunAgentIdentityResolution(
        tenant_id=_TENANT,
        user_id=_USER,
        principal=None,
        request_identity=_principal(),
    )
    with _parent_execution_context(task, run_id, include_task_id=False):
        with patch(
            "intergrax.applications._shared.mcp_catalog_invocation_bridge."
            "resolve_mcp_run_agent_identity",
            return_value=resolution,
        ):
            outcome = await execute_mcp_catalog_tool_invocation(
                composition,
                tool_id="child.ok",
                task_id=str(task.task_id),
                run_id=run_id,
                step_id=_STEP,
                asserted_tenant_id=_TENANT,
            )
    assert outcome.get("error") == "execution_admission_denied"
    assert invoker.results == []


@pytest.mark.asyncio
async def test_catalog_child_denied_on_governance_principal_mismatch() -> None:
    registry = ToolRegistry()
    registry.register(_contract("child.ok"), _handler())
    invoker = _RecordingCatalogInvoker()
    composition = _host_composition(registry, invoker)
    task, run_id = await _register_binding()
    resolution = McpRunAgentIdentityResolution(
        tenant_id=_TENANT,
        user_id=_USER,
        principal=None,
        request_identity=_principal(),
    )
    with _parent_execution_context(task, run_id, principal_id="other-principal"):
        with patch(
            "intergrax.applications._shared.mcp_catalog_invocation_bridge."
            "resolve_mcp_run_agent_identity",
            return_value=resolution,
        ):
            outcome = await execute_mcp_catalog_tool_invocation(
                composition,
                tool_id="child.ok",
                task_id=str(task.task_id),
                run_id=run_id,
                step_id=_STEP,
                asserted_tenant_id=_TENANT,
            )
    assert outcome.get("error") == "execution_admission_denied"
    assert invoker.results == []


@pytest.mark.asyncio
async def test_catalog_child_mse_uses_child_execution_id() -> None:
    @dataclass
    class _RecordingMsePort:
        seen_execution_ids: list[str] = field(default_factory=list)

        def authorize(
            self,
            request: CollaborativeWorkEnforcementRequest,
            *,
            source_agent_id: str = "platform.orchestration.tool_invocation",
            source_step_id: str | None = None,
        ) -> MeaningfulSideEffectAuthorizationResult:
            _ = source_agent_id, source_step_id
            mse = request.meaningful_side_effect_request
            assert mse is not None
            self.seen_execution_ids.append(str(mse.execution_id))
            return MeaningfulSideEffectAuthorizationResult(
                permitted=True,
                decision=PolicyDecision(action=PolicyAction.ALLOW, reason="ok"),
                enforcement_result=MagicMock(),
                requires_governed_continuation=False,
            )

    registry = ToolRegistry()
    registry.register(_contract("child.fx", side_effects=True), _handler())
    invoker = _RecordingCatalogInvoker()
    mse_port = _RecordingMsePort()
    composition = _host_composition(
        registry,
        invoker,
        meaningful_side_effect_authorization=mse_port,
    )
    task, run_id = await _register_binding()
    resolution = McpRunAgentIdentityResolution(
        tenant_id=_TENANT,
        user_id=_USER,
        principal=None,
        request_identity=_principal(),
    )
    with _parent_execution_context(task, run_id) as parent_execution_id:
        with patch(
            "intergrax.applications._shared.mcp_catalog_invocation_bridge."
            "resolve_mcp_run_agent_identity",
            return_value=resolution,
        ):
            outcome = await execute_mcp_catalog_tool_invocation(
                composition,
                tool_id="child.fx",
                task_id=str(task.task_id),
                run_id=run_id,
                step_id=_STEP,
                asserted_tenant_id=_TENANT,
            )
    assert outcome.get("success") is True
    assert len(mse_port.seen_execution_ids) == 1
    assert mse_port.seen_execution_ids[0] != str(parent_execution_id)
