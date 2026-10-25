# © Artur Czarnecki. All rights reserved.

"""F3 physical L3 — real MCP HTTP client, cross-loop ingress, parent cancel."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
from fastmcp import Client
from pydantic import BaseModel

from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_harness_control_plane_mutation_boundary,
)
from intergrax.applications._shared.mcp_catalog_invocation_bridge import (
    _METADATA_STEP_ID,
    _serialize_tool_result,
)
from intergrax.applications._shared.mcp_harness_principal import McpRunAgentIdentityResolution
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.applications._shared.production_catalog_tool_child_execution_wiring import (
    build_production_catalog_tool_child_execution_port,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
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
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.tool_execution_outcome import ToolExecutionResult
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.execution.host_root_catalog_child_ingress_draining_delegate import (
    HostRootCatalogChildIngressDrainingDelegate,
)
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.execution.root_execution_child_work_ingress import (
    RootExecutionChildWorkIngress,
)
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.network]

_TENANT = "tenant-f3-l3"
_USER = "operator-f3"
_STEP = "f3.catalog.step"


class _In(BaseModel):
    query: str = "x"


class _Out(BaseModel):
    value: str = "invoked"


@dataclass
class _SlowCatalogInvoker:
    caller_agent_id: str = "agent-f3"
    results: list[ExecutionBoundCatalogToolInvokeRequest] = field(default_factory=list)
    delay_s: float = 0.3

    def invoke(self, request: ExecutionBoundCatalogToolInvokeRequest) -> ToolExecutionResult[_Out]:
        self.results.append(request)
        if self.delay_s > 0:
            import time

            time.sleep(self.delay_s)
        return ToolExecutionResult.ok(_Out(value="child-done"))


def _contract(tool_id: str) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description="f3 l3",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        category="f3",
    )


def _handler() -> ToolHandler[_In, _Out]:
    async def _run(_input: _In, _ctx: object) -> _Out:
        return _Out(value="unused")

    return _run


def _principal() -> RequestIdentity:
    return RequestIdentity(
        tenant_id=_TENANT,
        user_id=_USER,
        principal_type=PrincipalType.USER,
        auth_subject=_USER,
    )


class _SlowDelegate:
    async def execute(self, _request: object) -> str:
        await asyncio.sleep(2.0)
        return "root-done"


class _IngressQualHostExecution(HostTaskExecutionPort):
    """Runs supervised root drain loop while MCP HTTP catalog uses ingress."""

    def __init__(self, ingress: RootExecutionChildWorkIngress) -> None:
        self._ingress = ingress
        self._task: Task | None = None

    async def execute(
        self,
        task: Task,
        *,
        run_id: object = None,
        attempt_id: object = None,
        execution_id: object = None,
        resume_checkpoint: object = None,
        restore_existing_execution: bool = False,
    ) -> TaskResult:
        self._task = task
        resolved_run = run_id if run_id is not None else mint_run_id()
        await ActiveTaskRegistry.register(task, resolved_run)
        self._ingress.bind_supervised_task(task.task_id)
        authority_token = bind_active_execution_authority(
            ParentExecutionAuthority.unrestricted_root(),
        )
        identity_token = bind_active_execution_identity(
            run_id=resolved_run,
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
            task_id=task.task_id,
        )
        governance_token = bind_active_execution_governance_identity(
            ActiveExecutionGovernanceIdentity(
                tenant_id=_TENANT,
                workspace_id="ws-f3",
                principal_id=_USER,
            ),
        )
        task.metadata["workspace_id"] = "ws-f3"
        draining = HostRootCatalogChildIngressDrainingDelegate(
            _SlowDelegate(),
            ingress=self._ingress,
        )
        try:
            await draining.execute(object())
            state = TaskState.COMPLETED
        except asyncio.CancelledError:
            state = TaskState.CANCELLED
            raise
        finally:
            self._ingress.unbind_supervised_task(task.task_id)
            reset_active_execution_governance_identity(governance_token)
            reset_active_execution_identity(identity_token)
            reset_active_execution_authority(authority_token)
            await ActiveTaskRegistry.unregister(task.task_id, resolved_run)
        return TaskResult(
            task_id=task.task_id,
            run_id=resolved_run,
            state=state,
            answer="done",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )


@pytest.fixture(autouse=True)
async def _clear_tasks() -> None:
    ActiveTaskRegistry.clear_for_tests()
    yield
    ActiveTaskRegistry.clear_for_tests()


@pytest.mark.asyncio
async def test_f3_http_cross_loop_child_admitted_under_active_root() -> None:
    registry = ToolRegistry()
    registry.register(_contract("f3.ok"), _handler())
    invoker = _SlowCatalogInvoker(delay_s=0.15)
    child_port = build_production_catalog_tool_child_execution_port()
    ingress = RootExecutionChildWorkIngress(
        catalog_child_execution=child_port,
        serialize_tool_result=_serialize_tool_result,
    )
    host = _IngressQualHostExecution(ingress)
    env = ApplicationEnvironmentProfile(application_profile=ApplicationProfile.PRODUCT)
    boundary = build_harness_control_plane_mutation_boundary(env)
    mcp = build_nexus_mcp_server(
        name="f3-cross-loop-l3",
        host_execution=host,
        registry=AgentRegistry(),
        default_capability="f3.cap",
        default_tenant_id=_TENANT,
        default_user_id=_USER,
        tool_registry=registry,
        catalog_tool_invoker=invoker,
        task_control_mutation_boundary=boundary,
        catalog_root_child_work_ingress=ingress,
        enable_cross_loop_catalog_child=True,
    )
    task_id = mint_task_id()
    run_id = mint_run_id()
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id=_USER,
        message="f3-root",
        context=TaskContext(capability="f3.cap"),
        metadata={_METADATA_STEP_ID: _STEP},
        agent_id="agent-f3",
    )
    await ActiveTaskRegistry.register(task, run_id)
    resolution = McpRunAgentIdentityResolution(
        tenant_id=_TENANT,
        user_id=_USER,
        principal=None,
        request_identity=_principal(),
    )
    root_task = asyncio.create_task(
        host.execute(task, run_id=run_id),
    )
    await asyncio.sleep(0.05)
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=resolution,
    ):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "invoke_catalog_tool",
                {
                    "tool_id": "f3.ok",
                    "task_id": str(task_id),
                    "run_id": str(run_id),
                    "step_id": _STEP,
                    "input": {"query": "hi"},
                },
            )
    await asyncio.wait_for(root_task, timeout=5.0)
    assert result.data.get("success") is True
    assert invoker.results
    assert str(invoker.results[0].task_id) == str(task_id)


@pytest.mark.asyncio
async def test_f3_http_cross_loop_parent_cancel_during_child() -> None:
    registry = ToolRegistry()
    registry.register(_contract("f3.slow"), _handler())
    invoker = _SlowCatalogInvoker(delay_s=0.5)
    child_port = build_production_catalog_tool_child_execution_port()
    ingress = RootExecutionChildWorkIngress(
        catalog_child_execution=child_port,
        serialize_tool_result=_serialize_tool_result,
    )
    host = _IngressQualHostExecution(ingress)
    env = ApplicationEnvironmentProfile(application_profile=ApplicationProfile.PRODUCT)
    boundary = build_harness_control_plane_mutation_boundary(env)
    mcp = build_nexus_mcp_server(
        name="f3-cancel-c-l3",
        host_execution=host,
        registry=AgentRegistry(),
        default_capability="f3.cap",
        default_tenant_id=_TENANT,
        default_user_id=_USER,
        tool_registry=registry,
        catalog_tool_invoker=invoker,
        task_control_mutation_boundary=boundary,
        catalog_root_child_work_ingress=ingress,
        enable_cross_loop_catalog_child=True,
    )
    task_id = mint_task_id()
    run_id = mint_run_id()
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id=_USER,
        message="f3-cancel",
        context=TaskContext(capability="f3.cap"),
        metadata={_METADATA_STEP_ID: _STEP},
        agent_id="agent-f3",
    )
    await ActiveTaskRegistry.register(task, run_id)
    resolution = McpRunAgentIdentityResolution(
        tenant_id=_TENANT,
        user_id=_USER,
        principal=None,
        request_identity=_principal(),
    )

    async def _cancel_soon() -> None:
        await asyncio.sleep(0.08)
        CancellationCoordinator.request(task, reason="f3_parent_cancel_l3")

    root_task = asyncio.create_task(host.execute(task, run_id=run_id))
    await asyncio.sleep(0.03)
    with patch(
        "intergrax.applications._shared.mcp_catalog_invocation_bridge.resolve_mcp_run_agent_identity",
        return_value=resolution,
    ):
        async with Client(mcp) as client:
            cancel_task = asyncio.create_task(_cancel_soon())
            result = await client.call_tool(
                "invoke_catalog_tool",
                {
                    "tool_id": "f3.slow",
                    "task_id": str(task_id),
                    "run_id": str(run_id),
                },
            )
            await cancel_task
    root_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await root_task
    assert result.data.get("success") is not True
    assert result.data.get("error") in {
        "execution_cancelled",
        "execution_admission_denied",
    }
