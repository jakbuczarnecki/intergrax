# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""FastMCP tools backed by canonical :class:`~intergrax.runtime.execution.host_task.HostTaskExecution`."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from intergrax.agents.tool_enablement import ToolEnablementProfile
from intergrax.applications._shared.mcp_catalog_invocation_bridge import (
    McpCatalogInvocationComposition,
)
from intergrax.applications._shared.mcp_catalog_tools import mount_catalog_tools_on_mcp
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.applications._shared.mcp_harness_principal import resolve_mcp_run_agent_identity
from intergrax.applications._shared.mcp_host_execution_transport_cancel import (
    McpHostExecutionTransportCancelContext,
    McpProtectedHostExecutionAdmissionError,
    admit_protected_mcp_root_execution_supervision,
    create_mcp_protected_host_execution_launch,
    resolve_mcp_task_control_mutation_boundary_from_request,
    run_host_execution_with_mcp_transport_cancel,
)
from intergrax.applications._shared.strict_production_serving_http_execution_principal import (
    bind_strict_serving_http_caller_principal,
)
from intergrax.tools.mcp_result_bounds import bound_mcp_structured_result, emit_bounded_mcp_list
from intergrax.applications._shared.mcp_transport_detachment_supervision_coupling import (
    resolve_transport_detachment_supervision_from_request,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.root_execution_transport_detachment_supervision import (
    RootExecutionTransportDetachmentSupervisionPort,
)
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.task.task import Task, TaskContext, TaskResult
from intergrax.tools.registry.runtime import ToolRegistry


def task_from_mcp_agent_intake(
    *,
    message: str,
    capability: str,
    tenant_id: str,
    user_id: str,
    session_id: str | None = None,
    intent: str | None = None,
) -> Task:
    """Map MCP agent intake fields to canonical platform Task."""
    context = (
        TaskContext(capability=capability, intent=intent)
        if intent
        else TaskContext(capability=capability)
    )
    return Task(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        message=message,
        context=context,
    )


async def execute_mcp_agent_task(
    host_execution: HostTaskExecutionPort,
    *,
    message: str,
    capability: str,
    tenant_id: str,
    user_id: str,
    session_id: str | None = None,
    intent: str | None = None,
    caller_request_identity: RequestIdentity | None = None,
    task_control_mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
    transport_detachment_supervision: (
        RootExecutionTransportDetachmentSupervisionPort | None
    ) = None,
) -> dict[str, object]:
    """Run one MCP agent task through canonical host task execution."""
    task = task_from_mcp_agent_intake(
        message=message,
        capability=capability,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        intent=intent,
    )
    boundary = task_control_mutation_boundary
    if boundary is None:
        boundary = resolve_mcp_task_control_mutation_boundary_from_request()
    cancel_ctx: McpHostExecutionTransportCancelContext | None = None
    if boundary is not None and caller_request_identity is not None:
        cancel_ctx = McpHostExecutionTransportCancelContext(
            mutation_boundary=boundary,
            caller_request_identity=caller_request_identity,
        )

    async def _run_execute() -> object:
        if caller_request_identity is not None:
            with bind_strict_serving_http_caller_principal(caller_request_identity):
                return await host_execution.execute(task)
        return await host_execution.execute(task)

    if cancel_ctx is None:
        execution_result = await _run_execute()
    else:
        supervision = transport_detachment_supervision
        if supervision is None:
            supervision = resolve_transport_detachment_supervision_from_request()
        if supervision is None:
            raise McpProtectedHostExecutionAdmissionError(
                "protected MCP run_agent requires root_execution_transport_detachment_supervision"
            )
        if caller_request_identity is None:
            raise McpProtectedHostExecutionAdmissionError(
                "protected MCP run_agent requires caller_request_identity for execution supervision"
            )
        launch = create_mcp_protected_host_execution_launch(_run_execute)
        await admit_protected_mcp_root_execution_supervision(
            supervision_port=supervision,
            task=task,
            caller_request_identity=caller_request_identity,
            launch=launch,
            transport_waiter_will_continue=False,
        )
        execution_result = await run_host_execution_with_mcp_transport_cancel(
            launch.host_task,
            task=task,
            cancel_ctx=cancel_ctx,
            transport_detachment_supervision=supervision,
            execution_start_gate=launch.execution_start_gate,
        )
    result = execution_result if isinstance(execution_result, TaskResult) else TaskResult.model_validate(execution_result)
    payload = {
        "task_id": result.task_id,
        "run_id": result.run_id,
        "state": result.state.value,
        "answer": result.answer,
        "agent_id": result.agent_id,
        "metadata": dict(result.metadata),
    }
    bounded, _ = bound_mcp_structured_result(payload)
    return bounded if isinstance(bounded, dict) else {"error": "mcp_payload_byte_limit_exceeded"}


def build_nexus_mcp_server(
    *,
    name: str,
    host_execution: HostTaskExecutionPort,
    registry: AgentRegistryRead,
    default_capability: str,
    default_tenant_id: str = "mcp",
    default_user_id: str = "mcp-user",
    tool_registry: ToolRegistry | None = None,
    catalog_tool_invoker: ExecutionBoundCatalogToolInvoker | None = None,
    tool_enablement: ToolEnablementProfile | None = None,
    task_control_mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None = None,
    transport_detachment_supervision: (
        RootExecutionTransportDetachmentSupervisionPort | None
    ) = None,
    auto_wire_catalog_transport_supervision: bool = True,
    catalog_root_child_work_ingress: object | None = None,
    enable_cross_loop_catalog_child: bool = False,
) -> FastMCP:
    """
    Register ``list_agents`` and ``run_agent`` on a FastMCP instance.

    Tools use the same registry and canonical host task execution as the Tier-3 FastAPI host.
    """
    mcp = FastMCP(name)

    @mcp.tool
    async def list_agents() -> list[dict[str, Any]]:
        """List registered agents and capabilities (same roster as the HTTP agents endpoint)."""
        agents: list[dict[str, Any]] = []
        for agent_id in registry.list_agent_ids():
            contract = registry.get(agent_id).get_contract()
            agents.append(
                {
                    "agent_id": contract.id,
                    "name": contract.name,
                    "capabilities": list(contract.capabilities),
                }
            )
        return emit_bounded_mcp_list(agents)

    @mcp.tool
    async def run_agent(
        message: str,
        capability: str = default_capability,
        tenant_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        intent: str | None = None,
    ) -> dict[str, object]:
        """Run one agent task via canonical host task execution (same path as HTTP POST /run)."""
        resolved = resolve_mcp_run_agent_identity(
            harness_auth=None,
            asserted_tenant_id=tenant_id,
            asserted_user_id=user_id,
            default_tenant_id=default_tenant_id,
            default_user_id=default_user_id,
        )
        return await execute_mcp_agent_task(
            host_execution,
            message=message,
            capability=capability,
            tenant_id=resolved.tenant_id,
            user_id=resolved.user_id,
            session_id=session_id,
            intent=intent,
            caller_request_identity=resolved.request_identity,
            task_control_mutation_boundary=task_control_mutation_boundary,
        )

    if tool_registry is not None:
        catalog_invocation: McpCatalogInvocationComposition | None = None
        if catalog_tool_invoker is not None:
            _ = auto_wire_catalog_transport_supervision, transport_detachment_supervision
            from intergrax.applications._shared.production_catalog_tool_child_execution_wiring import (
                build_production_catalog_tool_child_execution_port,
            )

            catalog_child_execution = build_production_catalog_tool_child_execution_port()
            catalog_invocation = McpCatalogInvocationComposition(
                catalog_invoker=catalog_tool_invoker,
                registry=tool_registry,
                tool_enablement=tool_enablement,
                task_control_mutation_boundary=task_control_mutation_boundary,
                meaningful_side_effect_authorization=meaningful_side_effect_authorization,
                catalog_tool_child_execution=catalog_child_execution,
                root_child_work_ingress=catalog_root_child_work_ingress,
                enable_cross_loop_catalog_child=enable_cross_loop_catalog_child,
                default_tenant_id=default_tenant_id,
                default_user_id=default_user_id,
            )
        mount_catalog_tools_on_mcp(
            mcp,
            tool_registry,
            catalog_invocation=catalog_invocation,
        )

    return mcp
