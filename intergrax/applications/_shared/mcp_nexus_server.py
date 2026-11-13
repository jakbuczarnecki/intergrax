# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""FastMCP tools backed by canonical :class:`~intergrax.runtime.execution.host_task.HostTaskExecution`."""

from __future__ import annotations

import asyncio
import re

from fastmcp import FastMCP

from intergrax.agents.tool_enablement import ToolEnablementProfile
from intergrax.applications._shared.mcp_harness_principal import resolve_mcp_run_agent_identity
from intergrax.applications._shared.mcp_host_execution_transport_cancel import (
    McpHostExecutionDeadlineExceeded,
    McpHostExecutionTransportCancelContext,
    McpProtectedHostExecutionAdmissionError,
    admit_protected_mcp_root_execution_supervision,
    create_mcp_protected_host_execution_launch,
    resolve_mcp_task_control_mutation_boundary_from_request,
    run_host_execution_with_mcp_transport_cancel,
)
from intergrax.applications._shared.mcp_identity_mode import McpIdentityMode, coerce_mcp_identity_mode
from intergrax.applications._shared.mcp_server_engine import (
    McpServerCompositionDeps,
    compose_mcp_server_from_definition,
    nexus_legacy_server_definition,
)
from intergrax.applications._shared.mcp_transport_detachment_supervision_coupling import (
    resolve_transport_detachment_supervision_from_request,
)
from intergrax.applications._shared.mcp_wire_errors import bound_mcp_result_or_raise, mcp_tool_error
from intergrax.applications._shared.strict_production_serving_http_execution_principal import (
    bind_strict_serving_http_caller_principal,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.contracts.mcp_server_definition import McpServerDefinitionErrorCode
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
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

MCP_IDENTITY_ASSERTION_METADATA_KEY = "mcp_identity_assertion"


def task_from_mcp_agent_intake(
    *,
    message: str,
    capability: str,
    tenant_id: str,
    user_id: str,
    session_id: str | None = None,
    intent: str | None = None,
    identity_verified: bool = True,
) -> Task:
    """Map MCP agent intake fields to canonical platform Task.

    ``identity_verified=False`` (``LOCAL_DEV_EXPLICIT`` assertion-only identity) labels the task so audit and
    governance can tell asserted tenant/user from a verified principal.
    """
    context = TaskContext(capability=capability, intent=intent) if intent else TaskContext(capability=capability)
    metadata = {} if identity_verified else {MCP_IDENTITY_ASSERTION_METADATA_KEY: "unverified"}
    return Task(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        message=message,
        context=context,
        metadata=metadata,
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
    transport_detachment_supervision: (RootExecutionTransportDetachmentSupervisionPort | None) = None,
    max_call_seconds: float | None = None,
) -> dict[str, object]:
    """Run one MCP agent task through canonical host task execution.

    ``max_call_seconds`` (optional) is the MCP transport call bound (OD-MCP-RC-02): on expiry the existing
    governed cancel is requested and the call ends with ``MCP_DEADLINE_EXCEEDED``; it never reports success.
    """
    task = task_from_mcp_agent_intake(
        message=message,
        capability=capability,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        intent=intent,
        identity_verified=caller_request_identity is not None,
    )
    boundary = task_control_mutation_boundary
    if boundary is None:
        boundary = resolve_mcp_task_control_mutation_boundary_from_request()
    cancel_ctx: McpHostExecutionTransportCancelContext | None = None
    if boundary is not None and caller_request_identity is not None:
        cancel_ctx = McpHostExecutionTransportCancelContext(
            mutation_boundary=boundary,
            caller_request_identity=caller_request_identity,
            max_call_seconds=max_call_seconds,
        )

    async def _run_execute() -> object:
        if caller_request_identity is not None:
            with bind_strict_serving_http_caller_principal(caller_request_identity):
                return await host_execution.execute(task)
        return await host_execution.execute(task)

    if cancel_ctx is None:
        if max_call_seconds is None:
            execution_result = await _run_execute()
        else:
            # No protected cancel context (local development identity): the bound cancels the awaited call.
            try:
                execution_result = await asyncio.wait_for(_run_execute(), timeout=max_call_seconds)
            except TimeoutError as exc:
                raise mcp_tool_error(
                    McpServerDefinitionErrorCode.MCP_DEADLINE_EXCEEDED,
                    f"call exceeded {max_call_seconds}s; outcome=cancelled",
                ) from exc
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
        try:
            execution_result = await run_host_execution_with_mcp_transport_cancel(
                launch.host_task,
                task=task,
                cancel_ctx=cancel_ctx,
                transport_detachment_supervision=supervision,
                execution_start_gate=launch.execution_start_gate,
            )
        except McpHostExecutionDeadlineExceeded as exc:
            raise mcp_tool_error(
                McpServerDefinitionErrorCode.MCP_DEADLINE_EXCEEDED,
                f"call exceeded {exc.max_call_seconds}s; outcome={exc.outcome}",
            ) from exc
    result = (
        execution_result if isinstance(execution_result, TaskResult) else TaskResult.model_validate(execution_result)
    )
    payload = {
        "task_id": result.task_id,
        "run_id": result.run_id,
        "state": result.state.value,
        "answer": result.answer,
        "agent_id": result.agent_id,
        "metadata": dict(result.metadata),
    }
    return bound_mcp_result_or_raise(payload)


async def execute_mcp_agent_task_with_resolved_identity(
    host_execution: HostTaskExecutionPort,
    *,
    message: str,
    capability: str,
    asserted_tenant_id: str | None,
    asserted_user_id: str | None,
    default_tenant_id: str,
    default_user_id: str,
    identity_mode: McpIdentityMode | str | None = None,
    session_id: str | None = None,
    intent: str | None = None,
    task_control_mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
    transport_detachment_supervision: (RootExecutionTransportDetachmentSupervisionPort | None) = None,
    max_call_seconds: float | None = None,
) -> dict[str, object]:
    """Resolve the MCP caller identity (fail-closed by default) and run one agent task.

    Single entry used by ``run_agent`` and by custom host tools that start agent tasks (for example the
    research pipeline), so no tool can bypass principal binding with client-asserted tenant/user values.
    """
    resolved = resolve_mcp_run_agent_identity(
        harness_auth=None,
        asserted_tenant_id=asserted_tenant_id,
        asserted_user_id=asserted_user_id,
        default_tenant_id=default_tenant_id,
        default_user_id=default_user_id,
        identity_mode=coerce_mcp_identity_mode(identity_mode),
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
        transport_detachment_supervision=transport_detachment_supervision,
        max_call_seconds=max_call_seconds,
    )


def legacy_server_id_from_display_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
    return slug or "mcp-server"


def _legacy_server_id_from_name(name: str) -> str:
    return legacy_server_id_from_display_name(name)


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
    transport_detachment_supervision: (RootExecutionTransportDetachmentSupervisionPort | None) = None,
    auto_wire_catalog_transport_supervision: bool = True,
    catalog_root_child_work_ingress: object | None = None,
    enable_cross_loop_catalog_child: bool = False,
    identity_mode: McpIdentityMode | str | None = None,
    max_call_seconds: float | None = None,
) -> FastMCP:
    """
    Register ``list_agents`` and ``run_agent`` on a FastMCP instance.

    Compatibility facade (OD-MCP-05-A) over :func:`compose_mcp_server_from_definition`.
    """
    catalog_tool_ids: list[str] = []
    if tool_registry is not None:
        catalog_tool_ids = [item.contract.tool_id for item in tool_registry.list()]

    definition = nexus_legacy_server_definition(
        server_id=_legacy_server_id_from_name(name),
        display_name=name,
        default_capability=default_capability,
        default_tenant_id=default_tenant_id,
        default_user_id=default_user_id,
        catalog_tool_ids=catalog_tool_ids,
        include_catalog_surface=tool_registry is not None,
    )
    deps = McpServerCompositionDeps(
        host_execution=host_execution,
        registry=registry,
        tool_registry=tool_registry,
        catalog_tool_invoker=catalog_tool_invoker,
        tool_enablement=tool_enablement,
        task_control_mutation_boundary=task_control_mutation_boundary,
        meaningful_side_effect_authorization=meaningful_side_effect_authorization,
        transport_detachment_supervision=transport_detachment_supervision,
        auto_wire_catalog_transport_supervision=auto_wire_catalog_transport_supervision,
        catalog_root_child_work_ingress=catalog_root_child_work_ingress,
        enable_cross_loop_catalog_child=enable_cross_loop_catalog_child,
        identity_mode=coerce_mcp_identity_mode(identity_mode),
        max_call_seconds=max_call_seconds,
    )
    return compose_mcp_server_from_definition(definition, deps)
