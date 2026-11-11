# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 MCP server composition via typed definition + engine (MCP-P4-04)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence

from fastmcp import FastMCP

from intergrax.agents.tool_enablement import ToolEnablementProfile
from intergrax.applications._shared.mcp_nexus_server import legacy_server_id_from_display_name
from intergrax.applications._shared.mcp_server_engine import (
    McpServerCompositionDeps,
    compose_mcp_server_from_definition,
    nexus_legacy_server_definition,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.contracts.mcp_server_definition import (
    McpCapabilityBinding,
    McpHostCustomBinding,
    McpServerDefinition,
    build_mcp_server_definition,
)
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
from intergrax.tools.registry.runtime import ToolRegistry


def compose_application_mcp_server(
    *,
    display_name: str,
    host_execution: HostTaskExecutionPort,
    registry: AgentRegistryRead,
    default_capability: str,
    default_tenant_id: str = "mcp",
    default_user_id: str = "mcp-user",
    tool_registry: ToolRegistry | None = None,
    extra_bindings: Sequence[McpCapabilityBinding] = (),
    custom_host_tools: Mapping[str, Callable[..., Awaitable[dict[str, object]]]] | None = None,
    catalog_tool_invoker: ExecutionBoundCatalogToolInvoker | None = None,
    tool_enablement: ToolEnablementProfile | None = None,
    task_control_mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None = None,
    transport_detachment_supervision: RootExecutionTransportDetachmentSupervisionPort | None = None,
    auto_wire_catalog_transport_supervision: bool = True,
    catalog_root_child_work_ingress: object | None = None,
    enable_cross_loop_catalog_child: bool = False,
) -> FastMCP:
    """Compose FastMCP from ``McpServerDefinition`` — preferred Tier-3 entry over ad-hoc decorators."""
    catalog_tool_ids: list[str] = []
    if tool_registry is not None:
        catalog_tool_ids = [item.contract.tool_id for item in tool_registry.list()]

    base = nexus_legacy_server_definition(
        server_id=legacy_server_id_from_display_name(display_name),
        display_name=display_name,
        default_capability=default_capability,
        default_tenant_id=default_tenant_id,
        default_user_id=default_user_id,
        catalog_tool_ids=catalog_tool_ids,
        include_catalog_surface=tool_registry is not None,
    )
    definition: McpServerDefinition = base
    if extra_bindings:
        merged_bindings = tuple(base.bindings) + tuple(extra_bindings)
        extra_ids = tuple(binding.binding_id for binding in extra_bindings)
        allowed = tuple(dict.fromkeys((*base.exposure_policy.allowed_binding_ids, *extra_ids)))
        definition = build_mcp_server_definition(
            server_id=base.server_id,
            display_name=base.display_name,
            default_capability=base.default_capability,
            default_tenant_id=base.default_tenant_id,
            default_user_id=base.default_user_id,
            bindings=merged_bindings,
            exposure_policy=base.exposure_policy.model_copy(update={"allowed_binding_ids": allowed}),
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
        custom_host_tools=custom_host_tools,
    )
    return compose_mcp_server_from_definition(definition, deps)


def research_pipeline_binding() -> McpHostCustomBinding:
    return McpHostCustomBinding(binding_id="research_pipeline", tool_name="run_research_pipeline")


__all__ = [
    "compose_application_mcp_server",
    "research_pipeline_binding",
]
