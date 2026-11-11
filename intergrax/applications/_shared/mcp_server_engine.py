# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Compose FastMCP from typed ``McpServerDefinition`` — delegates to existing MCP bridges."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fastmcp import FastMCP

from intergrax.agents.tool_enablement import ToolEnablementProfile
from intergrax.applications._shared.mcp_catalog_invocation_bridge import (
    McpCatalogInvocationComposition,
)
from intergrax.applications._shared.mcp_catalog_tools import mount_catalog_tools_on_mcp
from intergrax.applications._shared.mcp_harness_principal import resolve_mcp_run_agent_identity
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.contracts.mcp_server_definition import (
    McpCatalogToolBinding,
    McpExposurePolicy,
    McpHostBuiltinBinding,
    McpHostBuiltinId,
    McpHostCustomBinding,
    McpServerDefinition,
    McpServerDefinitionValidationError,
    build_mcp_server_definition,
    resolve_effective_bindings,
    validate_mcp_server_definition,
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
from intergrax.tools.mcp_result_bounds import emit_bounded_mcp_list
from intergrax.tools.registry.runtime import ToolRegistry


@dataclass(frozen=True, slots=True)
class McpServerCompositionDeps:
    """Runtime ports required to materialize a definition on FastMCP."""

    host_execution: HostTaskExecutionPort
    registry: AgentRegistryRead
    tool_registry: ToolRegistry | None = None
    catalog_tool_invoker: ExecutionBoundCatalogToolInvoker | None = None
    tool_enablement: ToolEnablementProfile | None = None
    task_control_mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None = None
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None = None
    transport_detachment_supervision: RootExecutionTransportDetachmentSupervisionPort | None = None
    auto_wire_catalog_transport_supervision: bool = True
    catalog_root_child_work_ingress: object | None = None
    enable_cross_loop_catalog_child: bool = False
    custom_host_tools: Mapping[str, Callable[..., Awaitable[dict[str, object]]]] | None = None


def _filtered_catalog_registry(registry: ToolRegistry, allowed_tool_ids: frozenset[str]) -> ToolRegistry:
    """Materialize allowlisted catalog entries into a throwaway registry (delegation, not a handler map)."""
    if not allowed_tool_ids:
        return ToolRegistry()
    filtered = ToolRegistry()
    for item in registry.list():
        if item.contract.tool_id not in allowed_tool_ids:
            continue
        filtered.register(item.contract, item.handler, activation=item.activation)
    return filtered


def _catalog_tool_ids_from_bindings(
    bindings: Sequence[McpCatalogToolBinding],
) -> frozenset[str]:
    return frozenset(binding.tool_id for binding in bindings)


def _host_builtin_ids(bindings: Sequence[McpHostBuiltinBinding]) -> frozenset[McpHostBuiltinId]:
    return frozenset(binding.builtin for binding in bindings)


def nexus_legacy_server_definition(
    *,
    server_id: str,
    display_name: str,
    default_capability: str,
    default_tenant_id: str,
    default_user_id: str,
    catalog_tool_ids: Sequence[str],
    include_catalog_surface: bool,
) -> McpServerDefinition:
    """Programmatic definition matching legacy ``build_nexus_mcp_server`` semantics."""
    binding_list: list = [
        McpHostBuiltinBinding(binding_id="list_agents", builtin=McpHostBuiltinId.LIST_AGENTS),
        McpHostBuiltinBinding(binding_id="run_agent", builtin=McpHostBuiltinId.RUN_AGENT),
    ]
    if include_catalog_surface:
        binding_list.extend(
            [
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
        )
        for tool_id in catalog_tool_ids:
            binding_list.append(
                McpCatalogToolBinding(
                    binding_id=f"catalog:{tool_id}",
                    tool_id=tool_id,
                )
            )
    return build_mcp_server_definition(
        server_id=server_id,
        display_name=display_name,
        default_capability=default_capability,
        default_tenant_id=default_tenant_id,
        default_user_id=default_user_id,
        bindings=tuple(binding_list),
        exposure_policy=McpExposurePolicy(
            enabled=True,
            allowed_binding_ids=tuple(item.binding_id for item in binding_list),
        ),
    )


def compose_mcp_server_from_definition(
    definition: McpServerDefinition,
    deps: McpServerCompositionDeps,
) -> FastMCP:
    """Validate definition and register exposed tools on FastMCP (OD-MCP-01-A)."""
    known_catalog: frozenset[str] | None = None
    if deps.tool_registry is not None:
        known_catalog = frozenset(item.contract.tool_id for item in deps.tool_registry.list())
    validate_mcp_server_definition(definition, known_catalog_tool_ids=known_catalog)

    exposed = resolve_effective_bindings(definition)
    host_builtins = [b for b in exposed if isinstance(b, McpHostBuiltinBinding)]
    catalog_bindings = [b for b in exposed if isinstance(b, McpCatalogToolBinding)]
    custom_bindings = [b for b in exposed if isinstance(b, McpHostCustomBinding)]
    builtin_ids = _host_builtin_ids(host_builtins)

    mcp = FastMCP(definition.display_name)

    if McpHostBuiltinId.LIST_AGENTS in builtin_ids:

        @mcp.tool
        async def list_agents() -> list[dict[str, Any]]:
            """List registered agents and capabilities (same roster as the HTTP agents endpoint)."""
            agents: list[dict[str, Any]] = []
            for agent_id in deps.registry.list_agent_ids():
                contract = deps.registry.get(agent_id).get_contract()
                agents.append(
                    {
                        "agent_id": contract.id,
                        "name": contract.name,
                        "capabilities": list(contract.capabilities),
                    }
                )
            return emit_bounded_mcp_list(agents)

    if McpHostBuiltinId.RUN_AGENT in builtin_ids:
        from intergrax.applications._shared.mcp_nexus_server import execute_mcp_agent_task

        default_capability = definition.default_capability
        default_tenant_id = definition.default_tenant_id
        default_user_id = definition.default_user_id
        host_execution = deps.host_execution
        task_control_mutation_boundary = deps.task_control_mutation_boundary

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

    catalog_surface = {
        McpHostBuiltinId.LIST_CATALOG_TOOLS,
        McpHostBuiltinId.DESCRIBE_CATALOG_TOOL,
        McpHostBuiltinId.INVOKE_CATALOG_TOOL,
    }
    wants_catalog = bool(builtin_ids & catalog_surface)
    if wants_catalog and deps.tool_registry is not None:
        allowed_ids = _catalog_tool_ids_from_bindings(catalog_bindings)
        registry_view = (
            _filtered_catalog_registry(deps.tool_registry, allowed_ids) if allowed_ids else deps.tool_registry
        )

        catalog_invocation: McpCatalogInvocationComposition | None = None
        if deps.catalog_tool_invoker is not None and McpHostBuiltinId.INVOKE_CATALOG_TOOL in builtin_ids:
            _ = deps.auto_wire_catalog_transport_supervision, deps.transport_detachment_supervision
            from intergrax.applications._shared.production_catalog_tool_child_execution_wiring import (
                build_production_catalog_tool_child_execution_port,
            )

            catalog_child_execution = build_production_catalog_tool_child_execution_port()
            catalog_invocation = McpCatalogInvocationComposition(
                catalog_invoker=deps.catalog_tool_invoker,
                registry=deps.tool_registry,
                tool_enablement=deps.tool_enablement,
                task_control_mutation_boundary=deps.task_control_mutation_boundary,
                meaningful_side_effect_authorization=deps.meaningful_side_effect_authorization,
                catalog_tool_child_execution=catalog_child_execution,
                root_child_work_ingress=deps.catalog_root_child_work_ingress,
                enable_cross_loop_catalog_child=deps.enable_cross_loop_catalog_child,
                default_tenant_id=definition.default_tenant_id,
                default_user_id=definition.default_user_id,
            )
        mount_catalog_tools_on_mcp(
            mcp,
            registry_view,
            catalog_invocation=catalog_invocation,
        )

    if custom_bindings:
        handlers = deps.custom_host_tools or {}
        from intergrax.contracts.mcp_server_definition import McpServerDefinitionErrorCode

        for binding in custom_bindings:
            handler = handlers.get(binding.tool_name)
            if handler is None:
                raise McpServerDefinitionValidationError(
                    McpServerDefinitionErrorCode.MCP_DEF_INVALID,
                    f"missing custom tool handler for {binding.tool_name!r}",
                )

            def _make_tool(
                name: str, fn: Callable[..., Awaitable[dict[str, object]]]
            ) -> Callable[..., Awaitable[dict[str, object]]]:
                @mcp.tool(name=name)
                async def _custom_tool(**kwargs: Any) -> dict[str, object]:
                    return await fn(**kwargs)

                return _custom_tool

            _make_tool(binding.tool_name, handler)

    return mcp


def exposed_catalog_tool_names(definition: McpServerDefinition) -> list[str]:
    """Discovery helper — catalog tool_ids exposed per allowlist (tests / L2)."""
    exposed = resolve_effective_bindings(definition)
    return [binding.tool_id for binding in exposed if isinstance(binding, McpCatalogToolBinding)]


def exposed_builtin_tool_names(definition: McpServerDefinition) -> list[str]:
    mapping = {
        McpHostBuiltinId.LIST_AGENTS: "list_agents",
        McpHostBuiltinId.RUN_AGENT: "run_agent",
        McpHostBuiltinId.LIST_CATALOG_TOOLS: "list_catalog_tools",
        McpHostBuiltinId.DESCRIBE_CATALOG_TOOL: "describe_catalog_tool",
        McpHostBuiltinId.INVOKE_CATALOG_TOOL: "invoke_catalog_tool",
    }
    exposed = resolve_effective_bindings(definition)
    names: list[str] = []
    for binding in exposed:
        if isinstance(binding, McpHostBuiltinBinding):
            name = mapping.get(binding.builtin)
            if name is not None:
                names.append(name)
        elif isinstance(binding, McpHostCustomBinding):
            names.append(binding.tool_name)
    return names


__all__ = [
    "McpServerCompositionDeps",
    "compose_mcp_server_from_definition",
    "exposed_builtin_tool_names",
    "exposed_catalog_tool_names",
    "nexus_legacy_server_definition",
]
