# © Artur Czarnecki. All rights reserved.

"""Build the MCP-SERVER-X P6 reference qualification FastMCP server (executable contract).

P7 (MCP-P7-01) extends the builder additively: every new keyword is optional and
defaults to the P6 behaviour, so the P6 qualification pack keeps its meaning.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence

from fastmcp import FastMCP
from intergrax.agents.tool_enablement import ToolEnablementProfile
from intergrax.applications._shared.mcp_developer_api import (
    McpBindingVisibility,
    McpCapabilityBinding,
    McpCatalogToolBinding,
    McpExposurePolicy,
    McpHostBuiltinBinding,
    McpHostBuiltinId,
    McpHostCustomBinding,
    McpIdentityMode,
    McpServerCompositionDeps,
    McpServerDefinition,
    build_mcp_server_definition,
    compose_mcp_server_from_definition,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.tools.registry.runtime import ToolRegistry

REFERENCE_SERVER_ID = "mcp-server-x-p6-reference"
REFERENCE_DISPLAY_NAME = "MCP-SERVER-X P6 Reference"
REFERENCE_CAPABILITY = "mcp.p6.reference"
REFERENCE_TENANT = "mcp-p6-reference"
REFERENCE_USER = "mcp-p6-reference-user"

CustomHandler = Callable[..., Awaitable[dict[str, object]]]


def build_mcp_server_x_reference_definition(
    *,
    catalog_tool_ids: Sequence[str] = (),
    include_catalog_tool_bindings: bool = False,
    extra_custom_binding_names: Sequence[str] = (),
    exposure_policy: McpExposurePolicy | None = None,
    retired_binding_ids: Sequence[str] = (),
    hidden_binding_ids: Sequence[str] = (),
    omit_from_allowlist: Sequence[str] = (),
) -> McpServerDefinition:
    """Reference definition (host builtins + qual echo + optional catalog surface).

    ``retired_binding_ids`` / ``hidden_binding_ids`` flip the matching binding flags;
    ``omit_from_allowlist`` keeps a binding declared but not allowlisted (disabled).
    An explicit ``exposure_policy`` overrides the derived allowlist.
    """
    retired = frozenset(retired_binding_ids)
    hidden = frozenset(hidden_binding_ids)

    def _flags(binding_id: str) -> dict[str, object]:
        return {
            "retired": binding_id in retired,
            "visibility": (McpBindingVisibility.HIDDEN if binding_id in hidden else McpBindingVisibility.PUBLIC),
        }

    bindings: list[McpCapabilityBinding] = [
        McpHostBuiltinBinding(
            binding_id="list_agents",
            builtin=McpHostBuiltinId.LIST_AGENTS,
            **_flags("list_agents"),
        ),
        McpHostBuiltinBinding(
            binding_id="run_agent",
            builtin=McpHostBuiltinId.RUN_AGENT,
            **_flags("run_agent"),
        ),
        McpHostCustomBinding(
            binding_id="qual_echo",
            tool_name="qual_echo",
            **_flags("qual_echo"),
        ),
    ]
    for name in extra_custom_binding_names:
        bindings.append(McpHostCustomBinding(binding_id=name, tool_name=name, **_flags(name)))
    if catalog_tool_ids:
        for builtin_id, builtin in (
            ("list_catalog_tools", McpHostBuiltinId.LIST_CATALOG_TOOLS),
            ("describe_catalog_tool", McpHostBuiltinId.DESCRIBE_CATALOG_TOOL),
            ("invoke_catalog_tool", McpHostBuiltinId.INVOKE_CATALOG_TOOL),
        ):
            bindings.append(
                McpHostBuiltinBinding(binding_id=builtin_id, builtin=builtin, **_flags(builtin_id)),
            )
        if include_catalog_tool_bindings:
            for tool_id in catalog_tool_ids:
                bindings.append(
                    McpCatalogToolBinding(
                        binding_id=f"catalog:{tool_id}",
                        tool_id=tool_id,
                        **_flags(f"catalog:{tool_id}"),
                    ),
                )

    policy = exposure_policy
    if policy is None and omit_from_allowlist:
        omitted = frozenset(omit_from_allowlist)
        policy = McpExposurePolicy(
            enabled=True,
            allowed_binding_ids=tuple(b.binding_id for b in bindings if b.binding_id not in omitted),
        )
    return build_mcp_server_definition(
        server_id=REFERENCE_SERVER_ID,
        display_name=REFERENCE_DISPLAY_NAME,
        default_capability=REFERENCE_CAPABILITY,
        default_tenant_id=REFERENCE_TENANT,
        default_user_id=REFERENCE_USER,
        bindings=tuple(bindings),
        exposure_policy=policy,
    )


def build_mcp_server_x_reference_qualification_server(
    *,
    host_execution: HostTaskExecutionPort,
    registry: AgentRegistryRead,
    tool_registry: ToolRegistry | None = None,
    custom_echo: CustomHandler | None = None,
    # --- P7 additive options (all default to the P6 behaviour) ---
    definition: McpServerDefinition | None = None,
    exposure_policy: McpExposurePolicy | None = None,
    retired_binding_ids: Sequence[str] = (),
    hidden_binding_ids: Sequence[str] = (),
    omit_from_allowlist: Sequence[str] = (),
    include_catalog_tool_bindings: bool = False,
    max_concurrent_tool_calls: int | None | object = ...,
    catalog_tool_invoker: ExecutionBoundCatalogToolInvoker | None = None,
    tool_enablement: ToolEnablementProfile | None = None,
    task_control_mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None = None,
    extra_custom_handlers: Mapping[str, CustomHandler] | None = None,
    identity_mode: McpIdentityMode = McpIdentityMode.LOCAL_DEV_EXPLICIT,
    max_call_seconds: float | None = None,
    public_custom_tool_names: frozenset[str] = frozenset(),
) -> FastMCP:
    """Reference server: host builtins + optional catalog surface + qual echo custom binding.

    With no P7 keyword the composition is identical to the P6 reference server.
    ``max_concurrent_tool_calls`` left at its ``...`` sentinel keeps the engine default.
    ``definition`` replaces the derived definition wholesale (the other definition
    keywords are then ignored).
    ``identity_mode`` defaults to ``LOCAL_DEV_EXPLICIT`` here because this is a qualification fixture driven
    in-process or over loopback; tests of the fail-closed production default pass ``HARNESS_BOUND``.
    """
    catalog_tool_ids: list[str] = []
    if tool_registry is not None:
        catalog_tool_ids = [item.contract.tool_id for item in tool_registry.list()]

    handlers: dict[str, CustomHandler] = dict(extra_custom_handlers or {})
    if definition is None:
        definition = build_mcp_server_x_reference_definition(
            catalog_tool_ids=catalog_tool_ids,
            include_catalog_tool_bindings=include_catalog_tool_bindings,
            extra_custom_binding_names=tuple(handlers),
            exposure_policy=exposure_policy,
            retired_binding_ids=retired_binding_ids,
            hidden_binding_ids=hidden_binding_ids,
            omit_from_allowlist=omit_from_allowlist,
        )
    if custom_echo is None:

        async def _default_echo(message: str = "") -> dict[str, object]:
            return {"echo": message, "server_id": REFERENCE_SERVER_ID}

        custom_echo = _default_echo
    handlers["qual_echo"] = custom_echo

    deps_kwargs: dict[str, object] = {}
    if max_concurrent_tool_calls is not ...:
        deps_kwargs["max_concurrent_tool_calls"] = max_concurrent_tool_calls
    deps = McpServerCompositionDeps(
        host_execution=host_execution,
        registry=registry,
        tool_registry=tool_registry,
        catalog_tool_invoker=catalog_tool_invoker,
        tool_enablement=tool_enablement,
        task_control_mutation_boundary=task_control_mutation_boundary,
        meaningful_side_effect_authorization=meaningful_side_effect_authorization,
        custom_host_tools=handlers,
        identity_mode=identity_mode,
        max_call_seconds=max_call_seconds,
        public_custom_tool_names=public_custom_tool_names,
        **deps_kwargs,
    )
    return compose_mcp_server_from_definition(definition, deps)
