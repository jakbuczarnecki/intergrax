# © Artur Czarnecki. All rights reserved.

"""Build ``ACPSessionHostContext`` for direct ``agent.run()`` from harness hosts (PAT-2)."""

from __future__ import annotations

from typing import Any

from intergrax.agents.authoring.acp_session_host import ACPSessionHostContext
from intergrax.applications._shared.declarative_tool_wiring import (
    build_declarative_invoker_from_tool_wiring,
)
from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.applications._shared.runtime_boundary_adapters import (
    agent_binding_to_run_binding,
    application_profile_to_runtime_profile,
)


def build_acp_session_host_context(
    *,
    app_profile: Any,
    binding: AgentBinding | None = None,
    declarative_tool_invoker: Any = None,
    critic_graph_hooks: Any = None,
) -> ACPSessionHostContext:
    return ACPSessionHostContext(
        runtime_profile=application_profile_to_runtime_profile(app_profile),
        binding=agent_binding_to_run_binding(binding),
        declarative_tool_invoker=declarative_tool_invoker,
        critic_graph_hooks=critic_graph_hooks,
    )


def build_acp_session_host_from_harness(
    runtime: HarnessHostRuntime,
    *,
    binding: AgentBinding | None = None,
) -> ACPSessionHostContext:
    """Attach critic CVL hooks and declarative tool invoker from a harness host."""
    invoker = build_declarative_invoker_from_tool_wiring(runtime.env_wiring.tool_wiring)
    return build_acp_session_host_context(
        app_profile=runtime.environment,
        binding=binding,
        declarative_tool_invoker=invoker,
        critic_graph_hooks=runtime.critic.graph_hooks,
    )
