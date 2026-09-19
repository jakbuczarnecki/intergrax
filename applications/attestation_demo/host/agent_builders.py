# © Artur Czarnecki. All rights reserved.

"""Type-keyed Tier-3 agent factories for attestation_demo."""

from __future__ import annotations

from boundary_demo.boundary_demo_agent import BoundaryDemoAgent
from intergrax.agents.agent_contract import Agent
from intergrax.agents.tool_enablement import ToolEnablementProfile, ToolWiringContextLike
from intergrax.applications._shared.lab_harness_context import lab_harness_context_from_build_context
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.runtime.attestation.settings import resolve_execution_boundary_export_runtime
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.tools.registry.wiring import ToolWiringContext


def build_attestation_demo_agent_builders(
    *,
    tool_profile: ToolEnablementProfile | None = None,
    tool_wiring_context: ToolWiringContext | ToolWiringContextLike | None = None,
    policy_bundle: RuntimePolicyBundle | None = None,
    boundary_event_buffer: BoundaryEventBuffer | None = None,
) -> dict[type[Agent], AgentFactory]:
    """Compose attestation builder map with host-bound boundary / tool dependencies."""
    resolved_wiring = (
        tool_wiring_context if isinstance(tool_wiring_context, ToolWiringContext) else None
    )

    def _build_boundary_demo_agent(
        ctx: ApplicationBuildContext, _binding: AgentBinding
    ) -> Agent:
        harness = lab_harness_context_from_build_context(
            ctx,
            policy_bundle=policy_bundle,
            tool_wiring_context=resolved_wiring,
        )
        export_settings = None
        if ctx.environment is not None:
            export_settings = resolve_execution_boundary_export_runtime(
                ctx.environment.execution_boundary_export_profile,
            )
        environment = ctx.environment
        resolved_profile = tool_profile
        if resolved_profile is None and environment is not None:
            resolved_profile = environment.tool_profile
        return BoundaryDemoAgent(
            harness,
            tool_profile=resolved_profile,
            tool_wiring_context=tool_wiring_context,
            execution_boundary_export=export_settings,
            boundary_event_buffer=boundary_event_buffer,
        )

    return {
        BoundaryDemoAgent: _build_boundary_demo_agent,
    }


ATTESTATION_DEMO_AGENT_BUILDERS: dict[type[Agent], AgentFactory] = (
    build_attestation_demo_agent_builders()
)
