# © Artur Czarnecki. All rights reserved.

"""Type-keyed Tier-3 agent factories for the lab application."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.agents.tool_enablement import ToolEnablementProfile, ToolWiringContextLike
from intergrax.applications._shared.lab_harness_context import lab_harness_context_from_build_context
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.tools.registry.wiring import ToolWiringContext
from echo.echo_agent import EchoAgent
from lab.mock_agents import (
    ComposerMockAgent,
    DocumentMockAgent,
    ResearchMockAgent,
    ValidatorMockAgent,
)
from research.research_agent import ResearchAgent
from research.summary_agent import SummaryAgent
from problem_radar.problem_radar_agent import ProblemRadarAgent
from signoff_probe.signoff_probe_agent import SignoffProbeAgent


def build_lab_agent_builders(
    *,
    tool_profile: ToolEnablementProfile | None = None,
    tool_wiring_context: ToolWiringContext | ToolWiringContextLike | None = None,
    policy_bundle: RuntimePolicyBundle | None = None,
) -> dict[type[Agent], AgentFactory]:
    """Compose lab builder map with host-bound harness / tool dependencies."""
    resolved_wiring = (
        tool_wiring_context if isinstance(tool_wiring_context, ToolWiringContext) else None
    )

    def _harness_agent_factory(agent_cls: type[Agent]) -> AgentFactory:
        def _build(ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
            harness = lab_harness_context_from_build_context(
                ctx,
                policy_bundle=policy_bundle,
                tool_wiring_context=resolved_wiring,
            )
            return agent_cls(harness)

        return _build

    def _build_research_agent(ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
        harness = lab_harness_context_from_build_context(
            ctx,
            policy_bundle=policy_bundle,
            tool_wiring_context=resolved_wiring,
        )
        environment = ctx.environment
        resolved_profile = tool_profile
        if resolved_profile is None and environment is not None:
            resolved_profile = environment.tool_profile
        return ResearchAgent(
            harness,
            tool_profile=resolved_profile,
            tool_wiring_context=tool_wiring_context,
            enable_websearch=True,
        )

    def _build_summary_agent(ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
        harness = lab_harness_context_from_build_context(
            ctx,
            policy_bundle=policy_bundle,
            tool_wiring_context=resolved_wiring,
        )
        return SummaryAgent(harness)

    return {
        EchoAgent: _harness_agent_factory(EchoAgent),
        ResearchMockAgent: _harness_agent_factory(ResearchMockAgent),
        DocumentMockAgent: _harness_agent_factory(DocumentMockAgent),
        ValidatorMockAgent: _harness_agent_factory(ValidatorMockAgent),
        ComposerMockAgent: _harness_agent_factory(ComposerMockAgent),
        SignoffProbeAgent: _harness_agent_factory(SignoffProbeAgent),
        ProblemRadarAgent: _harness_agent_factory(ProblemRadarAgent),
        ResearchAgent: _build_research_agent,
        SummaryAgent: _build_summary_agent,
    }


LAB_AGENT_BUILDERS: dict[type[Agent], AgentFactory] = build_lab_agent_builders()
