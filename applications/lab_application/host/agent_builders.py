# © Artur Czarnecki. All rights reserved.

"""Type-keyed Tier-3 agent factories for the lab application."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.agents.reference_harness import LabHarnessContext, default_reference_harness
from intergrax.agents.tool_enablement import ToolEnablementProfile
from intergrax.applications._shared.tool_enablement_binding import resolve_tool_enablement
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.manifest import AgentBinding
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
    lab_harness: LabHarnessContext | None = None,
) -> dict[type[Agent], AgentFactory]:
    """Compose lab builder map with host-prepared lab harness dependency."""
    harness = lab_harness if lab_harness is not None else default_reference_harness()

    def _harness_agent_factory(agent_cls: type[Agent]) -> AgentFactory:
        def _build(ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
            _ = ctx
            return agent_cls(harness)

        return _build

    def _build_research_agent(ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
        environment = ctx.environment
        env_profile = environment.tool_profile if environment is not None else None
        resolved_profile = resolve_tool_enablement(
            tool_profile,
            environment_tool_profile=env_profile,
        )
        return ResearchAgent(
            harness,
            tool_profile=resolved_profile,
            enable_websearch=True,
        )

    def _build_summary_agent(ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
        _ = ctx
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
