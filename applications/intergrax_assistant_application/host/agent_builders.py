# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.agents.reference_harness import LabHarnessContext, default_reference_harness
from intergrax.agents.tool_enablement import ToolEnablementProfile
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax_assistant.intergrax_assistant_agent import IntergraxAssistantAgent


def build_intergrax_assistant_agent_builders(
    *,
    tool_profile: ToolEnablementProfile | None = None,
    lab_harness: LabHarnessContext | None = None,
) -> dict[type[Agent], AgentFactory]:
    """Compose assistant builder map with host-prepared lab harness dependency."""
    harness = lab_harness if lab_harness is not None else default_reference_harness()

    def _build_concierge_agent(
        _ctx: ApplicationBuildContext, _binding: AgentBinding
    ) -> Agent:
        return IntergraxAssistantAgent()

    def _build_echo_agent(ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
        from echo.echo_agent import EchoAgent

        _ = ctx
        return EchoAgent(harness)

    def _build_legal_agent(_ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
        from legal.legal_agent import LegalAgent

        return LegalAgent()

    def _build_research_agent(ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
        from research.research_agent import ResearchAgent

        environment = ctx.environment
        resolved_profile = tool_profile
        if resolved_profile is None and environment is not None:
            resolved_profile = environment.tool_profile
        return ResearchAgent(
            harness,
            tool_profile=resolved_profile,
            enable_websearch=True,
        )

    def _build_summary_agent(ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
        from research.summary_agent import SummaryAgent

        _ = ctx
        return SummaryAgent(harness)

    from echo.echo_agent import EchoAgent
    from legal.legal_agent import LegalAgent
    from research.research_agent import ResearchAgent
    from research.summary_agent import SummaryAgent

    return {
        IntergraxAssistantAgent: _build_concierge_agent,
        EchoAgent: _build_echo_agent,
        LegalAgent: _build_legal_agent,
        ResearchAgent: _build_research_agent,
        SummaryAgent: _build_summary_agent,
    }


INTERGRAX_ASSISTANT_AGENT_BUILDERS: dict[type[Agent], AgentFactory] = (
    build_intergrax_assistant_agent_builders()
)
