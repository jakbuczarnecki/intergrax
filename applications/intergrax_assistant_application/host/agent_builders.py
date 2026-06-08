# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.applications._shared.lab_harness_context import lab_harness_context_from_build_context
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax_assistant.intergrax_assistant_agent import IntergraxAssistantAgent


def _harness_agent_factory(agent_cls: type[Agent]) -> AgentFactory:
    def _build(ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
        harness = lab_harness_context_from_build_context(ctx)
        return agent_cls(harness)

    return _build


def _build_concierge_agent(_ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
    return IntergraxAssistantAgent()


def _build_echo_agent(ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
    from echo.echo_agent import EchoAgent

    harness = lab_harness_context_from_build_context(ctx)
    return EchoAgent(harness)


def _build_legal_agent(_ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
    from legal.legal_agent import LegalAgent

    return LegalAgent()


def _build_research_agent(ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
    from research.research_agent import ResearchAgent

    harness = lab_harness_context_from_build_context(ctx)
    return ResearchAgent(
        harness,
        tool_profile=ctx.tool_profile,
        tool_wiring_context=ctx.tool_wiring_context,
        enable_websearch=True,
    )


def _build_summary_agent(ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
    from research.summary_agent import SummaryAgent

    harness = lab_harness_context_from_build_context(ctx)
    return SummaryAgent(harness)


def _agent_builders() -> dict[type[Agent], AgentFactory]:
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


INTERGRAX_ASSISTANT_AGENT_BUILDERS: dict[type[Agent], AgentFactory] = _agent_builders()
