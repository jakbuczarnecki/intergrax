# © Artur Czarnecki. All rights reserved.

"""Type-keyed Tier-3 agent factories for the lab application."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.applications._shared.lab_harness_context import lab_harness_context_from_build_context
from intergrax.applications.contracts.factory import AgentFactory
from echo.echo_agent import EchoAgent
from lab.mock_agents import (
    ComposerMockAgent,
    DocumentMockAgent,
    ResearchMockAgent,
    ValidatorMockAgent,
)
from research.research_agent import ResearchAgent
from research.summary_agent import SummaryAgent
from signoff_probe.signoff_probe_agent import SignoffProbeAgent


def _harness_agent_factory(agent_cls: type[Agent]) -> AgentFactory:
    def _build(ctx, _binding) -> Agent:
        harness = lab_harness_context_from_build_context(ctx)
        return agent_cls(harness)

    return _build


def _build_research_agent(ctx, _binding) -> Agent:
    return ResearchAgent(
        tool_profile=ctx.tool_profile,
        tool_wiring_context=ctx.tool_wiring_context,
    )


def _build_summary_agent(ctx, _binding) -> Agent:
    return SummaryAgent(
        tool_profile=ctx.tool_profile,
        tool_wiring_context=ctx.tool_wiring_context,
    )


LAB_AGENT_BUILDERS: dict[type[Agent], AgentFactory] = {
    EchoAgent: _harness_agent_factory(EchoAgent),
    ResearchMockAgent: _harness_agent_factory(ResearchMockAgent),
    DocumentMockAgent: _harness_agent_factory(DocumentMockAgent),
    ValidatorMockAgent: _harness_agent_factory(ValidatorMockAgent),
    ComposerMockAgent: _harness_agent_factory(ComposerMockAgent),
    SignoffProbeAgent: _harness_agent_factory(SignoffProbeAgent),
    ResearchAgent: _build_research_agent,
    SummaryAgent: _build_summary_agent,
}
