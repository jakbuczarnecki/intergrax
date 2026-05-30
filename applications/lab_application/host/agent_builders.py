# © Artur Czarnecki. All rights reserved.

"""Type-keyed Tier-3 agent factories for the lab application."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
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


def _zero_arg_factory(agent_cls: type[Agent]) -> AgentFactory:
    def _build(_ctx, _binding) -> Agent:
        return agent_cls()

    return _build


# Prefer type-keyed builders — no string builder_key required on AgentBinding.mount().
LAB_AGENT_BUILDERS: dict[type[Agent], AgentFactory] = {
    EchoAgent: _zero_arg_factory(EchoAgent),
    ResearchMockAgent: _zero_arg_factory(ResearchMockAgent),
    DocumentMockAgent: _zero_arg_factory(DocumentMockAgent),
    ValidatorMockAgent: _zero_arg_factory(ValidatorMockAgent),
    ComposerMockAgent: _zero_arg_factory(ComposerMockAgent),
    SignoffProbeAgent: _zero_arg_factory(SignoffProbeAgent),
    ResearchAgent: _zero_arg_factory(ResearchAgent),
    SummaryAgent: _zero_arg_factory(SummaryAgent),
}
