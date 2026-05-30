# © Artur Czarnecki. All rights reserved.

"""Type-keyed Tier-3 agent factories for the research application."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.factory import AgentFactory
from research.research_agent import ResearchAgent
from research.summary_agent import SummaryAgent


def _zero_arg_factory(agent_cls: type[Agent]) -> AgentFactory:
    def _build(_ctx, _binding) -> Agent:
        return agent_cls()

    return _build


RESEARCH_AGENT_BUILDERS: dict[type[Agent], AgentFactory] = {
    ResearchAgent: _zero_arg_factory(ResearchAgent),
    SummaryAgent: _zero_arg_factory(SummaryAgent),
}
