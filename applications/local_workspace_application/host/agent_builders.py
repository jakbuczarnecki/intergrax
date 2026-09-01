# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.manifest import AgentBinding
from local_indexer.local_indexer_agent import LocalIndexerAgent
from local_search.local_search_agent import LocalSearchAgent
from local_synthesizer.local_synthesizer_agent import LocalSynthesizerAgent
from tool_selection_qualifier.tool_selection_qualifier_agent import ToolSelectionQualifierAgent


def _zero_arg_factory(agent_cls: type[Agent]) -> AgentFactory:
    def _build(_ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
        return agent_cls()

    return _build


LOCAL_WORKSPACE_AGENT_BUILDERS: dict[type[Agent], AgentFactory] = {
    LocalIndexerAgent: _zero_arg_factory(LocalIndexerAgent),
    LocalSearchAgent: _zero_arg_factory(LocalSearchAgent),
    LocalSynthesizerAgent: _zero_arg_factory(LocalSynthesizerAgent),
    ToolSelectionQualifierAgent: _zero_arg_factory(ToolSelectionQualifierAgent),
}
