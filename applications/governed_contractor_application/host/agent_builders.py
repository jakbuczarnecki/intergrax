# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.manifest import AgentBinding
from external_contractor_adapter.external_contractor_adapter_agent import ExternalContractorAdapterAgent


def _zero_arg_factory(agent_cls: type[Agent]) -> AgentFactory:
    def _build(_ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
        return agent_cls()

    return _build


GOVERNED_CONTRACTOR_AGENT_BUILDERS: dict[type[Agent], AgentFactory] = {
    ExternalContractorAdapterAgent: _zero_arg_factory(ExternalContractorAdapterAgent),
}
