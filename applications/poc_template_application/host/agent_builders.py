# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.manifest import AgentBinding
from echo.echo_agent import EchoAgent


def _zero_arg_factory(agent_cls: type[Agent]) -> AgentFactory:
    def _build(_ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
        return agent_cls()

    return _build


POC_TEMPLATE_AGENT_BUILDERS: dict[type[Agent], AgentFactory] = {
    EchoAgent: _zero_arg_factory(EchoAgent),
}
