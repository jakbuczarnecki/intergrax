# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.manifest import AgentBinding
from dispute_analyst.dispute_analyst_agent import DisputeAnalystAgent
from dispute_intake.dispute_intake_agent import DisputeIntakeAgent
from dispute_scenario.dispute_scenario_agent import DisputeScenarioAgent
from dispute_strategist.dispute_strategist_agent import DisputeStrategistAgent


def _zero_arg_factory(agent_cls: type[Agent]) -> AgentFactory:
    def _build(_ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
        return agent_cls()

    return _build


DISPUTE_SIM_AGENT_BUILDERS: dict[type[Agent], AgentFactory] = {
    DisputeIntakeAgent: _zero_arg_factory(DisputeIntakeAgent),
    DisputeAnalystAgent: _zero_arg_factory(DisputeAnalystAgent),
    DisputeStrategistAgent: _zero_arg_factory(DisputeStrategistAgent),
    DisputeScenarioAgent: _zero_arg_factory(DisputeScenarioAgent),
}
