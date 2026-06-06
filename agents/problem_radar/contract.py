# © Artur Czarnecki. All rights reserved.

from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.skills.providers.research.manifests import RESEARCH_LITERATURE_SCAN
from problem_radar.capabilities import CAPABILITIES


def build_agent_contract() -> AgentContract:
    return AgentContract(
        id="problem_radar",
        name="Problem Radar Agent",
        description=(
            "Discovers repeated user pain signals from public sources and clusters "
            "them into opportunity themes (Phase K.1 prototype)."
        ),
        version="0.1.0",
        capabilities=CAPABILITIES,
        skills=[RESEARCH_LITERATURE_SCAN],
        extra_tools=[],
        risk_level=AgentRiskLevel.MEDIUM,
        lifecycle_state=AgentLifecycleState.EXPERIMENTAL,
        owner_team="platform",
        max_steps=12,
        validation_rules=["structured_output"],
    )
