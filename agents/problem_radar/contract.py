# © Artur Czarnecki. All rights reserved.

from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
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
        allowed_tools=["websearch.query", "rag.retrieve"],
        skill_ids=["research.literature_scan"],
        risk_level=AgentRiskLevel.MEDIUM,
        max_steps=12,
        validation_rules=["structured_output"],
    )
