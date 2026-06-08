# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax_assistant.capabilities import CAPABILITIES

# Register skill packs on the contract — see docs/SKILLS.md


def build_agent_contract() -> AgentContract:
    return AgentContract(
        id="intergrax_assistant",
        name="IntergraxAssistantAgent",
        description="Scaffolded UAEP agent for Intergrax experiments.",
        version="0.1.0",
        capabilities=CAPABILITIES,
        skills=[],
        extra_tools=[],
        risk_level=AgentRiskLevel.LOW,
        lifecycle_state=AgentLifecycleState.DEVELOPMENT,
        owner_team="platform",
        max_steps=10,
    )
