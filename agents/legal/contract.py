# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.skills.providers.legal.manifests import LEGAL_CONTRACT_REVIEW
from legal.capabilities import CAPABILITIES

# Skill packs: docs/architecture/SKILLS.md


def build_agent_contract() -> AgentContract:
    return AgentContract(
        id="legal",
        name="LegalAgent",
        description="Contract review agent (scaffold baseline — Phase AA-LEG).",
        version="0.1.0",
        capabilities=CAPABILITIES,
        skills=[LEGAL_CONTRACT_REVIEW],
        extra_tools=[],
        risk_level=AgentRiskLevel.HIGH,
        lifecycle_state=AgentLifecycleState.STAGING,
        owner_team="platform",
        max_steps=20,
    )
