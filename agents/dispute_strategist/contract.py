# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.agents.authoring.patterns.base import PATTERN_VERSION
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_run_enums import CognitivePattern
from dispute_strategist.capabilities import CAPABILITIES

# Register skill packs on the contract — see docs/architecture/SKILLS.md


def build_agent_contract() -> AgentContract:
    return AgentContract(
        id="dispute_strategist",
        name="DisputeStrategistAgent",
        description="DSW litigation strategy — attack/defense lines and emphasis map.",
        version="0.1.0",
        capabilities=CAPABILITIES,
        skills=[],
        extra_tools=[],
        risk_level=AgentRiskLevel.LOW,
        lifecycle_state=AgentLifecycleState.STAGING,
        owner_team="platform",
        max_steps=10,
        cognitive_pattern=CognitivePattern.REFLEX,
        pattern_version=PATTERN_VERSION,
    )
