# © Artur Czarnecki. All rights reserved.

from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.agents.authoring.patterns.base import PATTERN_VERSION
from external_contractor_adapter.capabilities import CAPABILITIES

_PATTERN = CognitivePattern.REFLEX


def build_agent_contract() -> AgentContract:
    return AgentContract(
        id="external_contractor_adapter",
        name="ExternalContractorAdapterAgent",
        description=(
            "Tier-2 adapter for governed external contractor agents — "
            "maps external A2A contractor lifecycle into Intergrax contracts."
        ),
        version="0.1.0",
        capabilities=CAPABILITIES,
        skills=[],
        extra_tools=[],
        risk_level=AgentRiskLevel.LOW,
        lifecycle_state=AgentLifecycleState.DEVELOPMENT,
        owner_team="platform",
        max_steps=10,
        cognitive_pattern=_PATTERN,
        pattern_version=PATTERN_VERSION,
        pattern_config={"primary_capability": "external_contractor.adapt"},
    )
