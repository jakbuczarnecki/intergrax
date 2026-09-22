# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from boundary_demo.capabilities import CAPABILITIES
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.skills.providers.data.manifests import DATA_RECORDS_ADMIN


def build_agent_contract() -> AgentContract:
    return AgentContract(
        id="boundary_demo_agent",
        name="Boundary Demo Agent",
        description="Partner PoC agent — writes a demo record via records.put.",
        version="0.1.0",
        capabilities=list(CAPABILITIES),
        skills=[DATA_RECORDS_ADMIN],
        extra_tools=[],
        risk_level=AgentRiskLevel.MEDIUM,
        lifecycle_state=AgentLifecycleState.STAGING,
        owner_team="platform",
        max_steps=1,
        cognitive_pattern=CognitivePattern.REFLEX,
        pattern_version="acp.v1",
    )
