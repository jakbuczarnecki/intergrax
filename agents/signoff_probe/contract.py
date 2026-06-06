# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.skills.providers.harness.manifests import HARNESS_TOOL_SMOKE
from signoff_probe.capabilities import CAPABILITIES


def build_agent_contract() -> AgentContract:
    return AgentContract(
        id="signoff_probe",
        name="SignoffProbeAgent",
        description="Scaffolded UAEP agent for Intergrax experiments.",
        version="0.1.0",
        capabilities=CAPABILITIES,
        skills=[HARNESS_TOOL_SMOKE],
        extra_tools=[],
        risk_level=AgentRiskLevel.LOW,
        lifecycle_state=AgentLifecycleState.DEVELOPMENT,
        owner_team="platform",
        max_steps=10,
    )
