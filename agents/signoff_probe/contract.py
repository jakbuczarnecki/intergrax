# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from signoff_probe.capabilities import CAPABILITIES


def build_agent_contract() -> AgentContract:
    return AgentContract(
        id="signoff_probe",
        name="SignoffProbeAgent",
        description="Scaffolded UAEP agent for Intergrax experiments.",
        version="0.1.0",
        capabilities=CAPABILITIES,
        allowed_tools=[],
        risk_level=AgentRiskLevel.LOW,
        max_steps=10,
    )
