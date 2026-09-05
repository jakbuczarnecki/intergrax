# © Artur Czarnecki. All rights reserved.

"""Declarative investigator agent contract for scenario composition."""

from __future__ import annotations

from intergrax.contracts.agent_contract_meta import AgentContract
from platform_proofs.scenarios.ai_incident_investigation.application.tools import SCENARIO_TOOL_IDS

INVESTIGATOR_AGENT_ID = "incident_investigator"
INVESTIGATOR_CAPABILITY = "incident_investigation.investigate"


def incident_investigator_contract() -> AgentContract:
    return AgentContract(
        id=INVESTIGATOR_AGENT_ID,
        name=INVESTIGATOR_AGENT_ID,
        description="Incident investigator — platform-native scenario",
        capabilities=[INVESTIGATOR_CAPABILITY],
        allowed_tools=list(SCENARIO_TOOL_IDS),
    )
