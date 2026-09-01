# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_run_enums import CognitivePattern
from tool_selection_qualifier.capabilities import CAPABILITIES


def build_agent_contract() -> AgentContract:
    return AgentContract(
        id="tool_selection_qualifier",
        name="Tool Selection Qualifier",
        description="DIAG-FUNCTIONAL-Q2 qualification agent — real LLM catalog tool selection.",
        version="0.1.0",
        capabilities=list(CAPABILITIES),
        risk_level=AgentRiskLevel.LOW,
        lifecycle_state=AgentLifecycleState.STAGING,
        owner_team="platform",
        max_steps=1,
        cognitive_pattern=CognitivePattern.REFLEX,
        pattern_version="acp.v1",
    )
