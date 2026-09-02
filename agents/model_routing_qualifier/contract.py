# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_run_enums import CognitivePattern
from model_routing_qualifier.capabilities import CAPABILITIES


def build_agent_contract() -> AgentContract:
    return AgentContract(
        id="model_routing_qualifier",
        name="Model Routing Qualifier",
        description="DIAG-FUNCTIONAL-Q4 qualification agent — real LLM routing and execution.",
        version="0.1.0",
        capabilities=list(CAPABILITIES),
        risk_level=AgentRiskLevel.LOW,
        lifecycle_state=AgentLifecycleState.STAGING,
        owner_team="platform",
        max_steps=1,
        cognitive_pattern=CognitivePattern.REFLEX,
        pattern_version="acp.v1",
    )
