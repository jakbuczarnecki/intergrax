# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.skills.providers.harness.manifests import HARNESS_TOOL_SMOKE


def build_agent_contract() -> AgentContract:
    return AgentContract(
        id="echo",
        name="Echo Agent",
        description="Echoes user input for runtime harness validation.",
        version="1.0.0",
        capabilities=["echo.basic"],
        skills=[HARNESS_TOOL_SMOKE],
        extra_tools=[],
        risk_level=AgentRiskLevel.LOW,
        lifecycle_state=AgentLifecycleState.PRODUCTION,
        production_eligible=True,
        owner_team="platform",
        owner_contact="harness@intergrax",
        on_call_contact="harness@intergrax",
        runbook_ref="docs/project/architecture/intergrax_runtime_architecture.md",
        modality_profile_id="lab.default",
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        validation_rules=["structured_output"],
        max_steps=5,
        cognitive_pattern=CognitivePattern.REFLEX,
        pattern_version="acp.v1",
    )
