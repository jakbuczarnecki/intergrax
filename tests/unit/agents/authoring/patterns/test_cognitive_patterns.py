# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.patterns.reference import (
    PatternReActProbe,
    PatternReflexProbe,
)
from intergrax.agents.authoring.patterns.types import CognitiveEvaluation
from intergrax.agents.authoring.uaep_step_bridge import agent_decision_to_step_outcome
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus, CognitivePattern, StepNextAction
from intergrax.contracts.agent_step import StepOutput
from intergrax.runtime.registry.agent_assembly_resolver import validate_cognitive_pattern_metadata


@pytest.mark.unit
@pytest.mark.gate
def test_cognitive_pattern_contract_validation_requires_version() -> None:
    from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel

    contract = AgentContract(
        id="demo",
        name="Demo",
        description="demo",
        capabilities=["demo.cap"],
        cognitive_pattern=CognitivePattern.REFLEX,
        risk_level=AgentRiskLevel.LOW,
        max_steps=1,
    )
    result = validate_cognitive_pattern_metadata(contract)
    assert not result.valid


@pytest.mark.unit
@pytest.mark.gate
async def test_reflex_probe_typed_run() -> None:
    agent = PatternReflexProbe()
    contract = agent.get_contract()
    assert contract.cognitive_pattern == CognitivePattern.REFLEX
    result = await agent.run(
        AgentRunRequest(
            input="probe",
            identity=RequestIdentity(tenant_id="t1", user_id="u1"),
        )
    )
    assert result.status == AgentRunStatus.SUCCEEDED


@pytest.mark.unit
@pytest.mark.gate
async def test_react_probe_bounded_loop() -> None:
    agent = PatternReActProbe()
    result = await agent.run(
        AgentRunRequest(
            input="react",
            identity=RequestIdentity(tenant_id="t1", user_id="u1"),
        )
    )
    assert result.status == AgentRunStatus.SUCCEEDED
    assert result.trace.total_steps >= 1


@pytest.mark.unit
@pytest.mark.gate
def test_agent_decision_complete_maps_to_step_outcome_enum() -> None:
    outcome = agent_decision_to_step_outcome(
        AgentDecision(type=AgentDecisionType.COMPLETE, reason="done"),
        StepOutput(step_id="s1", summary="answer"),
    )
    assert outcome.next_action == StepNextAction.CONTINUE or outcome.is_terminal
