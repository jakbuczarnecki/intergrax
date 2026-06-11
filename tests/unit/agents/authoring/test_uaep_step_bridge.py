# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.uaep_step_bridge import agent_decision_to_step_outcome
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_run_enums import StepNextAction, TerminalReason
from intergrax.contracts.agent_step import StepOutput


@pytest.mark.unit
@pytest.mark.gate
def test_agent_decision_continue_maps_to_step_outcome() -> None:
    output = StepOutput(step_id="s1", summary="partial")
    outcome = agent_decision_to_step_outcome(
        AgentDecision(type=AgentDecisionType.CONTINUE, reason="next"),
        output,
    )
    assert outcome.next_action == StepNextAction.CONTINUE
    assert outcome.is_terminal is False


@pytest.mark.unit
@pytest.mark.gate
def test_agent_decision_complete_maps_to_terminal_outcome() -> None:
    output = StepOutput(step_id="s2", summary="done")
    outcome = agent_decision_to_step_outcome(
        AgentDecision(type=AgentDecisionType.COMPLETE, reason="finished"),
        output,
    )
    assert outcome.is_terminal is True
    assert outcome.terminal_reason == TerminalReason.GOAL_MET
    assert outcome.output == "done"


@pytest.mark.unit
@pytest.mark.gate
def test_agent_decision_request_human_maps_to_pause_hitl() -> None:
    outcome = agent_decision_to_step_outcome(
        AgentDecision(type=AgentDecisionType.REQUEST_HUMAN, reason="approve"),
        StepOutput(step_id="s3", summary=""),
    )
    assert outcome.next_action == StepNextAction.PAUSE_HITL
    assert outcome.terminal_reason == TerminalReason.HUMAN_REQUIRED
