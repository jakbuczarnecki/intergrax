# © Artur Czarnecki. All rights reserved.

import warnings

import pytest

from intergrax.agents.authoring.decisions import (
    complete,
    continue_to,
    continue_with,
    delegate_handoff,
    delegate_to,
    finish,
    pause_for_human,
    request_replan,
    to_step_outcome,
)
from intergrax.contracts.agent_decision import AgentDecisionType
from intergrax.contracts.agent_run_enums import StepNextAction, TerminalReason
from intergrax.contracts.agent_step import StepOutput


@pytest.mark.unit
@pytest.mark.gate
def test_finish_maps_to_step_outcome_complete() -> None:
    outcome = finish({"answer": "ok"})
    assert outcome.is_terminal is True
    assert outcome.terminal_reason == TerminalReason.GOAL_MET
    assert outcome.output == {"answer": "ok"}


@pytest.mark.unit
@pytest.mark.gate
def test_continue_with_sets_continue_action() -> None:
    outcome = continue_with({"phase": "act"})
    assert outcome.is_terminal is False
    assert outcome.next_action == StepNextAction.CONTINUE
    assert outcome.state_delta == {"phase": "act"}


@pytest.mark.unit
@pytest.mark.gate
def test_pause_for_human_sets_pause_hitl() -> None:
    outcome = pause_for_human("need approval")
    assert outcome.next_action == StepNextAction.PAUSE_HITL
    assert outcome.terminal_reason == TerminalReason.HUMAN_REQUIRED


@pytest.mark.unit
@pytest.mark.gate
def test_request_replan_sets_replanned() -> None:
    outcome = request_replan({"phase": "replan"})
    assert outcome.terminal_reason == TerminalReason.REPLANNED
    assert outcome.next_action == StepNextAction.REPLAN


@pytest.mark.unit
@pytest.mark.gate
def test_delegate_handoff_embeds_handoff_in_diagnostics() -> None:
    outcome = delegate_handoff(
        "target_agent",
        from_agent_id="source_agent",
        reason="hand off",
    )
    assert outcome.next_action == StepNextAction.REPLAN
    assert outcome.diagnostics is not None
    handoff = outcome.diagnostics["handoff"]
    assert handoff["to_agent_id"] == "target_agent"
    assert handoff["from_agent_id"] == "source_agent"


@pytest.mark.unit
@pytest.mark.gate
def test_to_step_outcome_reexports_bridge() -> None:
    decision = continue_to("step_two", reason="next")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        outcome = to_step_outcome(
            decision,
            StepOutput(step_id="step_one", summary="partial"),
        )
    assert outcome.next_action == StepNextAction.CONTINUE
    assert outcome.diagnostics is not None
    assert outcome.diagnostics["next_step_id"] == "step_two"


@pytest.mark.unit
@pytest.mark.gate
def test_legacy_complete_emits_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="UAEP AgentDecision helpers are deprecated"):
        decision = complete(reason="done")
    assert decision.type == AgentDecisionType.COMPLETE


@pytest.mark.unit
@pytest.mark.gate
def test_legacy_delegate_to_uses_modify_plan_with_handoff() -> None:
    with pytest.warns(DeprecationWarning):
        decision = delegate_to("peer", from_agent_id="self", reason="shift")
    assert decision.type == AgentDecisionType.MODIFY_PLAN
    assert decision.handoff is not None
    assert decision.handoff.to_agent_id == "peer"
    outcome = to_step_outcome(decision, StepOutput(step_id="s1", summary=""))
    assert outcome.next_action == StepNextAction.REPLAN
    assert outcome.diagnostics is not None
    assert outcome.diagnostics["handoff"]["to_agent_id"] == "peer"
