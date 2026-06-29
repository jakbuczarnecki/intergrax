# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_run import AgentRunError
from intergrax.contracts.agent_run_enums import (
    AgentRunErrorCode,
    StepNextAction,
    TerminalReason,
)
from local_indexer.diagnostics import IndexSummaryDiagnostic


@pytest.mark.unit
@pytest.mark.gate
def test_step_outcome_continue_with_sets_continue_action() -> None:
    outcome = StepOutcome.continue_with({"phase": "execute"})
    assert outcome.is_terminal is False
    assert outcome.next_action == StepNextAction.CONTINUE
    assert outcome.state_delta == {"phase": "execute"}


@pytest.mark.unit
@pytest.mark.gate
def test_step_outcome_complete_sets_goal_met() -> None:
    outcome = StepOutcome.complete({"answer": "ok"})
    assert outcome.is_terminal is True
    assert outcome.terminal_reason == TerminalReason.GOAL_MET
    assert outcome.output == {"answer": "ok"}


@pytest.mark.unit
@pytest.mark.gate
def test_step_outcome_fail_sets_fail_action() -> None:
    error = AgentRunError(code=AgentRunErrorCode.POLICY_DENIED, message="denied")
    outcome = StepOutcome.fail([error], terminal_reason=TerminalReason.POLICY_DENIED)
    assert outcome.is_terminal is True
    assert outcome.next_action == StepNextAction.FAIL
    assert outcome.terminal_reason == TerminalReason.POLICY_DENIED


@pytest.mark.unit
@pytest.mark.gate
def test_step_outcome_pause_hitl_is_non_terminal() -> None:
    outcome = StepOutcome.pause_hitl("need approval")
    assert outcome.is_terminal is False
    assert outcome.next_action == StepNextAction.PAUSE_HITL
    assert outcome.terminal_reason == TerminalReason.HUMAN_REQUIRED


@pytest.mark.unit
@pytest.mark.gate
def test_step_outcome_replan_sets_replanned() -> None:
    outcome = StepOutcome.replan({"phase": "replan"})
    assert outcome.is_terminal is True
    assert outcome.terminal_reason == TerminalReason.REPLANNED
    assert outcome.next_action == StepNextAction.REPLAN


@pytest.mark.unit
@pytest.mark.gate
def test_step_outcome_complete_merges_diagnostic_payloads() -> None:
    payload = IndexSummaryDiagnostic(
        accepted_count=1,
        rejected_count=0,
        ingested_count=1,
        chunk_count=1,
        source_count=1,
    )
    outcome = StepOutcome.complete({"answer": "ok"}, diagnostic_payloads=[payload])
    assert outcome.diagnostics is not None
    assert "lkw.index_summary.v1" in outcome.diagnostics
    assert outcome.diagnostics["lkw.index_summary.v1"]["accepted_count"] == 1
