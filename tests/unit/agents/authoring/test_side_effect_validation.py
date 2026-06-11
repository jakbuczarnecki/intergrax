# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.side_effect_validation import validate_side_effect_mode
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_run_enums import SideEffectMode


@pytest.mark.unit
@pytest.mark.gate
def test_immediate_mode_rejects_requested_actions() -> None:
    outcome = StepOutcome.continue_with({}).model_copy(
        update={"requested_actions": [{"tool_id": "x"}]},
    )
    message = validate_side_effect_mode(outcome, SideEffectMode.IMMEDIATE)
    assert message is not None


@pytest.mark.unit
@pytest.mark.gate
def test_declarative_mode_rejects_immediate_tool_hint() -> None:
    outcome = StepOutcome.continue_with(
        {},
        diagnostics={"immediate_tool_calls": True},
    )
    message = validate_side_effect_mode(outcome, SideEffectMode.DECLARATIVE)
    assert message is not None
