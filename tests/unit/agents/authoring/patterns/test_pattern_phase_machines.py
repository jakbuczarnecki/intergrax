# © Artur Czarnecki. All rights reserved.

"""Phase machine unit tests for multi-step cognitive patterns (ACP-10)."""

from __future__ import annotations

import pytest

from intergrax.agents.authoring.patterns.decomposition import DecompositionAgent
from intergrax.agents.authoring.patterns.plan_execute import PlanExecuteAgent
from intergrax.agents.authoring.patterns.react import ReActAgent
from intergrax.agents.authoring.patterns.reflection import ReflectionAgent


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        ("plan", "execute"),
        ("execute", "synthesize"),
        ("synthesize", "done"),
        ("done", "done"),
    ],
)
def test_plan_execute_advance_phase(phase: str, expected: str) -> None:
    assert PlanExecuteAgent._advance_phase(phase) == expected


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        ("draft", "critique"),
        ("critique", "revise"),
        ("revise", "done"),
        ("done", "done"),
    ],
)
def test_reflection_advance_phase(phase: str, expected: str) -> None:
    assert ReflectionAgent._advance_phase(phase) == expected


@pytest.mark.unit
@pytest.mark.gate
def test_react_agent_default_iteration_budget_is_positive() -> None:
    assert ReActAgent.default_max_react_iterations > 0


@pytest.mark.unit
@pytest.mark.gate
def test_decomposition_agent_uses_typed_session_state() -> None:
    from intergrax.agents.authoring.patterns.states import DecompositionSessionState

    assert DecompositionAgent.session_state_type is DecompositionSessionState
