# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.patterns.react_budget import record_react_tool_calls, sync_react_budget
from intergrax.agents.authoring.patterns.states import ReActSessionState
from intergrax.contracts.acp_state import AcpBudgetState

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_sync_react_budget_mirrors_counters() -> None:
    state = ReActSessionState(react_iterations_used=2, max_react_iterations=5)
    synced = sync_react_budget(state, default_max_iterations=8)
    assert synced.budget is not None
    assert synced.budget.react_iterations_used == 2
    assert synced.budget.react_iterations_max == 5


def test_record_react_tool_calls_increments_budget() -> None:
    state = ReActSessionState(budget=AcpBudgetState(tool_calls=1))
    updated = record_react_tool_calls(state, 2)
    assert updated.budget is not None
    assert updated.budget.tool_calls == 3
