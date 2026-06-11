# © Artur Czarnecki. All rights reserved.

"""Shared ReAct budget keys in ``acp.state.v1.budget`` (ACP-CLOSE-PAT-1 · §25.2)."""

from __future__ import annotations

from intergrax.contracts.acp_state import AcpBudgetState
from intergrax.agents.authoring.patterns.states import ReActSessionState


def sync_react_budget(
    state: ReActSessionState,
    *,
    default_max_iterations: int,
) -> ReActSessionState:
    """Mirror pattern counters into ``budget`` for cross-layer observability."""
    max_iters = state.max_react_iterations if state.max_react_iterations > 0 else default_max_iterations
    budget = state.budget or AcpBudgetState()
    updated_budget = budget.model_copy(
        update={
            "react_iterations_used": state.react_iterations_used,
            "react_iterations_max": max_iters,
        }
    )
    if updated_budget == state.budget:
        return state
    return state.model_copy(update={"budget": updated_budget})


def record_react_tool_calls(state: ReActSessionState, tool_calls: int) -> ReActSessionState:
    if tool_calls <= 0:
        return state
    budget = state.budget or AcpBudgetState()
    return state.model_copy(
        update={
            "budget": budget.model_copy(
                update={"tool_calls": budget.tool_calls + tool_calls},
            )
        }
    )
