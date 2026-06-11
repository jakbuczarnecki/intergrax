# © Artur Czarnecki. All rights reserved.

"""Side-effect mode enforcement (architecture §32.8 · ACP-CON-3)."""

from __future__ import annotations

from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_run_enums import SideEffectMode


def validate_side_effect_mode(
    outcome: StepOutcome,
    mode: SideEffectMode,
) -> str | None:
    """
    Return an error message when outcome conflicts with side_effect_mode.

    IMMEDIATE: authors invoke gateways inside on_next_step — no requested_actions.
    DECLARATIVE: kernel executes requested_actions — no immediate tool hints.
    """
    has_actions = bool(outcome.requested_actions)
    immediate_hint = bool(
        outcome.diagnostics and outcome.diagnostics.get("immediate_tool_calls")
    )
    if mode == SideEffectMode.IMMEDIATE and has_actions:
        return "requested_actions are not allowed when side_effect_mode is immediate"
    if mode == SideEffectMode.DECLARATIVE and immediate_hint:
        return "immediate_tool_calls are not allowed when side_effect_mode is declarative"
    return None
