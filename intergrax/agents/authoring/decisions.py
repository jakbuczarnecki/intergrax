# © Artur Czarnecki. All rights reserved.

"""
Author decision helpers (ACP-7).

Primary surface: ``StepOutcome`` factories (architecture §32.0.4, ACP-DX-6).
Legacy UAEP ``AgentDecision`` helpers remain for ``decide_after_step`` bridge code
only — they emit :class:`DeprecationWarning` and map through :func:`to_step_outcome`.
"""

from __future__ import annotations

import warnings
from typing import Any

from intergrax.agents.authoring.step_outcome import StepOutcome, StateDelta
from intergrax.agents.authoring.uaep_step_bridge import agent_decision_to_step_outcome
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType, PlanDelta
from intergrax.contracts.agent_handoff import AgentHandoff
from intergrax.contracts.agent_run import AgentRunError
from intergrax.contracts.agent_run_enums import TerminalReason
from intergrax.contracts.agent_step import StepOutput

_UAEP_DEPRECATION = (
    "UAEP AgentDecision helpers are deprecated; use StepOutcome factories "
    "(continue_with, finish, fail_step, pause_for_human, request_replan, delegate_handoff) "
    "in on_next_step — see architecture §32.0.4."
)


def continue_with(
    state_delta: StateDelta | None = None,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> StepOutcome:
    """Continue the typed step loop after applying ``state_delta`` (§32.0.4)."""
    return StepOutcome.continue_with(state_delta, diagnostics=diagnostics)


def finish(
    output: str | dict[str, Any],
    *,
    terminal_reason: TerminalReason = TerminalReason.GOAL_MET,
    state_delta: StateDelta | None = None,
    confidence: float | None = None,
    artifacts: list[dict[str, Any]] | None = None,
) -> StepOutcome:
    """Terminate the run with a successful output (§32.0.4)."""
    return StepOutcome.complete(
        output,
        terminal_reason=terminal_reason,
        state_delta=state_delta,
        confidence=confidence,
        artifacts=artifacts,
    )


def fail_step(
    errors: list[AgentRunError],
    *,
    terminal_reason: TerminalReason = TerminalReason.ERROR,
    state_delta: StateDelta | None = None,
    is_terminal: bool = True,
) -> StepOutcome:
    """Terminate or signal failure with structured errors (§32.0.4)."""
    return StepOutcome.fail(
        errors,
        terminal_reason=terminal_reason,
        state_delta=state_delta,
        is_terminal=is_terminal,
    )


def pause_for_human(
    reason: str,
    *,
    governance_snapshot: dict[str, Any] | None = None,
    state_delta: StateDelta | None = None,
) -> StepOutcome:
    """Pause for HITL without ending the session (§32.0.4)."""
    return StepOutcome.pause_hitl(
        reason,
        governance_snapshot=governance_snapshot,
        state_delta=state_delta,
    )


def request_replan(
    state_delta: StateDelta | None = None,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> StepOutcome:
    """End this run so Nexus may schedule a replanned run (§32.0.4)."""
    return StepOutcome.replan(state_delta, diagnostics=diagnostics)


def delegate_handoff(
    to_agent_id: str,
    *,
    from_agent_id: str,
    reason: str = "delegate",
    payload: dict[str, Any] | None = None,
    state_delta: StateDelta | None = None,
) -> StepOutcome:
    """Request delegation via structured handoff embedded in a replan outcome."""
    handoff = AgentHandoff(
        from_agent_id=from_agent_id,
        to_agent_id=to_agent_id,
        reason=reason,
        payload=dict(payload or {}),
    )
    diagnostics = {"handoff": handoff.model_dump(mode="json")}
    return StepOutcome.replan(state_delta, diagnostics=diagnostics)


def to_step_outcome(
    decision: AgentDecision,
    output: StepOutput | None = None,
) -> StepOutcome:
    """Map legacy UAEP ``AgentDecision`` to typed ``StepOutcome`` (ACP-STEP-3 bridge)."""
    return agent_decision_to_step_outcome(decision, output)


def _warn_uaep_helper(name: str) -> None:
    warnings.warn(f"{name}() — {_UAEP_DEPRECATION}", DeprecationWarning, stacklevel=3)


def complete(*, reason: str = "step finished") -> AgentDecision:
    """Deprecated — use :func:`finish` or ``StepOutcome.complete`` in ``on_next_step``."""
    _warn_uaep_helper("complete")
    return AgentDecision(type=AgentDecisionType.COMPLETE, reason=reason)


def continue_to(step_id: str, *, reason: str = "continue") -> AgentDecision:
    """Deprecated — use :func:`continue_with` in ``on_next_step``."""
    _warn_uaep_helper("continue_to")
    return AgentDecision(
        type=AgentDecisionType.CONTINUE,
        reason=reason,
        payload={"next_step_id": step_id},
    )


def delegate_to(
    agent_id: str,
    *,
    from_agent_id: str = "",
    reason: str = "delegate",
) -> AgentDecision:
    """Deprecated — use :func:`delegate_handoff` or ``AgentHandoff`` via replan."""
    _warn_uaep_helper("delegate_to")
    handoff = AgentHandoff(
        from_agent_id=from_agent_id or "unknown",
        to_agent_id=agent_id,
        reason=reason,
    )
    return AgentDecision(
        type=AgentDecisionType.MODIFY_PLAN,
        reason=reason,
        handoff=handoff,
        suggested_plan_delta=PlanDelta(description=reason),
    )
