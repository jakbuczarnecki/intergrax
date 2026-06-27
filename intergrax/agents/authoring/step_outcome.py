# © Artur Czarnecki. All rights reserved.

"""StepOutcome factories for readable author control flow (architecture §32.0.4)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.agents.authoring.diagnostic_serialization import merge_diagnostic_payloads
from intergrax.contracts.agent_run import AgentRunError
from intergrax.contracts.agent_run_enums import StepNextAction, TerminalReason
from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload

StateDelta = dict[str, Any]


class StepOutcome(BaseModel):
    """Author decision for one harness iteration (architecture §32.3)."""

    model_config = ConfigDict(extra="forbid")

    is_terminal: bool = False
    terminal_reason: TerminalReason | None = None
    output: str | dict[str, Any] | None = None
    state_delta: StateDelta = Field(default_factory=dict)
    next_action: StepNextAction = StepNextAction.CONTINUE
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float | None = None
    errors: list[AgentRunError] = Field(default_factory=list)
    diagnostics: dict[str, Any] | None = None
    requested_actions: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def _terminal_reason_when_terminal(self) -> StepOutcome:
        if self.is_terminal and self.terminal_reason is None:
            raise ValueError("terminal_reason is required when is_terminal is true")
        return self

    @classmethod
    def continue_with(
        cls,
        state_delta: StateDelta | None = None,
        *,
        diagnostics: dict[str, Any] | None = None,
        diagnostic_payloads: Sequence[DiagnosticPayload] | None = None,
    ) -> StepOutcome:
        merged = merge_diagnostic_payloads(diagnostics, diagnostic_payloads or ())
        return cls(
            is_terminal=False,
            next_action=StepNextAction.CONTINUE,
            state_delta=dict(state_delta or {}),
            diagnostics=merged or None,
        )

    @classmethod
    def complete(
        cls,
        output: str | dict[str, Any],
        *,
        terminal_reason: TerminalReason = TerminalReason.GOAL_MET,
        state_delta: StateDelta | None = None,
        confidence: float | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        diagnostics: dict[str, Any] | None = None,
        diagnostic_payloads: Sequence[DiagnosticPayload] | None = None,
    ) -> StepOutcome:
        merged = merge_diagnostic_payloads(diagnostics, diagnostic_payloads or ())
        return cls(
            is_terminal=True,
            terminal_reason=terminal_reason,
            output=output,
            state_delta=dict(state_delta or {}),
            confidence=confidence,
            artifacts=list(artifacts or []),
            diagnostics=merged or None,
        )

    @classmethod
    def fail(
        cls,
        errors: list[AgentRunError],
        *,
        terminal_reason: TerminalReason = TerminalReason.ERROR,
        state_delta: StateDelta | None = None,
        is_terminal: bool = True,
        diagnostics: dict[str, Any] | None = None,
        diagnostic_payloads: Sequence[DiagnosticPayload] | None = None,
    ) -> StepOutcome:
        merged = merge_diagnostic_payloads(diagnostics, diagnostic_payloads or ())
        return cls(
            is_terminal=is_terminal,
            terminal_reason=terminal_reason,
            next_action=StepNextAction.FAIL,
            errors=list(errors),
            state_delta=dict(state_delta or {}),
            diagnostics=merged or None,
        )

    @classmethod
    def pause_hitl(
        cls,
        reason: str,
        *,
        governance_snapshot: dict[str, Any] | None = None,
        state_delta: StateDelta | None = None,
        diagnostic_payloads: Sequence[DiagnosticPayload] | None = None,
    ) -> StepOutcome:
        _ = governance_snapshot
        merged = merge_diagnostic_payloads({"pause_reason": reason}, diagnostic_payloads or ())
        return cls(
            is_terminal=False,
            terminal_reason=TerminalReason.HUMAN_REQUIRED,
            next_action=StepNextAction.PAUSE_HITL,
            state_delta=dict(state_delta or {}),
            diagnostics=merged or None,
        )

    @classmethod
    def replan(
        cls,
        state_delta: StateDelta | None = None,
        *,
        diagnostics: dict[str, Any] | None = None,
        diagnostic_payloads: Sequence[DiagnosticPayload] | None = None,
    ) -> StepOutcome:
        merged = merge_diagnostic_payloads(diagnostics, diagnostic_payloads or ())
        return cls(
            is_terminal=True,
            terminal_reason=TerminalReason.REPLANNED,
            next_action=StepNextAction.REPLAN,
            state_delta=dict(state_delta or {}),
            diagnostics=merged or None,
        )
