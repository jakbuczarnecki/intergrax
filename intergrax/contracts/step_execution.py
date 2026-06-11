# © Artur Czarnecki. All rights reserved.

"""Harness step execution record (architecture §32 · §38)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_run_enums import AgentRunErrorCode
from intergrax.contracts.runtime_policy import PolicyDecision


class AgentStepRecord(BaseModel):
    """Plane B step journal entry stub — expanded in ACP-OBS-1."""

    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(default=0, ge=0)
    next_action: str = ""
    is_terminal: bool = False
    terminal_reason: str | None = None
    policy_pre: PolicyDecision | None = None
    policy_post: PolicyDecision | None = None
    state_version: int | None = None
    error_code: AgentRunErrorCode | None = None
    diagnostics: dict[str, Any] | None = None


class StepExecutionRecord(BaseModel):
    """Result of HarnessKernel.execute_step — harness-owned, not author-facing."""

    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(default=0, ge=0)
    outcome_applied: bool = False
    policy_pre: PolicyDecision | None = None
    policy_post: PolicyDecision | None = None
    state_version: int | None = None
    error_code: AgentRunErrorCode | None = None
    budget_exceeded: bool = False
    side_effect_mode_violation: bool = False
    trace_event_count: int = 0
    step_record: AgentStepRecord | None = None
