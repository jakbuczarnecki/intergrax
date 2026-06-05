# © Artur Czarnecki. All rights reserved.

"""Agent lifecycle governance contracts for deprecation and retirement (Phase V-ALG.3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field

from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState


class AgentLifecycleTransitionRequest(BaseModel):
    agent_id: str
    agent_version: str
    current_state: AgentLifecycleState
    target_state: AgentLifecycleState
    requested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    migration_window_days: int = 30
    migration_guide_ref: str = ""
    deprecation_notice_ref: str = ""


class AgentLifecycleDecision(BaseModel):
    approved: bool
    reasons: list[str] = Field(default_factory=list)


def evaluate_agent_lifecycle_transition(
    request: AgentLifecycleTransitionRequest,
) -> AgentLifecycleDecision:
    reasons: list[str] = []
    if not _is_allowed_transition(request.current_state, request.target_state):
        reasons.append(
            "Unsupported lifecycle transition: "
            f"{request.current_state.value} -> {request.target_state.value}"
        )

    if request.target_state == AgentLifecycleState.DEPRECATED:
        if request.migration_window_days < 14:
            reasons.append("Migration window must be at least 14 days for deprecation")
        if not request.migration_guide_ref:
            reasons.append("Deprecation requires migration guide reference")
        if not request.deprecation_notice_ref:
            reasons.append("Deprecation requires notice reference")

    if request.target_state == AgentLifecycleState.RETIRED:
        if request.current_state != AgentLifecycleState.DEPRECATED:
            reasons.append("Retirement requires deprecated state before retirement")
        if not request.migration_guide_ref:
            reasons.append("Retirement requires migration guide reference")

    return AgentLifecycleDecision(approved=not reasons, reasons=reasons)


def compute_deprecation_deadline(request: AgentLifecycleTransitionRequest) -> datetime:
    return request.requested_at + timedelta(days=request.migration_window_days)


def _is_allowed_transition(current: AgentLifecycleState, target: AgentLifecycleState) -> bool:
    allowed: set[tuple[AgentLifecycleState, AgentLifecycleState]] = {
        (AgentLifecycleState.DEVELOPMENT, AgentLifecycleState.STAGING),
        (AgentLifecycleState.STAGING, AgentLifecycleState.PRODUCTION),
        (AgentLifecycleState.PRODUCTION, AgentLifecycleState.DEPRECATED),
        (AgentLifecycleState.DEPRECATED, AgentLifecycleState.RETIRED),
    }
    return (current, target) in allowed
