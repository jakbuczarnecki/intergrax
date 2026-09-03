# © Artur Czarnecki. All rights reserved.

"""Typed contracts for declarative policy REQUIRE_HITL bridge (ADR-PLATFORM-PLUGIN-001)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class DeclarativePolicyHitlSignal:
    """Bridge transport from tool boundary to orchestration. Not persisted as authoritative state."""

    invocation_scope_id: str
    task_id: str
    run_id: str
    step_id: str
    tool_id: str
    agent_id: str
    idempotency_key: str | None
    matched_rule_ids: tuple[str, ...]
    policy_provenance_digest: str | None
    reasons: tuple[str, ...]


class DeclarativeHitlPendingApproval(BaseModel):
    """Authoritative pre-approval scope for a paused REQUIRE_HITL invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    invocation_scope_id: str
    task_id: str
    run_id: str
    step_id: str
    tool_id: str
    idempotency_key: str | None
    matched_rule_ids: tuple[str, ...]
    human_request_id: str
    policy_provenance_digest: str | None
    agent_id: str
    pause_id: str
    created_at: str


class DeclarativeHitlApprovalGrant(BaseModel):
    """Single-use approval artifact created only from persisted pending after APPROVE."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grant_id: str
    invocation_scope_id: str
    task_id: str
    run_id: str
    step_id: str
    tool_id: str
    agent_id: str
    idempotency_key: str | None
    matched_rule_ids: tuple[str, ...]
    human_request_id: str
    policy_provenance_digest: str | None
    pause_id: str
    approved_at: str


class DeclarativeHitlDecisionPayload(BaseModel):
    """Optional mirror on AgentDecision.payload for UAEP diagnostics only."""

    model_config = ConfigDict(extra="forbid")

    invocation_scope_id: str
    task_id: str
    run_id: str
    step_id: str
    tool_id: str
    agent_id: str
    matched_rule_ids: tuple[str, ...] = Field(default_factory=tuple)
    policy_provenance_digest: str | None = None

    @classmethod
    def from_signal(cls, signal: DeclarativePolicyHitlSignal) -> DeclarativeHitlDecisionPayload:
        return cls(
            invocation_scope_id=signal.invocation_scope_id,
            task_id=signal.task_id,
            run_id=signal.run_id,
            step_id=signal.step_id,
            tool_id=signal.tool_id,
            agent_id=signal.agent_id,
            matched_rule_ids=signal.matched_rule_ids,
            policy_provenance_digest=signal.policy_provenance_digest,
        )

    def to_audit_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
