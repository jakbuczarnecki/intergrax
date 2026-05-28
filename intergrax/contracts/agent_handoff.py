# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nexus-mediated agent handoff contract (architecture §42.15)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from intergrax.contracts.agent_decision import AgentDecision
    from intergrax.contracts.agent_execution_result import AgentExecutionResult


class AgentHandoff(BaseModel):
    """
    Structured transfer of responsibility between agents within one task.

    Emitted via ``AgentDecision`` (typically ``MODIFY_PLAN``) — Nexus executes the handoff.
    """

    handoff_id: str = Field(default_factory=lambda: f"ho_{uuid4().hex[:12]}")
    from_agent_id: str
    to_agent_id: Optional[str] = None
    to_capability: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    artifacts: List[str] = Field(default_factory=list)
    required_validation: List[str] = Field(default_factory=list)
    schema_version: str = "handoff.v1"

    @model_validator(mode="after")
    def _require_target(self) -> AgentHandoff:
        if not self.to_agent_id and not self.to_capability:
            raise ValueError("handoff requires to_agent_id or to_capability")
        return self


def handoff_from_decision(decision: AgentDecision) -> Optional[AgentHandoff]:
    """Extract a handoff request from an agent decision."""
    if decision.handoff is not None:
        return decision.handoff
    raw = decision.payload.get("handoff")
    if isinstance(raw, AgentHandoff):
        return raw
    if isinstance(raw, dict):
        return AgentHandoff.model_validate(raw)
    return None


def is_handoff_decision(decision: AgentDecision) -> bool:
    from intergrax.contracts.agent_decision import AgentDecisionType

    if handoff_from_decision(decision) is not None:
        return True
    return decision.type == AgentDecisionType.MODIFY_PLAN and bool(
        decision.payload.get("handoff")
    )


def resolve_handoff_from_execution(execution: AgentExecutionResult) -> Optional[AgentHandoff]:
    """Resolve pending handoff from ``AgentExecutionResult``."""
    if execution.agent_decision is not None:
        handoff = handoff_from_decision(execution.agent_decision)
        if handoff is not None:
            return handoff
    raw = (execution.structured_data or {}).get("pending_handoff")
    if isinstance(raw, AgentHandoff):
        return raw
    if isinstance(raw, dict):
        return AgentHandoff.model_validate(raw)
    return None
