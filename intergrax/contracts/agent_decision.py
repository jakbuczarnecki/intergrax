# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical agent control-flow decisions (architecture §42.7)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AgentDecisionType(str, Enum):
    CONTINUE = "continue"
    COMPLETE = "complete"
    RETRY = "retry"
    REQUEST_HUMAN = "request_human"
    INTERRUPT = "interrupt"
    ESCALATE = "escalate"
    MODIFY_PLAN = "modify_plan"
    FAIL = "fail"
    CANCEL = "cancel"


class EventSeverity(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RetryHint(BaseModel):
    retryable: bool = True
    reason: str = ""
    backoff_ms: Optional[int] = None
    max_attempts: Optional[int] = None


class HumanRequest(BaseModel):
    request_id: str
    prompt: str
    options: list[str] = Field(default_factory=list)
    context_artifacts: list[str] = Field(default_factory=list)
    urgency: str = "normal"
    timeout_seconds: Optional[int] = None
    default_on_timeout: Optional[AgentDecisionType] = None


class PlanDelta(BaseModel):
    """Suggested plan change; interpreted by Nexus (§42.7)."""

    description: str = ""
    add_nodes: list[Dict[str, Any]] = Field(default_factory=list)
    remove_node_ids: list[str] = Field(default_factory=list)


class AgentDecision(BaseModel):
    """
    Formal control-flow intent from an agent to Nexus.

    Agents MUST NOT pause the runtime or perform side effects directly —
    they return ``AgentDecision`` instances only (§42.7, §42.41).
    """

    type: AgentDecisionType
    reason: str = ""
    severity: EventSeverity = EventSeverity.INFO
    payload: Dict[str, Any] = Field(default_factory=dict)
    interrupt_id: Optional[str] = None
    suggested_plan_delta: Optional[PlanDelta] = None
    human_request: Optional[HumanRequest] = None
    retry_hint: Optional[RetryHint] = None
    confidence: Optional[float] = None
    schema_version: str = "agent_decision.v1"
