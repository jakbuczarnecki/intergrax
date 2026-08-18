# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical agent control-flow decisions (architecture §42.7)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from intergrax.contracts.agent_handoff import AgentHandoff
from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.governed_continuation_correlation import GovernedContinuationCorrelation


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


class RetryHint(BaseModel):
    retryable: bool = True
    reason: str = ""
    backoff_ms: Optional[int] = None
    max_attempts: Optional[int] = None


class HumanRequestUrgency(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


HUMAN_REQUEST_TIMEOUT_DEFAULTS = frozenset(
    {
        AgentDecisionType.FAIL,
        AgentDecisionType.ESCALATE,
        AgentDecisionType.CANCEL,
    }
)


class HumanRequest(BaseModel):
    schema_version: str = "human_request.v2"
    request_id: str
    prompt: str
    options: list[str] = Field(default_factory=list)
    context_artifacts: list[str] = Field(default_factory=list)
    urgency: HumanRequestUrgency = HumanRequestUrgency.NORMAL
    timeout_seconds: Optional[int] = None
    default_on_timeout: Optional[AgentDecisionType] = None
    governed_continuation: Optional[GovernedContinuationCorrelation] = None

    @field_validator("urgency", mode="before")
    @classmethod
    def _normalize_urgency(cls, value: object) -> HumanRequestUrgency:
        if isinstance(value, HumanRequestUrgency):
            return value
        raw = str(value or HumanRequestUrgency.NORMAL.value).strip().lower()
        try:
            return HumanRequestUrgency(raw)
        except ValueError:
            return HumanRequestUrgency.NORMAL

    @field_validator("timeout_seconds")
    @classmethod
    def _validate_timeout_seconds(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        return value

    @model_validator(mode="after")
    def _validate_timeout_default(self) -> HumanRequest:
        if self.default_on_timeout is not None:
            if self.default_on_timeout not in HUMAN_REQUEST_TIMEOUT_DEFAULTS:
                allowed = ", ".join(sorted(d.value for d in HUMAN_REQUEST_TIMEOUT_DEFAULTS))
                raise ValueError(
                    f"default_on_timeout must be one of: {allowed}"
                )
            if self.timeout_seconds is None:
                raise ValueError("default_on_timeout requires timeout_seconds")
        return self


def human_request_fields_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract v2 HumanRequest fields from AgentDecision.payload."""
    fields: Dict[str, Any] = {}
    if "urgency" in payload:
        fields["urgency"] = payload["urgency"]
    if payload.get("timeout_seconds") is not None:
        fields["timeout_seconds"] = int(payload["timeout_seconds"])
    raw_default = payload.get("default_on_timeout")
    if raw_default is not None:
        try:
            fields["default_on_timeout"] = AgentDecisionType(str(raw_default))
        except ValueError:
            pass
    return fields


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
    handoff: Optional[AgentHandoff] = None
    schema_version: str = "agent_decision.v1"

    @model_validator(mode="before")
    @classmethod
    def _coerce_handoff_from_payload(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("handoff") is None:
            payload = data.get("payload")
            if isinstance(payload, dict) and payload.get("handoff") is not None:
                data = dict(data)
                data["handoff"] = payload["handoff"]
        return data
