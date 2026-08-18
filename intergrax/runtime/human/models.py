# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Human decision and escalation contracts (Phase F.3, §42.38)."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.utils.time_provider import SystemTimeProvider


class HumanResponseVerdict(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    UNKNOWN = "unknown"


class EscalationTarget(str, Enum):
    HUMAN_OPERATOR = "human_operator"
    APPLICATION_ADMIN = "application_admin"
    FAIL_TASK = "fail_task"


class HumanDecisionRecord(BaseModel):
    decision_id: str = Field(default_factory=lambda: f"hdec_{uuid4().hex[:16]}")
    task_id: str
    tenant_id: str
    user_id: str = ""
    human_request_id: str = ""
    verdict: HumanResponseVerdict
    response_text: str = ""
    escalation_level: int = 0
    escalation_target: Optional[EscalationTarget] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    notes: str = ""
    created_at_utc: str


class EscalationOutcome(BaseModel):
    target: EscalationTarget
    level: int
    message: str = ""
    fail_task: bool = False


def build_human_decision_record(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    verdict: HumanResponseVerdict,
    response_text: str,
    human_request_id: str = "",
    escalation_level: int = 0,
    escalation_target: Optional[EscalationTarget] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    notes: str = "",
) -> HumanDecisionRecord:
    """Vendor-neutral factory for persisted human decision records."""
    return HumanDecisionRecord(
        task_id=task_id,
        tenant_id=tenant_id,
        user_id=user_id,
        human_request_id=human_request_id,
        verdict=verdict,
        response_text=response_text,
        escalation_level=escalation_level,
        escalation_target=escalation_target,
        agent_id=agent_id,
        run_id=run_id,
        notes=notes,
        created_at_utc=SystemTimeProvider.utc_now().isoformat(),
    )
