# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class HitlListPendingInput(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    limit: int = Field(default=50, ge=1, le=200)


class HitlDecisionOutput(BaseModel):
    decision_id: str
    task_id: str
    tenant_id: str
    user_id: str = ""
    human_request_id: str = ""
    verdict: str
    response_text: str = ""
    escalation_level: int = 0
    escalation_target: str = ""
    agent_id: str = ""
    run_id: str = ""
    notes: str = ""
    created_at_utc: str = ""


class HitlListPendingOutput(BaseModel):
    used: bool = False
    decisions: list[HitlDecisionOutput] = Field(default_factory=list)
    total: int = 0
    reason: str = ""


class HitlGetDecisionInput(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    decision_id: str = Field(..., min_length=1)


class HitlGetDecisionOutput(BaseModel):
    used: bool = False
    found: bool = False
    decision: HitlDecisionOutput | None = None
    reason: str = ""


class HitlSummarizeQueueInput(BaseModel):
    tenant_id: str = Field(..., min_length=1)


class HitlSummarizeQueueOutput(BaseModel):
    used: bool = False
    counts_by_verdict: dict[str, int] = Field(default_factory=dict)
    pending_escalations: int = 0
    reason: str = ""


class HitlSubmitResponseInput(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    user_id: str = Field(default="")
    verdict: str = Field(..., min_length=1, description="approve | reject | escalate | unknown")
    response_text: str = Field(default="")
    human_request_id: str = Field(default="")
    escalation_level: int = Field(default=0, ge=0)
    escalation_target: str = Field(default="")
    agent_id: str = Field(default="")
    run_id: str = Field(default="")
    notes: str = Field(default="")


class HitlSubmitResponseOutput(BaseModel):
    used: bool = False
    recorded: bool = False
    decision: HitlDecisionOutput | None = None
    reason: str = ""


class HitlListForTaskInput(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)


class HitlListForTaskOutput(BaseModel):
    used: bool = False
    decisions: list[HitlDecisionOutput] = Field(default_factory=list)
    total: int = 0
    reason: str = ""
