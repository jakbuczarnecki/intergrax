# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class GovernanceDecisionStateV1(StrEnum):
    APPROVED = "APPROVED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"


class GovernanceApprovalResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subject_id: str = Field(..., min_length=1, max_length=64)
    decision_state: GovernanceDecisionStateV1
    approved: bool
    updated_at: datetime
    valid_from: datetime | None = None
    valid_until: datetime | None = None


class RequestCountResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    read_request_count: int = Field(..., ge=0)
