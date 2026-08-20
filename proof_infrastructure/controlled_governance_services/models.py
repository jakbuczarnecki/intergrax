# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SecurityBlockerStatusV1(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class SecurityBlockerV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(..., min_length=1, max_length=64)
    status: SecurityBlockerStatusV1


class SecurityStatusResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(..., min_length=1, max_length=64)
    blockers: list[SecurityBlockerV1] = Field(default_factory=list, max_length=32)
    status: str = Field(default="clear", min_length=1, max_length=32)
    updated_at: datetime


class ChangeApprovalStateV1(StrEnum):
    APPROVED = "APPROVED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"


class ChangeApprovalResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    change_id: str = Field(..., min_length=1, max_length=64)
    approval_state: ChangeApprovalStateV1
    approved: bool
    updated_at: datetime


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


class RequestCountResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    security_read_count: int = Field(..., ge=0)
    change_read_count: int = Field(..., ge=0)
    governance_read_count: int = Field(..., ge=0)
