# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


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


class RequestCountResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    read_request_count: int = Field(..., ge=0)
