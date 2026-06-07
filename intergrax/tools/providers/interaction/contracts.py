# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class InteractionListSessionsInput(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    limit: int = Field(default=20, ge=1, le=100)


class InteractionSessionOutput(BaseModel):
    session_id: str
    user_id: str = ""
    tenant_id: str = ""
    updated_at_utc: str = ""


class InteractionListSessionsOutput(BaseModel):
    used: bool = False
    sessions: list[InteractionSessionOutput] = Field(default_factory=list)
    total: int = 0
    reason: str = ""


class InteractionGetLastInputInput(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)


class InteractionGetLastInputOutput(BaseModel):
    used: bool = False
    found: bool = False
    message: str = ""
    reason: str = ""
