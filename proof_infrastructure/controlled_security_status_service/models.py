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


class RequestCountResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    read_request_count: int = Field(..., ge=0)


class SecurityStatusReadBehaviorV1(StrEnum):
    NORMAL = "normal"
    HTTP_503 = "http_503"
    MALFORMED_JSON = "malformed_json"


class SecurityStatusReadBehaviorControlV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    behavior: SecurityStatusReadBehaviorV1
