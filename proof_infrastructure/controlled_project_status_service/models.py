# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProjectBlockerStatusV1(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class ProjectBlockerV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(..., min_length=1, max_length=64)
    status: ProjectBlockerStatusV1


class ProjectStatusResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(..., min_length=1, max_length=64)
    readiness_score: int = Field(..., ge=0, le=100)
    blockers: list[ProjectBlockerV1] = Field(default_factory=list, max_length=32)
    status: str = Field(default="active", min_length=1, max_length=32)
    updated_at: datetime


class ProjectStatusControlUpdateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    readiness_score: int | None = Field(default=None, ge=0, le=100)
    blockers: list[ProjectBlockerV1] | None = Field(default=None, max_length=32)
    status: str | None = Field(default=None, min_length=1, max_length=32)

    @field_validator("blockers")
    @classmethod
    def _validate_blockers(
        cls,
        value: tuple[ProjectBlockerV1, ...] | None,
    ) -> tuple[ProjectBlockerV1, ...] | None:
        if value is None:
            return None
        seen: set[str] = set()
        for blocker in value:
            if blocker.id in seen:
                raise ValueError("duplicate_blocker_id")
            seen.add(blocker.id)
        return value


class RequestCountResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    read_request_count: int = Field(..., ge=0)
