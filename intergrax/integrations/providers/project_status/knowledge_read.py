# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_STATUS_PROVIDER_ID: Final[str] = "project_status"
PROJECT_STATUS_SOURCE_KIND: Final[str] = "project"


class ProjectStatusBlockerStatusV1(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class ProjectStatusBlockerV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(..., min_length=1, max_length=64)
    status: ProjectStatusBlockerStatusV1


class ProjectStatusSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(..., min_length=1, max_length=64)
    readiness_score: int = Field(..., ge=0, le=100)
    blockers: list[ProjectStatusBlockerV1] = Field(default_factory=list, max_length=32)
    status: str = Field(default="active", min_length=1, max_length=32)
    updated_at: datetime

    @field_validator("project_id")
    @classmethod
    def _validate_project_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or cleaned != value:
            raise ValueError("project_id_invalid")
        return cleaned


class ProjectStatusNotFoundError(LookupError):
    """Raised when the remote authority has no record for the project."""


class ProjectStatusReadError(RuntimeError):
    """Raised when the remote authority returns an invalid or failed response."""


class ProjectStatusReadClient:
    """Minimal async read client contract used by the live handler."""

    async def read_project_status(self, *, project_id: str) -> ProjectStatusSnapshotV1:
        raise NotImplementedError
