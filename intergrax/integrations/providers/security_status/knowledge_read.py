# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

SECURITY_STATUS_PROVIDER_ID: Final[str] = "security_status"
SECURITY_STATUS_SOURCE_KIND: Final[str] = "security"


class SecurityBlockerStatusV1(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class SecurityBlockerV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(..., min_length=1, max_length=64)
    status: SecurityBlockerStatusV1


class SecurityStatusSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(..., min_length=1, max_length=64)
    blockers: list[SecurityBlockerV1] = Field(default_factory=list, max_length=32)
    status: str = Field(default="clear", min_length=1, max_length=32)
    updated_at: datetime

    @field_validator("project_id")
    @classmethod
    def _validate_project_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or cleaned != value:
            raise ValueError("project_id_invalid")
        return cleaned


class SecurityStatusNotFoundError(LookupError):
    """Raised when the remote authority has no record for the project."""


class SecurityStatusReadError(RuntimeError):
    """Raised when the remote authority returns an invalid or failed response."""


class SecurityStatusReadClient:
    """Minimal async read client contract used by the live handler."""

    async def read_security_status(self, *, project_id: str) -> SecurityStatusSnapshotV1:
        raise NotImplementedError
