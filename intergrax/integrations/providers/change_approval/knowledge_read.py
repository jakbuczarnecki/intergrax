# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

CHANGE_APPROVAL_PROVIDER_ID: Final[str] = "change_approval"
CHANGE_APPROVAL_SOURCE_KIND: Final[str] = "change"


class ChangeApprovalStateV1(StrEnum):
    APPROVED = "APPROVED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"


class ChangeApprovalSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    change_id: str = Field(..., min_length=1, max_length=64)
    approval_state: ChangeApprovalStateV1
    approved: bool
    updated_at: datetime

    @field_validator("change_id")
    @classmethod
    def _validate_change_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or cleaned != value:
            raise ValueError("change_id_invalid")
        return cleaned


class ChangeApprovalNotFoundError(LookupError):
    """Raised when the remote authority has no record for the change."""


class ChangeApprovalReadError(RuntimeError):
    """Raised when the remote authority returns an invalid or failed response."""


class ChangeApprovalReadClient:
    """Minimal async read client contract used by the live handler."""

    async def read_change_approval(self, *, change_id: str) -> ChangeApprovalSnapshotV1:
        raise NotImplementedError
