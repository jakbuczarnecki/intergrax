# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

GOVERNANCE_APPROVAL_PROVIDER_ID: Final[str] = "governance_approval"
GOVERNANCE_APPROVAL_SOURCE_KIND: Final[str] = "approval"


class GovernanceDecisionStateV1(StrEnum):
    APPROVED = "APPROVED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"


class GovernanceApprovalSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subject_id: str = Field(..., min_length=1, max_length=64)
    decision_state: GovernanceDecisionStateV1
    approved: bool
    updated_at: datetime

    @field_validator("subject_id")
    @classmethod
    def _validate_subject_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or cleaned != value:
            raise ValueError("subject_id_invalid")
        return cleaned


class GovernanceApprovalNotFoundError(LookupError):
    """Raised when the remote authority has no record for the subject."""


class GovernanceApprovalReadError(RuntimeError):
    """Raised when the remote authority returns an invalid or failed response."""


class GovernanceApprovalReadClient:
    """Minimal async read client contract used by the live handler."""

    async def read_governance_approval(
        self,
        *,
        subject_id: str,
    ) -> GovernanceApprovalSnapshotV1:
        raise NotImplementedError
