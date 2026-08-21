# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    @field_validator("subject_id")
    @classmethod
    def _validate_subject_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or cleaned != value:
            raise ValueError("subject_id_invalid")
        return cleaned

    @field_validator("updated_at", "valid_from", "valid_until")
    @classmethod
    def _timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("governance_approval_timestamp_must_be_timezone_aware")
        return value

    @model_validator(mode="after")
    def _validate_validity_pair(self) -> GovernanceApprovalSnapshotV1:
        if (self.valid_from is None) ^ (self.valid_until is None):
            raise ValueError("governance_validity_interval_incomplete")
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until < self.valid_from
        ):
            raise ValueError("governance_validity_interval_invalid")
        return self


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
