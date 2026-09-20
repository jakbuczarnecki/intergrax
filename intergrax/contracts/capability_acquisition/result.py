# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability realization coordination result (UCA-2)."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_acquisition.evidence import (
    CapabilityRealizationEvidence,
)
from intergrax.contracts.capability_acquisition.outcome import (
    CapabilityRealizationOutcome,
)
from intergrax.contracts.capability_acquisition.reason_code import (
    CapabilityRealizationReasonCode,
)
from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey

SCHEMA_CAPABILITY_REALIZATION_RESULT_V1: Final = "capability_realization_result.v1"
_NON_EMPTY = Field(min_length=1)


class CapabilityRealizationResult(BaseModel):
    """Immutable auditable realization outcome — not a loose success flag."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_realization_result.v1"] = (
        SCHEMA_CAPABILITY_REALIZATION_RESULT_V1
    )
    request_id: str = _NON_EMPTY
    realization_need_id: str = _NON_EMPTY
    provider_id: str | None = None
    outcome: CapabilityRealizationOutcome
    reason_code: CapabilityRealizationReasonCode
    capability_identity: CapabilityIdentityKey
    started_at: datetime
    completed_at: datetime
    evidence: CapabilityRealizationEvidence | None = None
    reason_detail: str = ""

    @field_validator("request_id", "realization_need_id", "provider_id", "reason_detail")
    @classmethod
    def _validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value == "":
            return ""
        return require_non_empty_text(value, label="text")

    @field_validator("started_at", "completed_at")
    @classmethod
    def _validate_timestamps(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def _validate_success_evidence(self) -> CapabilityRealizationResult:
        if self.outcome is CapabilityRealizationOutcome.SUCCEEDED:
            if self.evidence is None or self.evidence.availability_evidence is None:
                raise ValueError(
                    "SUCCEEDED requires availability evidence for canonical projection",
                )
            if self.provider_id is None:
                raise ValueError("SUCCEEDED requires provider_id")
        return self


__all__ = [
    "SCHEMA_CAPABILITY_REALIZATION_RESULT_V1",
    "CapabilityRealizationResult",
]
