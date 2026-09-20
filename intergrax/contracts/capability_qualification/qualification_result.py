# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability qualification coordination result (UCA-4)."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_qualification.qualification_evidence import (
    CapabilityQualificationEvidence,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_reason_code import (
    CapabilityQualificationReasonCode,
)
from intergrax.contracts.capability_qualification.qualification_success_evidence import (
    validate_qualification_success_evidence,
)

SCHEMA_CAPABILITY_QUALIFICATION_RESULT_V1: Final = "capability_qualification_result.v1"
_NON_EMPTY = Field(min_length=1)


class CapabilityQualificationResult(BaseModel):
    """Immutable auditable qualification outcome — not lifecycle execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_qualification_result.v1"] = (
        SCHEMA_CAPABILITY_QUALIFICATION_RESULT_V1
    )
    qualification_request_id: str = _NON_EMPTY
    acquisition_request_id: str = _NON_EMPTY
    gap_id: str = _NON_EMPTY
    strategy_id: str = _NON_EMPTY
    provider_id: str | None = None
    outcome: CapabilityQualificationOutcome
    reason_code: CapabilityQualificationReasonCode
    started_at: datetime
    completed_at: datetime
    evidence: CapabilityQualificationEvidence | None = None
    reason_detail: str = ""
    correlation_id: str | None = None
    causation_id: str | None = None

    @field_validator(
        "qualification_request_id",
        "acquisition_request_id",
        "gap_id",
        "strategy_id",
        "provider_id",
        "correlation_id",
        "causation_id",
        "reason_detail",
    )
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
    def _validate_success_evidence(self) -> CapabilityQualificationResult:
        if self.outcome is CapabilityQualificationOutcome.QUALIFIED:
            validate_qualification_success_evidence(self.evidence)
            if self.provider_id is None:
                raise ValueError("QUALIFIED requires provider_id")
        return self


__all__ = [
    "SCHEMA_CAPABILITY_QUALIFICATION_RESULT_V1",
    "CapabilityQualificationResult",
]
