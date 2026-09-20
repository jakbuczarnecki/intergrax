# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability acquisition coordination result (UCA-3)."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    CapabilityAcquisitionEvidence,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_success_evidence import (
    validate_acquisition_success_evidence,
)
from intergrax.contracts.capability_catalog._validation import require_non_empty_text

SCHEMA_CAPABILITY_ACQUISITION_RESULT_V1: Final = "capability_acquisition_result.v1"
_NON_EMPTY = Field(min_length=1)


class CapabilityAcquisitionResult(BaseModel):
    """Immutable auditable acquisition outcome — not qualification or execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_acquisition_result.v1"] = (
        SCHEMA_CAPABILITY_ACQUISITION_RESULT_V1
    )
    request_id: str = _NON_EMPTY
    gap_id: str = _NON_EMPTY
    strategy_id: str | None = None
    outcome: CapabilityAcquisitionOutcome
    reason_code: CapabilityAcquisitionReasonCode
    started_at: datetime
    completed_at: datetime
    evidence: CapabilityAcquisitionEvidence | None = None
    reason_detail: str = ""
    correlation_id: str | None = None
    causation_id: str | None = None

    @field_validator(
        "request_id",
        "gap_id",
        "strategy_id",
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
    def _validate_success_evidence(self) -> CapabilityAcquisitionResult:
        if self.outcome is CapabilityAcquisitionOutcome.SUCCEEDED:
            validate_acquisition_success_evidence(self.evidence)
            if self.strategy_id is None:
                raise ValueError("SUCCEEDED requires strategy_id")
        return self


__all__ = [
    "SCHEMA_CAPABILITY_ACQUISITION_RESULT_V1",
    "CapabilityAcquisitionResult",
]
