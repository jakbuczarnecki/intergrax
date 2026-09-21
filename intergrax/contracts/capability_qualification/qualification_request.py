# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability qualification coordination request (UCA-4)."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_catalog._validation import require_non_empty_text

SCHEMA_CAPABILITY_QUALIFICATION_REQUEST_V1: Final = (
    "capability_qualification_request.v1"
)
_NON_EMPTY = Field(min_length=1)


def derive_capability_qualification_request_id(
    *,
    acquisition_request_id: str,
    qualification_nonce: str,
) -> str:
    """Deterministic qualification identity — distinct from acquisition request id."""
    normalized_acquisition = require_non_empty_text(
        acquisition_request_id,
        label="acquisition_request_id",
    )
    normalized_nonce = require_non_empty_text(
        qualification_nonce,
        label="qualification_nonce",
    )
    return (
        f"capability-qualification-request:{normalized_acquisition}:{normalized_nonce}"
    )


class CapabilityQualificationRequest(BaseModel):
    """Qualification dispatch envelope — only successful acquisition handoff."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_qualification_request.v1"] = (
        SCHEMA_CAPABILITY_QUALIFICATION_REQUEST_V1
    )
    qualification_request_id: str = _NON_EMPTY
    qualification_nonce: str = _NON_EMPTY
    acquisition_request_id: str = _NON_EMPTY
    gap_id: str = _NON_EMPTY
    strategy_id: str = _NON_EMPTY
    acquisition_result: CapabilityAcquisitionResult
    correlation_id: str | None = None
    causation_id: str | None = None
    requested_at: datetime

    @field_validator(
        "qualification_request_id",
        "qualification_nonce",
        "acquisition_request_id",
        "gap_id",
        "strategy_id",
        "correlation_id",
        "causation_id",
    )
    @classmethod
    def _validate_ids(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label="id")

    @field_validator("requested_at")
    @classmethod
    def _validate_requested_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("requested_at must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def _validate_handoff_boundary(self) -> CapabilityQualificationRequest:
        expected_id = derive_capability_qualification_request_id(
            acquisition_request_id=self.acquisition_request_id,
            qualification_nonce=self.qualification_nonce,
        )
        if self.qualification_request_id != expected_id:
            raise ValueError(
                f"qualification_request_id must be derived identity {expected_id!r}",
            )
        result = self.acquisition_result
        if result.outcome is not CapabilityAcquisitionOutcome.SUCCEEDED:
            raise ValueError(
                "qualification requires acquisition outcome SUCCEEDED",
            )
        if result.request_id != self.acquisition_request_id:
            raise ValueError("acquisition_request_id must match acquisition_result")
        if result.gap_id != self.gap_id:
            raise ValueError("gap_id must match acquisition_result")
        if result.strategy_id != self.strategy_id:
            raise ValueError("strategy_id must match acquisition_result")
        if result.strategy_id is None:
            raise ValueError("SUCCEEDED acquisition_result requires strategy_id")
        if self.correlation_id != result.correlation_id:
            raise ValueError("correlation_id must match acquisition_result")
        if self.causation_id != result.causation_id:
            raise ValueError("causation_id must match acquisition_result")
        return self


__all__ = [
    "SCHEMA_CAPABILITY_QUALIFICATION_REQUEST_V1",
    "CapabilityQualificationRequest",
    "derive_capability_qualification_request_id",
]
