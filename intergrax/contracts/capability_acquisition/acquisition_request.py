# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability acquisition coordination request (UCA-3)."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.capability_gap import CapabilityGap
from intergrax.contracts.capability_catalog.need import CapabilityNeed

SCHEMA_CAPABILITY_ACQUISITION_REQUEST_V1: Final = "capability_acquisition_request.v1"
_NON_EMPTY = Field(min_length=1)


def derive_capability_acquisition_request_id(
    *,
    gap_id: str,
    request_nonce: str,
) -> str:
    """Deterministic request identity for idempotent acquisition coordination."""
    normalized_gap = require_non_empty_text(gap_id, label="gap_id")
    normalized_nonce = require_non_empty_text(request_nonce, label="request_nonce")
    return f"capability-acquisition-request:{normalized_gap}:{normalized_nonce}"


class CapabilityAcquisitionRequest(BaseModel):
    """Generic acquisition dispatch envelope — anchored on canonical CapabilityGap."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_acquisition_request.v1"] = (
        SCHEMA_CAPABILITY_ACQUISITION_REQUEST_V1
    )
    request_id: str = _NON_EMPTY
    request_nonce: str = _NON_EMPTY
    capability_gap: CapabilityGap
    capability_need: CapabilityNeed | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    requested_at: datetime

    @field_validator(
        "request_id",
        "request_nonce",
        "correlation_id",
        "causation_id",
    )
    @classmethod
    def _validate_optional_ids(cls, value: str | None) -> str | None:
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
    def _validate_request_identity(self) -> CapabilityAcquisitionRequest:
        expected = derive_capability_acquisition_request_id(
            gap_id=self.capability_gap.gap_id,
            request_nonce=self.request_nonce,
        )
        if self.request_id != expected:
            raise ValueError(
                f"request_id must be deterministic derived identity {expected!r}",
            )
        if self.capability_need is not None:
            need_id = self.capability_need.need_id
            if need_id is not None and need_id != self.capability_gap.need_id:
                raise ValueError(
                    "capability_need.need_id must match capability_gap.need_id",
                )
        return self


__all__ = [
    "SCHEMA_CAPABILITY_ACQUISITION_REQUEST_V1",
    "CapabilityAcquisitionRequest",
    "derive_capability_acquisition_request_id",
]
