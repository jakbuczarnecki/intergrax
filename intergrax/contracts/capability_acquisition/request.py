# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability realization coordination request (UCA-2)."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.capability_realization_need import (
    CapabilityRealizationNeed,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind

SCHEMA_CAPABILITY_REALIZATION_REQUEST_V1: Final = "capability_realization_request.v1"
_NON_EMPTY = Field(min_length=1)


def derive_capability_realization_request_id(
    *,
    realization_need_id: str,
    request_nonce: str,
) -> str:
    """Deterministic request identity for idempotent coordination."""
    normalized_need = require_non_empty_text(
        realization_need_id,
        label="realization_need_id",
    )
    normalized_nonce = require_non_empty_text(request_nonce, label="request_nonce")
    return f"capability-realization-request:{normalized_need}:{normalized_nonce}"


class CapabilityRealizationRequest(BaseModel):
    """Generic realization dispatch envelope — no domain registries or installers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_realization_request.v1"] = (
        SCHEMA_CAPABILITY_REALIZATION_REQUEST_V1
    )
    request_id: str = _NON_EMPTY
    request_nonce: str = _NON_EMPTY
    realization_need: CapabilityRealizationNeed
    correlation_id: str | None = None
    causation_id: str | None = None
    host_profile_id: str | None = None
    requested_at: datetime

    @field_validator(
        "request_id",
        "request_nonce",
        "correlation_id",
        "causation_id",
        "host_profile_id",
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
    def _validate_request_identity(self) -> CapabilityRealizationRequest:
        expected = derive_capability_realization_request_id(
            realization_need_id=self.realization_need.realization_need_id,
            request_nonce=self.request_nonce,
        )
        if self.request_id != expected:
            raise ValueError(
                "request_id must be deterministic derived identity "
                f"{expected!r}",
            )
        return self

    @property
    def capability_kind(self) -> CapabilityKind:
        return self.realization_need.capability_identity.kind


__all__ = [
    "SCHEMA_CAPABILITY_REALIZATION_REQUEST_V1",
    "CapabilityRealizationRequest",
    "derive_capability_realization_request_id",
]
