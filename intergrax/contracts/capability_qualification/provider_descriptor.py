# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qualification provider descriptor for selection (UCA-4)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text

SCHEMA_CAPABILITY_QUALIFICATION_PROVIDER_DESCRIPTOR_V1: Final = (
    "capability_qualification_provider_descriptor.v1"
)
_NON_EMPTY = Field(min_length=1)


class CapabilityQualificationProviderDescriptor(BaseModel):
    """Minimal metadata for governance-aware provider selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_qualification_provider_descriptor.v1"] = (
        SCHEMA_CAPABILITY_QUALIFICATION_PROVIDER_DESCRIPTOR_V1
    )
    provider_id: str = _NON_EMPTY

    @field_validator("provider_id")
    @classmethod
    def _validate_provider_id(cls, value: str) -> str:
        return require_non_empty_text(value, label="provider_id")


__all__ = [
    "SCHEMA_CAPABILITY_QUALIFICATION_PROVIDER_DESCRIPTOR_V1",
    "CapabilityQualificationProviderDescriptor",
]
