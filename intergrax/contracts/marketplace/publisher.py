# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Publisher presentation metadata for marketplace listings (Stage 11)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from intergrax.contracts.capability_catalog._validation import (
    normalize_optional_text,
    require_non_empty_text,
)

SCHEMA_MARKETPLACE_PUBLISHER_METADATA_V1: Final = "marketplace_publisher_metadata.v1"
_NON_EMPTY = Field(min_length=1)


class MarketplacePublisherMetadata(BaseModel):
    """Product-layer publisher profile — canonical identity remains provenance.publisher."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["marketplace_publisher_metadata.v1"] = (
        SCHEMA_MARKETPLACE_PUBLISHER_METADATA_V1
    )
    publisher_id: str = _NON_EMPTY
    display_name: str = _NON_EMPTY
    website_reference: str | None = None

    @field_validator("publisher_id", "display_name")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        return require_non_empty_text(value, label=str(info.field_name))

    @field_validator("website_reference")
    @classmethod
    def _validate_website_reference(cls, value: str | None) -> str | None:
        return normalize_optional_text(value, label="website_reference")
