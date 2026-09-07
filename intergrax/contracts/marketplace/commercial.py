# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Display-only commercial metadata for marketplace listings (Stage 11)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import (
    normalize_optional_text,
    require_non_empty_text,
)

SCHEMA_MARKETPLACE_COMMERCIAL_METADATA_V1: Final = "marketplace_commercial_metadata.v1"


class CommercialModel(StrEnum):
    """Display-only commercial classification — not execution or entitlement authority."""

    FREE = "free"
    PAID = "paid"
    CONTACT_VENDOR = "contact_vendor"
    INTERNAL = "internal"


class MarketplaceCommercialMetadata(BaseModel):
    """Read-only commercial presentation fields — never authoritative for runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["marketplace_commercial_metadata.v1"] = (
        SCHEMA_MARKETPLACE_COMMERCIAL_METADATA_V1
    )
    commercial_model: CommercialModel
    display_price: str | None = None
    minor_units: int | None = Field(default=None, ge=0)
    currency_code: str | None = None
    pricing_reference: str | None = None

    @field_validator("display_price", "pricing_reference")
    @classmethod
    def _validate_optional_text(cls, value: str | None, info: ValidationInfo) -> str | None:
        return normalize_optional_text(value, label=str(info.field_name))

    @field_validator("currency_code")
    @classmethod
    def _validate_currency_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = require_non_empty_text(value, label="currency_code")
        if normalized != normalized.upper():
            raise ValueError("currency_code must be uppercase")
        if len(normalized) != 3:
            raise ValueError("currency_code must be a 3-letter ISO-like code")
        return normalized

    @model_validator(mode="after")
    def _validate_commercial_constraints(self) -> MarketplaceCommercialMetadata:
        if self.minor_units is not None and self.currency_code is None:
            raise ValueError("currency_code is required when minor_units is provided")
        if self.commercial_model is CommercialModel.FREE:
            if self.minor_units is not None or self.display_price is not None:
                raise ValueError("FREE commercial_model must not include active price fields")
        if self.commercial_model is CommercialModel.INTERNAL:
            if self.minor_units is not None or self.display_price is not None:
                raise ValueError("INTERNAL commercial_model must not include active price fields")
        return self
