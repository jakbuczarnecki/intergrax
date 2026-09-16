# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace listing visibility contract (ME-9)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text

SCHEMA_MARKETPLACE_VISIBILITY_V1: Final = "marketplace_visibility.v1"


class MarketplaceVisibilityScope(StrEnum):
    """Explicit marketplace discovery scope — not entitlement or governance."""

    PUBLIC = "public"
    TENANT_PRIVATE = "tenant_private"


class MarketplaceVisibility(BaseModel):
    """Typed visibility metadata for a marketplace listing projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["marketplace_visibility.v1"] = SCHEMA_MARKETPLACE_VISIBILITY_V1
    scope: MarketplaceVisibilityScope
    tenant_id: str | None = None

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label="tenant_id")

    @model_validator(mode="after")
    def _validate_scope_consistency(self) -> MarketplaceVisibility:
        if self.scope is MarketplaceVisibilityScope.PUBLIC:
            if self.tenant_id is not None:
                raise ValueError(
                    "PUBLIC marketplace visibility must not include tenant_id",
                )
            return self
        if self.tenant_id is None:
            raise ValueError(
                "TENANT_PRIVATE marketplace visibility requires tenant_id",
            )
        return self
