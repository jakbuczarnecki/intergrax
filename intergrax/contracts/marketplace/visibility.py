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
    ORGANIZATION_PRIVATE = "organization_private"


class MarketplaceVisibility(BaseModel):
    """Typed visibility metadata for a marketplace listing projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["marketplace_visibility.v1"] = SCHEMA_MARKETPLACE_VISIBILITY_V1
    scope: MarketplaceVisibilityScope
    tenant_id: str | None = None
    organization_id: str | None = None

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label="tenant_id")

    @field_validator("organization_id")
    @classmethod
    def _validate_organization_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label="organization_id")

    @model_validator(mode="after")
    def _validate_scope_consistency(self) -> MarketplaceVisibility:
        has_tenant = self.tenant_id is not None
        has_org = self.organization_id is not None
        if has_tenant and has_org:
            raise ValueError(
                "marketplace visibility must not set tenant_id and organization_id together",
            )
        if self.scope is MarketplaceVisibilityScope.PUBLIC:
            if has_tenant:
                raise ValueError(
                    "PUBLIC marketplace visibility must not include tenant_id",
                )
            if has_org:
                raise ValueError(
                    "PUBLIC marketplace visibility must not include organization_id",
                )
            return self
        if self.scope is MarketplaceVisibilityScope.TENANT_PRIVATE:
            if not has_tenant:
                raise ValueError(
                    "TENANT_PRIVATE marketplace visibility requires tenant_id",
                )
            if has_org:
                raise ValueError(
                    "TENANT_PRIVATE marketplace visibility must not include organization_id",
                )
            return self
        if self.scope is MarketplaceVisibilityScope.ORGANIZATION_PRIVATE:
            if not has_org:
                raise ValueError(
                    "ORGANIZATION_PRIVATE marketplace visibility requires organization_id",
                )
            if has_tenant:
                raise ValueError(
                    "ORGANIZATION_PRIVATE marketplace visibility must not include tenant_id",
                )
            return self
        return self
