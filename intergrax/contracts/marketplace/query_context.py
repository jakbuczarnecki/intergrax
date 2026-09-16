# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit marketplace query scope for tenant-aware discovery (ME-9)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text

SCHEMA_MARKETPLACE_QUERY_CONTEXT_V1: Final = "marketplace_query_context.v1"


class MarketplaceQueryContext(BaseModel):
    """Caller-authorized tenant scope for marketplace read surfaces.

    When ``tenant_id`` is absent, only PUBLIC listings are visible (fail-closed
    for TENANT_PRIVATE). This is not IAM — it carries no grant authority.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["marketplace_query_context.v1"] = (
        SCHEMA_MARKETPLACE_QUERY_CONTEXT_V1
    )
    tenant_id: str | None = None

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label="tenant_id")
