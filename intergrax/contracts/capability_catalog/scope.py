# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Discovery scope contract (CAPABILITY-CATALOG-1 Stage 3)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import (
    normalize_optional_text,
    require_non_empty_text,
)

SCHEMA_CAPABILITY_DISCOVERY_SCOPE_V1: Final = "capability_discovery_scope.v1"


class CapabilityDiscoveryScopeMode(StrEnum):
    """Explicit scope boundary — no implicit global discovery."""

    ENTERPRISE = "enterprise"
    GLOBAL = "global"


class CapabilityDiscoveryScope(BaseModel):
    """Scope parameters for enterprise or explicit global discovery paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_discovery_scope.v1"] = (
        SCHEMA_CAPABILITY_DISCOVERY_SCOPE_V1
    )
    mode: CapabilityDiscoveryScopeMode = CapabilityDiscoveryScopeMode.ENTERPRISE
    organization_id: str | None = None
    tenant_id: str | None = None
    application_id: str | None = None
    work_context_id: str | None = None

    @field_validator("organization_id", "tenant_id", "application_id")
    @classmethod
    def _validate_required_scope_ids(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label="scope identifier")

    @field_validator("work_context_id")
    @classmethod
    def _validate_work_context_id(cls, value: str | None) -> str | None:
        return normalize_optional_text(value, label="work_context_id")

    @model_validator(mode="after")
    def _validate_mode_scope_consistency(self) -> CapabilityDiscoveryScope:
        enterprise_ids = (
            self.organization_id,
            self.tenant_id,
            self.application_id,
        )
        if self.mode is CapabilityDiscoveryScopeMode.ENTERPRISE:
            if any(item is None for item in enterprise_ids):
                raise ValueError(
                    "enterprise discovery scope requires organization_id, "
                    "tenant_id, and application_id",
                )
            return self
        if any(item is not None for item in enterprise_ids):
            raise ValueError(
                "global discovery scope must not include organization_id, "
                "tenant_id, or application_id",
            )
        return self
