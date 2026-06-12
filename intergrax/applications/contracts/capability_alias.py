# © Artur Czarnecki. All rights reserved.

"""Capability alias and descriptor contracts (APP-EVOL-3 · §49.3 · UAEP §42.27)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.runtime.task.task_metadata_keys import TaskMetadataKey

CAPABILITY_ALIAS_REDIRECT_KEY = TaskMetadataKey.CAPABILITY_ALIAS_REDIRECT


class CapabilityDescriptor(BaseModel):
    """UAEP §42.27 routing metadata for a versioned capability token."""

    model_config = ConfigDict(extra="forbid")

    capability: str = Field(min_length=1)
    version: str = "1.0.0"
    agent_id: str = ""
    contract_version: str = "1.0.0"
    deprecated: bool = False
    superseded_by: str | None = None


class CapabilityAlias(BaseModel):
    """Legacy capability redirect during a bounded migration window (§49.3.1)."""

    model_config = ConfigDict(extra="forbid")

    alias: str = Field(min_length=1)
    canonical: str = Field(min_length=1)
    effective_from: str | None = Field(
        default=None,
        description="UTC ISO-8601 — redirect active at or after this instant",
    )
    sunset_at: str | None = Field(
        default=None,
        description="UTC ISO-8601 — alias stops redirecting; STRICT intake blocks after sunset",
    )
    notice_ref: str | None = None
    migration_guide_ref: str | None = None

    @model_validator(mode="after")
    def _alias_differs_from_canonical(self) -> CapabilityAlias:
        if self.alias.strip() == self.canonical.strip():
            raise ValueError("alias and canonical must differ")
        return self


class CapabilityGovernanceProfile(BaseModel):
    """Environment-scoped capability deprecation and alias registry (APP-EVOL-3)."""

    model_config = ConfigDict(extra="forbid")

    aliases: list[CapabilityAlias] = Field(default_factory=list)
    minimum_alias_window_days: int = Field(default=14, ge=0)
