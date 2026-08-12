# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Effective roster projection contracts (AGENT_DISTRIBUTION §13–§14)."""

from __future__ import annotations

from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution._digest import content_digest_for_model
from intergrax.agent_distribution.binding import AgentBindingFactoryReference, AgentBindingPolicyOverrides

_NON_EMPTY = Field(min_length=1)

SCHEMA_EFFECTIVE_ROSTER_V1: Final = "effective_roster.v1"
SCHEMA_EFFECTIVE_ROSTER_ENTRY_V1: Final = "effective_roster_entry.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class EffectiveRosterEntry(BaseModel):
    """One merged roster row — derived only, not durable SoT (§13.4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_EFFECTIVE_ROSTER_ENTRY_V1
    logical_agent_id: str = _NON_EMPTY
    installation_slot_id: str = _NON_EMPTY
    active_installation_id: str | None = None
    package_digest: str = _NON_EMPTY
    distribution_package_id: str = _NON_EMPTY
    effective_enablement: bool
    merged_config: dict[str, Any] = Field(default_factory=dict)
    secret_refs: tuple[str, ...] = ()
    policy_overrides: AgentBindingPolicyOverrides | None = None
    factory_reference: AgentBindingFactoryReference | None = None
    application_binding_id: str | None = None
    manifest_origin_ref: str | None = None

    @field_validator(
        "logical_agent_id",
        "installation_slot_id",
        "active_installation_id",
        "package_digest",
        "distribution_package_id",
        "application_binding_id",
        "manifest_origin_ref",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class EffectiveRoster(BaseModel):
    """Derived roster for dependency resolution and graph build (§13.4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_EFFECTIVE_ROSTER_V1
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    manifest_release_id: str = _NON_EMPTY
    binding_revisions: tuple[int, ...] = ()
    entries: tuple[EffectiveRosterEntry, ...]
    effective_roster_revision_id: str | None = None

    @field_validator("application_id", "application_environment_id", "manifest_release_id")
    @classmethod
    def _strip_required_fields(cls, value: str) -> str:
        return _strip_required(value)

    def compute_revision_id(self) -> str:
        """Content-addressed roster revision identity (§6.1)."""
        payload = self.model_copy(update={"effective_roster_revision_id": None})
        return content_digest_for_model(payload)

    def with_revision_id(self) -> EffectiveRoster:
        """Return roster with computed effective_roster_revision_id."""
        return self.model_copy(update={"effective_roster_revision_id": self.compute_revision_id()})
