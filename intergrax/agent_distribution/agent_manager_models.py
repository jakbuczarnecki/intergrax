# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent Manager read model contracts — discovery + lifecycle projection only."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution.catalog import (
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.installation import InstallationState
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState

_NON_EMPTY = Field(min_length=1)

SCHEMA_AGENT_MANAGER_ENTRY_V1: Final = "agent_manager_entry.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class AgentManagerDerivedStatus(StrEnum):
    """Derived projection — not persisted lifecycle authority."""

    UNAVAILABLE = "unavailable"
    DISCOVERABLE = "discoverable"
    INSTALLED = "installed"
    BOUND = "bound"
    ENABLED = "enabled"
    READY_FOR_REVISION = "ready_for_revision"
    SERVING = "serving"
    DEGRADED = "degraded"


class LifecycleMatchResolution(StrEnum):
    """Catalog ↔ lifecycle join outcome — never guessed from display names."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"
    NOT_APPLICABLE = "not_applicable"


class AgentManagerIdentityView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manager_entry_id: str = _NON_EMPTY
    catalog_entry_id: str | None = None
    package_id_line: str | None = None
    display_name: str
    catalog_source: CatalogSourceIdentity | None = None
    publisher: str | None = None

    @field_validator("manager_entry_id", "display_name")
    @classmethod
    def _strip_required_fields(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator("catalog_entry_id", "package_id_line", "publisher")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class AgentManagerDiscoveryView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    categories: tuple[str, ...] = ()
    trust_labels: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    compatibility_summary: str | None = None
    provider_kind: CatalogProviderKind | None = None

    @field_validator("compatibility_summary")
    @classmethod
    def _strip_compatibility(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class AgentManagerLifecycleView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    match_resolution: LifecycleMatchResolution = LifecycleMatchResolution.NOT_APPLICABLE
    installation_state: InstallationState | None = None
    installed: bool = False
    bound: bool = False
    enabled_in_desired_state: bool = False
    logical_agent_id: str | None = None
    installation_id: str | None = None
    installation_slot_id: str | None = None
    application_binding_id: str | None = None
    distribution_package_id: str | None = None
    package_version: str | None = None
    package_digest: str | None = None

    @field_validator(
        "logical_agent_id",
        "installation_id",
        "installation_slot_id",
        "application_binding_id",
        "distribution_package_id",
        "package_version",
        "package_digest",
    )
    @classmethod
    def _strip_optional_ids(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class AgentManagerRuntimeView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    included_in_active_revision: bool = False
    included_in_candidate_revision: bool = False
    traffic_serving_revision_id: str | None = None
    pending_candidate_revision_id: str | None = None
    pending_candidate_revision_state: RuntimeRevisionState | None = None
    serving: bool = False

    @field_validator(
        "traffic_serving_revision_id",
        "pending_candidate_revision_id",
    )
    @classmethod
    def _strip_optional_revision_ids(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class AgentManagerAvailabilityView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    installable: bool = False
    bindable: bool = False
    activatable: bool = False


class AgentManagerEntry(BaseModel):
    """Immutable Agent Manager row for one catalog/lifecycle identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_MANAGER_ENTRY_V1
    derived_status: AgentManagerDerivedStatus
    identity: AgentManagerIdentityView
    discovery: AgentManagerDiscoveryView
    lifecycle: AgentManagerLifecycleView
    runtime: AgentManagerRuntimeView
    availability: AgentManagerAvailabilityView


class AgentManagerListFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    catalog_source_id: str | None = None
    provider_kind: CatalogProviderKind | None = None
    category: str | None = None
    publisher: str | None = None
    installed: bool | None = None
    bound: bool | None = None
    enabled: bool | None = None
    capability: str | None = None

    @field_validator("catalog_source_id", "category", "publisher", "capability")
    @classmethod
    def _strip_optional_filters(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class AgentManagerListScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY

    @field_validator("application_id", "application_environment_id")
    @classmethod
    def _strip_scope_ids(cls, value: str) -> str:
        return _strip_required(value)


class AgentManagerListResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[AgentManagerEntry, ...] = ()
    total: int = Field(ge=0)
    scope: AgentManagerListScope
    filters: AgentManagerListFilters | None = None
