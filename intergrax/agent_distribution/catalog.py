# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog view contracts and CatalogSourceProvider port (AGENT_DISTRIBUTION §8)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution.identity import AgentPackageCandidate
from intergrax.agent_distribution._digest import normalize_optional_package_digest

_NON_EMPTY = Field(min_length=1)

SCHEMA_AGENT_CATALOG_ENTRY_V1: Final = "agent_catalog_entry.v1"
SCHEMA_CATALOG_SOURCE_IDENTITY_V1: Final = "catalog_source_identity.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class CatalogProviderKind(StrEnum):
    """Known catalog provider kinds — extensible via provider instance ids."""

    BUILTIN = "builtin"
    LOCAL_DEVELOPER = "local_developer"
    ENTERPRISE_PRIVATE = "enterprise_private"
    OFFICIAL_CATALOG = "official_catalog"
    GOVERNED_THIRD_PARTY = "governed_third_party"


class CatalogSourceIdentity(BaseModel):
    """Stable provider type + instance identity (§6.1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_CATALOG_SOURCE_IDENTITY_V1
    catalog_source_id: str = _NON_EMPTY
    provider_kind: CatalogProviderKind

    @field_validator("catalog_source_id")
    @classmethod
    def _validate_source_id(cls, value: str) -> str:
        return _strip_required(value)


class AgentCatalogVersionChannelRef(BaseModel):
    """Pointer to a resolvable version — not production authority alone."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version_label: str = _NON_EMPTY
    package_version: str | None = None
    package_digest: str | None = None

    @field_validator("version_label", "package_version")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    @field_validator("package_digest")
    @classmethod
    def _validate_optional_package_digest(cls, value: str | None) -> str | None:
        return normalize_optional_package_digest(value)


class AgentCatalogEntry(BaseModel):
    """Catalog metadata only — MUST NOT become execution truth (§6.3)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_CATALOG_ENTRY_V1
    catalog_entry_id: str = _NON_EMPTY
    catalog_source: CatalogSourceIdentity
    display_name: str = _NON_EMPTY
    publisher: str | None = None
    categories: tuple[str, ...] = ()
    package_id_line: str = _NON_EMPTY
    version_channel_refs: tuple[AgentCatalogVersionChannelRef, ...] = ()
    compatibility_summary: str | None = None
    trust_labels: tuple[str, ...] = ()

    @field_validator(
        "catalog_entry_id",
        "display_name",
        "package_id_line",
        "publisher",
        "compatibility_summary",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    @field_validator("categories", "trust_labels")
    @classmethod
    def _strip_tuple_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_strip_required(item) for item in value)


class CatalogPackageResolution(BaseModel):
    """Result of catalog package resolution — candidate identity + fetch locator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entry: AgentCatalogEntry
    package_candidate: AgentPackageCandidate
    artifact_locator: str = _NON_EMPTY

    @field_validator("artifact_locator")
    @classmethod
    def _validate_locator(cls, value: str) -> str:
        return _strip_required(value)


class CatalogEntryFilters(BaseModel):
    """Neutral catalog listing filters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    package_id_line: str | None = None
    category: str | None = None
    publisher: str | None = None


class ProviderHealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ProviderHealth(BaseModel):
    """Optional catalog provider health surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ProviderHealthStatus
    detail: str | None = None


class CatalogSourceProvider(Protocol):
    """Port for catalog discovery and package resolution (§8.1) — no implementation."""

    @property
    def catalog_source_id(self) -> str:
        """Stable provider instance id."""

    def list_entries(
        self,
        filters: CatalogEntryFilters | None = None,
    ) -> list[AgentCatalogEntry]:
        """List catalog entries for discoverability."""

    def resolve_package(
        self,
        entry: AgentCatalogEntry,
        *,
        version_selector: str,
    ) -> CatalogPackageResolution:
        """Resolve a catalog entry to a package candidate and artifact locator."""

    def health(self) -> ProviderHealth | None:
        """Optional provider health probe."""
